# -*- coding: utf-8 -*-

"""
Training script
"""

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import MultiplicativeLR
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.utils import setup_logger
from ignite.handlers import Checkpoint, DiskSaver
from ignite.contrib.handlers.param_scheduler import LinearCyclicalScheduler, LRScheduler
from ignite.contrib.handlers.tensorboard_logger import (
    TensorboardLogger,
    OutputHandler,
    OptimizerParamsHandler,
)

from shutil import make_archive, rmtree
import sys
import os
from os import path, environ
from typing import Union, List, Tuple, Callable
import zipfile
import pickle
import logging

from core.cmd import create_default_args
from core.utils import (
    init_torch,
    global_iteration_from_engine,
    all_log_dir,
    checkpoint_dir,
    init_logging,
    init_prngs,
    prepare_dirs
)
from core.data import create_data_manager
from models import factory as model_factory
from interface.handlers import StopOnInterpolateByAccuracy, StopOnInterpolateByLoss

logger = logging.getLogger(__name__)


def setup_tb_logger(
    args, trainer, evaluator, test_evaluator, val_evaluator, optimizer
):
    with SummaryWriter(log_dir=all_log_dir(args)) as writer:
        writer.add_text("data", args.data)
        writer.add_text("model", args.model)
        writer.add_text("model_additions", ",".join(args.model_additions))
        writer.add_text("learning_rate", str(args.learning_rate))
        writer.add_text("batch_size", str(args.batch_size))
        writer.add_text("epochs", str(args.epochs))
        writer.add_text("seed", str(args.seed))
        writer.add_text("optimizer", args.optimizer)
        writer.add_text("early_exit_accuracy", str(args.early_exit_accuracy))
        writer.add_text("dropout", str(args.dropout))
        writer.add_text("early_exit_loss", str(args.early_exit_loss))
        writer.add_text("lr_decay_rate", str(args.lr_decay_rate))
        writer.add_text("lr_step", str(args.lr_step))
        writer.add_text("stop_by_loss_threshold", str(args.stop_by_loss_threshold))
        writer.add_text(
            "stop_by_accuracy_threshold", str(args.stop_by_accuracy_threshold)
        )

        writer.add_text("E_DEVICE", args.e_device)
        writer.add_text("E_NAME", args.e_name)
        writer.add_text("E_SAVE_DIR", args.e_save_dir)
        writer.add_text("E_DATA_DIR", args.e_data_dir)
        writer.add_text("E_WORKERS", str(args.e_workers))

    logger = TensorboardLogger(log_dir=all_log_dir(args))
    logger.attach(
        trainer,
        log_handler=OutputHandler(
            tag="train",
            output_transform=lambda loss: {"batch_loss": loss},
            metric_names="all",
        ),
        event_name=Events.ITERATION_COMPLETED(every=args.l_loss_checkpoint),
    )

    logger.attach(
        evaluator,
        log_handler=OutputHandler(
            tag="train",
            metric_names=["loss", "accuracy"],
            global_step_transform=global_iteration_from_engine(trainer),
        ),
        event_name=Events.COMPLETED,
    )
    logger.attach(
        test_evaluator,
        log_handler=OutputHandler(
            tag="test",
            metric_names=["loss", "accuracy"],
            global_step_transform=global_iteration_from_engine(trainer),
        ),
        event_name=Events.COMPLETED,
    )
    if val_evaluator is not None:
        logger.attach(
            val_evaluator,
            log_handler=OutputHandler(
                tag="validation",
                metric_names=["loss", "accuracy"],
                global_step_transform=global_iteration_from_engine(trainer),
            ),
            event_name=Events.COMPLETED,
        )
    logger.attach(
        trainer,
        log_handler=OptimizerParamsHandler(optimizer),
        event_name=Events.ITERATION_STARTED,
    )

    return logger


def every_and_n_times(every, n):
    if every is None:
        every = 1
    if n is None:
        n = float("inf")

    def wrap_every_and_n_times(engine_, event_num):
        times_called = wrap_every_and_n_times.times_called
        wrap_every_and_n_times.times_called += 1
        if event_num % every == 0 and times_called < n:
            return True
        return False

    wrap_every_and_n_times.times_called = 0
    return wrap_every_and_n_times


def _every_or_specified(save_checkpoints: Union[List, Tuple]) -> Callable:
    def _never_event_filter(engine_, step):
        return False

    def _every_event_filter(engine_, step):
        return step % save_checkpoints[0] == 0

    def _specified_event_filter(engine_, step):
        if step in save_checkpoints:
            return True
        return False

    if type(save_checkpoints) == int:
        save_checkpoints = (save_checkpoints,)

    if len(save_checkpoints) == 0:
        return _never_event_filter
    elif len(save_checkpoints) == 1:
        return _never_event_filter if save_checkpoints[0] == 0 else _every_event_filter
    return _specified_event_filter


def run(args, data_manager, model, device):
    data_loader = data_manager.dloader
    num_iterations = len(data_loader)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
    )

    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    if args.l1_regularization:
        loss_fn = loss_fn_with_regularization(
            loss_fn, model, args.l1_regularization, 1
        )
    if args.l2_regularization:
        loss_fn = loss_fn_with_regularization(
            loss_fn, model, args.l2_regularization, 2
        )
    trainer = create_supervised_trainer(model, optimizer, loss_fn, device=device)
    evaluator = create_supervised_evaluator(
        model, metrics={"accuracy": Accuracy(), "loss": Loss(loss_fn)}, device=device
    )
    test_evaluator = create_supervised_evaluator(
        model, metrics={"accuracy": Accuracy(), "loss": Loss(loss_fn)}, device=device
    )
    val_evaluator = None
    if data_manager.vloader is not None:
        val_evaluator = create_supervised_evaluator(
            model,
            metrics={"accuracy": Accuracy(), "loss": Loss(loss_fn)},
            device=device,
        )
    trainer.logger = setup_logger("trainer", level=logging.INFO)

    def compute_metrics(engine_):
        evaluator.run(data_loader)
        test_evaluator.run(data_manager.tloader)
        if data_manager.vloader is not None:
            val_evaluator.run(data_manager.vloader)

    trainer.add_event_handler(
        Events.ITERATION_COMPLETED(every=args.eval_every * num_iterations), compute_metrics
    )
    trainer.add_event_handler(Events.COMPLETED, compute_metrics)
    
    if args.early_exit_accuracy:
        evaluator.add_event_handler(
            Events.COMPLETED,
            StopOnInterpolateByAccuracy(threshold=args.stop_by_accuracy_threshold),
            trainer,
        )
    elif args.early_exit_loss:
        evaluator.add_event_handler(
            Events.COMPLETED,
            StopOnInterpolateByLoss(threshold=args.stop_by_loss_threshold),
            trainer,
        )
    lr_scheduler = None

    def _lr_mult(epoch):
        if args.lr_step == 0:
            return 1  # constant lr
        if (epoch % args.lr_step == 0) or (
            epoch % int(args.epochs * 0.75) == 0
        ):
            return args.lr_decay_rate
        return 1

    lr_scheduler = LRScheduler(MultiplicativeLR(optimizer, lr_lambda=_lr_mult))
    trainer.add_event_handler(Events.EPOCH_COMPLETED, lr_scheduler)

    tb_logger = setup_tb_logger(
        args, trainer, evaluator, test_evaluator, val_evaluator, optimizer
    )

    objects_to_checkpoint = {"trainer": trainer, "model": model, "optimizer": optimizer}
    if lr_scheduler:
        objects_to_checkpoint["lr_scheduler"] = lr_scheduler
    training_checkpoint = Checkpoint(
        to_save=objects_to_checkpoint,
        save_handler=DiskSaver(checkpoint_dir(args), require_empty=False),
        n_saved=None,
    )

    trainer.add_event_handler(
        Events.ITERATION_COMPLETED(
            event_filter=_every_or_specified(
                tuple(it * num_iterations for it in args.l_save_checkpoint)
            )
        ),
        training_checkpoint,
    )

    if args.l_resume_from is not None:
        logger.info("Starting from previous checkpoint")
        chkpt_dir = checkpoint_dir(args)
        chkpt_zip = "{}.zip".format(chkpt_dir)
        if os.path.isfile(chkpt_zip):
            load_from_zipped_checkpoint(
                objects_to_checkpoint, chkpt_zip, args.l_resume_from
            )
        else:
            load_from_checkpoint(
                objects_to_checkpoint, chkpt_dir, args.l_resume_from
            )
    logger.info("Running")
    trainer.add_event_handler(Events.EPOCH_STARTED(once=1), training_checkpoint)
    logger.info("Starting training")
    trainer.run(data_loader, max_epochs=args.epochs)
    # save a checkpoint at the end. if the accuracy is 100%, @run will be stopped
    # and the current model may or may not be saved
    logger.info("Stopped training")
    training_checkpoint(trainer)

    tb_logger.close()


def load_from_checkpoint(objects_to_checkpoint, chkpt_dir, resume_from):
    checkpoint = torch.load("{}/checkpoint_{}.pth".format(chkpt_dir, resume_from))
    Checkpoint.load_objects(to_load=objects_to_checkpoint, checkpoint=checkpoint)


def load_from_zipped_checkpoint(objects_to_checkpoint, chkpt_zip, resume_from):
    with zipfile.ZipFile(chkpt_zip, "r") as zdir:
        with zdir.open("checkpoint_{}.pth".format(resume_from)) as chkpt:
            checkpoint = torch.load(chkpt)
            Checkpoint.load_objects(
                to_load=objects_to_checkpoint, checkpoint=checkpoint
            )


def compress_checkpoints(args):
    compress_dir = checkpoint_dir(args)
    make_archive(compress_dir, "zip", compress_dir)
    rmtree(compress_dir)


def add_local_args(parser):
    opt_group = parser.add_argument_group("train local")
    opt_group.add_argument(
        "--l_resume-from",
        type=int,
        default=None,
        help="Checkpoint to resume training from, if interrupted previously.",
    )
    opt_group.add_argument(
        "--l_save-checkpoint",
        nargs="*",
        default=(0,),
        type=int,
        help="When to save model checkpoints to disk, expressed in epochs. By default a checkpoint of the model at initialization and at convergence are saved. If one value is specified, it denotes the checkpoint frequency. If multiple values are given, they are used as explicit checkpoints.",
    )
    # @deprecate
    opt_group.add_argument("--l_accuracy-threshold", type=float, default=1.0)
    opt_group.add_argument(
        "--l_loss-checkpoint",
        type=int,
        default=382,
        help="Report/save the batch loss every L_LOSS-CHECKPOINT iterations.",
    )
    opt_group.add_argument(
        "--l_raw-checkpoints",
        help="If true, do not zip the model checkpoints into one zip file.",
    )


def main(args):
    logger = logging.getLogger(__name__)

    logger.info(args)
    init_torch(cmd_args=args, double_precision=False)
    init_prngs(args)
    model = model_factory.create_model(
        args.model, args.data, additions=args.model_additions, dropout_rate=args.dropout
    )
    
    if torch.cuda.is_available() and args.e_device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    model.train()
    model.to(device)
    tv_split = (None, None)
    if args.train_split and args.val_split:
        tv_split = (args.train_split, args.val_split)
    logger.info("Train val split {}".format(tv_split))
    run(
        args,
        create_data_manager(
            args,
            args.label_noise,
            augment=args.augmentation,
            seed=args.label_seed,
            train_validation_split=tv_split,
        ),
        model,
        device
    )
    if not args.l_raw_checkpoints:
        compress_checkpoints(args)


def get_args():
    parser = create_default_args()
    add_local_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    import signal
    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

    args = get_args()
    prepare_dirs(args)
    main(args)
