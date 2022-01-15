# -*- coding: utf-8 -*-

"""
Compute density and absolute deviation on a trained model loaded from one or more checkpoints.
"""

import torch
from torch.multiprocessing import set_start_method as torch_set_start_method

import sys
import os
from zipfile import ZipFile
from functools import partial
from multiprocessing import set_start_method, get_start_method
import logging

from core.cmd import create_default_args
from core.data import create_data_manager, DATASET_INFO_MAP
from core.utils import (
    init_torch,
    all_log_dir,
    checkpoint_dir,
    get_set_start_method,
    init_logging,
    init_prngs,
    prepare_dirs
)
from core import strategies
from models import factory
from core.metrics import Metrics, create_metric

logger = logging.getLogger(__name__)


def checkpointed_models(cmd_args, reverse=False, index_by_ordinal=True, device="cpu"):
    checkpoints_loc = "{}.zip".format(checkpoint_dir(cmd_args))

    def step_key(zip_info):
        return int(zip_info.filename.split("_")[1].split(".")[0])

    with ZipFile(checkpoints_loc) as checkpoints_zip:
        zip_infos = [zi for zi in checkpoints_zip.infolist() if zi.file_size > 0]
        for cur_key, zip_info in enumerate(sorted(zip_infos, key=step_key, reverse=reverse)):
            if not index_by_ordinal:
                cur_key = step_key(zip_info)
            if (
                cmd_args.l_load_checkpoints is not None
                and cur_key not in cmd_args.l_load_checkpoints
            ):
                continue
            with checkpoints_zip.open(zip_info.filename) as saved_file:
                saved_obj = torch.load(
                    saved_file, map_location=torch.device(device)
                )
                net = factory.create_model(
                    cmd_args.model, cmd_args.data, additions=cmd_args.model_additions, dropout_rate=cmd_args.dropout
                )
                net.load_state_dict(saved_obj["model"])
                net = net.to(device=device, dtype=torch.get_default_dtype())
                net.eval()
                yield cur_key, net


def get_metrics(args, data_manager):
    metrics = []
    for gen_strategy in args.l_gen_strategy:
        logger.debug("Initializing strategy: {}".format(gen_strategy))
        if "closed-path" in gen_strategy:
            split, data_source = strategies.GEN_STRATEGIES[gen_strategy](data_manager)
            metrics.append(
                create_metric(
                    "absolute_deviation", {split: data_source}, opts=args
                )
            )            
        else:
            raise ValueError("Unknown data generation strategy: {}".format(gen_strategy))

    return metrics


def compute_stats(metrics, args, device):
    checkpointed_models_gen = checkpointed_models(args, reverse=False, device=device)

    for step, model in checkpointed_models_gen:
        logger.info("Computing metrics for step {}".format(step))
        for metric in metrics:
            metric(model, step)
        logger.info("Done with metrics for step {}".format(step))
    return metrics


def add_local_args(parser):
    opt_group = parser.add_argument_group("compute_stats local")
    opt_group.add_argument(
        "--l_gen-strategy",
        default=None,
        choices=tuple(strategies.GEN_STRATEGIES.keys()),
        nargs="*",
        help="Strategy used for generating paths, based on the dataset split used.",
    )
    opt_group.add_argument(
        "--l_out-name",
        type=str,
        default="stats",
        help="Base filename for saving results",
    )
    opt_group.add_argument(
        "--l_num-paths",
        default=2,
        type=int,
        help="Number of paths to generate. It should be divisible by BATCH_SIZE, and its maximum value is the size of the dataset split considered.",
    )    
    opt_group.add_argument(
        "--l_buff-size",
        type=int,
        default=30000,
        help="Length of each preallocated CUDA buffer, used for storing density-dependent metrics (default = 30000).",
    )
    
    opt_group.add_argument(
        "--l_num-anchors",
        type=int,
        default=8,
        help="Number of 'anchor points', i.e. image transformation used to anchor a path to the support of the data distribution.",
    )
    opt_group.add_argument(
        "--l_closed-path-radius",
        type=int,
        default=4,
        help="The radius of the image transformation used to generate anchor points for each path [default = 4].",
    )
    opt_group.add_argument(
        "--l_load-checkpoints",
        type=int,
        nargs="*",
        default=None,
        help="Optional. Count regions only using the specified checkpoints, identified by step number [default = use all available checkpoints]."
        "NOTE: if the specified checkpoint is not valid, it will not be loaded. "
        "To find available checkpoints run with the --l_get-checkpoints.",
    )
    opt_group.add_argument(
        "--l_get-checkpoints",
        default=False,
        action="store_true",
        help="Print all available checkpoints and exit.",
    )

    opt_group.add_argument(
        "--l_skip-paths",
        default=0,
        type=int,
        help="Skip the first L_SKIP_PATHS paths [default = 0]. Useful when distributing workloads over several parallel jobs."
    )
    
    opt_group.add_argument(
        "--l_path-sample-seed",
        default=4321,
        type=int,
        help="Seed for sampling data points to generate paths. The seed is used to shuffle the dataset split where paths are generated from, to ensure reproducibility."
    )

    opt_group.add_argument(
        "--l_open-path",
        action="store_true",
        default=False,
        help="Generate open paths using weak augmentations."
    )

    opt_group.add_argument(
        "--l_random-dataset",
        action="store_true",
        default=False,
        help="Use in combination with DATA. Generate random dataset with the same pixel-wise statistics as DATA.",


def main(args, logs_path):
    logger.info(args)
    if __name__ == "__main__":
        try:
            get_set_start_method(cmd_opts)()
        except RuntimeError:
            pass
    else:
        logger.warn("Defaulting to start_method = {}".format(get_start_method()))

    init_torch(cmd_args=args, double_precision=True)
    init_prngs(args)

    if torch.cuda.is_available() and args.e_device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.l_get_checkpoints:
        all_available_checkpoints = [
            step_key for (step_key, _) in checkpointed_models(args, reverse=False)
        ]
        logger.info("Available checkpoints in {} are:".format(all_log_dir(args)))
        logger.info(all_available_checkpoints)
        sys.exit(0)

    tv_split = (None, None)
    if args.train_split and args.val_split:
        tv_split = (args.train_split, args.val_split)

    data_manager = create_data_manager(
        args,
        args.label_noise,
        seed=args.label_seed,
        train_validation_split=tv_split,
        normalize=True,
        gen_paths=True,
    )

    metrics = get_metrics(args, data_manager)
    metrics = compute_stats(metrics, args, device)
    logger.info("All metrics computed.")


def get_args():
    parser = create_default_args()
    add_local_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    logs_path = prepare_dirs(args)
    logger = logging.getLogger(__name__)
    main(args, logs_path)
