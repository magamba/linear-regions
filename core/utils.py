# -*- coding: utf-8 -*-

from torch import set_default_dtype, set_default_tensor_type, float64, manual_seed
from torch.cuda import DoubleTensor, FloatTensor
import torch
import numpy as np
from torch import Tensor
from torch.jit.annotations import List, Tuple
from os import environ
from multiprocessing import set_start_method
import random
import logging

logger = logging.getLogger(__name__)

def use_multiprocessing(cmd_args):
    if torch.cuda.device_count() > 1:
        return True
    if cmd_args.e_workers > 1 and cmd_args.e_device == "cpu":
        return True
    return False

def init_torch(double_precision=False, cmd_args=None):
    logger.info("Initializing torch")
    logger.info("double_precision={}".format(double_precision))
    if cmd_args is not None:
        logger.info("e_device={}".format(cmd_args.e_device))
    if double_precision:
        set_default_dtype(float64)
#    if cmd_args and cmd_args.e_device != "cpu":
#        if double_precision:
#            set_default_tensor_type(DoubleTensor)
#        else:
#            set_default_tensor_type(FloatTensor)


def init_prngs(cmd_args):
    torch.manual_seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    random.seed(cmd_args.seed)
    if cmd_args.e_device != "cpu":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if "PYTHONHASHSEED" not in environ:
        logger.warn(
            "PYTHONHASHSEED is not defined, this may cause reproducibility issues"
        )


def init_logging(logger_name, logfile, log_level: str, cmd_args):
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: %s" % log_level)

    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")

    f_handler = logging.FileHandler(logfile)
    f_handler.setLevel(numeric_level)
    c_handler = logging.StreamHandler()
    c_handler.setLevel(numeric_level)
    f_handler.setFormatter(formatter)
    c_handler.setFormatter(formatter)
    
    logging.basicConfig(level=numeric_level, handlers=[f_handler, c_handler])
    logger = logging.getLogger(logger_name)

    if use_multiprocessing(cmd_args):
        from multiprocessing_logging import install_mp_handler
        install_mp_handler(logger)

    return logger


def get_set_start_method(cmd_opts):
    def do_set_start_method(set_method):
        logger.info("Setting start_method to {}".format(cmd_opts.start_method))
        set_method(cmd_opts.start_method)

    if cmd_opts.e_device == "cpu":
        return lambda: do_set_start_method(set_start_method)
    return lambda: do_set_start_method(torch.multiprocessing.set_start_method)


def log_dir_base_args(base, name=None):
    if name:
        base = "{}/{}".format(base, name)
    return base


def log_dir_base(cmd_args):
    return log_dir_base_args(cmd_args.e_save_dir, cmd_args.e_name)


def log_additions(additions, l1reg, l2reg, augmentation, dropout=None, weight_decay=None):
    additions_list = [ item + "_" + str(dropout) if (item == "dropout" and dropout is not None) else item for item in additions ]
    name = "-".join(additions_list)
    if l1reg:
        name = "{}-l1-{:0.5f}".format(name, l1reg)
    if l2reg:
        name = "{}-l2-{:0.5f}".format(name, l2reg)
    if augmentation:
        name = "{}-{}".format(name, "augmentation")
    if weight_decay is not None:
        name = "{}-{}".format(name, "wd_" + str(weight_decay))
    if name == "":
        return "default"
    return name.strip("-")


def all_log_dir_args(
    base,
    model,
    data,
    label_noise,
    seed,
    l1reg=None,
    l2reg=None,
    name=None,
    augmentation=None,
    model_additions=(),
    dropout=0.,
    weight_decay=0.
):
    if weight_decay == 0.:
        weight_decay = None
    if dropout == 0.:
        dropout = None
    dir_name = "{}/{}/{}/{}".format(
        log_dir_base_args(base, name),
        model,
        data,
        log_additions(model_additions, l1reg, l2reg, augmentation, dropout, weight_decay),
    )
    sub_dir_name = ""
    if label_noise != 0:
        sub_dir_name = "noise-{:.4f}".format(label_noise)
    if seed is not None:
        sub_dir_name = "-".join([sub_dir_name, "seed-{}".format(seed)])
    else:
        sub_dir_name = "-".join([sub_dir_name, "no_seed"])
    return "{}/{}".format(dir_name, sub_dir_name.strip("-"))


def all_log_dir(cmd_opts):
    return all_log_dir_args(
        cmd_opts.e_save_dir,
        cmd_opts.model,
        cmd_opts.data,
        cmd_opts.label_noise,
        cmd_opts.seed,
        cmd_opts.l1_regularization,
        cmd_opts.l2_regularization,
        cmd_opts.e_name,
        cmd_opts.augmentation,
        cmd_opts.model_additions,
        cmd_opts.dropout,
        cmd_opts.weight_decay
    )

def prepare_dirs(args):
    """Prepare directories to store results and logs"""
    import os
    
    logs_path = all_log_dir(args)
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)
    logfile = os.path.join(logs_path, args.logfile)
    # configure root logger
    init_logging(None, logfile, args.log_level, args)
    return logs_path



def checkpoint_dir_args(
    base,
    model,
    data,
    label_noise,
    seed,
    l1reg=None,
    l2reg=None,
    name=None,
    augmentation=None,
    model_additions=(),
    dropout=0.,
    weight_decay=0.
):
    return "{}/checkpoints".format(
        all_log_dir_args(
            base, model, data, label_noise, seed, l1reg, l2reg, name, augmentation, model_additions, dropout, weight_decay
        )
    )


def checkpoint_dir(cmd_opts):
    return checkpoint_dir_args(
        cmd_opts.e_save_dir,
        cmd_opts.model,
        cmd_opts.data,
        cmd_opts.label_noise,
        cmd_opts.seed,
        cmd_opts.l1_regularization,
        cmd_opts.l2_regularization,
        cmd_opts.e_name,
        cmd_opts.augmentation,
        cmd_opts.model_additions,
        cmd_opts.dropout,
        cmd_opts.weight_decay
    )


def global_iteration_from_engine(engine):
    def _wrap_global_step(engine_, event_name_):
        return engine.state.iteration

    return _wrap_global_step


def normalize(tensor: Tensor, mean: List[float], std: List[float], inplace: bool = False) -> Tensor:
    """Normalize a tensor image with mean and standard deviation.

    .. note::
        This transform acts out of place by default, i.e., it does not mutates the input tensor.

    See :class:`~torchvision.transforms.Normalize` for more details.

    Args:
        tensor (Tensor): Tensor image of size (C, H, W) or (B, C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation inplace.

    Returns:
        Tensor: Normalized Tensor image.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError('Input tensor should be a torch tensor. Got {}.'.format(type(tensor)))

    if tensor.ndim < 3:
        raise ValueError('Expected tensor to be a tensor image of size (..., C, H, W). Got tensor.size() = '
                         '{}.'.format(tensor.size()))

    if not inplace:
        tensor = tensor.clone()

    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    if (std == 0).any():
        raise ValueError('std evaluated to zero after conversion to {}, leading to division by zero.'.format(dtype))
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    tensor.sub_(mean).div_(std)
    return tensor

