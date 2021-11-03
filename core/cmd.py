# -*- coding: utf-8 -*-

"""
A series of common functions for dealing with command line arguments. The naming
convention used is the following:
    * arguments which may be read from the bash environment are prefixed with "e_"
      and when passed via the environment with "E_". (e.g. --e_device or E_DEVICE).
    * arguments which are shared across scripts (reused) have no prefix (e.g.
      --batch-size)
    * arguments which are only used by one script are prefixed with "l_" (e.g.
      --l_resume-from)

The precedence for arguments is
    environment --> run_configs.py --> command line
with command line having the highest priority. If options are in the environment,
they can be overridden by run_configs and lastly those on the command line are given
absolute priority to override any previous value. Where possible, local arguments
should use a option group to emphasize locality of the arguments.
"""

from argparse import ArgumentParser, Action
from os import environ
from functools import wraps

import core.data as dsets
from models.concepts import ALL_ADDITIONS
from models.factory import MODEL_FACTORY_MAP

Datasets = dsets.Datasets

def ensure(ensure_func, n=1):
    def decorator_ensure(func):
        @wraps(func)
        def wrapper_ensure(self, *args, **kwargs):
            ensure_func(self, args[:n])
            return func(self, *args, **kwargs)

        return wrapper_ensure

    return decorator_ensure


class ReusableArgumentParser(ArgumentParser):
    def __init__(self, *args, **kwargs):
        self.opts_to_acts = {}
        self.injections = {}
        self.conditional_injections = {}
        self.set_op_to_require_op = {}
        super().__init__(*args, **kwargs)

    def add_argument(self, *args, **kwargs):
        opt_action = super().add_argument(*args, **kwargs)
        self.opts_to_acts[opt_action.dest] = opt_action
        return opt_action

    def parse_args(self, *args, **kwargs):
        parsed_args = super().parse_args(*args, **kwargs)
        self._set_opts_after_parsed(parsed_args, self.injections)

        for cond_injection_key in self.conditional_injections:
            cond_val, injections = self.conditional_injections[cond_injection_key]
            if getattr(parsed_args, cond_injection_key) == cond_val:
                self._set_opts_after_parsed(parsed_args, injections)
        for set_op, (is_set_func, require_op) in self.set_op_to_require_op.items():
            set_val = getattr(parsed_args, set_op)
            require_val = getattr(parsed_args, require_op)
            if is_set_func(set_val, require_val):
                raise ValueError(
                    "Under {}, {}={} but {}={}".format(
                        is_set_func, set_op, set_val, require_op, require_val
                    )
                )
        return parsed_args

    @staticmethod
    def _set_opts_after_parsed(parsed_args, injections):
        for injection_key in injections:
            setattr(parsed_args, injection_key, injections[injection_key])

    def _require_opt_in_parser(self, opt_names):
        for opt_name in opt_names:
            if opt_name not in self.opts_to_acts:
                raise ValueError("{} is not in parser".format(opt_name))

    @ensure(_require_opt_in_parser)
    def set_required(self, opt_name, required):
        self.opts_to_acts[opt_name].required = required

    @ensure(_require_opt_in_parser)
    def inject_opt(self, opt_name, opt_value):
        self.set_required(opt_name, False)
        self.injections[opt_name] = opt_value

    @ensure(_require_opt_in_parser)
    def if_set(self, opt_name, opt_val, injections):
        self.conditional_injections[opt_name] = (opt_val, injections)

    @ensure(_require_opt_in_parser, 2)
    def if_set_require(self, set_opt, required_opt):
        self.if_set_require_under(set_opt, required_opt, lambda s, r: s and not r)

    @ensure(_require_opt_in_parser, 2)
    def if_set_require_under(self, set_opt, required_opt, is_set_func):
        if not callable(is_set_func):
            raise ValueError("{} is not a function".format(is_set_func))
        self.set_op_to_require_op[set_opt] = (is_set_func, required_opt)


class DefaultToEnvOpt(Action):
    def __init__(
        self,
        option_strings,
        dest,
        const=None,
        required=False,
        default=None,
        type=None,
        **kwargs
    ):
        env_name = const if const else dest.upper()
        # First check ENV, then can overwrite with command line, fallback is the default
        old_default = default
        default = environ.get(env_name) or old_default
        if type:
            default = type(default)
        # if the opt is required, but we found it in environ, then the user doesn't
        # have to specify it
        if required and default is not None:
            required = False

        if default is None:
            default = old_default

        super().__init__(
            option_strings,
            dest,
            const=const,
            default=default,
            required=required,
            type=type,
            **kwargs
        )

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)


class DefaultToEnvFlag(DefaultToEnvOpt):
    def __init__(
        self, option_strings, dest, nargs=None, type=None, required=None, **kwargs
    ):
        super().__init__(
            option_strings, dest, nargs=0, type=bool, required=False, **kwargs
        )

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, True)


def add_env_opts(parser):
    parser.add_argument(
        "--e_device",
        action=DefaultToEnvOpt,
        default="cpu",
        help="Should follow PyTorch standards (i.e. cpu, cuda, cuda:1, etc.)",
    )
    parser.add_argument(
        "--e_name",
        help="the name of this experimental run, a subdirectory under --e_save-dir",
        action=DefaultToEnvOpt,
    )
    parser.add_argument(
        "--e_save-dir",
        default="./",
        help="the top level directory to save all logs to",
        action=DefaultToEnvOpt,
    )
    parser.add_argument(
        "--e_data-dir",
        default="",
        action=DefaultToEnvOpt,
        help="the top level directory where datasets are loaded from",
    )
    parser.add_argument(
        "--e_workers",
        type=int,
        action=DefaultToEnvOpt,
        default=1,
        help="If running on CPU, how many process level threads to use",
    )


def add_default_opts(parser):
    parser.add_argument(
        "--data",
        choices=dsets.DATASETS,
        help="which data set to use",
    )
    parser.add_argument(
        "--model",
        choices=MODEL_FACTORY_MAP.keys(),
        type=str,
        default=None,
        help="the model to use, see models/factory.py#MODEL_FACTORY_MAP for options",
    )
    parser.add_argument(
        "--model-additions",
        choices=ALL_ADDITIONS,
        nargs="*",
        default=(),
        help="network architecture additions, e.g. batch_norm and dropout",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        help="max number of epochs to train for",
        default=300,
    )
    parser.add_argument(
        "-l",
        "--learning-rate",
        type=float,
        default=0.1,
        help="learning rate for training",
    )
    parser.add_argument(
        "--lr-step",
        "--learning-rate-step",
        type=int,
        default=150,
        help="Every --lr-step, modify the learning rate",
    )
    parser.add_argument(
        "--l1-regularization",
        type=float,
        default=None,
        help="L1-regularization coefficient, may use in conjunction "
        "with --l2-regularization for elastic net regularization",
    )
    parser.add_argument(
        "--l2-regularization",
        type=float,
        default=None,
        help="L2-regularization coefficient, may use in conjunction "
        "with --l1-regularization for elastic net regularization",
    )
    parser.add_argument(
        "-o",
        "--optimizer",
        type=str,
        default="sgd",
        choices=["sgd", "adam"],
        help="optimizer to use for training",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=1,
        help="compute metrics (loss, accuracy) every EVAL_EVERY epochs",
    )
    parser.add_argument(
        "--augmentation",
        default=False,
        action="store_true",
        help="enable data augmentation",
    )
    parser.add_argument(
        "--label-noise",
        default=0.,
        type=float,
        help="the fration of corrupted training labels",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="seed used to control weight initialization and shuffling of batches"
    )
    parser.add_argument(
        "--label_seed", type=int, default=None, help="seed used for corrupting labels"
    )
    parser.add_argument(
        "--data-split-seed", type=int, default="42", help="seed used for making a validation split out of the train set"
    )
    parser.add_argument(
        "--early-exit-accuracy",
        action="store_true",
        help="true to stop training once accuracy is 1.0",
    )
    parser.add_argument(
        "--train-split",
        type=int,
        default=None,
        help="the number of samples in the train set. the sum of this"
        "and --val-split should equal the length of the true"
        "train set. If setting, --val-split must also be set.",
    )
    parser.add_argument(
        "--val-split",
        type=int,
        default=None,
        help="the number of samples in the validation set. the sum of"
        "this and --train-split should equal the length of the "
        "true train set. Must provide in conjunction with "
        "--train-split",
    )
    parser.if_set_require("train_split", "val_split")

    parser.add_argument(
        "--start-method",
        type=str,
        default="forkserver",
        choices=("spawn", "forkserver", "fork"),
        help="if using CUDA/GPUs you must choose spawn or forkserver, of which the "
        "latter is preferable since it should be faster. if using CPU only, "
        "fork may be preferable.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        help="log level",
    )
    parser.add_argument(
        "--logfile",
        type=str,
        default="linear_regions.log",
        help="file name to save logs to",
    )
    parser.add_argument(
        "--early-exit-loss",
        action="store_true",
        help="stop training based on training loss threshold",
    )
    parser.add_argument(
        "--lr-decay-rate",
        type=float,
        default=0.2,
        help="the value to use for multiplicative learning reate "
        "decay. set to 1.0 for no lr decay",
    )
    parser.add_argument(
        "--stop-by-loss-threshold",
        type=float,
        default=0.19,
        help="the training loss to stop training at. must set "
        "--early-exit-loss as well to true in conjunction",
    )
    parser.add_argument(
        "--stop-by-accuracy-threshold",
        type=float,
        default=0.0,
        help="the training accuracy to stop training at. must set "
        "--early-exit-accuracy to true in conjunction",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.,
        help="value for L2 weight decay in conjunction with SGD",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.,
        help="dropout rate",
    )
    parser.add_argument(
        "--momentum", type=float, default=0.9, help="momentum coefficient for SGD"
    )


def create_default_args():
    parser = ReusableArgumentParser()
    add_env_opts(parser)
    add_default_opts(parser)
    return parser


def create_env_args():
    parser = ReusableArgumentParser()
    add_env_opts(parser)
    return parser

