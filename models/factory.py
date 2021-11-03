from types import SimpleNamespace

from core.data import DATASET_INFO_MAP
from models.conv import MODEL_FACTORY_MAP as CONV_FACTORY_MAP
from models.resnet import MODEL_FACTORY_MAP as RESIDUAL_FACTORY_MAP
from models.concepts import NetworkAddition

MODEL_FACTORY_MAP = {**CONV_FACTORY_MAP, **RESIDUAL_FACTORY_MAP}

Models = SimpleNamespace(**MODEL_FACTORY_MAP)
Models.names = SimpleNamespace(**{name: name for name in MODEL_FACTORY_MAP})


def create_model(model_name, data_name, additions=(), *args, **kwargs):
    if model_name not in MODEL_FACTORY_MAP:
        raise KeyError("{} is not in MODELS_MAP".format(model_name))
    if data_name not in DATASET_INFO_MAP:
        raise KeyError("{} is not in ALL_DATASETS".format(data_name))
    dropout_rate = kwargs.pop("dropout_rate", 0.)
    network_builder = MODEL_FACTORY_MAP[model_name](
        DATASET_INFO_MAP[data_name], *args, **kwargs
    )
    for addition in additions:
        network_builder.add(NetworkAddition(addition), dropout_rate=dropout_rate)
    return network_builder.build_net()
