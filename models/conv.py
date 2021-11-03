import torch.nn as nn
from numpy import prod

from core.counting import Countable
from core.data import DatasetInfos
from models.concepts import NetworkBuilder, NetworkAddition
from functools import partial

def vgg_layers(blocks, in_channels, num_classes, batch_norm=False):
    layers = []
    for v in blocks:
        if v == 'A':
            layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = v

    if len(blocks) < 12: # vgg8
        spatial_res = 4 * 4
        layers.extend(
        [
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(64 * spatial_res, 120),
            nn.ReLU(True),
            nn.Linear(120, num_classes),
        ]
    )
    else:
        avgpool_spatial_res = 1 * 1
        layers += [
            nn.AdaptiveAvgPool2d(avgpool_spatial_res),
            nn.Flatten(),
            nn.Linear(512 * spatial_res, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, num_classes),
            ]
    return nn.Sequential(*layers)


class VGG(nn.Module):
    def __init__(self, blocks, num_classes=100, in_channels=3, init_weights=True):
        super(VGG, self).__init__()
        self._layers = vgg_layers(blocks, in_channels, num_classes)
        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    @property
    def use_batch_norm(self):
        return self._use_batch_norm
        
    def forward(self, x):
        return self._layers(x)


class VGGBuilder(NetworkBuilder):
    def __init__(self, vgg_cls, blocks, dataset_info):
        self.blocks = blocks
        self._model = vgg_cls(
            blocks,
            num_classes=dataset_info.output_dimension,
            in_channels=dataset_info.input_shape[0],
            init_weights=True
        )
        super().__init__(dataset_info)
        
    def add(self, addition: NetworkAddition, **kwargs):
        dropout_rate = kwargs.pop("dropout_rate", 0.)
        if addition == NetworkAddition.BATCH_NORM:
            self.add_batch_norm()
        if addition == NetworkAddition.DROPOUT:
            self.add_dropout(dropout_rate)

    def add_batch_norm(self):
        self._model.use_batch_norm = True
        layers_with_batch_norm = []
        for layer in self._layers:
            layers_with_batch_norm.append(layer)
            if isinstance(layer, nn.Conv2d):
                layers_with_batch_norm.append(nn.BatchNorm2d(layer.out_channels))
        self._layers = layers_with_batch_norm
        
    def add_dropout(self, dropout_rate=0.):
        layers_with_drop = []
        for idx, layer in enumerate(self._layers):
            layers_with_drop.append(layer)
            if idx != len(self._layers) - 1 and isinstance(layer, nn.Linear):
                layers_with_drop.append(nn.Dropout(p=droput_rate))
        self._layers = layers_with_drop

    def build_net(self) -> nn.Module:
        return self._model


MODEL_FACTORY_MAP = {
    "vgg8": partial(VGGBuilder, VGG, [6, 6, 'A', 16, 16, 'A', 64, 64]),
    "vgg11": partial(VGGBuilder, VGG, [64, 'A', 128, 'A', 256, 256, 'A', 512, 512, 'A', 512, 512]),
    "vgg13": partial(VGGBuilder, VGG, [64, 64, 'A', 128, 128, 'A', 256, 256, 'A', 512, 512, 'A', 512, 512]),
    "vgg16": partial(VGGBuilder, VGG, [64, 64, 'A', 128, 128, 'A', 256, 256, 256, 'A', 512, 512, 512, 'A', 512, 512, 512]),
    "vgg19": partial(VGGBuilder, VGG, [64, 64, 'A', 128, 128, 'A', 256, 256, 256, 256, 'A', 512, 512, 512, 512, 'A', 512, 512, 512, 512]),
}
