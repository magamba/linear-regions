# -*- coding: utf-8 -*-

"""ResNet model definitions
"""

import torch
import torch.nn as nn
import numpy as np
from torchvision.models import resnet as torch_resnet

from models.concepts import NetworkBuilder, NetworkAddition

from functools import partial

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, bias=True):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=bias, dilation=dilation)

class BasicBlock(nn.Module):
    """
    Residual block with optional batch normalization
    """

    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        use_batch_norm=False,
    ):
        super(BasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride, bias=not use_batch_norm)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes, bias=not use_batch_norm)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride
        self.use_batch_norm = use_batch_norm
        if not use_batch_norm: 
            # disable batch norm at this stage to ensure reproducibility with prev
            # versions of the code, which always initialized batch norm layers
            self.bn1, self.bn2 = None, None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        if self.use_batch_norm:
            out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        if self.use_batch_norm:
            out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu2(out)

        return out


class ResNetNoBN(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=1000,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        use_batch_norm=False,
        in_channels=3,
        inplanes=64,
        channels=(128, 256, 512),
    ):
        super(ResNetNoBN, self).__init__()

        num_layers = sum(layers)  # used for Fixup init
        self.inplanes = inplanes
        self.dilation = 1
        self._use_batch_norm = use_batch_norm
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            in_channels, self.inplanes, kernel_size=3, stride=1, padding=1, bias=not use_batch_norm
        )
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=False)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(
            block,
            channels[0],
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
        )
        self.layer3 = self._make_layer(
            block,
            channels[1],
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
        )
        self.layer4 = self._make_layer(
            block,
            channels[2],
            layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2],
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels[2] * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, BasicBlock):
                nn.init.normal_(
                    m.conv1.weight,
                    mean=0,
                    std=np.sqrt(
                        2.0
                        / (m.conv1.weight.shape[0] * np.prod(m.conv1.weight.shape[2:]))
                    )
                    * num_layers ** (-0.5),
                )
                #nn.init.constant_(m.conv2.weight, 0)
                nn.init.normal_(m.conv2.weight, mean=0, std=1e-6)
                if m.conv1.bias is not None:
                    nn.init.constant_(m.conv1.bias, 0)
                if m.conv2.bias is not None:
                    nn.init.constant_(m.conv2.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                #nn.init.constant_(m.weight, 0)
                nn.init.normal_(m.weight, mean=0, std=1e-6)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
            if not use_batch_norm:
                self.bn1 = None

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                torch_resnet.conv1x1(self.inplanes, planes * block.expansion, stride),
            )

        layers = [
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                use_batch_norm=self.use_batch_norm,
            )
        ]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    use_batch_norm=self.use_batch_norm,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        if self.use_batch_norm:
            x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

    @property
    def use_batch_norm(self):
        return self._use_batch_norm

    def _set_use_batch_norm_layer(self, seq_layer):
        for block in seq_layer:
            block.use_batch_norm = self.use_batch_norm

    @use_batch_norm.setter
    def use_batch_norm(self, truth_value):
        self._use_batch_norm = truth_value
        self._set_use_batch_norm_layer(self.layer1)
        self._set_use_batch_norm_layer(self.layer2)
        self._set_use_batch_norm_layer(self.layer3)
        self._set_use_batch_norm_layer(self.layer4)


class ResNetNoBNSmall(ResNetNoBN):
    def __init__(
        self,
        block,
        layers,
        num_classes=1000,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        use_batch_norm=False,
        in_channels=3,
    ):
        super().__init__(
            block,
            layers,
            num_classes=num_classes,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            replace_stride_with_dilation=replace_stride_with_dilation,
            use_batch_norm=use_batch_norm,
            in_channels=in_channels,
            inplanes=16,
            channels=(32, 64, 128),
        )


class ResNetBuilder(NetworkBuilder):
    def __init__(self, resnet_cls, block_cls, arch, dataset_info):
        self._model = resnet_cls(
            block_cls,
            arch,
            num_classes=dataset_info.output_dimension,
            in_channels=dataset_info.input_shape[0],
        )
        super().__init__(dataset_info)

    def add(self, addition: NetworkAddition):
        if addition == NetworkAddition.BATCH_NORM:
            self.add_batch_norm()
        if addition == NetworkAddition.DROPOUT:
            self.add_dropout()

    def add_batch_norm(self):
        self._model.use_batch_norm = True

    def add_dropout(self):
        raise NotImplementedError("Dropout for ResNet not supported yet")

    def build_net(self) -> nn.Module:
        return self._model


MODEL_FACTORY_MAP = {
    "resnet18": partial(ResNetBuilder, ResNetNoBN, BasicBlock, [2, 2, 2, 2]),
    "resnet34": partial(ResNetBuilder, ResNetNoBN, BasicBlock, [3, 4, 6, 3]),
    "resnet18_thin": partial(ResNetBuilder, ResNetNoBNSmall, BasicBlock, [2, 2, 2, 2]),
}
