from enum import Enum
import abc

from core.counting import Countable

import torch
import torch.nn as nn


class NetworkAddition(Enum):
    BATCH_NORM = "batch_norm"
    DROPOUT = "dropout"


ALL_ADDITIONS = {NetworkAddition.BATCH_NORM.value, NetworkAddition.DROPOUT.value}


class NetworkBuilder(abc.ABC):
    def __init__(self, dataset_info):
        self._dataset_info = dataset_info

    @abc.abstractmethod
    def add(self, addition: NetworkAddition, **kwargs):
        """Add the network component addition, to the network"""

    @abc.abstractmethod
    def build_net(self) -> nn.Module:
        """
        Take whatever internal state this keeps and convert it into a module
        object to be consumed metrics
        """

        
class AffineLayer(nn.Module):
    """ Wrapper for linear layers, useful for linear region counting.
    
        For a linear layer of type nn.Linear or nn.Conv2d, AffineLayer
        decomposes the forward pass into a linear transformation and a
        translation by @linear_layer.bias. Refer to self.forward for details
        about using this module.
    """

    def __init__(self, linear_module, batch_size):
        """@param linear_module: nn.Module linear module to wrap
           @param batch_size: int supported batch_size	       
        """
        super(AffineLayer, self).__init__()
        self._is_supported_module(linear_module)
        self.linear = linear_module
        self.bias = None
        self._retain_bias = False
        self.batch_size = batch_size * 2
        self.half_batch = batch_size
        bias_orig = self.linear.bias.clone() if self.linear.bias is not None else None
        if bias_orig is not None:
            try:
                weight_size = len(self.linear.weight.shape) -2
            except AttributeError:
                if isinstance(self.linear, nn.BatchNorm2d):
                    weight_size = 2 # assuming batch norm is always used after convolution
                else:
                    weight_size = 0
            shape = (1, -1,) + (1,) * weight_size
            bias_orig = nn.Parameter(bias_orig.reshape(shape), requires_grad=False)
        self.register_parameter('bias_orig', bias_orig)
        self.init_bias()
        
    @staticmethod
    def _is_supported_module(module):
        if not is_supported_(module):
            raise ValueError("Expected linear_module of type torch.nn.Linear, \
                              torch.nn.BatchNorm2d, or torch.nn.Conv2d")

    def retain_bias_(self, retain=False):
        self._retain_bias = retain and (self.bias_orig is not None)

    @property
    def weight(self) -> nn.Parameter or None:
        if self.linear is None:
            return None
        return self.linear.weight

    def init_bias(self) -> None:
        """ Copy the bias parameter (if defined) from @self.linear_layer
            and then disables bias computation in @self.linear_layer,
            effectively decomposing the layer-wise forward pass into a linear
            transformation (performed by @self.linear_layer), followed by a 
            translation (specified by @self.bias).
        """
        if self.linear is not None:
            if isinstance(self.linear, nn.BatchNorm2d):
                self.bias = self._copy_bias_bn()
            else:
                self.bias = self._copy_bias()
            self._zero_bias()

    def _copy_bias_bn(self) -> nn.Parameter or None:
        bias = self.linear.bias
        mean = self.linear.running_mean
        var = self.linear.running_var
        eps = self.linear.eps
        
        if bias is not None:
            output_shape = (self.batch_size, bias.shape[0]) + (1,1)
            bias_shape = (1, output_shape[1]) + (1,) * (len(output_shape) -2)
            bias_broadcast = torch.zeros(
                                output_shape,
                                dtype=self.linear.weight.dtype,
                                layout=self.linear.weight.layout,
                                device=self.linear.weight.device
            )
            half_batch = int(self.batch_size // 2)
            bias_broadcast[:half_batch] += bias.reshape(bias_shape)
            bias_broadcast[half_batch:] += ((mean / torch.sqrt(var + eps)) * \
                                             self.linear.weight).reshape(bias_shape)

            return nn.Parameter(bias_broadcast, requires_grad=False)
        else:
            return bias

    def _copy_bias(self) -> nn.Parameter or None:
        """ Copy the bias parameter from @self.linear and broadcast it to
            the batch dimension @self.batch_size, so that self.bias[:b] 
            is a copy of the bias parameter @self.linear.bias, while
            self.bias[b:] is all zeros, where b = batch_size // 2
        """
        bias = self.linear.bias

        if bias is not None:
            output_shape = (self.batch_size, self.linear.weight.shape[0]) + \
                           (1,) * (len(self.linear.weight.shape) -2)
            bias_shape = (1, output_shape[1]) + (1,) * (len(output_shape) -2)
            bias_broadcast = torch.zeros(
                                output_shape,
                                dtype=self.linear.weight.dtype,
                                layout=self.linear.weight.layout,
                                device=self.linear.weight.device
            )
            bias_broadcast[:int(self.batch_size // 2)] += bias.reshape(bias_shape)
            return nn.Parameter(bias_broadcast, requires_grad=False)
        else:
            return bias

    def _zero_bias(self) -> None:
        """ Reset the bias parameter of @self.linear, so that
            the module effectively computes a linear transformation
        """
        if self.linear is not None:
            self.linear.bias = None

    def extra_repr(self) -> str:
        zero_rows_idx = [ torch.all(self.bias.view(self.batch_size, -1)[i] == 0) for i in range(self.batch_size)]
        zero_rows = self.bias[zero_rows_idx, ].shape[0]
        repr = '(bias): ' + str(self.bias.shape) + '\n' + "{}/{} zero rows.".format(zero_rows, self.bias.shape[0])
        return repr

    def forward(self, x):
        """
        For an input tensor @x with batch dimension B, to speed up 
        region counting, the forward pass of AffineLayer, for 
        @self.batch_size = 2 * B, applies an affine transformation
        linear(x) + self.bias to @x[:B], while only a linear
        transformation linear(x) to @x[B:].
        
        Can take input of arbitrary even batch size b = 2k < B.
        """
        x = self.linear(x)
        
        if self._retain_bias:
            x += self.bias_orig
        elif self.bias is not None:
            b = int(x.shape[0] // 2)
            x += self.bias[self.half_batch -b : self.half_batch + b]
        return x


def is_supported_(module: nn.Module) -> bool :
    if isinstance(module, nn.Linear):
        return True
    elif isinstance(module, nn.Conv2d):
        return True
    elif isinstance(module, nn.BatchNorm2d):
        return True
    else:
        return False

def wrap_affine_layers(module: nn.Module, name: str, batch_size: int, parent: nn.Module or None) -> None:
    """Wrap every instance of @module with AffineLayer, recursively for @module and its children.
    """
    if is_supported_(module):
        parent._modules[name] = AffineLayer(module, batch_size)
    for name, child in module.named_children():
        if isinstance(module, AffineLayer) and name == 'linear':
            continue
        wrap_affine_layers(child, name, batch_size, parent=module)
