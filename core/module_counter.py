# -*- coding: utf-8 -*-

import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from core import counting
from models.concepts import wrap_affine_layers, AffineLayer

class ModuleCounter(counting.Countable):
    SUPPORTED_MODULES = {
        nn.Conv2d,
        nn.Linear,
        nn.AvgPool2d,
        nn.Flatten,
        nn.ReLU,
        nn.BatchNorm2d,
        nn.Dropout,
        nn.Softmax,
        nn.AdaptiveAvgPool2d, 
        AffineLayer
    }
    
    __constants__ = ['_epsilon']
    _epsilon: torch.Tensor
    _inf: torch.Tensor

    def __init__(self, model, input_shape=None, batch_size=None, buff_size=30000, device="cpu"):
        ModuleCounter.check_countable(model)
        self._model = model
        wrap_affine_layers(self._model, 'model', batch_size, None)
        self._model.eval()
        self._model.requires_grad_(False)
        self._input_shape = input_shape
        self._device = device
        if input_shape is None:
            self._input_shape = model.input_shape
        self._num_relus, self._relu_modules = self._get_num_relus(self._model)
        self._act_reg_state = self._ActivationRegionState(self._num_relus)
        self._model_handles = None
        self._epsilon = torch.tensor([counting.EPSILON], dtype=torch.float64, device=device)
        self._machine_eps = torch.tensor([torch.finfo(torch.float64).eps], dtype=torch.float64, device=device)
        self._inf = torch.tensor([np.inf], dtype=torch.float64, device=device)
        self._retain_bias = False # appy bias to second half of batch size for AffineLayers
        self._buff_size = buff_size

    @property
    def input_shape(self):
        return self._input_shape

    @property
    def model(self):
        return self._model

    @staticmethod
    def check_countable(model):
        for module in model.modules():
            if (
                module.children() == []
                and type(module) not in ModuleCounter.SUPPORTED_MODULES
            ):
                raise ValueError("{} not in list of supported modules".format(model))

    def _get_num_relus(self, model, ret_bn_modules=False):
        if self._input_shape is None:
            return self._get_num_relu_modules(model)

        def count_callbacks(relu_mod, inp_, outp_):
            count_callbacks.relus += 1
            count_callbacks.relu_modules.append(relu_mod)

        count_callbacks.relus = 0
        count_callbacks.relu_modules = []

        hooks = []
        for module in model.modules():
            if isinstance(module, nn.ReLU):
                hooks.append(module.register_forward_hook(count_callbacks))

        model(torch.ones((2,) + self._input_shape, device=self._device))
        for hook in hooks:
            hook.remove()
        for idx in range(len(count_callbacks.relu_modules)):
            count_callbacks.relu_modules[idx].register_buffer(
                "index", torch.tensor(idx)
            )
        return count_callbacks.relus, count_callbacks.relu_modules

    def _get_num_relu_modules(self, model):
        relus = 0
        relu_modules = []
        for module in model.modules():
            if isinstance(module, nn.ReLU):
                relus += 1
                relu_modules.append(module)
        return relus, relu_modules

    @property
    def num_relus(self):
        return self._num_relus

    def _model_fwd(self, x, retain_bias=False):
        self._retain_bias = retain_bias
        batch_size = int(x.shape[0] // 2)
        self._act_reg_state.reset(batch_size, device=x.device)
        # logits are correctly computed only for the first half of a batch
        output = self._model(x)
        lambdas, self._act_reg_state.boundary_layer_ids = torch.min(self._act_reg_state.lambdas_to_cross, dim=1)
        self._act_reg_state.best_lambdas_to_cross = lambdas
        return output[:batch_size], output[batch_size:]

    class _ActivationRegionState:
        def __init__(self, num_relus):
            self.act_region = None
            self.relu_counter = None
            self.relu_range = (0, num_relus)
            self.lambdas_to_cross = None
            self.best_lambdas_to_cross = None
            self.boundary_layer_ids = None

        def num_relus(self):
            return self.relu_range[1] - self.relu_range[0]

        def reset(self, batch_size, device="cpu"):
            self.lambdas_to_cross = torch.full(
                (batch_size, self.num_relus()), fill_value=np.inf, device=device
            )
            self.act_region = counting.ActivationRegion(
                [0] + [i + 1 for i in range(self.num_relus())] + [0]
            )
            self.relu_counter = 0

    def _retain_bias_pre_fwd_hook(self, module, input):
        """ To be used on concepts.AffineLayer, to instruct
            the module to execute a forward pass retaining the full bias
            parameter.
        """
        module.retain_bias_(retain=self._retain_bias)
        return input
    
    
    def _reset_retain_bias_hook(self, module, input, output):
        module.retain_bias_(retain=False)
        return output
    
    
    def _record_act_pattern_hook(self, module, input, output):
        """ Record several statistics for @module and store
            them in the corresponding entry of @self._act_reg_state.
            
            @param module: instance of torch.nn.ReLU
            @param input: torch.tensor of shape (2N, *), where input[:N] corresponds
                          to input data, while input[N:] are the respective 
                          directions in the input space.
            @param output: torch.tensor of shape (2N, *) Module's output
            
            Stats reported:
                act_pattern: torch.tensor of bool Activation pattern of output[:N]
                pre_activations: module's pre-activation for input[:N]
                dir_pre_activations: preactivations of directions, input[N:]
                lambdas_to_cross: tensor of size (N, *), whose entry represent 
                                  the values of lambda required to move along
                                  direction input[N:] in order to cross the current
                                  linear region boundary.
            @return
                output: torch.tensor the output of the module, with the directions
                        stored in output[half_batch:] transformed according to the
                        activation pattern of output[:half_batch].
        """
        with torch.no_grad():
            relu_counter = module.index.item()
            half_batch = int(output.shape[0] // 2)
          
            act_pattern = output[:half_batch] > 0
            self._act_reg_state.act_region.act_pattern[relu_counter +1] = act_pattern
            output, lambdas = self.find_lambdas_to_cross(
                input, output, act_pattern, half_batch, relu_counter, self._epsilon, self._inf, self._machine_eps
            )
            self._act_reg_state.lambdas_to_cross[:, relu_counter], _ = torch.min(lambdas, dim=1)
        return output

    @staticmethod
    @torch.jit.script
    def find_lambdas_to_cross(
        preact: Tuple[torch.Tensor], activation: torch.Tensor, act_pattern: torch.Tensor, half_batch_size: int, relu_counter: int, epsilon: torch.Tensor, inf: torch.Tensor, eps: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        activation[half_batch_size:] = preact[0][half_batch_size:] * act_pattern
        lambdas = torch.div(-preact[0][:half_batch_size], preact[0][half_batch_size:] + eps).view(half_batch_size, -1)
        lambdas[lambdas <= epsilon] = inf
        
        return (activation, lambdas)

    def _idx_in_relu_range(self, idx):
        relu_range = self._act_reg_state.relu_range
        return relu_range[0] <= idx < relu_range[1]

    def _attach_hooks_to_relus(self, relu_modules, callback):
        handles = []
        for idx in range(len(relu_modules)):
            if self._idx_in_relu_range(idx):
                handles.append(relu_modules[idx].register_forward_hook(callback))
        return handles

    def _attach_hooks_to_affine_layers(self, modules, pre_fwd_hook, fwd_hook):
        handles = []
        for module in modules:
            if isinstance(module, AffineLayer):
                handles.append(module.register_forward_pre_hook(pre_fwd_hook))
                handles.append(module.register_forward_hook(fwd_hook))
        return handles

    def _attach_hooks_to_layers(self, modules, callback):
        handles = []
        for module in modules:
            handles.append(module.register_forward_hook(callback))
        return handles

    def get_activation_region_state(self, x, d=None, retain_bias=False):
        if self._model_handles is None:
            self._model_handles = self._attach_hooks_to_relus(
                self._relu_modules, self._record_act_pattern_hook
            )
            self._model_handles += self._attach_hooks_to_affine_layers(
                self._model.modules(), self._retain_bias_pre_fwd_hook, self._reset_retain_bias_hook
            )
            
        if d is not None:
            x = torch.cat((x,d)).to(x.device)
        logits, directions = self._model_fwd(x, retain_bias)
        return self._act_reg_state, logits, directions

    def remove_model_handles(self):
        if self._model_handles is None:
            return
        for handle in self._model_handles:
            handle.remove()
        self._model_handles = None


    def remove_all_handles(self):
        self.remove_model_handles()
        

    def count(self, x, d, device="cpu"):
        def act_reg_callback_closure(x0s, directions=None, retain_bias=False):
            reg_state, logits, directions = self.get_activation_region_state(x0s, directions, retain_bias)
            return (
                reg_state.best_lambdas_to_cross,
                reg_state.boundary_layer_ids,
                reg_state.act_region,
                logits,
                directions
            )

        ret_vals = counting.find_act_patterns_between_points(
            act_reg_callback_closure,
            x,
            d,
            device=device,
            buff_size=self._buff_size
        )
        self.remove_all_handles()
        return ret_vals

