# -*- coding: utf-8 -*-

"""
Exact counting along a line segment joining two points in input space by counting boundary crossings.
"""

import torch
import torch.nn as nn
import numpy as np
import abc
import logging
from typing import Tuple, Union

logger = logging.getLogger(__name__)

__all__ = [
    "find_act_patterns_between_points",
]


# This assumes double precision of all weights and data points
# if using single precision floats, would need to change this to something like 1e-4
EPSILON = 1e-6

class ActivationRegion:
    def __init__(self, arch):

        self.arch = arch
        # list of list of 0's and 1s
        self.act_pattern = []
        for layer in arch:
            if isinstance(layer, (int, np.integer)):
                self.act_pattern.append([3 for _ in range(layer)])
            else:
                self.act_pattern.append([[3 for _ in range(dim)] for dim in layer])

    def __eq__(self, other):
        if len(self.act_pattern) != len(other.act_pattern):
            return False
        for idx in range(1, len(self.act_pattern) - 1):
            self_pattern = self.act_pattern[idx]
            other_pattern = other.act_pattern[idx]
            if not torch.all(self_pattern.eq(other_pattern)):
                return False
        return True

    def __hash__(self):
        str_pattern = "".join(
            [
                "1" if j else "0"
                for pat in self.act_pattern[1:-1]
                for j in list_flatten(pat.tolist())
            ]
        )
        return hash(str_pattern)

    def squeeze(self):
        for i in range(1, len(self.act_pattern) - 1):
            self.act_pattern[i] = self.act_pattern[i].flatten()

    @property
    def num_individual_regions(self):
        return len(self.act_pattern[1])
        
    def not_eq_to_idx(self, other_batch, diff_indices: torch.Tensor) -> torch.Tensor:
        """
        @param other_batch: Instance of `ActivationRegion` with batch dimension (4D)
        @return: A list of batch indices where @self.act_pattern does not equal
                 `other.act_pattern`
        """
        indices = torch.arange(self.num_individual_regions, device=self.act_pattern[1].device)
        mask = torch.full_like(indices, fill_value=False, dtype=torch.bool)

        for layer_idx in range(1, len(self.act_pattern) -1):
            self._not_eq_to_idx(
                self.act_pattern[layer_idx],
                other_batch.act_pattern[layer_idx][diff_indices],
                mask,
                self.act_pattern[1].shape[0]
            )

        return indices[mask]

    @staticmethod
    @torch.jit.script
    def _not_eq_to_idx(act_pattern: torch.Tensor, other: torch.Tensor, mask: torch.Tensor, nbatches: int) -> None:
        mask.logical_or(
            act_pattern.view(nbatches, -1).logical_xor(
                other.view(nbatches, -1)
            ).sum(dim=1),
            out=mask
        )


class Countable(abc.ABC):
    @property
    @abc.abstractmethod
    def input_shape(self):
        """@return the expected input shape"""

    @abc.abstractmethod
    def count(self, x, d, device="cpu"):
        """
        Count self on lines between x and d which are 4d tensors (B, C, H, W)
        """

"""
    Region counting code
"""

@torch.jit.script
def _norm_of_batches(batch_variables: torch.Tensor) -> torch.Tensor:
    """ Compute the L2 norm of a batch-indexed tensor @batch_variables of shape
        (N, *), along the non-batch dimension.
        
        @param batch_variables: torch.tensor of shape (N, *)
        @return norm: torch.tensor of shape (N, 1, .., 1) Per sample norm of 
                      each batch-indexed sample, with singleton dimensions 
                      matching the shape of @batch_variables, 
    """
    shape = (batch_variables.shape[0],) + (1,) * len(batch_variables.shape[1:])
    return torch.norm(
        batch_variables.view(shape[0], -1), p=2, dim=1
    ).view(shape)


@torch.jit.script
def _normalized_directions(x0s: torch.Tensor, x1s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ Normalized tensor @x1s - @x0s of shape (N, *) along all non-batch dimensions 
        (flattened), and return normalized tensor of directions.
        
        @param x0s: torch.Tensor (N, *) start points of x1s - x0s
        @param x1s: torch Tensor (N, *) endpoints
        @return torch.Tensor (N, *) unnormalized direction tensor
        @return torch.Tensor (N, *) normalized direction tensor
        @return torch.Tensor (N, *) norm tensor
    """
    directions_unnormalized = x1s - x0s
    norm_of_directions = _norm_of_batches(directions_unnormalized)
    return directions_unnormalized, directions_unnormalized / norm_of_directions, norm_of_directions


@torch.jit.script
def _normalize_lambdas(lambdas: torch.Tensor, norm_of_directions: torch.Tensor) -> torch.Tensor:
    return lambdas / norm_of_directions.view(-1)


@torch.jit.script
def _remove_lambdas_below_sensitivity(
    lambdas: torch.Tensor, 
    diff_indices: torch.Tensor,
    layer_change_indices_: torch.Tensor,
    norm_of_directions: torch.Tensor, 
    one: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ Filter out  numerically unstable lambda values
    """
    # should compare against norm_of_directions[diff_indices] if directions_normalized
    # are used for estimating lambdas
    diff_indices = diff_indices[
        torch.where(
            torch.le(lambdas[diff_indices], one)
        )[0]
    ]
    layer_change_indices_ = layer_change_indices_[diff_indices]
    lambdas = lambdas[diff_indices]
    return lambdas, diff_indices, layer_change_indices_


@torch.jit.script
def _index_directions(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """ For a tensor @x of stacked values and corresponding
        directions, and a tensor @indices indexing values
        in @x[:x.shape[0] // 2], extend indices so that they
        also match the corresponding values in @x[x.shape[0] // 2:]
    """
    half_batch = int(x.shape[0] // 2)
    full_batch_indices = torch.cat((indices, indices + half_batch))
    return x[full_batch_indices]


@torch.jit.script
def _override_directions(x: torch.Tensor, new: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """ For a tensor @x of stacked values, a tensor @new, 
        and a tensor @indices indexing values in @x[:x.shape[0]], 
        ovverride the values in @x[x.shape[0]:] with @new
    """
    new_tensor = x.detach().clone()
    new_tensor[len(indices):] = new[indices]
    return new_tensor


@torch.jit.script
def _cross_region_boundary(
    input: torch.Tensor, output: torch.Tensor, lambdas: torch.Tensor, directions: torch.Tensor, norm_of_directions: torch.Tensor, epsilon: torch.Tensor, indices: torch.Tensor
) -> None:
    output[indices] = input[indices] + (lambdas + epsilon) * directions[indices]
    

@torch.jit.script
def _step_along_direction(
    input: torch.Tensor, output: torch.Tensor, lambdas: torch.Tensor, directions: torch.Tensor, indices: torch.Tensor
) -> None:
    output[indices] = input[indices] + lambdas * directions[indices]


@torch.jit.script
def _remove_overshoot_indices(
    x: torch.Tensor, x0s: torch.Tensor, norm_of_directions: torch.Tensor, indices: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ For a batch of inputs @x of shape (N, *), a batch of start points @x0s, and a batch of
        finite directions @norm_of_directions of shape (N, *), return a tensor of indices
        such that || @x[@indices] - @x0s[@indices] || <= @norm_of_directions
    """
    idx_within_dist = torch.where(
        _norm_of_batches(x[indices] - x0s[indices]) <= norm_of_directions[indices]
    )[0]
    return indices[idx_within_dist], idx_within_dist


@torch.jit.script
def variation_along_direction_normalized(
    x_in: torch.Tensor, x_out: torch.Tensor, logits_in: torch.Tensor, logits_out: torch.Tensor
) -> torch.Tensor:
    """ Compute the directional derivative of the linear function
        represented by @logits_in and @logits_out along the direction
        @x_out - @x_in, normalized by the length of @x_out - @x_in
        
        Params:
            @x_in : 4D Tensor of shape (N,C,H,W) Batch of entry points to their
                    corresponding linear regions.
            @x_out: 4D Tensor of shape (N,C,H,W) Batch of exit points to their 
                    corresponding linear region.
            @logits_in: 2D Tensor of shape (N, K) Logit values at @x_in.
            @logits_out: 2D Tensor of shape (N, K) Logit values at @x_out.
        
        Return: Tensor of shape (N, K) The per-logit norm of directional 
                derivative of the network's output in the direction @x_out - @x_in.
    """
    norm = torch.norm((x_out - x_in).view(x_out.shape[0], -1), p=2, dim=1, keepdim=True)
    return (logits_out - logits_in) / norm


@torch.jit.script
def variation_along_direction(
    x_in: torch.Tensor, x_out: torch.Tensor, logits_in: torch.Tensor, logits_out: torch.Tensor
) -> torch.Tensor:
    """ Compute the directional derivative of the linear function
        represented by @logits_in and @logits_out along the direction
        @x_out - @x_in.
        
        Params:
            @x_in : 4D Tensor of shape (N,C,H,W) Batch of entry points to their
                    corresponding linear regions.
            @x_out: 4D Tensor of shape (N,C,H,W) Batch of exit points to their 
                    corresponding linear region.
            @logits_in: 2D Tensor of shape (N, K) Logit values at @x_in.
            @logits_out: 2D Tensor of shape (N, K) Logit values at @x_out.
        
        Return: Tensor of shape (N, K) The per-logit norm of directional 
                derivative of the network's output in the direction @x_out - @x_in.
    """
    return logits_out - logits_in


@torch.jit.script
def absolute_deviation(
    t_start: torch.Tensor, 
    t_end: torch.Tensor, 
    norm_of_directions: torch.Tensor, 
    logits_x0: torch.Tensor, 
    logits_x_at_x0: torch.Tensor, 
    variation_interpolated: torch.Tensor, 
    variation_linear: torch.Tensor
) -> torch.Tensor:
    """ Compute the absolute deviation between the network's output and an 
        affine baseline interpolating between the logits at x0 and x1
        along @directions.
        
        Parameters:
        @param t_start: torch.Tensor of shape (N, *) with singleton dimensions
                        beyond the batch dimension
        @param t_end: torch.Tensor of shape (N, *) with singleton dimensions 
                      beyond the batch dimension
        @param norm_of_directions: torch.Tensor of shape (N, *), batch of 
               L2 norms of flatted @directions, with singleton dimensions
               beyond the batch dimension
        @param logits_x0: torch.Tensor of shape (N,K) network output at x0s
        @param logits_x_at_x0: torch.Tensor (N, K) the output of the affine
               function that the network computes on the linear region of x,
               evaluated at x0
        @param variation_interpolated: torch.Tensor of shape (N,K) batched
               per-logit variation of the interpolating affine function.
        @param variation_linear: torch.Tensor of shape (N,K) batched
               variation of the function computed by the network at x, computed
               between x1 and x0.
        
        Return:
        @return absolute_deviation: torch.Tensor of shape (N, K) batched per-logit
                absolute_deviation deviation over the interval [@t_start, @t_end]
                of the real line R.
    """
    deviation_x0 = logits_x0 - logits_x_at_x0
    # if lambdas are computed using directions_normalized, then variation_linear
    # should be upscaled by norm_of_directions to compensate for the normalization
    linear_deviation = variation_linear - variation_interpolated
    zeros = torch.zeros_like(deviation_x0)
    
    t_at_intersection = torch.max(
        t_start, torch.min(deviation_x0 / linear_deviation, t_end)
    )
    mask_nans = deviation_x0.eq(zeros)
    t_at_intersection[mask_nans] = (zeros + t_start)[mask_nans]
    t_int_square = t_at_intersection.square()
    linear_deviation = linear_deviation.abs()
    deviation_x0 = deviation_x0.abs()
    
    return norm_of_directions * (
        (t_at_intersection - t_start) * deviation_x0 + \
        (t_end - t_at_intersection) * deviation_x0 + \
        0.5 * ((t_int_square - t_start.square()) * linear_deviation + \
        (t_end.square() - t_int_square) * linear_deviation)
    )


def find_act_patterns_between_points(
    get_act_region_fn,
    x0s,
    x1s,
    device="cpu",
    var_along_d=False,
    buff_size=30000
):
    """
    @param get_act_region_fn: a function with signature:
        get_act_region_fn(x, directions): ->
            (
                lambdas_to_cross,
                crossing_layer_indices,
                act_region,
                logits,
                directions
            )
        Parameters of @get_act_region: 
           For tensors @x0s, @x1s of shape (N, *), respectively denoting the 
           endpoints of a line segment, direction is the L2 normalized direction
           @x1s - @x0s (of shape (N)), and @x is taken as the starting endopoint
           @x0s.
        
        Return values of @get_act_region:
            lambdas_to_cross: tensor of shape (N) scalar coefficient required 
                              from crossing the the linear regions respectively
                              containing x0s, in the direction @direction.
            crossing_layer_indices: tensor of shape (N) layer indices at which 
                                    the crossing occurs.
            act_region: Instance of class ActivationRegion for x0s 
            logits : 2D (B, K) predictions, where K is the number of classes

    @param x0s: batch-indexed tensor of shape (N, *) where * is any tuple of 
                dimensions indexing the image/input data shape, denoting the
                starting point of a batch of N lines in the input space.
    @param x1s: Tensors with same shape as @x0, denoting the endpoints of a batch 
                of N lines in the input space.
    @param buff_size: int size of each preallocated cuda buffer used to store 
                      activation region statistics.
    @return:
        pts: List of lists of tensors. The outer list is indexed by batch size N
             Each inner list containts the entry and exit poit for each activation 
             region visited.
        absolute_deviation: List of lists of lists of floats. The outer list matches @pts
                  in length and is batch-indexed. Each inner list abs_deviation[i]
                  contains a list of K per-logit absolute deviations. The length
                  of abs_deviation[i] is the number of activation regions crossed for 
                  line @i.
        logits: List of lists of lists of floats. The outer list matches @pts
                in length and is batch-indexed. Each inner list logits[i]
                contains a list of K logit values, evaluated at the corresponding
                point in @pts (i.e. 2 points per activation region). 
                The length of logits[i] is twice the number of activation
                regions crossed by line @i.
    """
    directions, directions_normalized, norm_of_directions = _normalized_directions(x0s, x1s)
    batch_size = x0s.shape[0]
    flatten_keep_dim = (-1,) + (1,) * len(x0s.shape[1:])
    inf = torch.full((1,), fill_value=np.inf, device=device)
    epsilon = torch.full((1,), fill_value=EPSILON, device=device)
    one = torch.ones((1,), device=device)
    
    x = torch.cat((x0s, directions)).to(device, non_blocking=True)
    x1s = torch.cat((x1s, directions)).to(device, non_blocking=True)
    x0s = x0s.detach()
    
    lambdas, layer_change_indices_, act_pattern_x, logits_x, variation_linear = get_act_region_fn(x)
    _, _, act_pattern_x1, logits_x1, _ = get_act_region_fn(x1s)
    density = torch.zeros(batch_size, device=device, dtype=torch.long)
    t_start = torch.zeros(batch_size, device=device)
    t_end = torch.zeros_like(t_start)
    logits_x0 = logits_x.detach().clone()
        
    # variation along d for each segment in x1s - x0s
    variation_interpolated = variation_along_direction(
        x0s, x1s[:batch_size], logits_x, logits_x1
    )
    
    # To optimize computation, we compute each affine function at x0s
    # whenever we collect logits for x_next
    x_next = torch.cat((x0s, x0s)).to(device, non_blocking=True)
    
    # free up GPU-memory
    x1s = x1s.to("cpu", non_blocking=True) 
    logits_x1 = logits_x1.detach().to("cpu", non_blocking=True)

    act_patterns = None
    indices_to_batches = torch.arange(batch_size, device=device, dtype=torch.long)
    
    # preallocate results buffers
    logits = torch.full(
        logits_x.shape + (buff_size, 2), fill_value=np.inf, dtype=torch.float64, device=device
    )
    pts = torch.full(
        (batch_size, buff_size, 2), fill_value=np.inf, dtype=torch.float64, device=device
    )
    tot_vars = torch.full(
        logits_x.shape + (buff_size,), fill_value=np.inf, dtype=torch.torch.float64, device=device
    )
    abs_deviation = torch.full(
        logits_x.shape + (buff_size,), fill_value=np.inf, dtype=torch.torch.float64, device=device
    )
    logits[indices_to_batches, :, density, 0] = logits_x.detach().clone()
    pts[indices_to_batches, density, 0] = 0.
    pts[indices_to_batches, density, 1] = 1.
    
    while True:
        # tensor of batch indices for which counting is not complete
        diff_indices = act_pattern_x.not_eq_to_idx(act_pattern_x1, indices_to_batches)
        if len(diff_indices) == 0:
            break
        
        # filter out lambdas below sensitivity EPSILON
        lambdas, diff_indices, layer_change_indices_ = _remove_lambdas_below_sensitivity(
            lambdas, diff_indices, layer_change_indices_, norm_of_directions, one
        )
        if len(diff_indices) == 0:
            break

        t_end[indices_to_batches[diff_indices]] += lambdas.view(-1)
        lambdas = lambdas.view(flatten_keep_dim)
        
        with torch.no_grad():
            # move along directions until the linear region boundary
            _step_along_direction(x, x_next, lambdas, directions, diff_indices)
        _, _, _, logits_next, logits_x_at_x0 = get_act_region_fn(
            _index_directions(x_next, diff_indices), retain_bias=True
        )
        
        batch_indices = indices_to_batches[diff_indices]
        deviation = absolute_deviation(
            t_start[batch_indices].view(-1,1),
            t_end[batch_indices].view(-1,1),
            norm_of_directions[diff_indices].view(-1,1), 
            logits_x0[diff_indices], 
            logits_x_at_x0,
            variation_interpolated[batch_indices],
            variation_linear[diff_indices]
        )

        variation = variation_along_direction(
            x[diff_indices], x_next[diff_indices], logits_x[diff_indices], logits_next
        )
        
        # done with the current batch of regions, storing stats
        try:
            logits[batch_indices, :, density[batch_indices], 1] = logits_next.detach().clone()
        except IndexError:
            logits = torch.cat(
                (logits, torch.full(
                    logits.shape[:2] + (10000, 2), fill_value=np.inf, device=device)
                ), dim=2
            )
            pts = torch.cat(
                (pts, torch.full(
                    (batch_size, 10000, 2), fill_value=np.inf, device=device)
                ), dim=1
            )
            tot_vars = torch.cat(
                (tot_vars, torch.full(
                    tot_vars.shape[:2] + (10000,), fill_value=np.inf, device=device)
                ), dim=2
            )
            abs_deviation = torch.cat(
                (abs_deviation, torch.full(
                    abs_deviation.shape[:2] + (10000,), fill_value=np.inf, device=device)
                ), dim=2
            )
            logits[batch_indices, :, density[batch_indices], 1] = logits_next.detach().clone()
        
        pts[batch_indices, density[batch_indices], 1] = lambdas.view(-1).clone()
        tot_vars[batch_indices, :, density[batch_indices]] = variation.detach().clone()
        abs_deviation[batch_indices, :, density[batch_indices]] = deviation.clone()
        
        with torch.no_grad():
            # cross into the next linear region
            _cross_region_boundary(x, x, lambdas, directions, norm_of_directions, epsilon, diff_indices)
        
            # check whether we overshot
            diff_indices, lambda_indices = _remove_overshoot_indices(x, x0s, norm_of_directions, diff_indices)
            if len(diff_indices) == 0:
                break
        
        batch_indices = indices_to_batches[diff_indices]
        density[batch_indices] += 1
        t_start[batch_indices] += lambdas[lambda_indices].view(-1)
        try:
            pts[batch_indices, density[batch_indices], 0] = lambdas[lambda_indices].view(-1).clone()
        except IndexError:
            logits = torch.cat(
                (logits, torch.full(
                    logits.shape[:2] + (10000, 2), fill_value=np.inf, device=device)
                ), dim=2
            )
            pts = torch.cat(
                (pts, torch.full(
                    (batch_size, 10000, 2), fill_value=np.inf, device=device)
                ), dim=1
            )
            tot_vars = torch.cat(
                (tot_vars, torch.full(
                    tot_vars.shape[:2] + (10000,), fill_value=np.inf, device=device)
                ), dim=2
            )
            abs_deviation = torch.cat(
                (abs_deviation, torch.full(
                    abs_deviation.shape[:2] + (10000,), fill_value=np.inf, device=device)
                ), dim=2
            )
            pts[batch_indices, density[batch_indices], 0] = lambdas[lambda_indices].view(-1).clone()

        x = _index_directions(x, diff_indices)
        x_next = _override_directions(x, x0s, diff_indices)
        
        lambdas, layer_change_indices_, act_pattern_x, logits_x, variation_linear = get_act_region_fn(x) 
        logits[batch_indices, :, density[batch_indices], 0] = logits_x.detach().clone()
        pts[batch_indices, density[batch_indices], 1] = torch.zeros_like(lambdas.view(-1)) 
        
        x0s = x0s[diff_indices]
        logits_x0 = logits_x0[diff_indices]
        norm_of_directions = norm_of_directions[diff_indices]
        directions = directions[diff_indices]
        indices_to_batches = indices_to_batches[diff_indices]
    
    logits = logits.to("cpu", non_blocking=True)
    pts = pts.to("cpu", non_blocking=True)
    tot_vars = tot_vars.to("cpu", non_blocking=True)
    density = density.to("cpu", non_blocking=True)
    abs_deviation = abs_deviation.to("cpu", non_blocking=True)
    variation_interpolated = variation_interpolated.to("cpu", non_blocking=True)
    logits[torch.arange(batch_size), :, density, 1] = logits_x1
 
    return pts, logits, density, tot_vars, abs_deviation, variation_interpolated
