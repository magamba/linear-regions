# -*- coding: utf-8 -*-

import os
import abc
import random
import json
from enum import Enum
from collections.abc import Iterable
from copy import deepcopy
from multiprocessing import Pool as MPPool
from multiprocessing import current_process as MP_current_process
from multiprocessing import Lock as MPLock
import logging
import torch.nn as nn
import torch

from core import counting
from core.module_counter import ModuleCounter
from core.utils import all_log_dir

logger = logging.getLogger(__name__)

def _init_pool(inner_locks):
    global gpu_locks
    gpu_locks = inner_locks


class Metric(abc.ABC):
    def __init__(self, data_sources, opts=None, **kwargs):
        if not isinstance(data_sources, dict):
            raise ValueError("data_sources must be a dict")
        self.data_sources = data_sources
        self._data_sources_keys = [source for source in data_sources]
        self.results = {key: {} for key in data_sources}
        self._init_opts(opts, **kwargs)

    def _init_opts(self, opts, **kwargs):
        pass

    @property
    @abc.abstractmethod
    def name(self):
        """Return the metric name from Metrics
        Useful for logging
        """
    
    def __call__(self, model, key, **kwargs):
        """
        @param model (torch.nn.Module): the network on which to compute the metric.
        @param key (str): primary key to index the results. Each entry results[key]
                          is represented by a matrix, implemented as a list of
                          same-length lists.
        @param index_sec (str or None): secondary key to index the results.
        """
        index_sec = kwargs.pop("index_sec", None)
        for data_source in self.data_sources:
            try:
                _ = self.results[data_source][key]
            except KeyError:
                self.results[data_source][key] = [] if index_sec is None else {}

            if index_sec is not None:
                try:
                    _ = self.results[data_source][key][index_sec]
                except KeyError:
                    self.results[data_source][key][index_sec] = []

                self.results[data_source][key][index_sec].append(
                    self.compute(model, data_source)
                )
            else:
                self.results[data_source][key].append(self.compute(model, data_source))

    def __getstate__(self):
        # data_sources may contain pointers to PyTorch datasets, and pickling these
        # could quickly lead to serialized objects becoming too large and taking up
        # too much space
        state = self.__dict__.copy()
        try:
            del state["data_sources"]
        except KeyError:
            pass
        return state

    def __repr__(self):
        return str(self.results)

    @abc.abstractmethod
    def compute(self, model, data_source):
        """Compute the metric """


class PointMetric(Metric, abc.ABC):
    """
    Class of metrics used for line-based counting.
    Note: self.data_sources is assumed to be a map of iterables!
    """

    def _init_opts(self, opts, **kwargs):
        self._workers = opts.e_workers
        self.e_device = opts.e_device
        self._seed = opts.seed
        self._skip_paths = opts.l_skip_paths
        self._path_sample_seed = opts.l_path_sample_seed
        if any(['closed-path' in _ for _ in opts.l_gen_strategy]):
            if opts.e_device != "cpu":
                self._workers = torch.cuda.device_count()
                logger.info(
                    "Path-based counting with multiprocessing. Setting GPU workers to {}".format(self._workers)
                )
            self._num_paths = opts.l_num_paths
        else:
            raise ValueError("Invalid path generation strategy.")

    def compute(self, model, data_source):
        # if it's not iterable, then we assume it's a callable, and when given the
        # model, it can construct an iterable.
        maybe_iterable = self.data_sources[data_source]
        if not isinstance(maybe_iterable, Iterable):
            maybe_iterable = maybe_iterable(model)
            if not isinstance(maybe_iterable, Iterable):
                raise TypeError("Tried to create iterable but still is not iterable")

        if self._workers <= 1:
            self._sync_compute(model, maybe_iterable, data_source)
        else:
            self._thread_pool_compute(model, maybe_iterable, data_source)

    def _do_compute(self, args):
        """ Parallel function
        """
        rank = -1
        for idx in range(len(gpu_locks)):
            if gpu_locks[idx].acquire(block=False):
                rank = idx
                break
        if rank == -1:
            raise Exception("Unable to acquire GPU lock")

        p_model = args[0].to(rank, non_blocking=True)
        data, labels, data_indices = args[3]
        data = data.to(rank, non_blocking=True)
        labels = labels.to(rank, non_blocking=True)
        kwargs = {"rank": rank}
        self.compute_one(
            p_model, (data, labels, data_indices), args[2], seed=args[4], uid=args[1], **kwargs
        )
        gpu_locks[rank].release()
        
    def _sync_compute(self, model, data_iterable, data_source):
        # seeding torch to ensure paths are visited always in the same order
        rng_state = torch.get_rng_state()
        torch.manual_seed(self._path_sample_seed)
        
        data_iter_iter = iter(data_iterable)
        torch.set_rng_state(rng_state)        
        
        logger.info("Initializing data_loader with seed {} to ensure reproducibility.".format(self._path_sample_seed))
        
        if self._skip_paths is not None:
            batch_size = 1
            index = 0
            while index < self._skip_paths:
                batch = next(data_iter_iter)
                batch_size = batch[0].shape[0]
                index += batch_size
                
        model.to(self._device, non_blocking=True)
        
        ind_random = random.Random(self._path_sample_seed)

        def _infinite_prng_seed():
            while True:
                yield ind_random.randint(0, 2 ** 20)
                
        uid_offset = int(self._skip_paths // batch_size) if self._skip_paths is not None else 0
        for datum, uid in zip(
            [ next(data_iter_iter) for i in range(self._num_paths) ],
            range(uid_offset, self._num_paths + uid_offset),
        ):
            data, labels, data_indices = datum
            data = data.to(self._device, non_blocking=True)
            labels = labels.to(self._device, non_blocking=True)

            self.compute_one(
                model, (data, labels, data_indices), data_source, uid=uid
            )
        

    def _thread_pool_compute(self, model, data_iterable, data_source):
        logger.info("Creating process pool service")
        if self.e_device == "cuda":
            logger.info("CUDA wanted/detected, using torch.multiprocessing")
            proc_pool = torch.multiprocessing.Pool
            lock_cls = torch.multiprocessing.Lock
        else:
            proc_pool = MPPool
            lock_cls = MPLock

        def _infinite_copy(to_copy, deep=False):
            while True:
                if deep:
                    cpy = deepcopy(to_copy)
                else:
                    cpy = to_copy
                yield cpy

        ind_random = random.Random(self._path_sample_seed)

        def _infinite_prng_seed():
            while True:
                yield ind_random.randint(0, 2 ** 20)

        def _to_cpu(sample, label, indices):
            return sample.cpu().share_memory_(), label.cpu().share_memory_(), indices.cpu().share_memory_()

        # seeding torch to ensure paths are visited always in the same order
        rng_state = torch.get_rng_state()
        torch.manual_seed(self._path_sample_seed)
        
        data_iter_iter = iter(data_iterable)
        torch.set_rng_state(rng_state)        
        
        logger.info("Initializing data_loader with seed {} to ensure reproducibility.".format(self._path_sample_seed))
        
        if self._skip_paths is not None:
            batch_size = 1
            index = 0
            while index < self._skip_paths:
                batch = next(data_iter_iter)
                batch_size = batch[0].shape[0]
                index += batch_size
        
        model = model.cpu()
        
        uid_offset = int(self._skip_paths // batch_size) if self._skip_paths is not None else 0
        model_and_data_iterable = zip(
            _infinite_copy(model, deep=True),
            range(uid_offset, self._num_paths + uid_offset),
            _infinite_copy(data_source),
            [_to_cpu(*next(data_iter_iter)) for i in range(self._num_paths)],
            _infinite_prng_seed(),
        )
        logger.info("Starting proc pool")
        locks = [lock_cls() for _ in range(self._workers)]
        with proc_pool(
            self._workers, initializer=_init_pool, initargs=(locks,)
        ) as executor:
            logger.info("Submitting to proc pool")
            executor.imap(
                self._do_compute,
                model_and_data_iterable,
            )
            
            executor.close()
            executor.join()
        logger.info("Thread pool service exiting")

    @abc.abstractmethod
    def compute_one(self, model, datum, data_source, *args, **kwargs):
        """Compute the metric here"""


def convert_maybe_model_to_countable(maybe_model, shape, batch_size, buff_size=None, device="cpu"):
    if not isinstance(maybe_model, counting.Countable) or not isinstance(maybe_model, ModuleCounter):
        return ModuleCounter(maybe_model, input_shape=shape, device=device, batch_size=batch_size, buff_size=buff_size)
    return maybe_model


class LineMetric(PointMetric):
    def _init_opts(self, opts, **kwargs):
        self._device = opts.e_device
        self._save_dir = all_log_dir(opts)
        self._split = None
        self._buff_size = opts.l_buff_size
        if opts.l_load_checkpoints is not None:
            self._checkpoint = opts.l_load_checkpoints[0]
        else:
            self._checkpoint = None
        super()._init_opts(opts, **kwargs)
        self._batch_size = opts.batch_size
        rescale_num_paths_by_batch_size = True
        for gen_strategy in self.data_sources.keys():
            split = gen_strategy
            self._split = split
            avail_points = len(self.data_sources[gen_strategy]) * opts.batch_size
            assert self._num_paths <= avail_points, "Error:" + \
                "requested {} paths, but the {} split of dataset contains only {} points.".format(
                        self._num_paths,
                        split,
                        avail_points
                        )
            assert (self._num_paths % opts.batch_size == 0), "Error: BATCH_SIZE: {} must divide L_NUM_PATHS: {}".format(opts.batch_size, self._num_paths)
            if rescale_num_paths_by_batch_size:
                self._num_paths //= opts.batch_size
                self._batch_size *=  opts.l_num_anchors
                rescale_num_paths_by_batch_size = False
                logger.info(
                    "Path-based counting on {} batches of size {} for the {} split. Effective batch size is {}".format(
                        self._num_paths,
                        opts.batch_size,
                        split,
                        self._batch_size * 2
                    )
                )

    def compute_one(self, model, datum, data_key, *args, **kwargs):
        dev = "cpu" if self._device == "cpu" else "cuda:0"
        device = kwargs.pop("rank", dev)
        torch.set_default_dtype(torch.float64)
        real_data = datum
        if isinstance(datum, tuple):
            real_data, labels, data_indices = datum
        countable_obj = convert_maybe_model_to_countable(
            model,
            real_data[0][0].shape,
            device=device,
            batch_size=self._batch_size,
            buff_size=self._buff_size
        )
        self._device = device
        stats = self.closed_path_compute(countable_obj, datum)
        fname = 'path-counting'
        fname = os.path.join(self._save_dir, fname)
        if self._split is not None:
            fname += '-' + str(self._split)
        if self._checkpoint is not None:
            fname = "{}-checkpoint-{}".format(fname, self._checkpoint)
        fname = "{}-id-{}.json".format(fname, kwargs["uid"])
        logger.info("Saving results to {}".format(fname))
        with open(fname, "w") as intermediate_result_file:
            json.dump(stats, intermediate_result_file, allow_nan=False)

    @abc.abstractmethod
    def closed_path_compute(self, countable_obj, data):
        """"data is a list of lists. The last list level is a single path"""


class LineCounts(LineMetric):

    def _init_opts(self, opts):
        super()._init_opts(opts)

    def name(self):
        return Metrics.LINE_COUNTS.value

    def closed_path_compute(self, countable_obj, data):
        logger.info("Computing statistics over closed paths")
        paths, labels, data_indices = data

        local_batch_size, num_anchors = paths.shape[:2]
        from_pts = paths.reshape(
            (local_batch_size * num_anchors,) + tuple(paths.shape[2:])
        )
        to_pts = from_pts.roll(shifts=-1, dims=0)

        pts, logits, density, tot_vars, abs_deviation, variation_interpolated = countable_obj.count(
            from_pts,
            to_pts,
            device=self._device,
        )
        
        results = [
            {
                k : {
                        "density": 0,
                        "logits" : [],
                        "points" : [],
                        "variation" : [],
                        "variation_interpolated" : [],
                        "absolute_deviation" : [],
                } for k in range(num_anchors)
           } for _ in range(local_batch_size)
        ]
        
        nclasses = logits.shape[1]
        for i in range(len(logits)): # batch size
            l = int(i // num_anchors) # batch_idx
            k = i % num_anchors # line_idx
            
            # trim buffers, convert to nested lists
            results[l][k]["density"] = density[i].item()            
            results[l][k]["logits"] = logits[i, :, :density[i], :].reshape(nclasses, -1, 2).tolist()
            results[l][k]["points"] = pts[i, :density[i], :].reshape(-1, 2).tolist()
            results[l][k]["variation"] = tot_vars[i, :, :density[i]].reshape(nclasses, -1).tolist()
            results[l][k]["variation_interpolated"] = variation_interpolated[i].tolist()
            results[l][k]["absolute_deviation"] = abs_deviation[i, :, :density[i]].reshape(nclasses, -1).tolist()
            
            if k == 0:
                results[l]["label"] = labels[l].to("cpu", non_blocking=True).item()
                results[l]["item_idx"] = data_indices[l].to("cpu", non_blocking=True).item()

        if isinstance(countable_obj, ModuleCounter):
            countable_obj.remove_all_handles()
        return results        


class Metrics(Enum):
    LINE_COUNTS = "absolute_deviation"


METRICS = {
    Metrics.LINE_COUNTS.value: LineCounts,
}


def create_metric(metric_name, data_sources, opts=None, **kwargs):
    metric_factory = METRICS[metric_name]
    return metric_factory(data_sources, opts=opts, **kwargs)
