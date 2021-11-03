# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import torch
import logging
import json
import ijson

from core.cmd import create_default_args
from core.utils import all_log_dir, init_logging, prepare_dirs

class NumpyEncoder(json.JSONEncoder):
    """Json encoder for numpy ndarrays 
       Taken from: https://stackoverflow.com/a/49677241/14216894
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class TensorEncoder(json.JSONEncoder):
    """Json encoder for torch tensors
    """
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.to("cpu").tolist()
        return json.JSONEncoder.default(self, obj)


def init_buffers(npaths, nclasses, nlines):
    aggregated_stats = {
        "labels" : torch.zeros(npaths, dtype=torch.long, device="cpu"),
        
        "item_ids" : torch.zeros(npaths, dtype=torch.long, device="cpu"),
    
        "line_stats" : {
            "absolute_deviation" : torch.zeros((npaths, nclasses, nlines), device=device),
            "variation_interpolated" : torch.zeros((npaths, nclasses, nlines), device=device),
            "variation" : torch.zeros((npaths, nclasses, nlines), device=device),
            "density" : torch.zeros((npaths, nlines), device=device),
        },
        
        "cumulative_path_stats" : {
            "absolute_deviation" : torch.zeros((npaths, nclasses), device=device),
            "variation_interpolated" : torch.zeros((npaths, nclasses), device=device),
            "variation" : torch.zeros((npaths, nclasses), device=device),
            "density" : torch.zeros((npaths), device=device),
        },
        
        "average_stats" : {
            "absolute_deviation" : torch.zeros((nclasses, 2), device=device),
            "variation_interpolated" : torch.zeros((nclasses, 2), device=device),
            "variation" : torch.zeros((nclasses, 2), device=device),
            "density" : torch.zeros(2, device=device),
        }
    }
    
    return aggregated_stats


def aggregate_statistics(batches, npaths):
    """Aggregate per-path statistics:
           - cumulative absolute deviation over lines (N, K, L)
           - cumulative variation interpolated over lines (N, K, L)
           - cumulative directional variation over lines (N, K, L)
           - cumulative density over lines (N,L)
    """    
    for path_id, path in enumerate(batches):
        if path_id == 0:
            nlines = len(path.keys()) -2 # ignore 2 metadata keys
            nclasses = len(path["0"]["logits"])
            logger.info("Found {} classes, {} lines".format(nclasses, nlines))
            
            logger.info("Populating buffers")
            aggregated_stats = init_buffers(npaths, nclasses, nlines)
        
        for line_id, line in enumerate(path):
            if line == "label":
                aggregated_stats["labels"][path_id] = path[line]
                continue
            if line == "item_idx":
                aggregated_stats["item_ids"][path_id] = path[line]
                continue
            
            for key in aggregated_stats["line_stats"]:
                if key == "density":
                    aggregated_stats["line_stats"][key][path_id, line_id] = \
                        torch.as_tensor(path[line][key], device=device)
                elif key == "variation_interpolated":
                    aggregated_stats["line_stats"][key][path_id, :, line_id] = \
                        torch.as_tensor(path[line][key], device=device)
                else:
                    # sum over activation regions
                    aggregated_stats["line_stats"][key][path_id, :, line_id] = \
                        torch.as_tensor(path[line][key], device=device).sum(dim=1)
        
        for key in aggregated_stats["cumulative_path_stats"]:
            # cumulative sum over lines
            if key == "density":
                aggregated_stats["cumulative_path_stats"][key][path_id] = \
                    aggregated_stats["line_stats"][key][path_id, :].sum()
            else:
                aggregated_stats["cumulative_path_stats"][key][path_id, :] = \
                    aggregated_stats["line_stats"][key][path_id, :, :].sum(dim=1)

        if path_id > 0 and (path_id +1) % 64 == 0:
            logger.info("Processed {} paths".format(path_id+1))
        
    for key in aggregated_stats["average_stats"]:
        # average over all paths of cumulatives along each line
        if key == "density":
            aggregated_stats["average_stats"][key][0] = \
                aggregated_stats["line_stats"][key].view(-1).mean()
            aggregated_stats["average_stats"][key][1] = \
                aggregated_stats["line_stats"][key].view(-1).std()
        else:
            # average over all lines and paths
            aggregated_stats["average_stats"][key][:, 0] = \
                aggregated_stats["line_stats"][key].transpose(1,0).reshape((nclasses, -1)).mean(dim=1)
            aggregated_stats["average_stats"][key][:, 1] = \
                aggregated_stats["line_stats"][key].transpose(1,0).reshape((nclasses, -1)).std(dim=1)
        
    return aggregated_stats


def add_metadata(results, args):
    """Add metadata to the results dictionary to make the experiment self contained
    """
    metadata = {
        "model" : args.model,
        "dataset" : args.data,
        "split" : args.dataset_split,
        "seed" : args.seed,
        "label-noise" : args.label_noise,
        "augmentation" : args.augmentation,
        "checkpoint" : args.checkpoint_id,
        "npaths" : args.npaths,
        "label-noise-seed" : args.label_seed,
        "data-split-seed" : args.data_split_seed,
        "path-sample-seed" : args.path_sample_seed,
    }
    results["metadata"] = metadata
    
    return results
    
    
def parse_results(fp):
    """Parse the list of pickle files speficied 
       by file pointer @fp, and yields each entry
    """
    for line in fp:
        line = line.strip()
        if os.path.exists(line):
            logger.info("Loading {}".format(line))
        else:
            logger.info("Skipping {}: file not found.".format(line))
        with open(line, 'rb') as path_batch_results:
            batches = ijson.items(path_batch_results, "item", use_float=True)
            for batch in batches:
                yield batch


def main(args):
    """Main"""

    save_path = prepare_dirs(args)
    global logger
    logger = logging.getLogger()
    torch.set_default_dtype(torch.float64)
    
    global device
    if torch.cuda.is_available() and args.e_device == "cuda":
        logger.info("Using cuda")
        device = torch.device("cuda")
    else:
        logger.info("Using cpu")
        device = torch.device("cpu")

    if not os.path.exists(args.load_from):
        raise IOError("File not found: {}".format(args.load_from))
    
    logger.info("Loading results from {}".format(args.load_from))
    
    with open(args.load_from, 'r') as fp:
        results_gen = parse_results(fp)
        logger.info("Aggregating statistics")
        with torch.no_grad():
            path_stats = aggregate_statistics(results_gen, args.npaths)
    
    add_metadata(path_stats, args)
    
    results_filename = args.output + '_' + args.dataset_split + '_checkpoint-' + str(args.checkpoint_id)
    results_filename = os.path.join(save_path, results_filename)
    results_filename += ".json"
    logger.info("Saving results to {}".format(results_filename))
    
    # move all tensors to cpu and serialize
    json_dict = json.dumps(path_stats, cls=TensorEncoder, allow_nan=False)
    
    with open(
        results_filename, "w"
    ) as write_stats:
        json.dump(json_dict, write_stats)


def add_local_args(parser):
    """Parse command line arguments
    """
    args_group = parser.add_argument_group("Aggregates statistics of path-based activation region analysis.")    
    args_group.add_argument("--npaths", type=int, default=1024, help="Total number of paths.")

    # paths to saved json results
    args_group.add_argument("--load-from", type=str, default='', help="File with the list of saved results, one for each line.")
    args_group.add_argument("--output", type=str, default='', help="Output filename.")
    
    args_group.add_argument("--checkpoint-id", type=str, default='', help="Checkpoint id. Used to add metadata to the results dictionary.")
    
    args_group.add_argument("--dataset-split", type=str, default='', help="Dataset split used to compute the statistics being aggregated. Used to add metadata to the results dictionary.")
    
    args_group.add_argument("--path-sample-seed", type=int, default=4321, help="The seed that was used to sample the paths used to compute the statistics. Used to add metadata to the results dictionary.")
    

def get_arg_parser():
    parser = create_default_args()
    add_local_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    import signal
    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

    args = get_arg_parser()
    main(args)
