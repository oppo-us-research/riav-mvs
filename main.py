"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under license: MIT
* For full license text, see LICENSE file in the repo root
"""

import os
import torch
import numpy as np

# this project related #
from src.options import MVSdepthOptions
from config.config import Config

if __name__ == "__main__":

    #----- global config -----#
    options = MVSdepthOptions()
    opts = options.parse()
    mycfg = Config(opts)
    
    torch.manual_seed(mycfg.opt.seed)
    torch.cuda.manual_seed(mycfg.opt.seed)
    np.random.seed(mycfg.opt.seed)
    
    if mycfg.opt.mode in ["train", "resume"]:
        from trainer import run_train
        print (f"[!!!] Run {mycfg.opt.mode} for {opts.network_class_name} !!!")
        run_train(cfg = mycfg)
    
    elif mycfg.opt.mode in ["test", "benchmark"]:
        from test import run_evaluate
        print ("run evaluate")
        run_evaluate(opt = mycfg.opt)
    
    elif mycfg.opts.eval_task == "eval_metric_only":
        from test import run_evaluate_saved_results
        print ("Already got the results, just run metric evaluation")
        print("    --->  loading image paths")

        if opts.your_exp_name: # you specified dir
            pred_depth_dir = os.path.join(opts.run_dir, "results", opts.your_exp_name)
        else:
            pred_depth_dir = opts.result_dir

        name_map_txt_file = os.path.join(
            pred_depth_dir,
            "{}_imgs_path.txt".format(opts.split))
        run_evaluate_saved_results(
            opts,
            name_map_txt_file,
            pred_depth_dir
            )
    
    
    if 0: # check config
        print ("[!!!] cehcking Config for {}".format(opts.network_class_name))
        mycfg.print_paths()
        mycfg.save_opts()
