"""
* Copyright (c) 2024 OPPO. All rights reserved.
* Under license: MIT
* For full license text, see LICENSE file in the repo root
"""

import os
import argparse
import yaml
try:
    from collections.abc import defaultdict
    from collections.abc import namedtuple
except ImportError:
    from collections import defaultdict
    from collections import namedtuple


# the directory this file options.py resides in
#file_dir = os.path.dirname(__file__)  

def parse_yaml(file_path: str) -> namedtuple:
     '''Parse yaml configuration file and return the object 
     in `namedtuple`.
     '''
     with open(file_path, "rb") as f:
          cfg: dict = yaml.safe_load(f)
     #args = namedtuple("train_args", cfg.keys())(*cfg.values())
     args = cfg
     return args

class MVSdepthOptions:
     def __init__(self):
          self.parser = argparse.ArgumentParser(
               description= "A PyTorch Implementation of RIAV-MVS"
               )

          # general
          self.parser.add_argument('--default_yaml_file',
                        help='default configure yaml file',
                        required=True,
                        type=str)

          self.parser.add_argument('--extra_yaml_file',
                        help='extra experiment configure yaml file',
                        required=True,
                        type=str) 

          
          self.parser.add_argument("--dataset",
                                  type=str,
                                  help="dataset to train on",
                                  required=True,
                                  )

          self.parser.add_argument("--height",
                                   type=int,
                                   required=True,
                                   help="input image height"
                                   )
          
          self.parser.add_argument("--width",
                                   type=int,
                                   required=True,
                                   help="input image width",
                                   )


          self.parser.add_argument("--min_depth",
                                   type=float,
                                   help="minimum depth",
                                   required=True,
                                   )
          self.parser.add_argument("--max_depth",
                                   type=float,
                                   required=True,
                                   help="maximum depth",
                                   )

          # OPTIMIZATION options
          self.parser.add_argument("--batch_size",
                                   type=int,
                                   default=4,
                                   help='mini-batch size, this is the total '
                                        'batch size of all GPUs on the current node when '
                                        'using Data Parallel or Distributed Data Parallel'
                                   )

          self.parser.add_argument("--num_workers",
                                   type=int,
                                   default=12)

          self.parser.add_argument("--learning_rate",
                                   type=float,
                                   help="initial learning rate",
                                   default=1e-4)
          

          self.parser.add_argument('--lr_scheduler',
                                   type = str,
                                   default='constant',
                                   choices = ['constant', 'piecewise_epoch', 'OneCycleLR'],
                                   help="learning rate scheduler"
                                   )


          self.parser.add_argument("--num_epochs",
                                   type=int,
                                   help="number of epochs",
                                   required=True,
                                   )

          self.parser.add_argument("--scheduler_step_size",
                                   help = "milestones in MultiStepLR",
                                   #type = lambda s: [int(item) for item in s.split("-")],
                                   type = str,
                                   default = ''
                                   )

          self.parser.add_argument("--load_weights_path",
                                   type=str,
                                   default='',
                                   help="checkpoint file to load model weights"
                                   )

          # LOGGING options
          self.parser.add_argument("--log_frequency",
                                   type=int,
                                   help="number of batches between each tensorboard log",
                                   default=250)

          self.parser.add_argument("--save_frequency",
                                   type=int,
                                   help="number of epochs between each save",
                                   default=1)
                                   
          self.parser.add_argument('--eval_epoch',
                                   default=0,
                                   type=int,
                                   help='which epoch checkpoint to load for evaluation')


          self.parser.add_argument('--eval_gpu_id',
                                   default='0',
                                   type=str,
                                   help='which gpu will be used for evaluation')


          self.parser.add_argument('--print_freq',
                                   default=10,
                                   type=int, metavar='N',
                                   )
          self.parser.add_argument('--resume_path',
                                   default='',
                                   type=str,
                                   help='path to latest checkpoint (default: none)')

          self.parser.add_argument('--mode',
                                   default='train',
                                   type=str,
                                   choices=["resume", "train", "val", "test", "debug"],
                                   help='path to latest checkpoint (default: none)'
                                  )

          # added for RAFT:
          self.parser.add_argument('--raft_iters',
                                   type = lambda s: [int(item) for item in s.split(",")],
                                   default="8,8",
                                   help='iteration number for depth/flow update')


          self.parser.add_argument('--exp_idx',
                                   type = str,
                                   required=True,
                                   help= "Experiment idx, e.g., exp50C"
                                   )

          self.parser.add_argument('--machine_name',
                                   type = str,
                                   required=True,
                                   #default='rtxa6000s3',
                                   help= "machine name flag used in the exp name"
                                   )


          self.parser.add_argument('--eval_task',
                                    type = str,
                                    default='all', # prediction + evaluation;
                                    )
          self.parser.add_argument('--your_exp_name',
                                    type = str,
                                    default='', # specified result_dir 
                                    )


          self.parser.add_argument("--m_2_mm_scale_loss",
                                   type=float,
                                   help="change loss values in meter to those in milimeter",
                                   default=1.0e3
                                   )

     
     ## priority: parse_args from command line > extra_yaml > default_yaml;
     def update_args_from_yaml(self, default_yaml_file, extra_yaml_file):
          # default args from yaml
          args = parse_yaml(default_yaml_file)
          # new args from yaml
          if extra_yaml_file:
               new_args = parse_yaml(extra_yaml_file)
               args.update(new_args)
          
          # parameters from command lines, have priority than yaml;
          for k, v in args.items():
               if k not in self.options:
                    setattr(self.options, k, v)
                    #print (f"Added (k, v)=({k}, {getattr(self.options, k)}")
               else:
                    #print (f"  Skipped (k, v)=({k}, {v})")
                    continue
          #print (self.options.__dict__.keys())

     
     def parse(self):
          self.options = self.parser.parse_args()

          # load parameters from yaml files
          self.update_args_from_yaml(
               default_yaml_file = self.options.default_yaml_file, 
               extra_yaml_file = self.options.extra_yaml_file
               )

          if self.options.scheduler_step_size == '':
               self.options.scheduler_step_size = [-1]
          else:
               tmp_str = self.options.scheduler_step_size
               self.options.scheduler_step_size = [int(item) for item in tmp_str.split("-")]


          if self.options.freeze_layers == '':
               self.options.freeze_layers = None
          else:
               tmp_str = self.options.freeze_layers
               self.options.freeze_layers = [item for item in tmp_str.split(",")]
               for layer in self.options.freeze_layers:
                    assert layer in [
                         'pose_encoder',  'pose',
                         'fnet', 'cnet', 'gru', 'attention',
                         
                         'encoder', # for baseline methods
                         'depth', # for baseline methods
                         ]
                    #print ("will freeze layer ", layer)

          if self.options.scheduler_step_size[0] <= 0:
               # i.e., to disable lr scheduling
               # we set scheduler_step_size=(#epoch+1)
               self.options.scheduler_step_size = [self.options.num_epochs + 1]

          return self.options
