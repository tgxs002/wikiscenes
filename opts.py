from __future__ import print_function

import os
import torch
import argparse
from core.config import cfg

def add_global_arguments(parser):

    parser.add_argument("--dataset", type=str,
                        help="Determines dataloader to use (only Pascal VOC supported)")
    parser.add_argument("--exp", type=str, default="main",
                        help="ID of the experiment (multiple runs)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Snapshot \"ID,iter\" to load")
    parser.add_argument("--run", type=str, help="ID of the run")
    parser.add_argument('--workers', type=int, default=8,
                        metavar='N', help='dataloader threads')
    parser.add_argument("--snapshot-dir", type=str, default='./snapshots',
                        help="Where to save snapshots of the model.")
    parser.add_argument("--logdir", type=str, default='./logs',
                        help="Where to save log files of the model.")

    # used at inference only
    parser.add_argument("--infer-list", type=str, default='data/val_augvoc.txt',
                        help="Path to a file list")
    parser.add_argument("--mask-output-dir", type=str, default='results/',
                        help="Path where to save masks")

    # hypter parameter
    parser.add_argument('--feature', type=str, choices=["conv3", "backbone", "sg", "decoder", "fc8"], default="decoder", help="Which layer to put constraint on.")
    parser.add_argument("--loss_3d", type=float, default=0.0, help="the coefficint of 3d_consistency loss")
    parser.add_argument("--triplet_margin", type=float, default=3.0, help="the margin value for triplet loss used in 3D consistency")
    parser.add_argument("--use_triplet", action='store_true')
    parser.add_argument('--export', type=str, choices=["features","kp_score", "AP", "None", "segmentation", "acc", "kp_feature"], default="None", help="What to export.")
    parser.add_argument('--export_set', type=str, default="train", help="on which set to export")
    parser.add_argument("--normalize_feature", action='store_true', help="whether to normalize the feature used for consistency")
    parser.add_argument("--multi_label", action='store_true')
    parser.add_argument("--use_contrastive", action='store_true')
    parser.add_argument("--use_contrastive_easy", action='store_true')
    parser.add_argument('--num_negative', type=int, default=10, metavar='N', help='number of negative cases to sample')
    parser.add_argument("--contrastive_tau", type=float, default=0.07, help="the temperature of contrastive loss")
    parser.add_argument('--pretrain', type=int, default=-1, metavar='N', help='finetune (do not use 3d loss) after a certain epochs. -1 means disabled.')
    parser.add_argument('--lr_milestones', nargs="+", default=[20, 25])

    #
    # Configuration
    #
    parser.add_argument(
        '--cfg', dest='cfg_file', required=True,
        help='Config file for training (and optionally testing)')
    parser.add_argument(
        '--set', dest='set_cfgs',
        help='Set config keys. Key value sequence seperate by whitespace.'
             'e.g. [key] [value] [key] [value]',
        default=[], nargs='+')

    parser.add_argument("--random-seed", type=int, default=64, help="Random seed")


def maybe_create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def check_global_arguments(args):

    torch.set_num_threads(args.workers)
    if args.workers != torch.get_num_threads():
        print("Warning: # of threads is only ", torch.get_num_threads())

    setattr(args, "fixed_batch_path", os.path.join(args.logdir, args.dataset, args.exp, "fixed_batch.pt"))
    args.logdir = os.path.join(args.logdir, args.dataset, args.exp, args.run)
    maybe_create_dir(args.logdir)
    #print("Saving events in: {}".format(args.logdir))

    #
    # Model directories
    #
    args.snapshot_dir = os.path.join(args.snapshot_dir, args.dataset, args.exp, args.run)
    maybe_create_dir(args.snapshot_dir)
    #print("Saving snapshots in: {}".format(args.snapshot_dir))

def get_arguments(args_in):
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Model Evaluation")

    add_global_arguments(parser)
    args = parser.parse_args(args_in)
    check_global_arguments(args)

    return args
