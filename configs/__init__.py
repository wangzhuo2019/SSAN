import argparse
import os
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    # build dirs
    parser.add_argument('--data_dir', type=str, default="", help='YOUR_Data_Dir')
    parser.add_argument('--result_path', type=str, default='./results', help='root result directory')
    parser.add_argument('--result_name', type=str, default='demo', help='result directory')
    # training settings
    parser.add_argument('--model_type', type=str, default="SSAN-M", help='model_type')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--img_size', type=int, default=256, help='img size')
    parser.add_argument('--map_size', type=int, default=32, help='depth map size')
    parser.add_argument('--protocol', type=str, default="O_C_I_to_M", help='protocal')
    parser.add_argument('--device', type=str, default='0,1', help='device id, format is like 0,1,2')
    parser.add_argument('--base_lr', type=float, default=0.0001, help='base learning rate')
    parser.add_argument('--start_epoch', type=int, default=0, help='start epoch')
    parser.add_argument('--num_epochs', type=int, default=1200, help='total training epochs')
    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--step_size', type=int, default=500, help='how many epochs lr decays once')
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma of optim.lr_scheduler.StepLR, decay of lr')
    parser.add_argument('--trans', type=str, default="p", help="different pre-process")
    # optimizer
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.00005)
    # debug
    parser.add_argument('--debug_subset_size', type=int, default=None)
    return parser.parse_args()


def str2bool(x):
    return x.lower() in ('true')
    
