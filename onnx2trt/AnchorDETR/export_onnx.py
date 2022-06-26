import os
import argparse
import datetime
import json
import random
import time
from pathlib import Path
from idna import check_hyphen_ok
from matplotlib.pyplot import style

import numpy as np
import torch
from torch.utils.data import DataLoader
# import datasets
import util.misc as utils
import datasets.samplers as samplers
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model
import onnx



def get_args_parser():
    parser = argparse.ArgumentParser('AnchorDETR Detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=[], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    parser.add_argument('--sgd', action='store_true')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', default=True,
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--num_feature_levels', default=1, type=int, help='number of feature levels')
    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0., type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_query_position', default=300, type=int,
                        help="Number of query positions")
    parser.add_argument('--num_query_pattern', default=3, type=int,
                        help="Number of query patterns")
    parser.add_argument('--spatial_prior', default='learned', choices=['learned', 'grid'],
                        type=str,help="Number of query patterns")
    parser.add_argument('--attention_type',
                        # default='nn.MultiheadAttention',
                        default="RCDA",
                        choices=['RCDA', 'nn.MultiheadAttention'],
                        type=str,help="Type of attention module")
    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--eval_set', default='val', choices=['val', 'test'],
                        type=str,help="dataset to evaluate")
    parser.add_argument('--coco_path', default='/data/coco', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--output_dir', default='/data/detr-workdir/r50-dc5',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', default=False, action='store_true', help='whether to resume from last checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')

    # ONNX params
    parser.add_argument('--onnx_path', default='AnchorDETR.onnx', type=str)
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument('--dynamic_shape', default=False, action='store_true')
    parser.add_argument('--do_constant_folding', default=False, action='store_true')


    return parser

def check_onnx(onnx_path):
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    print(f"[INFO]  ONNX model: {onnx_path} check success!")


def main(args):
    print(args)
    model, criterion, postprocessors = build_model(args)
    model_without_ddp = model

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
    
    model.eval()
    input_names = ["image", "mask"]
    output_names = ["pred_logits", "pred_boxes"]

    x=(torch.randn((1, 3, 800, 800)), torch.randn((1, 800, 800)))
    

    if not args.dynamic_shape:
        print('Export ONNX model with static shape!')
        torch.onnx.export(model, 
                            x, 
                            args.onnx_path, 
                            input_names=input_names, 
                            output_names=output_names, 
                            export_params=True, 
                            verbose=args.verbose, 
                            training=False, 
                            opset_version=13, 
                            do_constant_folding=args.do_constant_folding)
    else:
        print('Export ONNX model with dynamic shape!')
        # dynamic_axes = {'image':{0:'batch', 2:'height', 3:'width'}, 'mask':{0:'batch', 1:'height', 2:'width'}, 
        #                 'pred_logits': {0:'batch', 1:'query_size'}, 'pred_boxes': {0:'batch', 1:'query_size'}}
        dynamic_axes = {'image':{2:'height', 3:'width'}, 'mask':{1:'height', 2:'width'}}
        # dynamic_axes = {'image':{0:'batch'}, 'mask':{0:'batch'}, 
        #                 'pred_logits': {0:'batch'}, 'pred_boxes': {0:'batch'}}

        torch.onnx.export(model, 
                            x, 
                            args.onnx_path, 
                            input_names=input_names, 
                            output_names=output_names, 
                            export_params=True, 
                            verbose=args.verbose, 
                            training=False, 
                            dynamic_axes=dynamic_axes,
                            opset_version=13, 
                            do_constant_folding=args.do_constant_folding)
        # raise ValueError("no impletement for dynamic shape")
    check_onnx(args.onnx_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('AnchorDETR Pytorch2ONNX script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)