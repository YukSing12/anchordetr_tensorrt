# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import os
import argparse
import datetime
import json
import random
import time
from pathlib import Path
from glob import glob 
import numpy as np
import torch
from torch.utils.data import DataLoader
import datasets
import util.misc as utils
import datasets.samplers as samplers
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, evaluate_trt, train_one_epoch, evaluate_onnx, evaluate_trt_async
from models import build_model
import ctypes
import tensorrt as trt

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
    parser.add_argument('--num_images', default=5000, type=int, help="numbers of image to be inferenced")
     # TensorRT params
    parser.add_argument('--trt_path', default=['AnchorDETR.plan'], type=str, nargs='+')
    parser.add_argument('--dynamic_shape', default=False, action='store_true', help='whether to use dynamic shape as input')
    parser.add_argument('--use_memory_pool', default=False, action='store_true', help='whether to use memeory pool for reuse')
    parser.add_argument('--execute_async', default=False, action='store_true', help='asynchronous inference')
    parser.add_argument('--plugins', required=True, type=str, help='Directory of trt plugins to load')
    parser.add_argument('--deepstream', action='store_true', default=False, help='Export onnx for deepstream')

    return parser


def main(args):
    print(args)
    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, '')

    soFileList = glob(os.path.join(args.plugins,"*.so"))

    if len(soFileList) > 0:
        print("Find Plugin %s!"%soFileList)
    else:
        print("No Plugin!")
    for soFile in soFileList:
        ctypes.cdll.LoadLibrary(soFile)

    engines = []
    contexts = []
    for trt_file in args.trt_path:
        if os.path.isfile(trt_file):
            with open(trt_file, 'rb') as encoderF:
                engine = trt.Runtime(logger).deserialize_cuda_engine(encoderF.read())
            if engine is None:
                print("Failed loading %s"%trt_file)
                return
            print("Succeeded loading %s"%trt_file)
        else:
            print("Failed finding %s"%trt_file)
            return
        engines.append(engine)
        contexts.append(engine.create_execution_context())
    # assert len(engines) == 2

    device = torch.device(args.device)
    _, criterion, postprocessors = build_model(args)
    dataset_val = build_dataset(image_set=args.eval_set, args=args)
    if args.num_images < 5000:
        dataset_val.ids = dataset_val.ids[:args.num_images]
        
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)


    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                 pin_memory=True)


    
    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

   
    output_dir = Path(args.output_dir)
    if args.execute_async:
        stream = torch.cuda.Stream()
        test_stats, coco_evaluator = evaluate_trt_async(engines, contexts, stream.cuda_stream, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir, save_json=True, args=args)
    else:
        test_stats, coco_evaluator = evaluate_trt(engines, contexts, criterion, postprocessors,
                                                data_loader_val, base_ds, device, args.output_dir, save_json=True, args=args)


    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('AnchorDETR TensorRT inference script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
