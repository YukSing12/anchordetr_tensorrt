# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable
from unittest import result

import torch
import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from datasets.data_prefetcher import data_prefetcher
import numpy as np
import tensorrt as trt
from cuda import cudart

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()

    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    for _ in metric_logger.log_every(range(len(data_loader)), print_freq, header):
        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()

        # metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(loss=loss_value)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

        samples, targets = prefetcher.next()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, save_json=False):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types, save_json=save_json)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )
    x = []
    y = []
    for samples, targets in metric_logger.log_every(data_loader, 100, header):

        x.append(samples.tensors.shape[2])
        y.append(samples.tensors.shape[3])
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples.tensors, samples.mask)
        # outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        # metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
        #                      **loss_dict_reduced_scaled,
        #                      **loss_dict_reduced_unscaled)
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()))
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    print("min x :{}, max x: {} ".format(min(x), max(x)))
    print("min y :{}, max y: {} ".format(min(y), max(y)))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    if save_json:
        all_res = utils.all_gather(coco_evaluator.results['bbox'])
        results=[]
        for p in all_res:
            results.extend(p)
        coco_evaluator.results['bbox'] = results

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator


@torch.no_grad()
def evaluate_onnx(ort_session, criterion, postprocessors, data_loader, base_ds, device, output_dir, save_json=False):
    # model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types, save_json=save_json)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    for samples, targets in metric_logger.log_every(data_loader, 100, header):
        # samples = samples.to(device)
        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        samples = samples
        targets = [{k: v for k, v in t.items()} for t in targets]

        # outputs = model(samples)
        ort_inputs = {"image":samples.tensors.numpy(), "mask":samples.mask.float().numpy()}
        if len(ort_session) == 1:
            result = ort_session[0].run(None, ort_inputs)
        else:
            backbone_result = ort_session[0].run(None, ort_inputs)
            # 1541-feat  1566-mask_out 
            transformer_inputs = {"1541":backbone_result[0], "1568":backbone_result[1]}
            result = ort_session[1].run(None, transformer_inputs)
        # result = ort_session.run(None, ort_inputs)
        # scores, boxs = ort_session.run(None, ort_inputs)
        outputs = {}
        outputs['pred_logits'] = torch.from_numpy(result[0])
        outputs['pred_boxes'] = torch.from_numpy(result[1])
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        # metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
        #                      **loss_dict_reduced_scaled,
        #                      **loss_dict_reduced_unscaled)
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()))
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    if save_json:
        all_res = utils.all_gather(coco_evaluator.results['bbox'])
        results=[]
        for p in all_res:
            results.extend(p)
        coco_evaluator.results['bbox'] = results

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator



@torch.no_grad()
def evaluate_trt(engines, contexts, criterion, postprocessors, data_loader, base_ds, device, output_dir, save_json=False, args=None):
    # model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types, save_json=save_json)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    if len(engines) < 2:
        if args.use_memory_pool:
            input_memory_pool = {}
            output_memory_pool = {}
        anchorDETR_nInput = np.sum([ engines[0].binding_is_input(i) for i in range(engines[0].num_bindings) ])
        anchorDETR_nOutput = engines[0].num_bindings - anchorDETR_nInput
    else:
        backbone_nInput = np.sum([ engines[0].binding_is_input(i) for i in range(engines[0].num_bindings) ])
        backbone_nOnput = engines[0].num_bindings - backbone_nInput
        transformer_nInput = np.sum([ engines[1].binding_is_input(i) for i in range(engines[1].num_bindings) ])
        transformer_nOnput = engines[1].num_bindings - transformer_nInput
    for samples, targets in metric_logger.log_every(data_loader, 100, header):
        samples = samples
        targets = [{k: v for k, v in t.items()} for t in targets]

        if len(engines) < 2:
            if args.use_memory_pool:
                contexts[0].set_binding_shape(0, samples.tensors.shape)
                contexts[0].set_binding_shape(1, samples.mask.shape)
                anchorDETR_bufferH = []
                anchorDETR_bufferH.append( samples.tensors.numpy().astype(np.float32).reshape(-1) )
                anchorDETR_bufferH.append( samples.mask.numpy().astype(np.float32).reshape(-1) )
                for i in range(anchorDETR_nInput, anchorDETR_nInput + anchorDETR_nOutput):                
                    anchorDETR_bufferH.append(np.empty(contexts[0].get_binding_shape(i), dtype=trt.nptype(engines[0].get_binding_dtype(i))) )
                assert len(anchorDETR_bufferH) == 4
                anchorDETR_bufferD = []
                for i in range( anchorDETR_nInput):   
                    if anchorDETR_bufferH[i].nbytes not in input_memory_pool.keys():
                        input_memory_pool[anchorDETR_bufferH[i].nbytes] = cudart.cudaMalloc(anchorDETR_bufferH[i].nbytes)[1]
                    anchorDETR_bufferD.append(input_memory_pool[anchorDETR_bufferH[i].nbytes])
                for i in range(anchorDETR_nInput, anchorDETR_nInput + anchorDETR_nOutput):   
                    if anchorDETR_bufferH[i].nbytes not in output_memory_pool.keys():
                        output_memory_pool[anchorDETR_bufferH[i].nbytes] = cudart.cudaMalloc(anchorDETR_bufferH[i].nbytes)[1]
                    anchorDETR_bufferD.append(output_memory_pool[anchorDETR_bufferH[i].nbytes])

                for i in range(anchorDETR_nInput):
                    cudart.cudaMemcpy(anchorDETR_bufferD[i], anchorDETR_bufferH[i].ctypes.data, anchorDETR_bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
                contexts[0].execute_v2(anchorDETR_bufferD)
                for i in range(anchorDETR_nInput, anchorDETR_nInput + anchorDETR_nOutput):  
                    cudart.cudaMemcpy(anchorDETR_bufferH[i].ctypes.data, anchorDETR_bufferD[i], anchorDETR_bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
                pred_logit_idx = engines[0].get_binding_index('pred_logits')
                pred_boxes_idx = engines[0].get_binding_index('pred_boxes')
                outputs = {}
                outputs['pred_logits'] = torch.from_numpy(anchorDETR_bufferH[pred_logit_idx].reshape(engines[0].get_binding_shape(pred_logit_idx)))
                outputs['pred_boxes'] = torch.from_numpy(anchorDETR_bufferH[pred_boxes_idx].reshape(engines[0].get_binding_shape(pred_boxes_idx)))

                # for d in anchorDETR_bufferD:
                #     cudart.cudaFree(d)

            else:
                contexts[0].set_binding_shape(0, samples.tensors.shape)
                contexts[0].set_binding_shape(1, samples.mask.shape)
                anchorDETR_bufferH = []
                anchorDETR_bufferH.append( samples.tensors.numpy().astype(np.float32).reshape(-1) )
                anchorDETR_bufferH.append( samples.mask.numpy().astype(np.float32).reshape(-1) )
                for i in range(anchorDETR_nInput, anchorDETR_nInput + anchorDETR_nOutput):                
                    anchorDETR_bufferH.append(np.empty(contexts[0].get_binding_shape(i), dtype=trt.nptype(engines[0].get_binding_dtype(i))) )
                assert len(anchorDETR_bufferH) == 4
                anchorDETR_bufferD = []
                for i in range( anchorDETR_nInput + anchorDETR_nOutput):                
                    anchorDETR_bufferD.append( cudart.cudaMalloc(anchorDETR_bufferH[i].nbytes)[1] )
                for i in range(anchorDETR_nInput):
                    cudart.cudaMemcpy(anchorDETR_bufferD[i], anchorDETR_bufferH[i].ctypes.data, anchorDETR_bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
                contexts[0].execute_v2(anchorDETR_bufferD)
                for i in range(anchorDETR_nInput, anchorDETR_nInput + anchorDETR_nOutput):  
                    cudart.cudaMemcpy(anchorDETR_bufferH[i].ctypes.data, anchorDETR_bufferD[i], anchorDETR_bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
                pred_logit_idx = engines[0].get_binding_index('pred_logits')
                pred_boxes_idx = engines[0].get_binding_index('pred_boxes')
                outputs = {}
                outputs['pred_logits'] = torch.from_numpy(anchorDETR_bufferH[pred_logit_idx].reshape(engines[0].get_binding_shape(pred_logit_idx)))
                outputs['pred_boxes'] = torch.from_numpy(anchorDETR_bufferH[pred_boxes_idx].reshape(engines[0].get_binding_shape(pred_boxes_idx)))

                for d in anchorDETR_bufferD:
                    cudart.cudaFree(d)

        else:
            # backbone input shape
            contexts[0].set_binding_shape(0, samples.tensors.shape) # image
            contexts[0].set_binding_shape(1, samples.mask.shape)    # mask
            feat_idx
            if args.dynamic_shape:
                feat_h = math.ceil(samples.tensors.shape[2] / 16)
                feat_w = math.ceil(samples.tensors.shape[3] / 16)
                # feat_shape = samples.tensors.shape
                contexts[1].set_binding_shape(0, (samples.tensors.shape[0], 2048, feat_h, feat_w))
                contexts[1].set_binding_shape(1, (1, samples.tensors.shape[0], feat_h, feat_w)) # mask_out
            else:
                # contexts[1].set_binding_shape(0, engines[0].get_binding_shape(2))
                # contexts[1].set_binding_shape(1, engines[0].get_binding_shape(3))
                contexts[1].set_binding_shape(0, engines[0].get_binding_shape(3))  # 1541, feat
                contexts[1].set_binding_shape(1, engines[0].get_binding_shape(2))  # 1568, mask_out


            # prepare backbone input buffer
            backbone_bufferH = []
            backbone_bufferH.append( samples.tensors.numpy().astype(np.float32).reshape(-1) )
            backbone_bufferH.append( samples.mask.numpy().astype(np.float32).reshape(-1) )
            for i in range(backbone_nInput, backbone_nInput + backbone_nOnput):                
                backbone_bufferH.append(np.empty(contexts[0].get_binding_shape(i), dtype=trt.nptype(engines[0].get_binding_dtype(i))) )
            assert len(backbone_bufferH) == 4
            backbone_bufferD = []
            for i in range(backbone_nInput + backbone_nOnput):                
                backbone_bufferD.append( cudart.cudaMalloc(backbone_bufferH[i].nbytes)[1] )
            for i in range(backbone_nInput):
                cudart.cudaMemcpy(backbone_bufferD[i], backbone_bufferH[i].ctypes.data, backbone_bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
            
            # backbone inference
            contexts[0].execute_v2(backbone_bufferD)

            # prepare transformer input and output buffer
            transformer_bufferH = []
            transformer_bufferD = []
            # transformer_bufferD.append(backbone_bufferD[2])
            # transformer_bufferD.append(backbone_bufferD[3])
            transformer_bufferD.append(backbone_bufferD[3]) # feat-1541
            transformer_bufferD.append(backbone_bufferD[2]) # mask_out-1568
            for i in range(transformer_nInput, transformer_nInput + transformer_nOnput):                
                transformer_bufferH.append( np.empty(contexts[1].get_binding_shape(i), dtype=trt.nptype(engines[1].get_binding_dtype(i))) )
            assert len(transformer_bufferH) == 2
            for i in range(transformer_nOnput):                     
                transformer_bufferD.append( cudart.cudaMalloc(transformer_bufferH[i].nbytes)[1] )

            # transformer inference
            contexts[1].execute_v2(transformer_bufferD)
            for i in range(transformer_nInput, transformer_nInput + transformer_nOnput):  
                cudart.cudaMemcpy(transformer_bufferH[i-transformer_nInput].ctypes.data, transformer_bufferD[i], transformer_bufferH[i-transformer_nInput].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

            pred_logit_idx = engines[1].get_binding_index('pred_logits')
            pred_boxes_idx = engines[1].get_binding_index('pred_boxes')
            outputs = {}
            outputs['pred_logits'] = torch.from_numpy(transformer_bufferH[pred_logit_idx-transformer_nInput].reshape(engines[1].get_binding_shape(pred_logit_idx)))
            outputs['pred_boxes'] = torch.from_numpy(transformer_bufferH[pred_boxes_idx-transformer_nInput].reshape(engines[1].get_binding_shape(pred_boxes_idx)))

            for d in backbone_bufferD+transformer_bufferD:
                cudart.cudaFree(d)

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        # metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
        #                      **loss_dict_reduced_scaled,
        #                      **loss_dict_reduced_unscaled)
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()))
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

        
    if args.use_memory_pool:
        for k in input_memory_pool:
            cudart.cudaFree(input_memory_pool[k])
        for k in output_memory_pool:
            cudart.cudaFree(output_memory_pool[k])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    if save_json:
        all_res = utils.all_gather(coco_evaluator.results['bbox'])
        results=[]
        for p in all_res:
            results.extend(p)
        coco_evaluator.results['bbox'] = results

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator


@torch.no_grad()
def evaluate_trt_async(engines, contexts, stream, criterion, postprocessors, data_loader, base_ds, device, output_dir, save_json=False, args=None):
    # model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types, save_json=save_json)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    if len(engines) < 2:
        if args.use_memory_pool:
            input_memory_pool = {}
            output_memory_pool = {}
        anchorDETR_nInput = np.sum([ engines[0].binding_is_input(i) for i in range(engines[0].num_bindings) ])
        anchorDETR_nOutput = engines[0].num_bindings - anchorDETR_nInput
    else:
        backbone_nInput = np.sum([ engines[0].binding_is_input(i) for i in range(engines[0].num_bindings) ])
        backbone_nOnput = engines[0].num_bindings - backbone_nInput
        transformer_nInput = np.sum([ engines[1].binding_is_input(i) for i in range(engines[1].num_bindings) ])
        transformer_nOnput = engines[1].num_bindings - transformer_nInput
    
    for samples, targets in metric_logger.log_every(data_loader, 100, header):
        samples = samples
        targets = [{k: v for k, v in t.items()} for t in targets]

        if len(engines) < 2:
            if args.use_memory_pool:
                contexts[0].set_binding_shape(0, samples.tensors.shape)
                contexts[0].set_binding_shape(1, samples.mask.shape)
                anchorDETR_bufferH = []
                anchorDETR_bufferH.append( samples.tensors.numpy().astype(np.float32).reshape(-1) )
                anchorDETR_bufferH.append( samples.mask.numpy().astype(np.float32).reshape(-1) )
                for i in range(anchorDETR_nInput, anchorDETR_nInput + anchorDETR_nOutput):                
                    anchorDETR_bufferH.append(np.empty(contexts[0].get_binding_shape(i), dtype=trt.nptype(engines[0].get_binding_dtype(i))) )
                assert len(anchorDETR_bufferH) == 4
                anchorDETR_bufferD = []
                for i in range( anchorDETR_nInput):   
                    if anchorDETR_bufferH[i].nbytes not in input_memory_pool.keys():
                        input_memory_pool[anchorDETR_bufferH[i].nbytes] = cudart.cudaMallocAsync(anchorDETR_bufferH[i].nbytes, stream)[1]
                    anchorDETR_bufferD.append(input_memory_pool[anchorDETR_bufferH[i].nbytes])
                for i in range(anchorDETR_nInput, anchorDETR_nInput + anchorDETR_nOutput):   
                    if anchorDETR_bufferH[i].nbytes not in output_memory_pool.keys():
                        output_memory_pool[anchorDETR_bufferH[i].nbytes] = cudart.cudaMallocAsync(anchorDETR_bufferH[i].nbytes, stream)[1]
                    anchorDETR_bufferD.append(output_memory_pool[anchorDETR_bufferH[i].nbytes])

                for i in range(anchorDETR_nInput):
                    cudart.cudaMemcpyAsync(anchorDETR_bufferD[i], anchorDETR_bufferH[i].ctypes.data, anchorDETR_bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
                contexts[0].execute_async_v2(anchorDETR_bufferD, stream)
                for i in range(anchorDETR_nInput, anchorDETR_nInput + anchorDETR_nOutput):  
                    cudart.cudaMemcpyAsync(anchorDETR_bufferH[i].ctypes.data, anchorDETR_bufferD[i], anchorDETR_bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
                pred_logit_idx = engines[0].get_binding_index('pred_logits')
                pred_boxes_idx = engines[0].get_binding_index('pred_boxes')
                outputs = {}
                outputs['pred_logits'] = torch.from_numpy(anchorDETR_bufferH[pred_logit_idx].reshape(engines[0].get_binding_shape(pred_logit_idx)))
                outputs['pred_boxes'] = torch.from_numpy(anchorDETR_bufferH[pred_boxes_idx].reshape(engines[0].get_binding_shape(pred_boxes_idx)))

                # for d in anchorDETR_bufferD:
                #     cudart.cudaFree(d)

            else:
                contexts[0].set_binding_shape(0, samples.tensors.shape)
                contexts[0].set_binding_shape(1, samples.mask.shape)
                anchorDETR_bufferH = []
                anchorDETR_bufferH.append( samples.tensors.numpy().astype(np.float32).reshape(-1) )
                anchorDETR_bufferH.append( samples.mask.numpy().astype(np.float32).reshape(-1) )
                for i in range(anchorDETR_nInput, anchorDETR_nInput + anchorDETR_nOutput):                
                    anchorDETR_bufferH.append(np.empty(contexts[0].get_binding_shape(i), dtype=trt.nptype(engines[0].get_binding_dtype(i))) )
                assert len(anchorDETR_bufferH) == 4
                anchorDETR_bufferD = []
                for i in range( anchorDETR_nInput + anchorDETR_nOutput):                
                    anchorDETR_bufferD.append( cudart.cudaMallocAsync(anchorDETR_bufferH[i].nbytes, stream)[1])
                for i in range(anchorDETR_nInput):
                    cudart.cudaMemcpyAsync(anchorDETR_bufferD[i], anchorDETR_bufferH[i].ctypes.data, anchorDETR_bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
                contexts[0].execute_async_v2(anchorDETR_bufferD, stream)
                for i in range(anchorDETR_nInput, anchorDETR_nInput + anchorDETR_nOutput):  
                    cudart.cudaMemcpyAsync(anchorDETR_bufferH[i].ctypes.data, anchorDETR_bufferD[i], anchorDETR_bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
                pred_logit_idx = engines[0].get_binding_index('pred_logits')
                pred_boxes_idx = engines[0].get_binding_index('pred_boxes')
                outputs = {}
                outputs['pred_logits'] = torch.from_numpy(anchorDETR_bufferH[pred_logit_idx].reshape(engines[0].get_binding_shape(pred_logit_idx)))
                outputs['pred_boxes'] = torch.from_numpy(anchorDETR_bufferH[pred_boxes_idx].reshape(engines[0].get_binding_shape(pred_boxes_idx)))

                for d in anchorDETR_bufferD:
                    cudart.cudaFreeAsync(d, stream)

        else:
            # backbone input shape
            contexts[0].set_binding_shape(0, samples.tensors.shape) # image
            contexts[0].set_binding_shape(1, samples.mask.shape)    # mask
            feat_idx
            if args.dynamic_shape:
                feat_h = math.ceil(samples.tensors.shape[2] / 16)
                feat_w = math.ceil(samples.tensors.shape[3] / 16)
                # feat_shape = samples.tensors.shape
                contexts[1].set_binding_shape(0, (samples.tensors.shape[0], 2048, feat_h, feat_w))
                contexts[1].set_binding_shape(1, (1, samples.tensors.shape[0], feat_h, feat_w)) # mask_out
            else:
                # contexts[1].set_binding_shape(0, engines[0].get_binding_shape(2))
                # contexts[1].set_binding_shape(1, engines[0].get_binding_shape(3))
                contexts[1].set_binding_shape(0, engines[0].get_binding_shape(3))  # 1541, feat
                contexts[1].set_binding_shape(1, engines[0].get_binding_shape(2))  # 1568, mask_out


            # prepare backbone input buffer
            backbone_bufferH = []
            backbone_bufferH.append( samples.tensors.numpy().astype(np.float32).reshape(-1) )
            backbone_bufferH.append( samples.mask.numpy().astype(np.float32).reshape(-1) )
            for i in range(backbone_nInput, backbone_nInput + backbone_nOnput):                
                backbone_bufferH.append(np.empty(contexts[0].get_binding_shape(i), dtype=trt.nptype(engines[0].get_binding_dtype(i))) )
            assert len(backbone_bufferH) == 4
            backbone_bufferD = []
            for i in range(backbone_nInput + backbone_nOnput):                
                backbone_bufferD.append( cudart.cudaMallocAsync(backbone_bufferH[i].nbytes, stream)[1] )
            for i in range(backbone_nInput):
                cudart.cudaMemcpyAsync(backbone_bufferD[i], backbone_bufferH[i].ctypes.data, backbone_bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
            
            # backbone inference
            contexts[0].execute_async_v2(backbone_bufferD, stream)

            # prepare transformer input and output buffer
            transformer_bufferH = []
            transformer_bufferD = []
            # transformer_bufferD.append(backbone_bufferD[2])
            # transformer_bufferD.append(backbone_bufferD[3])
            transformer_bufferD.append(backbone_bufferD[3]) # feat-1541
            transformer_bufferD.append(backbone_bufferD[2]) # mask_out-1568
            for i in range(transformer_nInput, transformer_nInput + transformer_nOnput):                
                transformer_bufferH.append( np.empty(contexts[1].get_binding_shape(i), dtype=trt.nptype(engines[1].get_binding_dtype(i))) )
            assert len(transformer_bufferH) == 2
            for i in range(transformer_nOnput):                     
                transformer_bufferD.append( cudart.cudaMallocAsync(transformer_bufferH[i].nbytes, stream)[1] )

            # transformer inference
            contexts[1].execute_async_v2(transformer_bufferD, stream)
            for i in range(transformer_nInput, transformer_nInput + transformer_nOnput):  
                cudart.cudaMemcpyAsync(transformer_bufferH[i-transformer_nInput].ctypes.data, transformer_bufferD[i], transformer_bufferH[i-transformer_nInput].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)

            pred_logit_idx = engines[1].get_binding_index('pred_logits')
            pred_boxes_idx = engines[1].get_binding_index('pred_boxes')
            outputs = {}
            outputs['pred_logits'] = torch.from_numpy(transformer_bufferH[pred_logit_idx-transformer_nInput].reshape(engines[1].get_binding_shape(pred_logit_idx)))
            outputs['pred_boxes'] = torch.from_numpy(transformer_bufferH[pred_boxes_idx-transformer_nInput].reshape(engines[1].get_binding_shape(pred_boxes_idx)))

            for d in backbone_bufferD+transformer_bufferD:
                cudart.cudaFreeAsync(d, stream)

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        # metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
        #                      **loss_dict_reduced_scaled,
        #                      **loss_dict_reduced_unscaled)
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()))
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

        
    if args.use_memory_pool:
        for k in input_memory_pool:
            cudart.cudaFree(input_memory_pool[k])
        for k in output_memory_pool:
            cudart.cudaFree(output_memory_pool[k])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    if save_json:
        all_res = utils.all_gather(coco_evaluator.results['bbox'])
        results=[]
        for p in all_res:
            results.extend(p)
        coco_evaluator.results['bbox'] = results

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator