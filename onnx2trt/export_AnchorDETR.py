import os
import sys
import numpy as np
from time import time
import tensorrt as trt
import ctypes
from glob import glob
import argparse

def get_args():
    parser = argparse.ArgumentParser('Export AnchorDETR TensorRT', add_help=False)
    parser.add_argument('--onnx', type=str, help='Path of onnx file to load')
    parser.add_argument('--trt', default='', type=str, help='Path of trt engine to save')
    parser.add_argument('--mergeparams', action='store_true', default=False, help='Merge parameters or not')
    parser.add_argument('--ln', action='store_true', default=False, help='Replace ops with LayernormPlugin or not')
    parser.add_argument('--softmax', action='store_true', default=False, help='Replace ops with SoftmaxPlugin or not')
    parser.add_argument('--maskedsoftmax', action='store_true', default=False, help='Replace ops with MaskedSoftmaxPlugin or not')
    parser.add_argument('--addQbiastranspose', '-Q', action='store_true', default=False, help='Replace ops with AddQBiasTransposePlugin or not')
    parser.add_argument('--addVbiastranspose', '-V', action='store_true', default=False, help='Replace ops with AddVBiasTransposePlugin or not')
    parser.add_argument('--mask2pos', action='store_true', default=False, help='Replace ops with Mask2PosPlugin or not')
    parser.add_argument('--optshape', action='store_true', default=False, help='Optimize input shape or not')
    parser.add_argument('--twodmm', action='store_true', default=False, help='Convert 4dmm to 2dmm or not')
    parser.add_argument('--threedmm', action='store_true', default=False, help='Convert 4dmm to 3dmm or not')
    parser.add_argument('--rcda', action='store_true', default=False, help='Replace ops with RCDAPlugin or not')
    parser.add_argument('--pos2d', action='store_true', default=False, help='Reuse pos2d or not')
    parser.add_argument('--pos1d', action='store_true', default=False, help='Reuse pos1d or not')
    parser.add_argument('--posemb', action='store_true', default=False, help='Reuse posemb or not')
    parser.add_argument('--rmslice', action='store_true', default=False, help='Remove nodes of slice or not')
    parser.add_argument('--fp16', action='store_true', default=False, help='Enable FP16 mode or not, default is TF32 if it is supported')
    parser.add_argument('--log_level', default=1, type=int, help='Logger level. (0:VERBOSE, 1:INFO, 2:WARNING, 3:ERROR, 4:INTERNAL_ERROR)')
    args = parser.parse_args()
    return args

args = get_args()
if args.onnx:
    onnxFile = args.onnx
else:
    onnxFile = './engines/onnx/modified_AnchorDETR.onnx'

if args.mergeparams:
    onnxFile =  onnxFile.replace(".onnx", "_mergeparams.onnx")
if args.ln:
    onnxFile = onnxFile.replace(".onnx", "_ln.onnx")
if args.softmax:
    onnxFile = onnxFile.replace(".onnx", "_softmax.onnx")
if args.maskedsoftmax and not args.rcda:
    onnxFile = onnxFile.replace(".onnx", "_msoftmax.onnx")
if args.addQbiastranspose and not args.rcda:
    onnxFile = onnxFile.replace(".onnx", "_Q.onnx")
if args.addVbiastranspose and not args.rcda:
    onnxFile = onnxFile.replace(".onnx", "_V.onnx")
if args.optshape:
    onnxFile = onnxFile.replace(".onnx", "_optshape.onnx")
if args.twodmm:
    onnxFile = onnxFile.replace(".onnx", "_2dmm.onnx")
if args.rcda:
    onnxFile = onnxFile.replace(".onnx", "_rcda.onnx")
if args.pos2d:
    onnxFile = onnxFile.replace(".onnx", "_pos2d.onnx")
if args.pos1d:
    onnxFile = onnxFile.replace(".onnx", "_pos1d.onnx")
if args.posemb:
    onnxFile = onnxFile.replace(".onnx", "_posemb.onnx")
if args.rmslice:
    onnxFile = onnxFile.replace(".onnx", "_rmslice.onnx")
if args.mask2pos:
    onnxFile = onnxFile.replace(".onnx", "_mask2pos.onnx")
if args.trt == '':
    trtFile = onnxFile.replace('onnx','plan')
else:
    trtFile = args.trt

if args.fp16:
    trtFile = trtFile.replace('.plan','_fp16.plan')
    
timeCacheFile = "/target/onnx2trt/AnchorDETR.cache"
soFileList = glob("./*.so")
useTimeCache = False

log_level = {0:trt.Logger.VERBOSE,
             1:trt.Logger.INFO,
             2:trt.Logger.WARNING,
             3:trt.Logger.ERROR,
             4:trt.Logger.INTERNAL_ERROR}

logger = trt.Logger(log_level[args.log_level])                                       # Logger Level: VERBOSE,INFO,WARNING,ERROR,INTERNAL_ERROR
if os.path.isfile(trtFile):                                                
    with open(trtFile, 'rb') as f:
        engineString = f.read()
    if engineString == None:
        print("Failed getting serialized engine!")
        exit(1)
    print("Succeeded getting serialized engine in {}!".format(trtFile))
else:                                                                     
    timeCache = b""
    if useTimeCache and os.path.isfile(timeCacheFile):
        with open(timeCacheFile, 'rb') as f:
            timeCache = f.read()
        if timeCache == None:
            print("Failed getting serialized timing cache!")
            exit(1)
        print("Succeeded getting serialized timing cache!")
    if len(soFileList) > 0:
        print("Find Plugin %s!"%soFileList)
    else:
        print("No Plugin!")
    for soFile in soFileList:
        ctypes.cdll.LoadLibrary(soFile)

    builder = trt.Builder(logger)                                         
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()
    config = builder.create_builder_config()
    config.max_workspace_size = 24 << 30                   
    if useTimeCache:
        cache = config.create_timing_cache(timeCache)
        config.set_timing_cache(cache, False)
    # config.set_tactic_sources(1 << int(trt.TacticSource.CUBLAS) | 1 << int(trt.TacticSource.CUBLAS_LT) | 1 << int(trt.TacticSource.CUDNN))
    if args.fp16:
        config.flags = 1 << int(trt.BuilderFlag.STRICT_TYPES)
        config.flags = 1 << int(trt.BuilderFlag.FP16)
    else:
        config.flags = 0
    # config.int8_calibrator = calibrator.MyCalibrator(calibrationDataPath, calibrationCount, (1, 1, imageHeight, imageWidth), cacheFile)
    
    parser = trt.OnnxParser(network, logger)
    if not os.path.exists(onnxFile):
        print("Failed finding onnx file in {}!".format(onnxFile))
        exit()
    print("Succeeded finding onnx file!")
    with open(onnxFile, 'rb') as model:
        if not parser.parse(model.read()):
            print("Failed parsing onnx file!")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            exit()
        print("Succeeded parsing onnx file!")

    if args.fp16:
        # pass
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            # if 'LayerNorm' in layer.name or 'SoftmaxPlugin' in layer.name:
            if 'LayerNorm' in layer.name or 'RCDA' in layer.name:
                layer.precision = trt.float32
                layer.set_output_type(0, trt.float32)

    inputTensor = network.get_input(0)  # image
    print("inputTensor.name:{}".format(inputTensor.name))
    profile.set_shape(inputTensor.name, (1, 3, 320, 512), (1, 3, 800, 800), (1, 3, 1344, 1344))             

    # inputTensor = network.get_input(1)  # mask
    # print("inputTensor.name:{}".format(inputTensor.name))
    # profile.set_shape(inputTensor.name, (1, 320, 512), (1, 800, 800), (1, 1344, 1344))           
    config.add_optimization_profile(profile)

    t0 = time()
    engineString = builder.build_serialized_network(network, config)     
    t1 = time()
    print("%s timing cache, %f mins"%("With" if useTimeCache/60 else "Without",(t1-t0)))

    if useTimeCache and not os.path.isfile(timeCacheFile):
        timeCache = config.get_timing_cache()
        timeCacheString = timeCache.serialize()
        with open(timeCacheFile, 'wb') as f:
            f.write(timeCacheString)
            print("Succeeded saving .cache file!")

    if engineString == None:
        print("Failed getting serialized engine!")
        exit(1)
    print("Succeeded getting serialized engine!")
    with open('./AnchorDETR.plan', 'wb') as f:                                         
        f.write(engineString)
        print("Succeeded saving .plan file in ./AnchorDETR.plan !")
    with open(trtFile, 'wb') as f:                                         
        f.write(engineString)
        print("Succeeded saving .plan file in {}!".format(trtFile))
