#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import ctypes
import numpy as np
from cuda import cudart  # 使用 cuda runtime API
import tensorrt as trt
import onnx
import onnx_graphsurgeon as gs
import numpy as np

soFilePath      = './MaskedSoftmaxPlugin.so'
nBS             = 1
nHead           = 8
nSL             = 2500
nEmbedding      = 50
epsilon         = 1e-6
npDataType      = np.float16
np.random.seed(97)
globalMask     = npDataType(np.round(np.ones([1, nEmbedding]) - np.random.rand(1,nEmbedding)))

def check(a, b, weak = False):
    if weak:
        return np.all( np.abs(a - b) < epsilon)
    else:
        return np.all( a == b )

def masked_maskesoftmaxCPU(bufferH):
    _mask = (1 - globalMask) * -10000.0
    _x = bufferH[0] + _mask
    _0 = np.max(_x,axis=-1,keepdims=True)
    _1 = np.exp(_x - _0)
    _2 = np.sum(_1,axis=-1,keepdims=True)
    _3 = _1 / (_2 + epsilon)
    return _3

def getMaskedSoftmaxOnnx():
    onnx_file = "temp.onnx"
    shape = ('BH', 'L', nEmbedding)
    x = gs.Variable(name="x", dtype=npDataType, shape=shape)
    mask = gs.Constant(name="mask", values=np.int32(globalMask))
    y = gs.Variable(name="y", dtype=npDataType, shape=shape)
    layernorm = gs.Node(op="MaskedSoftmaxPlugin", 
                        name="MaskedSoftmax_1", 
                        inputs=[x, mask], 
                        outputs=[y], 
                        attrs={"epsilon":epsilon})
    graph = gs.Graph(nodes=[layernorm], inputs=[x], outputs=[y])
    onnx.save(gs.export_onnx(graph), onnx_file)
    return onnx_file

def run():
    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, '')
    ctypes.cdll.LoadLibrary(soFilePath)

    builder         = trt.Builder(logger)
    network         = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config          = builder.create_builder_config()
    config.max_workspace_size = 6 << 30
    config.flags    = 1 << int(trt.BuilderFlag.FP16) if int(npDataType == np.float16) else 0
    profile = builder.create_optimization_profile()
    
    parser = trt.OnnxParser(network, logger)
    onnxFile = getMaskedSoftmaxOnnx()
    if not os.path.exists(onnxFile):
        print("Failed finding onnx file!")
        exit()
    print("Succeeded finding onnx file!")
    with open(onnxFile, 'rb') as model:
        if not parser.parse(model.read()):
            print("Failed parsing onnx file!")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            exit()
        print("Succeeded parsing onnx file!")
    


    inputTensor = network.get_input(0)  # x
    print("inputTensor.name:{}".format(inputTensor.name))
    profile.set_shape(inputTensor.name, (nBS*nHead, nSL, nEmbedding), (nBS*nHead, nSL, nEmbedding), (nBS*nHead*2, nSL*2, nEmbedding))  
    config.add_optimization_profile(profile)

    engineString = builder.build_serialized_network(network, config)
    engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)

    context = engine.create_execution_context()
    context.set_binding_shape(0,[nBS*nHead, nSL, nEmbedding])
    print("Binding all? %s"%(["No","Yes"][int(context.all_binding_shapes_specified)]))
    
    nInput = np.sum([ engine.binding_is_input(i) for i in range(engine.num_bindings) ])
    nOutput = engine.num_bindings - nInput
    for i in range(engine.num_bindings):
        print("input ->" if engine.binding_is_input(i) else "output->",engine.get_binding_dtype(i),engine.get_binding_shape(i),context.get_binding_shape(i))

    bufferH = []
    bufferH.append( np.random.rand(nBS*nHead,nSL,nEmbedding).astype(npDataType).reshape(nBS*nHead,nSL,nEmbedding) * 2 - 1)
    bufferH.append(np.empty(context.get_binding_shape(1),dtype=trt.nptype(engine.get_binding_dtype(1))))

    bufferD = []
    for i in range(engine.num_bindings):
        bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

    for i in range(nInput):
        cudart.cudaMemcpy(bufferD[i], bufferH[i].ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

    context.execute_v2(bufferD)

    for i in range(nInput, nInput + nOutput):
        cudart.cudaMemcpy(bufferH[i].ctypes.data, bufferD[i], bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

    print("check result:")
    temp1 = bufferH[-1]
    temp2 = masked_maskesoftmaxCPU(bufferH[:1])
    print(check(temp1,temp2,True), "max diff=%f"%(np.max(np.abs(temp1 - temp2))) )
    
    for b in bufferD:
        cudart.cudaFree(b)

if __name__ == '__main__':
    os.system("rm -f ./*.trt")
    np.set_printoptions(precision = 4, linewidth = 200, suppress = True)
    run()