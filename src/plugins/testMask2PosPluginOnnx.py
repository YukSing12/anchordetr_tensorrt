import torch
import torch.nn as nn
import os
import ctypes
import numpy as np
from cuda import cudart  # 使用 cuda runtime API
import tensorrt as trt
import onnx
import onnx_graphsurgeon as gs
import numpy as np

soFilePath      = './Mask2PosPlugin.so'
nBS             = 1
nH              = 32
nW              = 20
nEmbedding      = 256
temperature     = 10000
dim_t = np.arange(nEmbedding)
dim_t = temperature ** (2 * (dim_t // 2) / nEmbedding)
epsilon         = 1e-5
npDataType      = np.float16
np.random.seed(97)

globalMask     = npDataType(np.round(np.ones([nBS, nW]) - np.random.rand(nBS,nW)))
def check(a, b, weak = False):
    if weak:
        return np.all( np.abs(a - b) < epsilon)
    else:
        return np.all( a == b )

def pos2posemb1d(pos, num_pos_feats=256, temperature=10000):
    scale = 6.2831854820251465
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos_x = pos[..., None] / dim_t
    posemb = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    return posemb


def mask2pos(mask_x, mask_y):
    y_embed = mask_y.cumsum(1, dtype=torch.float32)
    x_embed = mask_x.cumsum(1, dtype=torch.float32)
    y_embed = (y_embed - 0.5) / y_embed[:, -1].unsqueeze(1)
    x_embed = (x_embed - 0.5) / x_embed[:, -1].unsqueeze(1)
    return y_embed, x_embed

def test_cpu(mask_x, mask_y):
    mask_x = torch.Tensor(mask_x)
    mask_y = torch.Tensor(mask_y)
    pos_col, pos_row = mask2pos(mask_x,mask_y)
    posemb_row = pos2posemb1d(pos_row)
    posemb_col = pos2posemb1d(pos_col)
    return posemb_row.numpy()

def get_onnx():
    onnx_file = "temp.onnx"
    mask_x = gs.Variable(name="mask_x", dtype=npDataType, shape=(1, 'W'))
    t = gs.Constant(name="dim_t", values=npDataType(dim_t))
    embed = gs.Variable(name="embed", dtype=npDataType)
    plugin = gs.Node(op="Mask2PosPlugin", 
                        name="Mask2PosPlugin_1", 
                        inputs=[mask_x, t], 
                        outputs=[embed])
    graph = gs.Graph(nodes=[plugin], inputs=[mask_x], outputs=[embed])
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
    onnxFile = get_onnx()
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
    
    inputTensor = network.get_input(0)  # mask_x
    print("inputTensor.name:{}".format(inputTensor.name))
    profile.set_shape(inputTensor.name, (nBS, nW), (nBS, nW), (nBS, 2*nW))  

    config.add_optimization_profile(profile)

    engineString = builder.build_serialized_network(network, config)
    engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)

    context = engine.create_execution_context()
    context.set_binding_shape(0,[nBS,nW])
    print("Binding all? %s"%(["No","Yes"][int(context.all_binding_shapes_specified)]))
    
    nInput = np.sum([ engine.binding_is_input(i) for i in range(engine.num_bindings) ])
    nOutput = engine.num_bindings - nInput
    for i in range(engine.num_bindings):
        print("input ->" if engine.binding_is_input(i) else "output->",engine.get_binding_dtype(i),engine.get_binding_shape(i),context.get_binding_shape(i))

    bufferH = []
    bufferH.append( globalMask.astype(npDataType).reshape(nBS,nW) )
    bufferH.append(np.empty(context.get_binding_shape(nInput),dtype=trt.nptype(engine.get_binding_dtype(nInput))))

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
    temp2 = test_cpu(bufferH[0], bufferH[0])
    print(check(temp1,temp2,True), "max diff=%f"%(np.max(np.abs(temp1 - temp2))) )
    
    for b in bufferD:
        cudart.cudaFree(b)

if __name__ == '__main__':
    os.system("rm -f ./*.trt")
    np.set_printoptions(precision = 4, linewidth = 200, suppress = True)
    run()