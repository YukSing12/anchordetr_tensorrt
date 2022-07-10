import onnx
import onnx_graphsurgeon as gs
# import onnxruntime as rt
import numpy as np
import argparse

def get_args():
    parser = argparse.ArgumentParser('Export AnchorDETR TensorRT', add_help=False)
    parser.add_argument('--odir', default='engines', type=str, help='Directory of onnx to save, default is ./engines')
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
    parser.add_argument('--debug', '-D', action='store_true', default=False, help='Enable debug mode')
    args = parser.parse_args()
    return args

args = get_args()

ENABLE_MERGE_PARAM = args.mergeparams
ENABLE_LAYERNORM_PLUGIN = args.ln
ENABLE_SOFTMAX_PLUGIN = args.softmax
ENABLE_MASKEDSOFTMAX_PLUGIN = args.maskedsoftmax
ENABLE_ADDQBIASTRANSPOSE_PLUGIN = args.addQbiastranspose
ENABLE_ADDVBIASTRANSPOSE_PLUGIN = args.addVbiastranspose
ENABLE_OPTDYNAMICSHAPE = args.optshape
ENABLE_CONVERT2DMM = args.twodmm
ENABLE_CONVERT3DMM = args.threedmm
ENABLE_RCDA_PLUGIN = args.rcda
ENABLE_REUSE_POS2D = args.pos2d
ENABLE_REUSE_POS1D = args.pos1d
ENABLE_REUSE_POSEMB = args.posemb
ENABLE_REMOVE_SLICE = args.rmslice
ENABLE_GEMMBATCHED = False
ENABLE_MASK2POS_PLUGIN = args.mask2pos
DEBUG = args.debug

src_onnx_path = './AnchorDETR.onnx'
if DEBUG:
    src_onnx_path = './debug.onnx'
dst_onnx_path = './engines/onnx/modified_AnchorDETR.onnx'

def replace_with_mask2pos(nodes_dict, target_node, nodes):
    node_id = int(target_node.name.split("_")[-1])
    if (target_node.outputs[0].outputs[0].outputs[0].outputs[0].outputs[0].outputs[0].outputs[0].outputs[0].outputs[0].outputs[0].outputs[0].outputs[0].outputs[0].outputs[0].outputs[0].outputs[0].outputs[0].outputs[0].outputs[0].outputs[1].op != 'Reshape' or
        target_node.outputs[0].outputs[0].outputs[0].outputs[0].outputs[0].outputs[0].outputs[0].outputs[0].outputs[0].outputs[0].op != 'Div'):
        print("Can not replace node {} with mask2pos".format(target_node.name))
        return

    output_node = target_node.outputs[0].outputs[0].outputs[0].outputs[0].outputs[0].outputs[0].outputs[0].outputs[0].outputs[0].outputs[0].outputs[0].outputs[0].outputs[0].outputs[0].outputs[0].outputs[0].outputs[0].outputs[0].outputs[0].outputs[1]
    dim_t_node = target_node.outputs[0].outputs[0].outputs[0].outputs[0].outputs[0].outputs[0].outputs[0].outputs[0].outputs[0].outputs[0]

    plugin = gs.Node(op='Mask2PosPlugin', name='Mask2PosPlugin_{}'.format(node_id), inputs=[target_node.inputs[0],dim_t_node.inputs[1]], outputs=[output_node.outputs[0]])
    nodes.append(plugin)

    target_node.inputs.clear()
    output_node.outputs.clear()


def convert_to_bemmbatched(nodes_dict, target_node, nodes):
    node_id = int(target_node.name.split("_")[-1])
    if ('Add_{}'.format(node_id+55) not in nodes_dict or
        'Add_{}'.format(node_id+65) not in nodes_dict or
        'Add_{}'.format(node_id+75) not in nodes_dict or
        'Add_{}'.format(node_id+76) not in nodes_dict or
        'MatMul_{}'.format(node_id+117) not in nodes_dict or
        'MatMul_{}'.format(node_id+167) not in nodes_dict or
        'MatMul_{}'.format(node_id+142) not in nodes_dict or
        'MatMul_{}'.format(node_id+192) not in nodes_dict or
        'MatMul_{}'.format(node_id+217) not in nodes_dict):
        print("Can not merge parameters from {}".format(target_node.name))
        return
    
    q_row_node = nodes_dict['Add_{}'.format(node_id+55)]
    q_col_node = nodes_dict['Add_{}'.format(node_id+65)]
    k_row_node = nodes_dict['Add_{}'.format(node_id+75)]
    k_col_node = nodes_dict['Add_{}'.format(node_id+76)]
    value_node = target_node
    
    # Concat valuables with shape of b x h x w x 256 to 5b x h x w x 256
    concat_out = gs.Variable(name='Concat_{}_out'.format(node_id))
    concat_node = gs.Node(op='Concat', name='Concat_{}'.format(node_id), 
                        inputs=[q_row_node.outputs[0],
                                q_col_node.outputs[0],
                                k_row_node.outputs[0],
                                k_col_node.outputs[0],
                                value_node.outputs[0]], outputs=[concat_out], attrs={"axis": 0})
    nodes.append(concat_node)
    
    # Concat weights with shape of 256 x 256 to 5 x 256 x 256
    q_row_mm_node = q_row_node.outputs[0].outputs[0].outputs[0].outputs[-1]
    q_col_mm_node = q_col_node.outputs[0].outputs[0].outputs[0].outputs[-1]
    k_row_mm_node = k_row_node.outputs[0].outputs[1]
    k_col_mm_node = k_col_node.outputs[0].outputs[1]
    value_mm_node = value_node.outputs[0].outputs[4]

    q_row_w = q_row_node.outputs[0].outputs[0].outputs[0].outputs[-1].inputs[-1]
    q_col_w = q_col_node.outputs[0].outputs[0].outputs[0].outputs[-1].inputs[-1]
    k_row_w = k_row_node.outputs[0].outputs[1].inputs[1]
    k_col_w = k_col_node.outputs[0].outputs[1].inputs[1]
    value_w = value_node.outputs[0].outputs[4].inputs[1]

    concat_w = gs.Constant(name='MatMul_Batched_{}_w'.format(node_id), values=np.concatenate([q_row_w.values.reshape([1, q_row_w.values.shape[0], q_row_w.values.shape[1]]),
                                                                              q_col_w.values.reshape([1, q_col_w.values.shape[0], q_col_w.values.shape[1]]),
                                                                              k_row_w.values.reshape([1, k_row_w.values.shape[0], k_row_w.values.shape[1]]),
                                                                              k_col_w.values.reshape([1, k_col_w.values.shape[0], k_col_w.values.shape[1]]),
                                                                              value_w.values.reshape([1, value_w.values.shape[0], value_w.values.shape[1]])], axis=0))
    

    # Reshape value with shape of 5b x h x w x 256 to 5b x hw x 256
    concat_shape_out = gs.Variable(name='Concat_shape_{}'.format(node_id))
    concat_shape_0 = gs.Constant(name='Concat_shape_{}_0', values=np.array([5]))
    concat_shape = gs.Node(op='Concat', name='Concat_shape_{}'.format(node_id), 
                        inputs=[concat_shape_0,
                                q_row_node.outputs[0].outputs[0].inputs[1].inputs[0].inputs[1],
                                q_row_node.outputs[0].outputs[0].inputs[1].inputs[0].inputs[2]], outputs=[concat_shape_out], attrs={"axis": 0})
    nodes.append(concat_shape)

    reshape_out = gs.Variable(name='Reshape_{}_out'.format(node_id))
    reshape_node = gs.Node(op='Reshape', name='Reshape_{}'.format(node_id), inputs=[concat_out, concat_shape_out],outputs=[reshape_out])
    nodes.append(reshape_node)

    # Replace five 2dMatMul to one MatMulBatched
    gemmBatch_out = gs.Variable(name='MatMul_Batched_{}_out'.format(node_id))
    gemmBatch = gs.Node(op='MatMul',name='MatMul_Batched_{}'.format(node_id),inputs=[reshape_out, concat_w], outputs=[gemmBatch_out])
    nodes.append(gemmBatch)

    # Get each output from MatMulBatched
    q_row_gather_indices = gs.Constant(name='Gather_{}_q_row_indices'.format(node_id), values=np.array(0, dtype=np.int64))
    q_row_gather = gs.Node(op='Gather', name='Gather_{}_q_row'.format(node_id), inputs=[gemmBatch_out, q_row_gather_indices], outputs=[q_row_mm_node.outputs[0]], attrs={"axis": 0})
    nodes.append(q_row_gather)

    q_col_gather_indices = gs.Constant(name='Gather_{}_q_col_indices'.format(node_id), values=np.array(1, dtype=np.int64))
    q_col_gather = gs.Node(op='Gather', name='Gather_{}_q_col'.format(node_id), inputs=[gemmBatch_out, q_col_gather_indices], outputs=[q_col_mm_node.outputs[0]], attrs={"axis": 0})
    nodes.append(q_col_gather)

    # Reshape k_row, k_col, and value from shape of 1 x hw x 256 to 1 x h x w x 256
    shape_in = target_node.outputs[0].outputs[-2].inputs[1].inputs[0].inputs[1]
    k_row_gather_out = gs.Variable(name='Gather_{}_k_row_out'.format(node_id))
    k_row_gather_indices = gs.Constant(name='Gather_{}_k_row_indices'.format(node_id), values=np.array(2, dtype=np.int64))
    k_row_gather = gs.Node(op='Gather', name='Gather_{}_k_row'.format(node_id), inputs=[gemmBatch_out, k_row_gather_indices], outputs=[k_row_gather_out], attrs={"axis": 0})
    k_row_reshape = gs.Node(op='Reshape', name='Reshape_{}_k_row'.format(node_id), inputs=[k_row_gather_out, shape_in], outputs=[k_row_mm_node.outputs[0]])
    nodes.append(k_row_gather)
    nodes.append(k_row_reshape)

    k_col_gather_out = gs.Variable(name='Gather_{}_k_col_out'.format(node_id))
    k_col_gather_indices = gs.Constant(name='Gather_{}_k_col_indices'.format(node_id), values=np.array(3, dtype=np.int64))
    k_col_gather = gs.Node(op='Gather', name='Gather_{}_k_col'.format(node_id), inputs=[gemmBatch_out, k_col_gather_indices], outputs=[k_col_gather_out], attrs={"axis": 0})
    k_col_reshape = gs.Node(op='Reshape', name='Reshape_{}_k_col'.format(node_id), inputs=[k_col_gather_out, shape_in], outputs=[k_col_mm_node.outputs[0]])
    nodes.append(k_col_gather)
    nodes.append(k_col_reshape)
    
    value_gather_out = gs.Variable(name='Gather_{}_value_out'.format(node_id))
    value_gather_indices = gs.Constant(name='Gather_{}_value_indices'.format(node_id), values=np.array(4, dtype=np.int64))
    value_gather = gs.Node(op='Gather', name='Gather_{}_value'.format(node_id), inputs=[gemmBatch_out, value_gather_indices], outputs=[value_gather_out], attrs={"axis": 0})
    value_reshape = gs.Node(op='Reshape', name='Reshape_{}_value'.format(node_id), inputs=[value_gather_out, shape_in], outputs=[value_mm_node.outputs[0]])
    nodes.append(value_gather)
    nodes.append(value_reshape)

    # Clear inputs and outputs to remove nodes
    q_row_mm_node.inputs.clear()
    q_col_mm_node.inputs.clear()
    k_row_mm_node.inputs.clear()
    k_col_mm_node.inputs.clear()
    value_mm_node.inputs.clear()

    q_row_mm_node.outputs.clear()
    q_col_mm_node.outputs.clear()
    k_row_mm_node.outputs.clear()
    k_col_mm_node.outputs.clear()
    value_mm_node.outputs.clear()

def fuse_slice_to_concat(nodes_dict, target_node, num):
    node_id = int(target_node.name.split("_")[-1])
    if ('Concat_{}'.format(node_id+2) not in nodes_dict):
        print("Can not merge parameters from {}".format(target_node.name))
        return
    target_node.outputs.clear()
    
    concat_node = nodes_dict['Concat_{}'.format(node_id+2)]
    
    concat_input0 = gs.Constant(name=concat_node.inputs[0].name, values=np.array([1]))
    concat_input1 = gs.Constant(name=concat_node.inputs[1].name, values=np.array([-1]))
    concat_input2 = gs.Constant(name=concat_node.inputs[1].name+'_{}'.format(num), values=np.array([num]))

    concat_node.inputs[0].name += '_Del'
    concat_node.inputs[1].name += '_Del'

    concat_node.inputs[0] = concat_input0
    concat_node.inputs[1] = concat_input1
    concat_node.inputs.append(concat_input2)

def merge_parameters(nodes_dict, target_node):
    node_id = int(target_node.name.split("_")[-1])
    if ('Slice_{}'.format(node_id+30) not in nodes_dict):
        print("Can not merge parameters from {}".format(target_node.name))
        return
    
    q_row_b_node = nodes_dict['Slice_{}'.format(node_id+30)]
    q_row_b = q_row_b_node.inputs[0]
    new_q_row_b = gs.Constant(name=q_row_b_node.outputs[0].name, values=q_row_b.values[0:256])
    
    q_row_b_node.outputs[0].name += 'Del'
    q_row_b_node.outputs[0].outputs[0].inputs[0] = new_q_row_b
    q_row_b_node.outputs.clear()

    q_col_b_node = nodes_dict['Slice_{}'.format(node_id+55)]
    q_col_b = q_col_b_node.inputs[0]
    new_q_col_b = gs.Constant(name=q_col_b_node.outputs[0].name, values=q_col_b.values[256:512])
    
    q_col_b_node.outputs[0].name += 'Del'
    q_col_b_node.outputs[0].outputs[0].inputs[0] = new_q_col_b
    q_col_b_node.outputs.clear()

    k_row_b_node = nodes_dict['Slice_{}'.format(node_id+80)]
    k_row_b = k_row_b_node.inputs[0]
    new_k_row_b = gs.Constant(name=k_row_b_node.outputs[0].name, values=k_row_b.values[512:768])
    
    k_row_b_node.outputs[0].name += 'Del'
    k_row_b_node.outputs[0].outputs[0].inputs[0] = new_k_row_b
    k_row_b_node.outputs.clear()

    k_col_b_node = nodes_dict['Slice_{}'.format(node_id+105)]
    k_col_b = k_col_b_node.inputs[0]
    new_k_col_b = gs.Constant(name=k_col_b_node.outputs[0].name, values=k_col_b.values[768:1024])
    
    k_col_b_node.outputs[0].name += 'Del'
    k_col_b_node.outputs[0].outputs[0].inputs[0] = new_k_col_b
    k_col_b_node.outputs.clear()

    value_b_node = nodes_dict['Slice_{}'.format(node_id+130)]
    value_b = value_b_node.inputs[0]
    new_value_b = gs.Constant(name=value_b_node.outputs[0].name, values=value_b.values[1024:1280])
    
    value_b_node.outputs[0].name += 'Del'
    value_b_node.outputs[0].outputs[0].inputs[0] = new_value_b
    value_b_node.outputs.clear()

    q_row_w_node = nodes_dict['Slice_{}'.format(node_id+20)]
    q_row_w = q_row_w_node.inputs[0]
    new_q_row_w = gs.Constant(name=q_row_w_node.outputs[0].outputs[0].outputs[0].name, values=q_row_w.values[0:256].transpose([1,0]))
    
    q_row_w_node.outputs[0].outputs[0].outputs[0].name += 'Del'
    q_row_w_node.outputs[0].outputs[0].outputs[0].outputs[0].inputs[1] = new_q_row_w
    q_row_w_node.outputs[0].outputs[0].outputs.clear()

    q_col_w_node = nodes_dict['Slice_{}'.format(node_id+46)]
    q_col_w = q_col_w_node.inputs[0]
    new_q_col_w = gs.Constant(name=q_col_w_node.outputs[0].outputs[0].outputs[0].name, values=q_col_w.values[256:512].transpose([1,0]))
    
    q_col_w_node.outputs[0].outputs[0].outputs[0].name += 'Del'
    q_col_w_node.outputs[0].outputs[0].outputs[0].outputs[0].inputs[1] = new_q_col_w
    q_col_w_node.outputs[0].outputs[0].outputs.clear()

    k_row_w_node = nodes_dict['Slice_{}'.format(node_id+71)]
    k_row_w = k_row_w_node.inputs[0]
    new_k_row_w = gs.Constant(name=k_row_w_node.outputs[0].outputs[0].outputs[0].name, values=k_row_w.values[512:768].transpose([1,0]))
    
    k_row_w_node.outputs[0].outputs[0].outputs[0].name += 'Del'
    k_row_w_node.outputs[0].outputs[0].outputs[0].outputs[0].inputs[1] = new_k_row_w
    k_row_w_node.outputs[0].outputs[0].outputs.clear()

    k_col_w_node = nodes_dict['Slice_{}'.format(node_id+96)]
    k_col_w = k_col_w_node.inputs[0]
    new_k_col_w = gs.Constant(name=k_col_w_node.outputs[0].outputs[0].outputs[0].name, values=k_col_w.values[768:1024].transpose([1,0]))
    
    k_col_w_node.outputs[0].outputs[0].outputs[0].name += 'Del'
    k_col_w_node.outputs[0].outputs[0].outputs[0].outputs[0].inputs[1] = new_k_col_w
    k_col_w_node.outputs[0].outputs[0].outputs.clear()

    value_node = nodes_dict['Slice_{}'.format(node_id+120)]
    value = value_node.inputs[0]
    new_value = gs.Constant(name=value_node.outputs[0].outputs[0].outputs[0].name, values=value.values[1024:1280].transpose([1,0]))
    
    value_node.outputs[0].outputs[0].outputs[0].name += 'Del'
    value_node.outputs[0].outputs[0].outputs[0].outputs[0].inputs[1] = new_value
    value_node.outputs[0].outputs[0].outputs.clear()

def reuse_posemb(nodes_dict, target_node, reuse_id):
    node_id = int(target_node.name.split("_")[-1])
    if (False):
        print("Can not reuse {}".format(target_node.name))
        return

    posemb = nodes_dict['Tile_{}'.format(reuse_id)]

    num = len(target_node.outputs[0].outputs)
    for i in range(num):
        target_node.outputs[0].outputs[0].inputs[1] = posemb.outputs[0]

    target_node.inputs.clear()
    target_node.outputs.clear()

def reuse_pos2posemb1d(nodes_dict, target_node, reuse_id):
    node_id = int(target_node.name.split("_")[-1])
    if ('Reshape_{}'.format(node_id+31) not in nodes_dict or
        'MatMul_{}'.format(node_id+33) not in nodes_dict):
        print("Can not reuse {}".format(target_node.name))
        return

    pos2posemb1d = nodes_dict['Gather_{}'.format(reuse_id)]
    pos2posemb1d_output = nodes_dict['Reshape_{}'.format(reuse_id + 31)]

    target_node.inputs.clear()
    reshape_node = nodes_dict['Reshape_{}'.format(node_id+31)]
    reshape_node.outputs.clear()

    nodes_dict['MatMul_{}'.format(node_id+33)].inputs[0] = pos2posemb1d_output.outputs[0]

def reuse_pos2posemb2d(nodes_dict, target_node, reuse_id):
    node_id = int(target_node.name.split("_")[-1])
    if ('Concat_{}'.format(node_id+63) not in nodes_dict or
        'MatMul_{}'.format(node_id+65) not in nodes_dict):
        print("Can not reuse {}".format(target_node.name))
        return
    pos2posemb2d = nodes_dict['Mul_{}'.format(reuse_id)]
    pos2posemb2d_output = nodes_dict['Concat_{}'.format(reuse_id + 63)]

    target_node.inputs.clear()
    concat_node = nodes_dict['Concat_{}'.format(node_id+63)]
    concat_node.outputs.clear()

    nodes_dict['MatMul_{}'.format(node_id+65)].inputs[0] = pos2posemb2d_output.outputs[0]


def replace_with_rcda(nodes_dict, q_row_mm_node, nodes):
    node_id = int(q_row_mm_node.name.split("_")[-1])
    if ('MatMul_{}'.format(node_id+25) not in nodes_dict or
        'MatMul_{}'.format(node_id+50) not in nodes_dict or
        'MatMul_{}'.format(node_id+75) not in nodes_dict or
        'MatMul_{}'.format(node_id+100) not in nodes_dict or
        'Gather_{}'.format(node_id+175) not in nodes_dict or
        'Gather_{}'.format(node_id+181) not in nodes_dict or
        'MatMul_{}'.format(node_id+286) not in nodes_dict ):
        print("Can not replace {} with rcda".format(q_row_mm_node.name))
        return

    q_col_mm_node       = nodes_dict['MatMul_{}'.format(node_id+25)]
    k_row_mm_node       = nodes_dict['MatMul_{}'.format(node_id+50)]
    k_col_mm_node       = nodes_dict['MatMul_{}'.format(node_id+75)]
    value_mm_node       = nodes_dict['MatMul_{}'.format(node_id+100)]
    k_row_mask_node     = nodes_dict['Gather_{}'.format(node_id+175)]
    k_col_mask_node     = nodes_dict['Gather_{}'.format(node_id+181)]
    ff_node             = nodes_dict['MatMul_{}'.format(node_id+286)]


    q_row   = q_row_mm_node.inputs[0]
    q_row_w = q_row_mm_node.inputs[1]
    q_row_b = q_row_mm_node.outputs[0].outputs[0].inputs[0]
    
    q_col   = q_col_mm_node.inputs[0]
    q_col_w = q_col_mm_node.inputs[1]
    q_col_b = q_col_mm_node.outputs[0].outputs[0].inputs[0]
    
    k_row   = k_row_mm_node.inputs[0]
    k_row_w = k_row_mm_node.inputs[1]
    k_row_b = k_row_mm_node.outputs[0].outputs[0].inputs[0]

    k_col   = k_col_mm_node.inputs[0]
    k_col_w = k_col_mm_node.inputs[1]
    k_col_b = k_col_mm_node.outputs[0].outputs[0].inputs[0]

    value   = value_mm_node.inputs[0]
    value_w = value_mm_node.inputs[1]
    value_b = value_mm_node.outputs[0].outputs[0].inputs[0]

    k_row_mask = k_row_mask_node.outputs[0]
    k_col_mask = k_col_mask_node.outputs[0]

    ff_w = ff_node.inputs[1]
    ff_b = ff_node.outputs[0].outputs[0].inputs[0]
    
    out_node = ff_node.outputs[0].outputs[0].outputs[0].outputs[0].outputs[0].outputs[0]
    rcda_out = out_node.outputs[0]
    plugin = gs.Node(op="RCDAPlugin", 
                        name="RCDAPlugin_{}".format(node_id), 
                        inputs=[q_row, q_col, k_row, k_col, value, 
                                q_row_w, q_col_w, k_row_w, k_col_w, value_w,
                                q_row_b, q_col_b, k_row_b, k_col_b, value_b,
                                k_row_mask, k_col_mask,
                                ff_w, ff_b], 
                        outputs=[rcda_out])

    nodes.append(plugin)

    q_row_mm_node.inputs.clear()
    q_col_mm_node.inputs.clear()
    k_row_mm_node.inputs.clear()
    k_col_mm_node.inputs.clear()
    value_mm_node.inputs.clear()
    k_row_mask_node.outputs[0].outputs[0].inputs.clear()
    k_col_mask_node.outputs[0].outputs[0].inputs.clear()
    out_node.outputs.clear()

def replace_with_addVbiastranspose(nodes_dict, add_node, nodes):
    node_id = int(add_node.name.split("_")[-1])
    if ('Transpose_{}'.format(node_id + 55) not in nodes_dict or
        'Reshape_{}'.format(node_id + 67) not in nodes_dict or
        'Transpose_{}'.format(node_id + 68) not in nodes_dict or
        'Reshape_{}'.format(node_id + 158) not in nodes_dict or
        'Unsqueeze_{}'.format(node_id + 61) not in nodes_dict or
        'Unsqueeze_{}'.format(node_id + 63) not in nodes_dict or
        'Unsqueeze_{}'.format(node_id + 65) not in nodes_dict or
        'Concat_{}'.format(node_id + 168) not in nodes_dict 
        
    ):
        return

    bias = add_node.inputs[0]
    v = add_node.inputs[1]
    output = nodes_dict['Reshape_{}'.format(node_id + 158)].outputs[0]
    plugin = gs.Node(op='AddVBiasTransposePlugin', 
                    name='AddVBiasTransposePlugin_{}'.format(node_id), 
                    inputs=[v, bias], 
                    outputs=[output])

    add_node.inputs.clear()
    nodes_dict['Reshape_{}'.format(node_id + 158)].outputs.clear()

    nodes_dict['Concat_{}'.format(node_id + 168)].inputs[0] = nodes_dict['Unsqueeze_{}'.format(node_id + 63)].outputs[0]
    nodes_dict['Concat_{}'.format(node_id + 168)].inputs[2] = nodes_dict['Unsqueeze_{}'.format(node_id + 61)].outputs[0]
    nodes_dict['Concat_{}'.format(node_id + 168)].inputs[3] = nodes_dict['Unsqueeze_{}'.format(node_id + 65)].outputs[0]
    nodes.append(plugin)

def replace_with_addQbiastranspose(nodes_dict, add_node, nodes):
    node_id = int(add_node.name.split("_")[-1])
    try:
        if (add_node.outputs[0].outputs[0].outputs[0].outputs[0].outputs[0].outputs[0].outputs[0].outputs[0].op != 'Transpose' or
            add_node.outputs[0].outputs[0].outputs[0].outputs[0].outputs[0].outputs[0].inputs[1].inputs[0].inputs[0].inputs[0].op != 'Unsqueeze' or
            add_node.outputs[0].outputs[0].outputs[0].outputs[0].outputs[0].outputs[0].inputs[1].inputs[0].inputs[2].inputs[0].op != 'Unsqueeze' or
            add_node.outputs[0].outputs[0].outputs[0].outputs[0].outputs[0].outputs[0].inputs[1].inputs[0].inputs[1].inputs[0].inputs[0].inputs[0].op != 'Mul'
            ):
            return
    except:
        return

    bias = add_node.inputs[0]
    q = add_node.inputs[1]
    output = add_node.outputs[0].outputs[0].outputs[0].outputs[0].outputs[0].outputs[0].outputs[0].outputs[0].outputs[0]
    plugin = gs.Node(op='AddQBiasTransposePlugin', 
                    name='AddQBiasTransposePlugin_{}'.format(node_id), 
                    inputs=[q, bias], 
                    outputs=[output])

    add_node.outputs[0].outputs[0].outputs[0].outputs[0].outputs[0].outputs[0].outputs[0].outputs[0].outputs.clear()
    add_node.outputs[0].outputs[0].outputs[0].outputs[0].outputs[0].outputs[0].inputs[1].inputs[0].inputs[0].inputs[0].inputs.clear()
    add_node.outputs[0].outputs[0].outputs[0].outputs[0].outputs[0].outputs[0].inputs[1].inputs[0].inputs[2].inputs[0].inputs.clear()
    add_node.outputs[0].outputs[0].outputs[0].outputs[0].outputs[0].outputs[0].inputs[1].inputs[0].inputs[1].inputs[0].inputs[0].inputs[0].inputs.clear()
    add_node.inputs.clear()

    nodes.append(plugin)

def replace_with_maskedsoftmax(nodes_dict, softmax_node, nodes):
    node_id = int(softmax_node.name.split("_")[-1])
    try:
        if (softmax_node.inputs[0].inputs[0].inputs[0].inputs[0].inputs[2].inputs[0].inputs[0].inputs[0].op != 'MatMul' or
            softmax_node.inputs[0].inputs[0].inputs[0].inputs[0].inputs[0].inputs[0].inputs[0].inputs[0].inputs[0].inputs[0].inputs[0].inputs[0].op != 'Gather' or
            softmax_node.inputs[0].inputs[0].inputs[0].inputs[0].inputs[2].inputs[0].op != 'Reshape' or
            softmax_node.inputs[0].inputs[0].inputs[0].inputs[0].inputs[0].inputs[0].inputs[0].inputs[0].inputs[0].inputs[0].op != 'Unsqueeze' or
            softmax_node.inputs[0].inputs[0].op != 'Reshape' or
            softmax_node.inputs[0].inputs[0].inputs[1].inputs[0].inputs[0].inputs[0].inputs[0].inputs[0].op != 'Mul' or
            softmax_node.inputs[0].inputs[0].inputs[1].inputs[0].inputs[1].inputs[0].op != 'Unsqueeze' or
            softmax_node.inputs[0].inputs[0].inputs[1].inputs[0].inputs[2].inputs[0].op != 'Unsqueeze' or
            softmax_node.inputs[0].inputs[0].inputs[0].inputs[0].inputs[2].inputs[0].inputs[1].inputs[0].inputs[0].inputs[0].op != 'Unsqueeze' or
            softmax_node.inputs[0].inputs[0].inputs[0].inputs[0].inputs[2].inputs[0].inputs[1].inputs[0].inputs[2].inputs[0].op != 'Unsqueeze' or
            softmax_node.inputs[0].inputs[0].inputs[0].inputs[0].inputs[2].inputs[0].inputs[1].inputs[0].inputs[3].inputs[0].op != 'Unsqueeze'
            ):
            print("Can not fuse pos into maskedsoftmax {}".format(softmax_node.name))
            return
    except:
        print("Can not fuse pos into maskedsoftmax {}".format(softmax_node.name))
        return
    
            
    matmul_node = softmax_node.inputs[0].inputs[0].inputs[0].inputs[0].inputs[2].inputs[0].inputs[0].inputs[0]
    gather_node = softmax_node.inputs[0].inputs[0].inputs[0].inputs[0].inputs[0].inputs[0].inputs[0].inputs[0].inputs[0].inputs[0].inputs[0].inputs[0]

    mask = gather_node.outputs[0]
    inp = matmul_node.outputs[0]

    maskedsoftmax = gs.Node(op='MaskedSoftmaxPlugin', 
                            name='MaskedSoftmaxPlugin_{}'.format(node_id), 
                            inputs=[inp, mask], 
                            outputs=[softmax_node.outputs[0]])

    softmax_node.inputs[0].inputs[0].inputs[0].inputs[0].inputs[0].inputs[0].inputs[0].inputs[0].inputs[0].inputs[0].inputs.clear()
    softmax_node.inputs[0].inputs[0].inputs[0].inputs[0].inputs[2].inputs[0].inputs[1].inputs[0].inputs[0].inputs[0].inputs.clear()
    softmax_node.inputs[0].inputs[0].inputs[0].inputs[0].inputs[2].inputs[0].inputs[1].inputs[0].inputs[2].inputs[0].inputs.clear()
    softmax_node.inputs[0].inputs[0].inputs[0].inputs[0].inputs[2].inputs[0].inputs[1].inputs[0].inputs[3].inputs[0].inputs.clear()
    softmax_node.inputs[0].inputs[0].inputs[1].inputs[0].inputs[0].inputs[0].inputs[0].inputs[0].inputs.clear()
    softmax_node.inputs[0].inputs[0].inputs[0].inputs[0].inputs[2].inputs[0].inputs.clear()
    softmax_node.inputs[0].inputs[0].inputs[1].inputs[0].inputs[1].inputs[0].inputs.clear()
    softmax_node.inputs[0].inputs[0].inputs[1].inputs[0].inputs[2].inputs[0].inputs.clear()
    softmax_node.inputs[0].inputs[0].outputs.clear()

    softmax_node.inputs.clear()
    softmax_node.outputs.clear()

    nodes.append(maskedsoftmax)

def replace_with_layernorm(nodes_dict, mean_node):
    node_id = int(mean_node.name.split("_")[-1])
    assert ('Sub_{}'.format(node_id+1) in nodes_dict) 
    assert ('Div_{}'.format(node_id+8) in nodes_dict)
    sub_node = nodes_dict['Sub_{}'.format(node_id+1)]
    div_node = nodes_dict['Div_{}'.format(node_id+8)]
    name = 'LayerNorm_{}'.format(node_id)
    layernorm = gs.Node(op="LayerNorm", name=name, inputs=[mean_node.inputs[0]], outputs=[div_node.outputs[0]], attrs={"epsilon": 1e-6})
    mean_node.inputs.clear()
    sub_node.inputs.clear()
    div_node.outputs.clear()
    return layernorm
    
def replace_with_layernorm_gamma_beta(nodes_dict, mean_node):
    node_id = int(mean_node.name.split("_")[-1])
    if not (('Sub_{}'.format(node_id+1) in nodes_dict)  
    and ('Div_{}'.format(node_id+8) in nodes_dict)  
    and ('Mul_{}'.format(node_id+9) in nodes_dict) 
    and ('Add_{}'.format(node_id+10) in nodes_dict)):
        return None

    sub_node = nodes_dict['Sub_{}'.format(node_id+1)]
    div_node = nodes_dict['Div_{}'.format(node_id+8)]
    mul_node = nodes_dict['Mul_{}'.format(node_id+9)]
    add_node = nodes_dict['Add_{}'.format(node_id+10)]

    gamma = mul_node.inputs[1]
    beta = add_node.inputs[1]
    assert (gamma.shape[0] == beta.shape[0])

    name = 'LayerNorm_{}'.format(node_id)
    layernorm = gs.Node(op="LayerNorm", 
                        name=name, 
                        inputs=[mean_node.inputs[0], gamma, beta], 
                        outputs=[add_node.outputs[0]], 
                        attrs={"epsilon": 1e-5})
    mean_node.inputs.clear()
    sub_node.inputs.clear()
    add_node.outputs.clear()
    return layernorm

def cast_to_int32(nodes_dict, gather_node):
    node_id = int(gather_node.name.split("_")[-1])
    assert ('Cast_{}'.format(node_id-2) in nodes_dict)

    cast = nodes_dict['Cast_{}'.format(node_id-2)]
    cast.attrs['to'] = onnx.TensorProto.INT32

def replace_slice_sith_gather(nodes_dict, slice_node, nodes):
    node_id = int(slice_node.name.split("_")[-1])

    gather_out = gs.Variable("{}_out".format(node_id), dtype=None)
    indices = gs.Constant(name = "Gather_{}_indices".format(node_id), values=np.array([-1], dtype=np.int32))
    gather = gs.Node(op="Gather", name="Gather_{}".format(node_id), inputs=[slice_node.inputs[0], indices], outputs=[gather_out])


    axes = gs.Constant(name = "Unsqueeze_{}_axes".format(node_id), values=np.array([1], dtype=np.int32))
    unsqueeze = gs.Node(op="Unsqueeze", name="Unsqueeze_{}".format(node_id), inputs=[gather_out, axes], outputs=[slice_node.outputs[0]])

    slice_node.inputs.clear()
    slice_node.outputs.clear()

    nodes.append(gather)
    nodes.append(unsqueeze)

def convert_4dmm_2dmm(nodes_dict, matmul_node, nodes):
    node_id = int(matmul_node.name.split("_")[-1])

    if (('Add_{}'.format(node_id+1) not in nodes_dict) or
        ('Relu_{}'.format(node_id+2) not in nodes_dict) or 
        ('MatMul_{}'.format(node_id+4) not in nodes_dict) or
        ('Add_{}'.format(node_id+5) not in nodes_dict)):

        return None
    add_node = nodes_dict['Add_{}'.format(node_id+5)]

    shape_out = gs.Variable("shape_{}_out".format(node_id), dtype=None)
    shape = gs.Node(op="Shape", name="Shape_{}".format(node_id), inputs=[matmul_node.inputs[0]], outputs=[shape_out])
    nodes.append(shape)

    gather_0_out = gs.Variable("gather_{}_0_out".format(node_id), dtype=None)
    gather_0_in = gs.Constant("gather_{}_0_in".format(node_id), values=np.array([0], dtype=np.int64))
    gather0 = gs.Node(op="Gather", name="Gather_{}_0".format(node_id), inputs=[shape_out, gather_0_in], outputs=[gather_0_out], attrs={"axis": 0})
    nodes.append(gather0)

    # unsqueeze_0_out = gs.Variable("unsqueeze_{}_0_out".format(node_id), dtype=None)
    # unsqueeze_0_in = gs.Constant("unsqueeze_{}_0_in".format(node_id), values=np.array([0], dtype=np.int32))
    # unsqueeze0 = gs.Node(op="Unsqueeze", name="Unsqueeze_{}_0".format(node_id), inputs=[gather_0_out, unsqueeze_0_in], outputs=[unsqueeze_0_out])
    # nodes.append(unsqueeze0)

    gather_1_out = gs.Variable("gather_{}_1_out".format(node_id), dtype=None)
    gather_1_in = gs.Constant("gather_{}_1_in".format(node_id), values=np.array([1], dtype=np.int64))
    gather1 = gs.Node(op="Gather", name="Gather_{}_1".format(node_id), inputs=[shape_out, gather_1_in], outputs=[gather_1_out], attrs={"axis": 0})
    nodes.append(gather1)

    # unsqueeze_1_out = gs.Variable("unsqueeze_{}_1_out".format(node_id), dtype=None)
    # unsqueeze_1_in = gs.Constant("unsqueeze_{}_1_in".format(node_id), values=np.array([0], dtype=np.int32))
    # unsqueeze1 = gs.Node(op="Unsqueeze", name="Unsqueeze_{}_1".format(node_id), inputs=[gather_1_out, unsqueeze_1_in], outputs=[unsqueeze_1_out])
    # nodes.append(unsqueeze1)

    gather_2_out = gs.Variable("gather_{}_2_out".format(node_id), dtype=None)
    gather_2_in = gs.Constant("gather_{}_2_in".format(node_id), values=np.array([2], dtype=np.int64))
    gather2 = gs.Node(op="Gather", name="Gather_{}_2".format(node_id), inputs=[shape_out, gather_2_in], outputs=[gather_2_out], attrs={"axis": 0})
    nodes.append(gather2)

    # unsqueeze_2_out = gs.Variable("unsqueeze_{}_2_out".format(node_id), dtype=None)
    # unsqueeze_2_in = gs.Constant("unsqueeze_{}_2_in".format(node_id), values=np.array([0], dtype=np.int32))
    # unsqueeze2 = gs.Node(op="Unsqueeze", name="Unsqueeze_{}_2".format(node_id), inputs=[gather_2_out, unsqueeze_2_in], outputs=[unsqueeze_2_out])
    # nodes.append(unsqueeze2)

    mul_01_out = gs.Variable("Mul_{}_01_out".format(node_id), dtype=None)
    mul_01 = gs.Node(op="Mul", name="Mul_{}_01".format(node_id), inputs=[gather_0_out, gather_1_out], outputs=[mul_01_out])
    nodes.append(mul_01)

    mul_012_out = gs.Variable("Mul_{}_012_out".format(node_id), dtype=None)
    mul_012 = gs.Node(op="Mul", name="Mul_{}_012".format(node_id), inputs=[gather_2_out, mul_01_out], outputs=[mul_012_out])
    nodes.append(mul_012)

    concat_0_out = gs.Variable("Concat_{}_0_out".format(node_id), dtype=None)
    concat_0_in = gs.Constant("Concat_{}_0_in".format(node_id), values=np.array([-1], dtype=np.int64))
    concat_0 = gs.Node(op="Concat", name="Concat_{}_0".format(node_id), inputs=[mul_012_out, concat_0_in], outputs=[concat_0_out], attrs={"axis": 0})
    nodes.append(concat_0)

    reshape_0_out  = gs.Variable("reshape_{}_0_out".format(node_id), dtype=None)
    reshape_0 = gs.Node(op="Reshape", name="Reshape_{}_0".format(node_id), inputs=[matmul_node.inputs[0], concat_0_out], outputs=[reshape_0_out])
    nodes.append(reshape_0)

    matmul_node.inputs[0] = reshape_0_out   

    concat_1_out = gs.Variable("Concat_{}_1_out".format(node_id), dtype=None)
    concat_1 = gs.Node(op="Concat", name="Concat_{}_1".format(node_id), inputs=[shape_out], outputs=[concat_1_out], attrs={"axis": 0})
    nodes.append(concat_1)

    reshape_1_in  = gs.Variable("reshape_{}_1_in".format(node_id), dtype=None)
    reshape_1 = gs.Node(op="Reshape", name="Reshape_{}_1".format(node_id), inputs=[reshape_1_in, concat_1_out], outputs=[add_node.outputs[0]])
    nodes.append(reshape_1)
    
    add_node.outputs[0] = reshape_1_in

def convert_4dmm_3dmm(nodes_dict, target_node, nodes):
    node_id = int(target_node.name.split("_")[-1])

    if (('Concat_{}'.format(node_id-4) not in nodes_dict)):

        return None

    concat_b = nodes_dict['Concat_{}'.format(node_id-4)]
    concat_b_shape0 = gs.Constant(name=concat_b.inputs[0].name, values=np.array([-1], dtype=np.int64))
    concat_b.inputs[0].name += 'del'
    concat_b.inputs[0] = concat_b_shape0
    del concat_b.inputs[1]

    shape_a_out = gs.Variable("shape_{}_out".format(node_id), dtype=None)
    shape_a = gs.Node(op="Shape", name="Shape_{}".format(node_id), inputs=[target_node.inputs[0]], outputs=[shape_a_out])
    nodes.append(shape_a)
    
    gather_a0_out = gs.Variable("gather_{}_a0_out".format(node_id), dtype=None)
    gather_a0_in = gs.Constant("gather_{}_a0_in".format(node_id), values=np.array([9], dtype=np.int64))
    gathera0 = gs.Node(op="Gather", name="Gather_{}_a0".format(node_id), inputs=[shape_a_out, gather_a0_in], outputs=[gather_a0_out], attrs={"axis": 0})
    nodes.append(gathera0)

    gather_a1_out = gs.Variable("gather_{}_a1_out".format(node_id), dtype=None)
    gather_a1_in = gs.Constant("gather_{}_a1_in".format(node_id), values=np.array([1], dtype=np.int64))
    gathera1 = gs.Node(op="Gather", name="Gather_{}_a1".format(node_id), inputs=[shape_a_out, gather_a1_in], outputs=[gather_a1_out], attrs={"axis": 0})
    nodes.append(gathera1)

    gather_a2_out = gs.Variable("gather_{}_a2_out".format(node_id), dtype=None)
    gather_a2_in = gs.Constant("gather_{}_a2_in".format(node_id), values=np.array([2], dtype=np.int64))
    gathera2 = gs.Node(op="Gather", name="Gather_{}_a2".format(node_id), inputs=[shape_a_out, gather_a2_in], outputs=[gather_a2_out], attrs={"axis": 0})
    nodes.append(gathera2)

    gather_a3_out = gs.Variable("gather_{}_a3_out".format(node_id), dtype=None)
    gather_a3_in = gs.Constant("gather_{}_a3_in".format(node_id), values=np.array([3], dtype=np.int64))
    gathera3 = gs.Node(op="Gather", name="Gather_{}_a3".format(node_id), inputs=[shape_a_out, gather_a3_in], outputs=[gather_a3_out], attrs={"axis": 0})
    nodes.append(gathera3)

    concat_a_out = gs.Variable("Concat_{}_a_out".format(node_id), dtype=None)
    concat_a_shape0 = gs.Constant(name="Concat_{}_a_shape0".format(node_id), values=np.array([-1], dtype=np.int64))
    concat_a = gs.Node(op="Concat", name="Concat_{}_a".format(node_id), inputs=[concat_a_shape0, gather_a1_out, gather_a2_out], outputs=[concat_a_out], attrs={"axis": 0})
    nodes.append(concat_a)

    reshape_a_out  = gs.Variable("reshape_{}_a_out".format(node_id), dtype=None)
    reshape_a = gs.Node(op="Reshape", name="Reshape_{}_a".format(node_id), inputs=[target_node.inputs[0], concat_a_out], outputs=[reshape_a_out])
    nodes.append(reshape_a)

    target_node.inputs[0] = reshape_a_out

    concat_c_out = gs.Variable("Concat_{}_c_out".format(node_id), dtype=None)
    concat_c_shape2 = gs.Constant(name="Concat_{}_c_shape2".format(node_id), values=np.array([1], dtype=np.int64))
    concat_c_shape3 = gs.Constant(name="Concat_{}_c_shape3".format(node_id), values=np.array([-1], dtype=np.int64))
    concat_c = gs.Node(op="Concat", name="Concat_{}_c".format(node_id), inputs=[gather_a0_out, gather_a1_out, concat_c_shape2, concat_c_shape3], outputs=[concat_c_out], attrs={"axis": 0})
    nodes.append(concat_c)

    reshape_c_in  = gs.Variable("reshape_{}_c_in".format(node_id), dtype=None)
    reshape_c = gs.Node(op="Reshape", name="Reshape_{}_c".format(node_id), inputs=[reshape_c_in, concat_c_out], outputs=[target_node.outputs[0]])
    nodes.append(reshape_c)

    target_node.outputs[0] = reshape_c_in   

def replace_with_softmax(nodes_dict, softmax_node, nodes):
    node_id = int(softmax_node.name.split("_")[-1])
    if softmax_node.inputs[0].inputs[0].op != 'MatMul':
        return
    
    softmax_node.op = "SoftmaxPlugin"
    softmax_node.name = softmax_node.name.replace("Softmax", "SoftmaxPlugin")

def optimize_dynamic_shape(nodes_dict, graph, nodes):
    image = graph.inputs[0]
    mask = graph.inputs[1]

    shape_out = gs.Variable("myShape_out", dtype=None)
    shape = gs.Node(op='Shape', name="myShape", inputs=[image], outputs=[shape_out])
    nodes.append(shape)

    gather2_out = gs.Variable("myGather_2_out", dtype=None)
    gather2_in = gs.Constant("myGather_2_in", values=np.array(2, dtype=np.int64))
    gather2 = gs.Node(op='Gather', name='myGather_2', inputs=[shape_out,gather2_in], outputs=[gather2_out], attrs={"axis": 0})
    nodes.append(gather2)

    gather3_out = gs.Variable("myGather_3_out", dtype=None)
    gather3_in = gs.Constant("myGather_3_in", values=np.array(3, dtype=np.int64))
    gather3 = gs.Node(op='Gather', name='myGather_3', inputs=[shape_out,gather3_in], outputs=[gather3_out], attrs={"axis": 0})
    nodes.append(gather3)

    unsqueeze0_out = gs.Variable("myUnsqueeze_0_out", dtype=None)
    unsqueeze0_in = gs.Constant("myUnsqueeze_0_in", values=np.array([0], dtype=np.int64))
    unsqueeze0 = gs.Node(op="Unsqueeze", name="myUnsqueeze_0", inputs=[gather2_out, unsqueeze0_in], outputs=[unsqueeze0_out])
    nodes.append(unsqueeze0)

    unsqueeze1_out = gs.Variable("myUnsqueeze_1_out", dtype=None)
    unsqueeze1_in = gs.Constant("myUnsqueeze_1_in", values=np.array([0], dtype=np.int64))
    unsqueeze1 = gs.Node(op="Unsqueeze", name="myUnsqueeze_1", inputs=[gather3_out, unsqueeze1_in], outputs=[unsqueeze1_out])
    nodes.append(unsqueeze1)

    
    slice1_out  = gs.Variable("mySlice1_out", dtype=None)
    slice1_starts  = gs.Constant("mySlice1_starts", values=np.array([0], dtype=np.int64))
    slice1_axes  = gs.Constant("mySlice1_axes", values=np.array([1], dtype=np.int64))
    slice1_steps  = gs.Constant("mySlice1_steps", values=np.array([1], dtype=np.int64))
    slice1_node = gs.Node(op="Slice", name="mySlice1", inputs=[mask, slice1_starts, unsqueeze0_out, slice1_axes, slice1_steps], outputs=[slice1_out])
    nodes.append(slice1_node)

    slice2_out  = gs.Variable("mySlice2_out", dtype=None)
    slice2_starts  = gs.Constant("mySlice2_starts", values=np.array([0], dtype=np.int64))
    slice2_axes  = gs.Constant("mySlice2_axes", values=np.array([2], dtype=np.int64))
    slice2_steps  = gs.Constant("mySlice2_steps", values=np.array([1], dtype=np.int64))
    slice2_node = gs.Node(op="Slice", name="mySlice2", inputs=[slice1_out, slice2_starts, unsqueeze1_out, slice2_axes, slice2_steps], outputs=[slice2_out])
    nodes.append(slice2_node)

    mask.outputs[0].inputs[0] = slice2_out

def add_cast(nodes_dict, not_node, nodes):
    node_id = int(not_node.name.split("_")[-1])

    cast_out  = gs.Variable("cast_{}_out".format(node_id), dtype=None)
    cast1 = gs.Node(op="Cast", name="Cast_{}_1".format(node_id), inputs=[not_node.inputs[0]], outputs=[cast_out], attrs={"to": onnx.TensorProto.BOOL})



    cast_in  = gs.Variable("cast_{}_in".format(node_id), dtype=None)
    cast2  = gs.Node(op="Cast", name="Cast_{}_2".format(node_id), inputs=[cast_in], outputs=[not_node.outputs[0]], attrs={"to": onnx.TensorProto.INT32})
    
    not_node.inputs[0] = cast_out
    not_node.outputs[0] = cast_in
    
    nodes.append(cast1)
    nodes.append(cast2)


print("Load onnx model from {}".format(src_onnx_path))
graph = gs.import_onnx(onnx.load(src_onnx_path))
print("Nodes:{}".format(len(graph.nodes)))
graph.fold_constants().cleanup()
nodes = graph.nodes

nodes_dict = {}
for node in nodes:
    name = node.name
    nodes_dict.update({name: node})

if not DEBUG:
    print("Change cast op for sucessfully pasering")
    cast_to_int32(nodes_dict, nodes_dict['Gather_1097'])

    print("Add cast op for sucessfully pasering")
    add_cast(nodes_dict, nodes_dict['Not_1235'], nodes)

if ENABLE_MERGE_PARAM:
    print("Fuse ops into params")
    ops = [1444, 1903, 2362, 2821, 3280, 3739,
           4524, 5286, 6048, 6810, 7572, 8334]
    for op_name in ops:
        op_name = 'Gather_{}'.format(op_name)
        if op_name in nodes_dict:
            merge_parameters(nodes_dict, nodes_dict[op_name])
    dst_onnx_path =  dst_onnx_path.replace(".onnx", "_mergeparams.onnx")

if ENABLE_LAYERNORM_PLUGIN:
    print("Fuse ops into LayerNorm")
    for op_name in nodes_dict:
        if 'ReduceMean' not in op_name:
            continue
        layernorm = replace_with_layernorm_gamma_beta(nodes_dict, nodes_dict[op_name])
        if layernorm:
            nodes.append(layernorm)
    dst_onnx_path =  dst_onnx_path.replace(".onnx", "_ln.onnx")
if ENABLE_SOFTMAX_PLUGIN:
    print("Replace ops into Softmax")
    ops = [4284, 5046, 5808, 6570, 7332, 8094]
    for op_name in ops:
        op_name = "Softmax_{}".format(op_name)
        if op_name in nodes_dict:
            replace_with_softmax(nodes_dict, nodes_dict[op_name], nodes)
    dst_onnx_path = dst_onnx_path.replace(".onnx", "_softmax.onnx")
if ENABLE_MASKEDSOFTMAX_PLUGIN and not ENABLE_RCDA_PLUGIN:
    print("Fuse ops into MaskedSoftmax in encoder")
    ops = [1710, 1711, 2169, 2170, 2628, 2629, 3087, 3088, 3546, 3547, 4005, 4006]
    for op_name in ops:
        op_name = "Softmax_{}".format(op_name)
        if op_name in nodes_dict:
            replace_with_maskedsoftmax(nodes_dict, nodes_dict[op_name], nodes)

    ops = [4790, 4791, 5552, 5553, 6314, 6315, 7076, 7077, 7838, 7839, 8600, 8601]
    print("Fuse ops into MaskedSoftmax in decoder")
    for op_name in ops:
        op_name = "Softmax_{}".format(op_name)
        if op_name in nodes_dict:
            replace_with_maskedsoftmax(nodes_dict, nodes_dict[op_name], nodes)

    dst_onnx_path = dst_onnx_path.replace(".onnx", "_msoftmax.onnx")
if ENABLE_ADDQBIASTRANSPOSE_PLUGIN and not ENABLE_RCDA_PLUGIN:
    print("Fuse ops into AddQBiasTranspose in encoder")
    ops = [1502, 1477, 1961, 1936, 2420, 2395, 2879, 2854, 3338, 3313, 3797, 3772]
    for op_name in ops:
        op_name = "Add_{}".format(op_name)
        if op_name in nodes_dict:
            replace_with_addQbiastranspose(nodes_dict, nodes_dict[op_name], nodes)

    print("Fuse ops into AddQBiasTranspose in decoder")
    ops = [4557, 4582, 5319, 5344, 6081, 6106, 6843, 6868, 7605, 7630, 8367, 8392]
    for op_name in ops:
        op_name = "Add_{}".format(op_name)
        if op_name in nodes_dict:
            replace_with_addQbiastranspose(nodes_dict, nodes_dict[op_name], nodes)
    dst_onnx_path = dst_onnx_path.replace(".onnx", "_Q.onnx")
if ENABLE_ADDVBIASTRANSPOSE_PLUGIN and not ENABLE_RCDA_PLUGIN:
    print("Fuse ops into AddVBiasTranspose in encoder")
    ops = [1577, 2036, 2495, 2954, 3413, 3872]
    for op_name in ops:
        op_name = "Add_{}".format(op_name)
        if op_name in nodes_dict:
            replace_with_addVbiastranspose(nodes_dict, nodes_dict[op_name], nodes)
    
    print("Fuse ops into AddVBiasTranspose in decoder")
    ops = [4657, 5419, 6181, 6943, 7705, 8467]
    for op_name in ops:
        op_name = "Add_{}".format(op_name)
        if op_name in nodes_dict:
            replace_with_addVbiastranspose(nodes_dict, nodes_dict[op_name], nodes)

    dst_onnx_path = dst_onnx_path.replace(".onnx", "_V.onnx")
if ENABLE_OPTDYNAMICSHAPE:
    print("Optimize dynamic shpae warning")
    optimize_dynamic_shape(nodes_dict, graph, nodes)
    dst_onnx_path = dst_onnx_path.replace(".onnx", "_optshape.onnx")
if ENABLE_CONVERT2DMM:
    print("Convert 4d MM(Matric Multiplication) to 2d MM")
    ops = [1788, 2247, 2706, 3165, 3624, 4083]
    for op_name in ops:
        op_name = "MatMul_{}".format(op_name)
        if op_name in nodes_dict:
            convert_4dmm_2dmm(nodes_dict, nodes_dict[op_name], nodes)
    dst_onnx_path = dst_onnx_path.replace(".onnx", "_2dmm.onnx")

if ENABLE_CONVERT3DMM:
    print("Convert 4d MM(Matric Multiplication) to 3d MM")
    ops = [1749, 2208, 2667, 3126, 3585, 4044]
    for op_name in ops:
        op_name = "MatMul_{}".format(op_name)
        if op_name in nodes_dict:
            convert_4dmm_3dmm(nodes_dict, nodes_dict[op_name], nodes)
    dst_onnx_path = dst_onnx_path.replace(".onnx", "_3dmm.onnx")

if ENABLE_RCDA_PLUGIN:
    # TODO(YukSing:) 4556, 5318, 6080, 6842, 7604, 8366 for decoder
    print("Fuse ops into RCDAPlugin in encoder")
    ops = [1476, 1935, 2394, 2853, 3312, 3771]
    for op_name in ops:
        op_name = 'MatMul_{}'.format(op_name)
        if op_name in nodes_dict:
            replace_with_rcda(nodes_dict, nodes_dict[op_name],  nodes)
    dst_onnx_path = dst_onnx_path.replace(".onnx", "_rcda.onnx")
if ENABLE_REUSE_POS2D:
    print("Reuse pos2posemb2d (Mul_4115)")
    ops = [4877, 5639, 6401, 7163, 7925]
    for op_name in ops:
        op_name = 'Mul_{}'.format(op_name)
        if op_name in nodes_dict:
            reuse_pos2posemb2d(nodes_dict, nodes_dict[op_name],  4115)
    dst_onnx_path = dst_onnx_path.replace(".onnx", "_pos2d.onnx")
if ENABLE_REUSE_POS1D:
    print("Reuse pos2posemb1d (Gather_4339, Gather_4379)")
    ops = [5101, 5863, 6625, 7387, 8149]
    for id in ops:
        op_name = 'Gather_{}'.format(id)
        if op_name in nodes_dict:
            reuse_pos2posemb1d(nodes_dict, nodes_dict[op_name],  4339)
        op_name = 'Gather_{}'.format(id+40)
        if op_name in nodes_dict:
            reuse_pos2posemb1d(nodes_dict, nodes_dict[op_name],  4379)
    dst_onnx_path = dst_onnx_path.replace(".onnx", "_pos1d.onnx")
if ENABLE_REUSE_POSEMB:
    print("Reuse posemb_row (Tile_1386, 1768)")
    ops = [1845,2304,2763,3222,3681,
           4444,5206,5968,6730,7492,8254]
    for id in ops:
        op_name = 'Tile_{}'.format(id)
        if op_name in nodes_dict:
            reuse_posemb(nodes_dict, nodes_dict[op_name], 1386)

    print("Reuse posemb_col (Tile_1413), 1806")
    ops = [1872,2331,2790,3249,3708,
           4471,5233,5995,6757,7519,8281]
    for id in ops:
        op_name = 'Tile_{}'.format(id)
        if op_name in nodes_dict:
            reuse_posemb(nodes_dict, nodes_dict[op_name], 1413)
    dst_onnx_path = dst_onnx_path.replace(".onnx", "_posemb.onnx")
if ENABLE_REMOVE_SLICE:
    print("Remove Slice_1287, Slice_1325")
    ops = [1287, 1325, 4407, 4367]
    for id in ops:
        op_name = 'Slice_{}'.format(id)
        if op_name in nodes_dict:
            fuse_slice_to_concat(nodes_dict, nodes_dict[op_name],256)

    ops = [4149, 4174]
    for id in ops:
        op_name = 'Slice_{}'.format(id)
        if op_name in nodes_dict:
            fuse_slice_to_concat(nodes_dict, nodes_dict[op_name],128)
    dst_onnx_path = dst_onnx_path.replace(".onnx", "_rmslice.onnx")
if ENABLE_GEMMBATCHED and DEBUG:
    # TODO: Fixed accuracy

    # Transpose node
    ops = [1359]
    for id in ops:
        # continue
        op_name = 'Transpose_{}'.format(id)
        if op_name in nodes_dict:
            convert_to_bemmbatched(nodes_dict, nodes_dict[op_name], nodes)

if ENABLE_MASK2POS_PLUGIN:
    print("Insert Mask2PosPlugin")
    ops = [1245, 1240]
    for id in ops:
        op_name = 'CumSum_{}'.format(id)
        if op_name in nodes_dict:
            replace_with_mask2pos(nodes_dict, nodes_dict[op_name], nodes)
    dst_onnx_path = dst_onnx_path.replace(".onnx", "_mask2pos.onnx")
if DEBUG:
    graph.cleanup().toposort()
    dst_onnx_path = './debug.onnx'
    # graph.toposort()
else:
    graph.cleanup().toposort()
print("Nodes:{}".format(len(graph.nodes)))
onnx.save(gs.export_onnx(graph), dst_onnx_path)
print("Save modified onnx model to {}".format(dst_onnx_path))