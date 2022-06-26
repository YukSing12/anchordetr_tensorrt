#include "AddVBiasTransposePlugin.h"
using namespace nvinfer1;

PluginFieldCollection    AddVBiasTransposePluginCreator::fc_ {};
std::vector<PluginField> AddVBiasTransposePluginCreator::attr_;

template<typename T>
__global__ void add_v_bias_transpose(T* v_out,
                        const T* __restrict v_in,
                        const T* __restrict v_bias,
                        const int batch_size,
                        const int seq_len,
                        const int head_num,
                        const int size_per_head)
{
    const int n = head_num * size_per_head;
    const int batch_id = blockIdx.x;
    const int word_id = blockIdx.y;
    const int row_id = batch_id * seq_len + word_id;
    for (int col_id = threadIdx.x; col_id < n; col_id += blockDim.x) {
        const int head_id = col_id / size_per_head;
        const int size_id = col_id % size_per_head;
        const int target_id = batch_id * (head_num * seq_len * size_per_head) + head_id * seq_len * size_per_head
                              + word_id * size_per_head + size_id;
        const int src_id = row_id * n + col_id;

        v_out[target_id] = (__ldg(&v_in[src_id]) + __ldg(&v_bias[col_id]));
    }
}

int32_t AddVBiasTransposePlugin::enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept
{
    WHERE_AM_I();
    const int head_num = 8;
    const int batch_size = inputDesc[0].dims.d[0];
    const int seq_len = inputDesc[0].dims.d[1] * inputDesc[0].dims.d[2];
    const int n = inputDesc[0].dims.d[3];
    const int size_per_head = n / head_num;
    dim3 grid(batch_size, seq_len);
    dim3 block(min(n, 512));
    if (inputDesc[0].type == DataType::kFLOAT)
    {
        (add_v_bias_transpose<float>) <<<grid, block, 0, stream>>>((float *)outputs[0], (const float *)inputs[0], (const float *)inputs[1], batch_size, seq_len, head_num, size_per_head);
    }
    else if (inputDesc[0].type == DataType::kHALF)
    {
        (add_v_bias_transpose<half>) <<<grid, block, 0, stream>>>((half *)outputs[0], (const half *)inputs[0], (const half *)inputs[1], batch_size, seq_len, head_num, size_per_head);
    }
    return 0;
}

REGISTER_TENSORRT_PLUGIN(AddVBiasTransposePluginCreator);