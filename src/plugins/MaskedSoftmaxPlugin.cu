#include "MaskedSoftmaxPlugin.h"
using namespace nvinfer1;

PluginFieldCollection    MaskedSoftmaxPluginCreator::fc_ {};
std::vector<PluginField> MaskedSoftmaxPluginCreator::attr_;

template<int ITEMS_PER_THREAD, typename T>
__global__ void masked_softmax_kernel_v4(T* qk_buf_,
                                  const T* qk_buf_src,
                                  const int* attr_mask,
                                  const int batch_size,
                                  const int head_num,
                                  const int tgt_seq_len,
                                  const int src_seq_len)
{
    for (int seq_id = blockIdx.x; seq_id < tgt_seq_len; seq_id += gridDim.x) {
        float data[ITEMS_PER_THREAD];
        int qk_offset;
        __shared__ float s_mean, s_max;
        float local_max = -1e20f;
        for (int i = 0; blockDim.x * i + threadIdx.x < src_seq_len; i++) {
            qk_offset =
                ((blockIdx.y * head_num + blockIdx.z) * tgt_seq_len + seq_id) * src_seq_len + blockDim.x * i + threadIdx.x;
            int mask_offset = blockDim.x * i + threadIdx.x;

            float qk       = static_cast<float>(__ldg(&qk_buf_src[qk_offset]));
            float mask_val = static_cast<float>(__ldg(&attr_mask[mask_offset]));

            mask_val = (1.0f - mask_val) * -10000.0f;

            data[i] = qk + mask_val;
            local_max = fmax(local_max, data[i]);
        }

        float max_val = blockDim.x <= 32 ? fastertransformer::warpReduceMax<float>(local_max) : fastertransformer::blockReduceMax<float>(local_max);
        if (threadIdx.x == 0) {
            s_max = max_val;
        }
        __syncthreads();

        float local_sum = 0;
        for (int i = 0; blockDim.x * i + threadIdx.x < src_seq_len; i++) {
            data[i] = __expf(data[i] - s_max);
            local_sum += data[i];
        }
        float sum_val = blockDim.x <= 32 ? fastertransformer::warpReduceSum(local_sum) : fastertransformer::blockReduceSum<float>(local_sum);
        if (threadIdx.x == 0) {
            s_mean = sum_val + 1e-6f;
            s_mean = __fdividef(1.0f, s_mean);
        }
        __syncthreads();

        for (int i = 0; blockDim.x * i + threadIdx.x < src_seq_len; i++) {
            qk_offset =
                ((blockIdx.y * head_num + blockIdx.z) * tgt_seq_len + seq_id) * src_seq_len + blockDim.x * i + threadIdx.x;
            qk_buf_[qk_offset] = (T)(data[i] * s_mean);
        }
    }
}

int32_t MaskedSoftmaxPlugin::enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept
{
    WHERE_AM_I();
    const int head_num = 8;
    const int batch_size = inputDesc[0].dims.d[0] / head_num;
    const int tgt_seq_len = inputDesc[0].dims.d[1];
    const int src_seq_len = inputDesc[0].dims.d[2];
    dim3 grid(tgt_seq_len, batch_size, head_num);
    if (batch_size * head_num > 360) {
        grid.x = ceil(float(tgt_seq_len) / 32.0f);
    }
    if (inputDesc[0].type == DataType::kFLOAT)
    {
        dim3 block((src_seq_len / 1 + 31) / 32 * 32);
        if (block.x > 3072 && block.x <= 4096) {
            block.x /= 4;
            assert(block.x <= 1024);
            masked_softmax_kernel_v4<4, float>                                                                   
                <<<grid, block, 0, stream>>>((float *)outputs[0], (const float *)inputs[0], (const int *)inputs[1], batch_size, head_num, tgt_seq_len, src_seq_len);        
        }
        if (block.x > 2048) {
            block.x /= 3;
            assert(block.x <= 1024);
            masked_softmax_kernel_v4<3, float>                                                                   
                <<<grid, block, 0, stream>>>((float *)outputs[0], (const float *)inputs[0], (const int *)inputs[1], batch_size, head_num, tgt_seq_len, src_seq_len);        
        }
        else if (block.x > 1024) {
            block.x /= 2;
            assert(block.x <= 1024);
            masked_softmax_kernel_v4<2, float>                                                                   
                <<<grid, block, 0, stream>>>((float *)outputs[0], (const float *)inputs[0], (const int *)inputs[1], batch_size, head_num, tgt_seq_len, src_seq_len);        
        }
        else if (block.x > 0) {
            block.x /= 1;
            assert(block.x <= 1024);
            masked_softmax_kernel_v4<1, float>                                                                   
                <<<grid, block, 0, stream>>>((float *)outputs[0], (const float *)inputs[0], (const int *)inputs[1], batch_size, head_num, tgt_seq_len, src_seq_len);        
        }
        else {
            printf("Error: tgt_seq_len can not meet tgt_seq_len <= 4096");
        }
    }
    else if (inputDesc[0].type == DataType::kHALF)
    {
        dim3 block((src_seq_len / 1 + 31) / 32 * 32);
        if (block.x > 3072 && block.x <= 4096) {
            block.x /= 4;
            assert(block.x <= 1024);
            masked_softmax_kernel_v4<4, half>                                                                   
                <<<grid, block, 0, stream>>>((half *)outputs[0], (const half *)inputs[0], (const int *)inputs[1], batch_size, head_num, tgt_seq_len, src_seq_len);        
        }
        if (block.x > 2048) {
            block.x /= 3;
            assert(block.x <= 1024);
            masked_softmax_kernel_v4<3, half>                                                                   
                <<<grid, block, 0, stream>>>((half *)outputs[0], (const half *)inputs[0], (const int *)inputs[1], batch_size, head_num, tgt_seq_len, src_seq_len);        
        }
        else if (block.x > 1024) {
            block.x /= 2;
            assert(block.x <= 1024);
            masked_softmax_kernel_v4<2, half>                                                                   
                <<<grid, block, 0, stream>>>((half *)outputs[0], (const half *)inputs[0], (const int *)inputs[1], batch_size, head_num, tgt_seq_len, src_seq_len);        
        }
        else if (block.x > 0) {
            block.x /= 1;
            assert(block.x <= 1024);
            masked_softmax_kernel_v4<1, half>                                                                   
                <<<grid, block, 0, stream>>>((half *)outputs[0], (const half *)inputs[0], (const int *)inputs[1], batch_size, head_num, tgt_seq_len, src_seq_len);        
        }
        else {
            printf("Error: tgt_seq_len can not meet tgt_seq_len <= 4096");
        }
    }
    return 0;
}

REGISTER_TENSORRT_PLUGIN(MaskedSoftmaxPluginCreator);