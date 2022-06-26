#include "Mask2PosPlugin.h"

using namespace nvinfer1;

PluginFieldCollection    Mask2PosPluginCreator::fc_ {};
std::vector<PluginField> Mask2PosPluginCreator::attr_;

// ALIGNPTR
int8_t *alignPtr(int8_t *ptr, uintptr_t to)
{
    uintptr_t addr = (uintptr_t)ptr;
    if (addr % to)
    {
        addr += to - addr % to;
    }
    return (int8_t *)addr;
}

// NEXTWORKSPACEPTR
int8_t *nextWorkspacePtr(int8_t *ptr, uintptr_t previousWorkspaceSize)
{
    uintptr_t addr = (uintptr_t)ptr;
    addr += previousWorkspaceSize;
    return alignPtr((int8_t *)addr, CUDA_MEM_ALIGN);
}

template <typename T, int TPB>
__global__ void pos2posemb1d(const T* input, const T* dim_t, T* output)
{
    const int bidx = blockIdx.x, idx = threadIdx.x;
    float thread_data = static_cast<float>(__ldg(&input[bidx])) / static_cast<float>(__ldg(&dim_t[idx]));
    int oidx = blockIdx.x * TPB + threadIdx.x;
    if(threadIdx.x % 2 == 0)  // Even
    {
        thread_data = sin(thread_data);
    }else                   // Odd
    {
        thread_data = cos(thread_data);
    }
    output[oidx] = (T)thread_data;
}

template <typename T, int TPB>
__global__ void mask2pos(const T* input, T* output, const int len, const T scalar)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    T thread_data = input[idx];

    typedef cub::BlockScan<T, TPB>             BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;
    BlockScan(temp_storage).InclusiveSum(thread_data, thread_data);

    __shared__ T m;
    if(threadIdx.x == len - 1)
    {
        m = thread_data;
    }
    __syncthreads();


    output[idx] = (thread_data - (T)0.5) / m * T(scalar);
}

int32_t Mask2PosPlugin::enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept
{
    WHERE_AM_I();
    const int batch_size = inputDesc[0].dims.d[0];
    const int n = inputDesc[0].dims.d[1];
    if (inputDesc[0].type == DataType::kFLOAT)
    {
        // allocate memory from workspace
        auto *    mul_out_d    = reinterpret_cast<float *>(workspace);
        uintptr_t mul_out_size = CEIL_TO(batch_size * n * sizeof(float), CUDA_MEM_ALIGN);

        // constexpr int VPT = 16 / sizeof(float);
        constexpr int VPT = 1;
        constexpr int TPB = 256 / VPT;
        const float scalar{6.2831854820251465f};
        mask2pos<float, TPB><<<batch_size, TPB, 0, stream>>>((const float *)inputs[0], (float *)mul_out_d, n, scalar);

        pos2posemb1d<float, 256><<<n, 256, 0, stream>>>((const float *)mul_out_d, (const float *)inputs[1], (float *)outputs[0]);
    }else if (inputDesc[0].type == DataType::kHALF)
    {
        // allocate memory from workspace
        auto *    mul_out_d    = reinterpret_cast<half *>(workspace);
        uintptr_t mul_out_size = CEIL_TO(batch_size * n * sizeof(half), CUDA_MEM_ALIGN);

        // constexpr int VPT = 16 / sizeof(half);
        constexpr int VPT = 1;
        constexpr int TPB = 256 / VPT;
        const half scalar{6.2831854820251465f};
        mask2pos<half, TPB><<<batch_size, TPB, 0, stream>>>((const half *)inputs[0], (half *)mul_out_d, n, scalar);

        pos2posemb1d<half, 256><<<n, 256, 0, stream>>>((const half *)mul_out_d, (const half *)inputs[1], (half *)outputs[0]);
    }
    return 0;
}

REGISTER_TENSORRT_PLUGIN(Mask2PosPluginCreator);