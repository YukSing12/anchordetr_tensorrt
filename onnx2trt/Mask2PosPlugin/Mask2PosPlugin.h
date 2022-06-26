#include <NvInfer.h>
#include <cub/cub.cuh>
#include <cuda_fp16.h>
#include "common.cuh"
#include <string>
#include <vector>

#define CEIL_DIVIDE(X, Y) (((X) + (Y)-1) / (Y))
#define CEIL_TO(X, Y)     (((X) + (Y)-1) / (Y) * (Y))

#define CUDA_MEM_ALIGN 256

#if NDEBUG
    #define WHERE_AM_I()                                 \
        do                                               \
        {                                                \
            printf("[%s]: this=->%p\n", __func__, this); \
        } while (0);
#else
    #define WHERE_AM_I()
#endif // #ifndef NDEBUG

namespace nvinfer1
{
static const char *PLUGIN_NAME {"Mask2PosPlugin"};
static const char *PLUGIN_VERSION {"1"};

class Mask2PosPlugin : public IPluginV2DynamicExt
{
private:
    std::string name_;
    std::string namespace_;

public:
    Mask2PosPlugin(const std::string &name):
        name_(name)
    {
        WHERE_AM_I();
    }

    Mask2PosPlugin(const std::string &name, const void *data, size_t length):
        name_(name)
    {
        WHERE_AM_I();
    }

    Mask2PosPlugin() = delete;

    ~Mask2PosPlugin()
    {
        WHERE_AM_I();
    }

    size_t getSerializationSize() const noexcept override
    {
        WHERE_AM_I();
        return 0;
    }

    void serialize(void *buffer) const noexcept override
    {
        WHERE_AM_I();
    }

    IPluginV2DynamicExt *clone() const noexcept override
    {
        WHERE_AM_I();
        return new Mask2PosPlugin(name_);
    }

    int getNbOutputs() const noexcept override
    {
        WHERE_AM_I();
        return 1;
    }

    DimsExprs getOutputDimensions(int32_t outputIndex, const DimsExprs *inputs, int32_t nbInputs, IExprBuilder &exprBuilder) noexcept override
    {
        WHERE_AM_I();
        DimsExprs outputs;
        outputs.nbDims = 3;
        outputs.d[0] = inputs[0].d[0];
        outputs.d[1] = inputs[0].d[1];
        outputs.d[2] = exprBuilder.constant(256);
        return outputs;
    }

    bool supportsFormatCombination(int32_t pos, const PluginTensorDesc *inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override
    {
        WHERE_AM_I();       
        if (inOut[pos].format != TensorFormat::kLINEAR)
        {
            return false;
        }

        bool res = false;
        switch (pos)
        {
        case 0: // mask
            res = (inOut[pos].type == DataType::kHALF);
            break;
        case 1: // dim_t
            res = (inOut[pos].type == DataType::kHALF);
            break;
        case 2: // output
            res = (inOut[pos].type == DataType::kHALF);
            break;
        default: // should NOT be here
            break;
        }
        return res;
    }

    DataType getOutputDataType(int outputIndex, const DataType *inputTypes, int nbInputs) const noexcept override
    {
        WHERE_AM_I();
        return inputTypes[0];
    }

    void configurePlugin(const DynamicPluginTensorDesc *in, int32_t nbInputs, const DynamicPluginTensorDesc *out, int32_t nbOutputs) noexcept override
    {
        WHERE_AM_I();
    }

    size_t getWorkspaceSize(const PluginTensorDesc *inputs, int32_t nbInputs, const PluginTensorDesc *outputs, int32_t nbOutputs) const noexcept override
    {
        WHERE_AM_I();
        const int batch_size        = inputs[0].dims.d[0];
        const int n                 = inputs[0].dims.d[1];
        const size_t element_size   = (inputs[0].type == DataType::kFLOAT) ? sizeof(float) : sizeof(half);
        size_t       realSize = 0, workspaceSize = 0;

        realSize = batch_size * n * element_size; // mul_out
        workspaceSize += CEIL_TO(realSize, CUDA_MEM_ALIGN);

        return workspaceSize;
    }

    void setPluginNamespace(const char *szNamespace) noexcept override
    {
        WHERE_AM_I();
        namespace_ = szNamespace;
    }
    const char *getPluginNamespace() const noexcept override
    {
        WHERE_AM_I();
        return namespace_.c_str();
    }
    const char *getPluginType() const noexcept override
    {
        WHERE_AM_I();
        return PLUGIN_NAME;
    }
    const char *getPluginVersion() const noexcept override
    {
        WHERE_AM_I();
        return PLUGIN_VERSION;
    }
    int initialize() noexcept override
    {
        WHERE_AM_I();
        return 0;
    }
    void terminate() noexcept override
    {
        WHERE_AM_I();
        return;
    }

    void destroy() noexcept override
    {
        WHERE_AM_I();
    }

    int32_t enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept override;
}; // class Mask2PosPlugin

class Mask2PosPluginCreator : public IPluginCreator
{
private:
    static PluginFieldCollection    fc_;
    static std::vector<PluginField> attr_;
    std::string                     namespace_;

public:
    Mask2PosPluginCreator()
    {
        WHERE_AM_I();
        fc_.nbFields = attr_.size();
        fc_.fields   = attr_.data();
    }

    ~Mask2PosPluginCreator() {}

    IPluginV2 *createPlugin(const char *name, const PluginFieldCollection *fc) noexcept override
    {
        WHERE_AM_I();
        return new Mask2PosPlugin(name);
    }

    IPluginV2 *deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept override
    {
        WHERE_AM_I();
        return new Mask2PosPlugin(name, serialData, serialLength);
    }

    void setPluginNamespace(const char *szNamespace) noexcept override
    {
        namespace_ = szNamespace;
    }

    const char *getPluginNamespace() const noexcept override
    {
        return namespace_.c_str();
    }

    const char *getPluginName() const noexcept override
    {
        return PLUGIN_NAME;
    }

    const char *getPluginVersion() const noexcept override
    {
        return PLUGIN_VERSION;
    }

    const PluginFieldCollection *getFieldNames() noexcept override
    {
        return &fc_;
    }
}; // class Mask2PosPluginCreator

} // namespace nvinfer1