#include <NvInfer.h>
#include <cuda_fp16.h>
#include <string>
#include <vector>
#include "reduce_kernel_utils.cuh"

#define CEIL_DIVIDE(X, Y) (((X) + (Y)-1) / (Y))
#define CEIL_TO(X, Y)     (((X) + (Y)-1) / (Y) * (Y))

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
static const char *PLUGIN_NAME {"AddQBiasTransposePlugin"};
static const char *PLUGIN_VERSION {"1"};

class AddQBiasTransposePlugin : public IPluginV2DynamicExt
{
private:
    std::string name_;
    std::string namespace_;

public:
    AddQBiasTransposePlugin(const std::string &name):
        name_(name)
    {
        WHERE_AM_I();
    }

    AddQBiasTransposePlugin(const std::string &name, const void *data, size_t length):
        name_(name)
    {
        WHERE_AM_I();
    }

    AddQBiasTransposePlugin() = delete;

    ~AddQBiasTransposePlugin()
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
        return new AddQBiasTransposePlugin(name_);
    }

    int getNbOutputs() const noexcept override
    {
        WHERE_AM_I();
        return 1;
    }

    DimsExprs getOutputDimensions(int32_t outputIndex, const DimsExprs *inputs, int32_t nbInputs, IExprBuilder &exprBuilder) noexcept override
    {
        WHERE_AM_I();
        // Input is B*HW*256
        assert(outputIndex == 0);
        // Copy over everything
        DimsExprs output;
        output.nbDims = 3;
        output.d[0] = exprBuilder.constant(8);              // head number
        output.d[1] = inputs[0].d[1];                       // h * w
        output.d[2] = exprBuilder.constant(32);
        return output;
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
        case 0: // q_row
            res = (inOut[pos].type == DataType::kHALF);
            // res = (inOut[pos].type == DataType::kFLOAT) || (inOut[pos].type == DataType::kHALF);
            break;
        case 1: // q_bias
            res = (inOut[pos].type == DataType::kHALF);
            // res = (inOut[pos].type == DataType::kFLOAT) || (inOut[pos].type == DataType::kHALF);
            break;
        case 2: // q_out
            res = inOut[pos].type == inOut[0].type;
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
        return 0;
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
}; // class AddQBiasTransposePlugin

class AddQBiasTransposePluginCreator : public IPluginCreator
{
private:
    static PluginFieldCollection    fc_;
    static std::vector<PluginField> attr_;
    std::string                     namespace_;

public:
    AddQBiasTransposePluginCreator()
    {
        WHERE_AM_I();
        fc_.nbFields = attr_.size();
        fc_.fields   = attr_.data();
    }

    ~AddQBiasTransposePluginCreator() {}

    IPluginV2 *createPlugin(const char *name, const PluginFieldCollection *fc) noexcept override
    {
        WHERE_AM_I();
        return new AddQBiasTransposePlugin(name);
    }

    IPluginV2 *deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept override
    {
        WHERE_AM_I();
        return new AddQBiasTransposePlugin(name, serialData, serialLength);
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
}; // class AddQBiasTransposePluginCreator

} // namespace nvinfer1