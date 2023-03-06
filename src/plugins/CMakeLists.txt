set(plugin_files 
AddQBiasTransposePlugin.cu 
AddVBiasTransposePlugin.cu 
LayerNormPlugin.cu 
Mask2PosPlugin.cu 
MaskedSoftmaxPlugin.cu)

set(LIB_NAME "Plugins")
add_library(${LIB_NAME} SHARED ${plugin_files})
set_target_properties(${LIB_NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(${LIB_NAME} cudart nvinfer nvinfer_plugin nvonnxparser)

install(TARGETS ${LIB_NAME} LIBRARY DESTINATION ${PROJECT_SOURCE_DIR}/so/plugins)