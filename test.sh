workdir=$(cd $(dirname $0); pwd)
clear
echo "========================== Evaluate performance of model ========================"
# trtexec \
#     --loadEngine=$workdir/model/plan/AnchorDETR.plan \
#     --profilingVerbosity=detailed \
#     --fp16 \
#     --minShapes=image:1x3x320x512 \
#     --optShapes=image:1x3x800x800 \
#     --maxShapes=image:1x3x1344x1344 \
#     --plugins=$workdir/so/AddQBiasTransposePlugin.so \
#     --plugins=$workdir/so/AddVBiasTransposePlugin.so \
#     --plugins=$workdir/so/LayerNormPlugin.so \
#     --plugins=$workdir/so/Mask2PosPlugin.so \
#     --plugins=$workdir/so/MaskedSoftmaxPlugin.so

echo "========================== Evaluate accuracy of model ========================"
cd $workdir/src/AnchorDETR
bash inference_trt_dynamic_shape.sh