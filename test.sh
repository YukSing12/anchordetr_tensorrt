workdir=$(cd $(dirname $0); pwd)
clear
echo "========================== Evaluate performance of model ========================"
trtexec \
    --loadEngine=$workdir/model/plan/AnchorDETR.plan \
    --profilingVerbosity=detailed \
    --fp16 \
    --minShapes=image:1x3x320x512 \
    --optShapes=image:1x3x800x800 \
    --maxShapes=image:1x3x1344x1344 \
    --plugins=$workdir/so/plugins/libPlugins.so

echo "========================== Evaluate accuracy of model ========================"
bash inference_dynamic_shape_trt.sh