workdir=$(cd $(dirname $0); pwd)
rm *.onnx *.plan *.so *.cache *.log -f
rm engines/onnx/*.onnx engines/plan/*.plan
clear
echo "===================== Convert model from pytorch to onnx ===================="
cd $workdir/AnchorDETR/
bash export_onnx_dynamic_shape.sh
mv AnchorDETR_dynamic.onnx  $workdir/AnchorDETR.onnx

echo "=========================== Start building plugins =========================="
cd $workdir
plugins="
    AddQBiasTransposePlugin
    AddVBiasTransposePlugin
    LayerNormPlugin
    Mask2PosPlugin
    MaskedSoftmaxPlugin
"
for plugin in $plugins
do
    echo "========================= Start building $plugin ========================"
    plugin_dir="${workdir}/${plugin}"
    cd $plugin_dir
    make clean
    make all
    cp "$plugin.so" $workdir
done

echo "===================== Convert model from onnx to tensorrt ===================="
cd $workdir
python modify_AnchorDETR.py --mergeparams --ln --maskedsoftmax -Q -V --mask2pos --pos2d --pos1d --posemb --rmslice
python export_AnchorDETR.py --mergeparams --ln --maskedsoftmax -Q -V --mask2pos --pos2d --pos1d --posemb --rmslice --fp16

echo "========================== Evaluate performance of model ========================"
trtexec \
    --loadEngine=AnchorDETR.plan \
    --profilingVerbosity=detailed \
    --fp16 \
    --minShapes=image:1x3x320x512,mask:1x320x512 \
    --optShapes=image:1x3x800x800,mask:1x800x800 \
    --maxShapes=image:1x3x1344x1344,mask:1x1344x1344 \
    --plugins=./AddQBiasTransposePlugin.so \
    --plugins=./AddVBiasTransposePlugin.so \
    --plugins=./LayerNormPlugin.so \
    --plugins=./Mask2PosPlugin.so \
    --plugins=./MaskedSoftmaxPlugin.so

echo "========================== Evaluate accuracy of model ========================"
cd $workdir/AnchorDETR
bash inference_trt_dynamic_shape.sh


