workdir=$(cd $(dirname $0); pwd)
clear
echo "===================== Convert model from pytorch to onnx ===================="
bash export_dynamic_shape_onnx.sh
mv $workdir/model/onnx/AnchorDETR_dynamic.onnx  $workdir/model/onnx/AnchorDETR.onnx

echo "=========================== Start building plugins =========================="
cd $workdir
mkdir -p build
cd build
cmake ..
make -j8
make install

echo "===================== Convert model from onnx to tensorrt ===================="
cd $workdir/src/python
modify_cmd="python modify_AnchorDETR.py --src_onnx $workdir/model/onnx/AnchorDETR.onnx --dst_onnx $workdir/model/onnx/modified_AnchorDETR.onnx --mergeparams --ln --maskedsoftmax -Q -V --mask2pos --pos2d --pos1d --posemb --rmslice"
rst=`eval $modify_cmd | grep "Save modified onnx model to"`
rst=($rst)
idx=$((${#rst[@]}-1))
onnx_path=${rst[$idx]}
echo $onnx_path
python export_AnchorDETR.py --onnx $onnx_path --trt $workdir/model/plan/AnchorDETR.plan --fp16 --plugins $workdir/so


