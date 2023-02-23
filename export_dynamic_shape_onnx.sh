workdir=$(cd $(dirname $0); pwd)
python $workdir/src/AnchorDETR/export_onnx.py  --resume $workdir/model/AnchorDETR_r50_dc5.pth --dynamic_shape  --onnx_path $workdir/model/onnx/AnchorDETR_dynamic.onnx