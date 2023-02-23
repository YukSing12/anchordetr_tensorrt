workdir=$(cd $(dirname $0); pwd)
python $workdir/src/AnchorDETR/inference_onnx.py --eval --coco_path $workdir/data/coco  --device cuda --onnx_path $workdir/model/onnx/AnchorDETR_dynamic.onnx --batch_size 1 --dynamic_shape