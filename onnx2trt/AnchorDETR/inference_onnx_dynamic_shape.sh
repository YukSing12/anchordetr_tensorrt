
python inference_onnx.py --eval --coco_path ../../datasets/coco  --device cuda --onnx_path AnchorDETR_dynamic.onnx --batch_size 1 --dynamic_shape
