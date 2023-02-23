# 准备数据

## COCO 2017数据集

前往[官网](https://cocodataset.org/#home)下载COCO 2017数据集COCO_COCO_2017_Train_Val_annotations.zip和val2017.zip
将它们解压在 coco目录下


## 原始模型权重
前往[AnchorDETR原始模型仓库](https://github.com/megvii-research/AnchorDETR)下载AnchorDETR-DC5-R50模型的权重
将AnchorDETR_r50_dc5.pth放置在当前目录下

## 最终目录结构
```
datasets/
├── AnchorDETR_r50_dc5.pth
└── coco
    ├── annotations
    ├── COCO_COCO_2017_Train_Val_annotations.zip
    ├── val2017
    └── val2017.zip
```