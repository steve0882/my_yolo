# -*- coding: utf-8 -*-
# @Time : 2025/11/28 14:25
# @Author : shifu wang
# @File : model_info
# @Project : ultralytics-main
from ultralytics import YOLO

if __name__ == '__main__':

    # 加载训练好的YOLO模型
    model = YOLO('ultralytics/cfg/models/rt-detr/rtdetr-resnet101.yaml',task='detect') # 替换为您的模型路径
    # 打印模型的基本信息
    print(model.info())
    # 打印详细的每层结构信息
    print(model.info(detailed=True))