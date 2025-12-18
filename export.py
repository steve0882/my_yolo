# -*- coding: utf-8 -*-
# @Time : 2025/11/13 16:49
# @Author : shifu wang
# @File : export
# @Project : ultralytics-main
from ultralytics import YOLO

if __name__ == '__main__':

    # Load a pretrained YOLO11n model
    model = YOLO("weight/yolo11n.pt")

    path = model.export(format="onnx")  # Returns the path to the exported model