# -*- coding: utf-8 -*-
# @Time : 2025/11/13 16:48
# @Author : shifu wang
# @File : predict
# @Project : ultralytics-main
from ultralytics import YOLO

if __name__ == '__main__':

    # Load a pretrained YOLO11n model
    model = YOLO(r"runs\detect\train2\weights\best.pt")

    # Perform object detection on an image
    # results = model("IMG_20250712_174738.jpg")  # Predict on an image
    # results[0].show()  # Display results

    # model.predict(source="IMG_20250712_174738.jpg",
    #               save=True,
    #               conf=0.5,  # 低于这个置信度，被忽略
    #               iou=0.7,  # 越小的值，消除的重复框越多。
    #               save_txt=True,
    #               )

    model.predict(source=r"pre_img.jpg",
                  save=True,
                  conf=0.25,  # 默认0.25 低于这个置信度，被忽略
                  iou=0.7,  # 默认0.7 越小的值，消除的重复框越多。
                  save_txt=False,
                  max_det=300, #最大检测数量
                  visualize=False,
                  )