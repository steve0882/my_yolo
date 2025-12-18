# -*- coding: utf-8 -*-
# @Time : 2025/11/13 16:48
# @Author : shifu wang
# @File : train
# @Project : ultralytics-main
from ultralytics import YOLO

if __name__ == '__main__':

    # Load a pretrained YOLO11n model
    model = YOLO(r"ultralytics\cfg\models\v8\my_yolov8n.yaml",task='detect')

    # Train the model on the COCO8 dataset for 100 epochs
    train_results = model.train(
        data="datasets/chili_detect/chili-detect.yaml",
        epochs=100,  # Number of training epochs
        imgsz=640,  # Image size for training
        device=0,  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
        amp=True,
        batch=16,
        workers=0,
        cache=True,
    )

    # resume train
    # model = YOLO("runs/pose/train15/weights/last.pt")
    # train_results = model.train(resume=True,)
