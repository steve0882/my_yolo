# @Time : 2025/11/13 16:48
# @Author : shifu wang
# @File : validate
# @Project : ultralytics-main
from ultralytics import RTDETR

if __name__ == "__main__":
    # Load a pretrained YOLO11n model
    model = RTDETR(r"runs\detect\train2\weights\best.pt")

    # Evaluate the model's performance on the validation set
    metrics = model.val(
        iou=0.5,
        plots=True,
        workers=0,
    )
