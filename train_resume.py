# @Time : 2025/11/20 22:08
# @Author : shifu wang
# @File : train_resume
# @Project : ultralytics-main
from ultralytics import YOLO

if __name__ == "__main__":
    # resume train
    model = YOLO(r"runs\detect\train8\weights\last.pt")
    train_results = model.train(
        resume=True,
    )
