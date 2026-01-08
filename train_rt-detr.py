# @Time : 2025/11/13 16:48
# @Author : shifu wang
# @File : train
# @Project : ultralytics-main
from ultralytics import RTDETR

if __name__ == "__main__":
    # Load a pretrained YOLO11n model
    model = RTDETR(r"ultralytics\cfg\models\rt-detr\rtdetr-resnet101.yaml")

    # Train the model on the COCO8 dataset for 100 epochs
    # train_results = model.train(
    #     data="datasets/chili_detect/chili-detect.yaml",
    #     epochs=100,  # Number of training epochs
    #     imgsz=640,  # Image size for training
    #     device=0,  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
    #     amp=True,
    #     batch=4,
    #     workers=0,
    #     cache=True,
    #     optimizer='SGD',
    #     #multi_scale=True,
    # )

    # resume train
    model = RTDETR(r"runs\detect\train2\weights\last.pt")
    train_results = model.train(
        resume=True,
    )
