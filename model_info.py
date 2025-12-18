from ultralytics import YOLO

if __name__ == '__main__':
    # 加载训练好的YOLO模型
    model = YOLO(r'ultralytics\cfg\models\v3\yolov3-spp.yaml') # 替换为您的模型路径
    # 打印模型的基本信息
    print(model.info())
    # 打印详细的每层结构信息
    print(model.info(detailed=True))