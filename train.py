import os
from ultralytics import YOLO


def main():
    dataset_path = os.path.join(os.path.abspath(os.getcwd()), 'datasets/grip-detection.v1i.yolov8/')
    data_yaml_path = os.path.join(dataset_path, 'data.yaml')

    # Check if the data.yaml file exists
    if not os.path.exists(data_yaml_path):
        raise FileNotFoundError(f"data.yaml could not be found in path: {data_yaml_path}")
    
    model = YOLO('yolov8n.pt') # load a pretrained model (recommended for training)
    model.train(data=os.path.join(dataset_path, 'data.yaml'), epochs=50, imgsz=640, device=0) # use GPU
    
    
if __name__ == '__main__':
    main()