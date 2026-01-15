import os
from ultralytics import YOLO

def train_model():
    # Load the model
    model = YOLO("yolo11n.pt")

    # This finds the root directory of your project automatically
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    # EXACT FOLDER NAME from your error log: 'RDD2022_India-1'
    data_path = os.path.join(base_dir, "data", "RDD2022-India-1", "data.yaml")

    if not os.path.exists(data_path):
        print(f" Still can't find it! Check this path manually: {data_path}")
        return

    # Start training
    model.train(
        data=data_path,
        epochs=50,
        imgsz=640,
        device="cpu", # Your i7-1360P will handle this well
        workers=4
    )

if __name__ == "__main__":
    train_model()