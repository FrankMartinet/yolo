from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")  # load an official model
model = YOLO("runs/train/train/weights/best.pt")  # load a custom trained model

# Export the model
model.export(format="onnx")
