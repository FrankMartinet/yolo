from ultralytics import YOLO

# Load a model
model = YOLO("yolov8s.yaml")  # build a new model from YAML
model = YOLO("yolov8s.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolov8s.yaml").load("yolov8s.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="coco8.yaml", 
                      epochs=20, 
                      imgsz=640, 
                      lr0=0.01, 
                      batch=16,
                      save=True, 
                      save_period=1, 
                      workers=4, 
                      device="cpu", 
                      pretrained=True,
                      seed=123,
                      resume=False,
                      val=True,
                      plots=True,
                      project="runs/detect",)