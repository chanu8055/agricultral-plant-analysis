from ultralytics import YOLO

# Load YOLOv8 nano model (fast and small)
model = YOLO("yolov8n.pt")

# Train the model
model.train(
    data="data.yaml",
    epochs=50,  # increase epochs for better learning
    imgsz=640,
    batch=8,
    name="plant_health_growth_detection",
    device="cpu"
)
