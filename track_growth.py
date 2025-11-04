from ultralytics import YOLO
import os

# Load your trained YOLO model
model = YOLO(r"C:\Users\shash\Downloads\Plant_Health_Monitoring_YOLOv8\runs\detect\plant_health_training2\weights\best.pt")

# Run prediction on your growth images
model.predict(
    source="growth_images",     # folder containing your test/growth images
    conf=0.25,                  # lower confidence threshold to catch weak detections
    save=True,
    project="growth_results",   # save folder
    name="predict5"
)
