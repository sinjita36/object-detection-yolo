from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

# Detect objects in image
results = model("sample.jpg")

# Print detected objects
for result in results:
    print(result.boxes)
