import cv2
from services.anpr import ANPRService

anpr = ANPRService(plate_model_path="weights/license_plate_yolov8n.pt")
frame = cv2.imread("night7.jpg")

result = anpr.read_plates(frame, frame_id="test_001", camera_id="cam_01")
print(f"Plate count: {len(result.plates)}")
for p in result.plates:
    print(f"  bbox={p.bbox}  text='{p.cleaned_text}'  confidence={p.confidence:.3f}")