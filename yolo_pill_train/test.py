from ultralytics import YOLO

model = YOLO("C:/Users/knifo/Desktop/BH13tesk/Pill_image/yolo_pill_train/pill_detection/yolo_pill_v7/weights/best.pt")

results = model.predict(
    source="C:/Users/knifo/Desktop/BH13tesk/Pill_image/yolo_pill_train/sample/3293.png",
    conf=0.2,
    save=True,
    show=True
)

for r in results:
    for box in r.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        print(f"Detected class: {model.names[cls]} with confidence {conf:.2f}")
