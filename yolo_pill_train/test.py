from ultralytics import YOLO

model = YOLO(*yolo_pill_v7/weights/best.pt")

results = model.predict(
    source="*/sample/sampleimage.png",
    conf=0.2,
    save=True,
    show=True
)

for r in results:
    for box in r.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        print(f"Detected class: {model.names[cls]} with confidence {conf:.2f}")
