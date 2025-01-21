from collections import Counter
from fastapi import FastAPI, UploadFile, File, HTTPException
from matplotlib import pyplot as plt
import torch
from ultralytics import YOLO
from PIL import Image
import numpy as np
from io import BytesIO
import cv2

app = FastAPI()

model = YOLO("best.pt")

def process_image(frame):
    results = model(frame, stream=True, save=False)  
    output = [result for result in results]
    orig_img = output[0].orig_img
    boxes = output[0].boxes.xyxy
    class_names = output[0].names
    confidences = output[0].boxes.conf

    if isinstance(orig_img, torch.Tensor):
        orig_img = orig_img.cpu().numpy()

    for idx, box in enumerate(boxes):
        x_min, y_min, x_max, y_max = box.cpu().numpy()
        class_id = int(output[0].boxes.cls[idx].item()) 
        class_name = class_names[class_id]  
        confidence = confidences[idx].item()  

        cv2.rectangle(orig_img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
        label = f"{class_name} ({confidence*100:.2f}%)"
        cv2.putText(orig_img, label, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    orig_img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    plt.imshow(orig_img_rgb)
    plt.axis('off')
    plt.show()

    return {"predictions": "Image processing completed and displayed.", "Values": orig_img_rgb}

@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image = Image.open(BytesIO(image_data)).convert("RGB")
        image_np = np.array(image)
        results = model(image_np)
        result = results[0]
        predictions = [{"label": result.names[int(box.cls)], "confidence": float(box.conf), "bbox": box.xyxy.tolist()} for box in result.boxes]
        labels = [result.names[int(box.cls)] for box in result.boxes]
        predictions.append({'count': Counter(labels)})
        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/capture-and-detect")
async def capture_and_detect():
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise HTTPException(status_code=500, detail="Webcam not accessible")

        ret, frame = cap.read()
        if not ret:
            cap.release()
            raise HTTPException(status_code=500, detail="Failed to capture image from webcam")

        results = model(frame)
        process_image(frame)

        result = results[0]
        predictions = [{"label": result.names[int(box.cls)], "confidence": float(box.conf), "bbox": box.xyxy.tolist()} for box in result.boxes]
        labels = [result.names[int(box.cls)] for box in result.boxes]
        predictions.append({'count': Counter(labels)})

        cap.release()
        return {"predictions": predictions}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during live capture and detection: {str(e)}")

@app.get("/")
def read_root():
    return {"message": "Welcome to the YOLO Detection API!"}
