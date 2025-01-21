from collections import Counter
from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
from PIL import Image
import numpy as np
from io import BytesIO
import cv2

app = FastAPI()

model = YOLO("app/best.pt")

@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image = Image.open(BytesIO(image_data)).convert("RGB")
        
        image_np = np.array(image)

        results = model(image_np)

        result = results[0] 
        predictions = []
        labels = []
        for box in result.boxes:
            predictions.append({
                "label": result.names[int(box.cls)],  
                "confidence": float(box.conf),        
                "bbox": box.xyxy.tolist()              
            })
            labels.append(result.names[int(box.cls)])
        predictions.append({
            'count': Counter(labels)
        })
        return {"predictions": predictions}
    
    except Exception as e:
        return {"error": f"Internal server error: {str(e)}"}

@app.post("/predict/video")
async def predict_video(file: UploadFile = File(...)):
    try:
        video_data = await file.read()
        video_bytes = np.frombuffer(video_data, np.uint8)
        video = cv2.imdecode(video_bytes, cv2.IMREAD_COLOR)
        
        results = model(video)
        
        result = results[0]
        predictions = []
        labels = []
        for box in result.boxes:
            predictions.append({
                "label": result.names[int(box.cls)],  
                "confidence": float(box.conf),        
                "bbox": box.xyxy.tolist()              
            })
            labels.append(result.names[int(box.cls)])
        predictions.append({
            'count': Counter(labels)
        })
        return {"predictions": predictions}
    
    except Exception as e:
        return {"error": f"Internal server error: {str(e)}"}

@app.post("/predict/live")
async def predict_live(file: UploadFile = File(...)):
    try:
        video_data = await file.read()
        video_bytes = np.frombuffer(video_data, np.uint8)
        video = cv2.imdecode(video_bytes, cv2.IMREAD_COLOR)
        
        results = model(video)
        
        result = results[0]
        predictions = []
        labels = []
        for box in result.boxes:
            predictions.append({
                "label": result.names[int(box.cls)],  
                "confidence": float(box.conf),        
                "bbox": box.xyxy.tolist()              
            })
            labels.append(result.names[int(box.cls)])
        predictions.append({
            'count': Counter(labels)
        })
        return {"predictions": predictions}
    
    except Exception as e:
        return {"error": f"Internal server error: {str(e)}"}
