from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
from io import BytesIO
import torch
import os
from pathlib import Path
import base64

# Import YOLOv5 model
# from models.experimental import attempt_load
from utils.general import non_max_suppression
# from utils.plots import plot_one_box
import cv2
import numpy as np

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load YOLOv5 model
model = torch.hub.load(r'C:\Users\hashi\yolov5', 'custom', path=r'C:/Users/hashi/Downloads/best.pt', source='local')
# device = torch.device("cuda" if torch.cuda.is_available() el.
# se "cpu")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("test0.html", {"request": request})

@app.post("/upload")
async def upload_file(request: Request, file: UploadFile = File(None)):

    
    # Read the contents of the uploaded file as bytes
    contents = await file.read()
    if not contents:
        # Return an HTTPException with status code 400 (Bad Request) and an error message
        return templates.TemplateResponse("error.html", {"request": request})
    
    # Convert bytes to PIL Image
    image = Image.open(BytesIO(contents)).convert("RGB")

    # Perform inference
    pred = model(image)
    pred = pred.xyxy  # Assuming you have only one image
    
    if len(pred[0])==0:
        return templates.TemplateResponse("error.html", {"request": request})

    # Convert image to OpenCV format
    image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    for box in pred:
    # Iterate over each bounding box in the prediction
        for bbox in box:
        # Extracting bounding box coordinates, confidence, and class label from the prediction
            x1, y1, x2, y2, conf, cls = map(int, bbox[:6].tolist())
        
        # Drawing a rectangle around the detected object
            cv2.rectangle(image_cv2, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # writing class name
            class_name = model.names[cls]
            cv2.putText(image_cv2, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            


    # Convert OpenCV image to PIL Image
    image_with_boxes = Image.fromarray(cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB))

    # Convert PIL Image to bytes
    img_byte_array = BytesIO()
    image_with_boxes.save(img_byte_array, format="JPEG")
    img_byte_array = img_byte_array.getvalue()

    # Encode the inferred image as a base64 string
    img_str = base64.b64encode(img_byte_array).decode("utf-8")
    
    return templates.TemplateResponse("test1.html", {"request": request, "image": img_str})



import uvicorn
import ngrok
ngrok.set_auth_token("2fNLadwPdufTdrbMS01J49Qr7r3_6hAZuck3PyzMGJ3gktpev")
uvicorn.run(app, host="localhost", port=8000)
# ColabCode.run_app(app=app)
# colab.run_app(app)
# colab.run_app(app=wsgi_app)


