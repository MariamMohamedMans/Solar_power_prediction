import os
import zipfile
import pickle
import numpy as np
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
zip_path = os.path.join(BASE_DIR, "models.zip")
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")
model_path = os.path.join(BASE_DIR, "solar_model.pkl")

# فك ضغط ملفات الموديل أوتوماتيك بأمان
if os.path.exists(zip_path) and not os.path.exists(model_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(BASE_DIR)

# تحميل الملفات سليمة 100%
scaler = pickle.load(open(scaler_path, "rb"))
model = pickle.load(open(model_path, "rb"))

templates = Jinja2Templates(directory=BASE_DIR)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    ambient_temp: float = Form(...),
    module_temp: float = Form(...),
    irradiation: float = Form(...)
):
    features = np.array([[ambient_temp, module_temp, irradiation]])
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)[0]
    
    return templates.TemplateResponse(
        "index.html", 
        {"request": request, "prediction_text": f"المخرجات المتوقعة: {prediction:.2f}"}
    )
