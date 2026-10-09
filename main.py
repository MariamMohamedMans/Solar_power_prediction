import os
import urllib.request
import pickle
import numpy as np
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

scaler_path = os.path.join(BASE_DIR, "scaler.pkl")
model_path = os.path.join(BASE_DIR, "solar_model.pkl")

# روابط التنزيل المباشرة لملفاتك على Google Drive
SCALER_URL = "https://drive.google.com/uc?export=download&id=1P0HuBC5FstOWVoHpgfJ0Crwf-ORGkeJc"
MODEL_URL = "https://drive.google.com/uc?export=download&id=1GzFWGs1tHBWtGawDcFeHW0fWsjqNr8JA"

# تنزيل الملفات تلقائياً لو مش موجودة في بيئة Vercel
if not os.path.exists(scaler_path):
    urllib.request.urlretrieve(SCALER_URL, scaler_path)

if not os.path.exists(model_path):
    urllib.request.urlretrieve(MODEL_URL, model_path)

# تحميل الـ Scaler والموديل
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
