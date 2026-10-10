import os
import joblib
import numpy as np
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")
model_path = os.path.join(BASE_DIR, "solar_model.pkl")

# استخدام joblib بدلاً من pickle لتجنب أخطاء STACK_GLOBAL
scaler = joblib.load(scaler_path)
model = joblib.load(model_path)

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
