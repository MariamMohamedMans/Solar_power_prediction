import os
import joblib
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

app = FastAPI()

# تحديد المسار المطلق للمجلد الحالي
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

scaler_path = os.path.join(BASE_DIR, "scaler.pkl")
model_path = os.path.join(BASE_DIR, "solar_model.pkl")

# تحميل الـ Scaler والموديل باستخدام joblib
scaler = joblib.load(scaler_path)
model = joblib.load(model_path)

# إعداد ملفات الـ Templates
templates = Jinja2Templates(directory=BASE_DIR)

# تعريف هيكل البيانات القادمة من الـ Frontend
class SolarInput(BaseModel):
    ambient_temperature: float
    module_temperature: float
    irradiation: float

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html")

@app.post("/predict")
async def predict_api(data: SolarInput):
    # تجهيز المدخلات بنفس ترتيب الـ Features اللي اتدرب عليها الموديل
    features = np.array([[data.ambient_temperature, data.module_temperature, data.irradiation]])
    
    # عمل Scale للمدخلات ثم التنبؤ
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)[0]
    
    # إرجاع النتيجة بالشکل اللي صفحة الـ HTML بتستقبله (predicted_power)
    return {"predicted_power": round(float(prediction), 2)}
