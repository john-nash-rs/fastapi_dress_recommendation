from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import joblib
import os

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load model and encoders
class DressAPIModel:
    def __init__(self, model_path="model/"):
        self.model = tf.keras.models.load_model(os.path.join(model_path, "dress_model.keras"))
        self.encoders = {
            name.split("_encoder.pkl")[0]: joblib.load(os.path.join(model_path, name))
            for name in os.listdir(model_path)
            if name.endswith("_encoder.pkl")
        }
        self.vocab_sizes = {k: len(enc.classes_) for k, enc in self.encoders.items()}

    def predict(self, occasion, country, formality=None, context=None):
        try:
            formality = formality or {
                'business_meeting': 'formal', 'picnic': 'casual',
                'wedding': 'formal', 'job_interview': 'formal',
                'beach_party': 'party', 'nightclub': 'party'
            }.get(occasion, 'casual')

            context = context or {
                'saudi_arabia': 'conservative', 'uae': 'conservative',
                'usa': 'moderate', 'uk': 'moderate', 'brazil': 'relaxed'
            }.get(country, 'moderate')

            encoded_input = {
                'occasion': np.array([self.encoders['occasion'].transform([occasion])[0]]),
                'country': np.array([self.encoders['country'].transform([country])[0]]),
                'formality': np.array([self.encoders['occasion_formality'].transform([formality])[0]]),
                'context': np.array([self.encoders['cultural_context'].transform([context])[0]]),
            }

            probs = self.model.predict(encoded_input)
            top3_idx = np.argsort(probs[0])[-3:][::-1]
            top3_labels = self.encoders['dress_recommendation'].inverse_transform(top3_idx)
            top3_conf = probs[0][top3_idx]

            return {
    "top_recommendation": str(top3_labels[0]),
    "confidence": float(top3_conf[0]),
    "top_3": [(str(label), float(conf)) for label, conf in zip(top3_labels, top3_conf)]
}

        except Exception as e:
            return {"error": str(e)}

# Instantiate model
dress_model = DressAPIModel()

# API Input model
class PredictRequest(BaseModel):
    occasion: str
    country: str
    formality: str = None
    context: str = None

# JSON prediction endpoint
@app.post("/predict")
def predict_dress(request: PredictRequest):
    return dress_model.predict(
        occasion=request.occasion,
        country=request.country,
        formality=request.formality,
        context=request.context
    )

# HTML form endpoint
@app.get("/form", response_class=HTMLResponse)
def get_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/form", response_class=HTMLResponse)
async def submit_form(
    request: Request,
    occasion: str = Form(...),
    country: str = Form(...),
    formality: str = Form(None),
    context: str = Form(None)
):
    result = dress_model.predict(occasion, country, formality, context)
    return templates.TemplateResponse("form.html", {
        "request": request,
        "result": result,
        "occasion": occasion,
        "country": country,
        "formality": formality,
        "context": context
    })

