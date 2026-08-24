from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib


app = FastAPI(
    title="Spam Detection API",
    description="API for predicting whether an SMS is Spam or Ham using an XGBoost model.",
    version="1.0.0"
)


# CORS middleware for Streamlit / React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load trained models
model = joblib.load("xgb_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")


# Request schema
class SMSRequest(BaseModel):
    text: str


@app.get("/")
def home():
    return {
        "message": "Spam Detection API is running!",
        "status": "success"
    }


@app.post("/predict")
def predict(request: SMSRequest):

    # Convert SMS text into TF-IDF features
    text_features = vectorizer.transform([request.text])

    # Make prediction
    prediction = model.predict(text_features)

    # Convert encoded prediction back to Spam/Ham
    predicted_label = label_encoder.inverse_transform(prediction)

    return {
        "input_text": request.text,
        "prediction": predicted_label[0]
    }