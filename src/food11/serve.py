"""Serve the registered Food-11 model with FastAPI."""

import io
import os

import mlflow
import mlflow.pyfunc
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from torchvision import transforms


# MLflow tracking server
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000",
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


# Load the champion model once when the API starts
model = mlflow.pyfunc.load_model("models:/food11@champion")


# Same preprocessing used during training
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


# Food-11 categories
CLASSES = [
    "Bread",
    "Dairy product",
    "Dessert",
    "Egg",
    "Fried food",
    "Meat",
    "Noodles-Pasta",
    "Rice",
    "Seafood",
    "Soup",
    "Vegetable-Fruit",
]


app = FastAPI(title="Food11 API")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    image = Image.open(io.BytesIO(contents)).convert("RGB")

    image_tensor = transform(image).unsqueeze(0)

    predictions = model.predict(image_tensor.numpy())

    logits = torch.tensor(np.asarray(predictions))

    probabilities = torch.softmax(logits, dim=1)

    confidence, predicted_index = torch.max(probabilities, dim=1)

    return {
        "category": CLASSES[predicted_index.item()],
        "confidence": confidence.item(),
    }