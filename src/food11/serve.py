
"""Serve the registered Food-11 model with FastAPI."""

import io
import os

import mlflow.pyfunc
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision import transforms

import mlflow

# MLflow tracking server
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000",
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


# Model is loaded when the API starts, not during import
model = None


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


# Load champion model only when the API starts
@app.on_event("startup")
def load_model():
    global model

    # Skip real model loading during unit tests and CI
    if os.environ.get("SKIP_MODEL"):
        return

    model = mlflow.pyfunc.load_model(
        "models:/food11@champion"
    )


# Health endpoint
@app.get("/health")
def health():
    return {"status": "ok"}


# Prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Return 503 if the model is not loaded
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded"
        )

    contents = await file.read()

    image = Image.open(
        io.BytesIO(contents)
    ).convert("RGB").resize((128, 128))

    image_tensor = transform(image).unsqueeze(0)

    predictions = model.predict(image_tensor.numpy())

    logits = torch.tensor(np.asarray(predictions))

    probabilities = torch.softmax(logits, dim=1)

    confidence, predicted_index = torch.max(
        probabilities, dim=1
    )

    return {
        "category": CLASSES[predicted_index.item()],
        "confidence": confidence.item(),
    }
