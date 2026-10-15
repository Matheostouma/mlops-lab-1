import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")

model_uri = (
    "runs:/21588ff4666343389ae2c10bf9259fad/"
    "food11_model"
)

result = mlflow.register_model(
    model_uri=model_uri,
    name="food11"
)

print("Model name:", result.name)
print("Model version:", result.version)
print("Model source:", result.source)