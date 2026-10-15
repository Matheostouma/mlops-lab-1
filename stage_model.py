import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:5000")

client = MlflowClient()

client.transition_model_version_stage(
    name="food11",
    version=1,
    stage="Staging"
)

print("Food-11 version 1 assigned to Staging!")