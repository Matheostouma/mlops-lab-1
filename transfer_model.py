from pathlib import Path
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("food11-compose")

model_path = Path(
    "mlruns/1/models/"
    "m-5c9283d0c20744d189eb5f178cde4117/"
    "artifacts"
)

with mlflow.start_run(run_name="import-lab3-model") as run:
    mlflow.log_artifacts(str(model_path), artifact_path="food11_model")
    print("Upload complete!")
    print("Run ID:", run.info.run_id)