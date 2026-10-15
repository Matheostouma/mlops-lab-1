Question 1: Without a volume, MLflow data is stored inside the container. If we remove the container and create a new one, the old database and model records are lost.

Question 2: A named volume is managed by Docker and keeps data separate from the container and project folder. A bind mount would also work, but it depends on a specific folder on the host computer.

Question 3: Docker Compose creates a network where containers can communicate using service names. The name mlflow resolves automatically to the MLflow container.

Question 4: Using an environment variable makes the frontend flexible. We can change the inference server address without modifying the Python code, including when running the frontend outside Compose.

Question 5: Inference does not need a published port because the frontend accesses it through the internal Compose network using http://inference:8000. Only MLflow and Streamlit need ports accessible from the host.

Question 6: depends_on controls startup order, not readiness. If MLflow is not ready, inference may fail to load the model and exit. We can use a health check, retries, or a restart policy.

Question 7: docker compose ps showed MLflow publishing port 5000 and frontend publishing port 8501. Inference had no published host port. This matches our Compose configuration.

Question 8: The old model remains loaded because inference loads the model at startup. After moving another model version to Staging, run docker compose restart inference to load the new Staging version.

Question 9: Restarting works because the inference image contains the application code and dependencies, while the model is loaded from MLflow at startup. We only need to rebuild if the code or dependencies change.

Question 10: MLflow data is stored in the named volume, so it survives container recreation with docker compose down and docker compose up. Running docker compose down -v would delete the volume and its stored database and artifacts. We did not run this destructive test.

Question 11: For multiple inference replicas with load balancing and recovery from machine failures, we would need an orchestration system such as Kubernetes. We would also need shared persistent storage and a reliable external database for MLflow.