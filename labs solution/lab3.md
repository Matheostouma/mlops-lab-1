# Lab 3 - Containerizing the Model with Docker

## Question 1

**What version number was your model given? What's the difference between a run's logged model artifact and a registered model?**

My model was first registered as Version 1. Later, I created Version 2 with the same trained model because Version 1 had an artifact path that was local to my Windows machine, which caused a problem when loading the model from Docker. I assigned the `champion` alias to Version 2.

A logged model artifact is the actual model files produced and stored by a specific MLflow run. A registered model is a model managed in the MLflow Model Registry with a name, such as `food11`, and it can have multiple versions.

---

## Question 2

**What aliases replaced the old built-in stages in MLflow? Why version a model separately from the run that produced it, and why is an alias more flexible than a fixed stage name?**

Aliases such as `champion` and `challenger` replace the old MLflow stages such as `Staging` and `Production`.

Versioning the model separately from the run makes it easier to manage different versions of the model without depending directly on the training run.

An alias is more flexible because it can be moved from one model version to another. For example, I can move the `champion` alias from Version 1 to Version 2 without changing my serving code.

---

## Question 3

**Why load the model through an MLflow model URI (`models:/food11@champion`) instead of pointing directly at the `.pth` file on disk? What would you have to change to serve a newer model version?**

Using `models:/food11@champion` allows MLflow to manage the location and version of the model. The serving application does not need to know the exact path of the `.pth` file.

To serve a newer model version, I can register the new version and move the `champion` alias to it. I do not need to change the model URI in `serve.py`.

Since the model is loaded once when the API starts, I would restart the API or Docker container so that it loads the new `champion` model.

---

## Question 4

**Why copy `pyproject.toml`/`uv.lock` and run `uv sync` before copying the rest of the source code? What happens to the build cache when you only change a line in `serve.py`?**

The dependency files are copied first so Docker can cache the dependency installation layer.

If I only change a line in `serve.py`, Docker can reuse the cached dependency layer instead of installing all the packages again. Only the source code layer and the layers after it need to be rebuilt.

This makes rebuilding the Docker image faster.

---

## Question 5

**What's the size difference between a naive single-stage image and your multi-stage one? Use `docker history <image>` to see which layers are the biggest.**

My `food11-api:latest` image showed approximately 1.99 GB of disk usage and 426 MB of content size.

Using `docker history`, I found that the largest layer was the copied virtual environment `/app/.venv`, which was approximately 1.43 GB.

I did not build a separate naive single-stage image, so I cannot give an exact measured size difference between the two.

The multi-stage build helps keep unnecessary build tools and builder files out of the final runtime image. Only what is needed to run the application is copied from the builder stage.

---

## Question 6

**What happens to build speed and image size if you forget the `.dockerignore`? Which of the excluded folders would actually break the build if they were sent to the Docker daemon?**

Without `.dockerignore`, Docker may send unnecessary files and folders such as `.venv`, `data`, `mlruns`, `.git`, and `__pycache__` to the Docker build context.

This makes the build context larger and can make Docker builds slower.

The local `.venv` folder is especially problematic because it contains dependencies created for the host operating system, while the Docker container is running Linux. Large folders such as `data` and `mlruns` are also unnecessary and would increase the amount of data sent during the build.

---

## Question 7

**Why can't the container simply use `127.0.0.1:5000` to reach the MLflow server on your host? What does `host.docker.internal` resolve to?**

Inside a Docker container, `127.0.0.1` refers to the container itself, not my Windows host machine.

Because my MLflow server is running on my host machine, the container cannot use `127.0.0.1:5000` to access it.

On Windows with Docker Desktop, `host.docker.internal` allows the container to reach the host machine.

Therefore, I used:

`MLFLOW_TRACKING_URI=http://host.docker.internal:5000`

to allow my Docker container to connect to the MLflow server running on my computer.

---

## Question 8

**Stop the container and start a new one from the same image. Does the model still load correctly without you rebuilding? What does that tell you about what's baked into the image versus fetched at runtime?**

Yes. I stopped the container and started another container using the same `food11-api:latest` image.

The model loaded correctly and the API worked without rebuilding the Docker image.

The application code and its dependencies are included in the Docker image, while the registered model is retrieved through MLflow at runtime.

This means the same Docker image can be restarted and load `food11@champion` as long as the container can access the MLflow server and the model artifacts.

I tested the `/predict` endpoint from the Docker container and received:

`{"category":"Bread","confidence":0.38816604018211365}`

The `/health` endpoint also returned:

`{"status":"ok"}`

---

## Question 9

**The Dockerfile and image are versioned differently — one lives in Git, the other doesn't yet. What's still missing before another machine, such as a CI runner or Kubernetes cluster, could reliably pull and run the exact image you just built?**

The Dockerfile is stored and versioned in Git, but the built Docker image currently exists only on my local machine.

Before another machine can reliably pull and run the image, I need to tag the Docker image and push it to a container registry such as Docker Hub or another container registry.

A CI runner or Kubernetes cluster could then pull the image from that registry.

For better reproducibility, I should use a specific immutable image tag or image digest instead of relying only on the `latest` tag.