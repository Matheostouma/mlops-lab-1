# Lab 5 — CI/CD Questions and Answers

## Question 1 — Why shouldn't unit tests load the real MLflow model?

Unit tests should be fast and independent of external services. If we load the real model, a test could fail because the MLflow server is offline, the network is unavailable, or the model cannot be downloaded. These failures do not necessarily mean our code is wrong. Using a fake model allows us to test the application without these dependencies.

## Question 2 — What does the fake model test prove?

The test verifies that the API accepts an image, processes the request, and returns a valid JSON response containing the expected category and a confidence value between 0 and 1.

It does not prove that the real model predicts correctly or achieves good accuracy. The fake model always returns a predefined prediction.

## Question 3 — Which check detects duplicates, and why are they dangerous?

The **No cross-split duplicates** check detects images that appear in both training and evaluation datasets.

Duplicates are dangerous because the model might be evaluated on images it has already seen during training. This is called data leakage and can make the accuracy look better than it really is.

A corrupted image usually causes an obvious error, while duplicated images can silently produce misleading evaluation results.

## Question 4 — Where should data validation run?

Data validation can run in two places:

1. **Locally before training**, after downloading or processing the dataset.
2. **In a CI job with access to the dataset**, after retrieving it from a DVC remote.

This ensures the dataset satisfies the required conditions before training or evaluation.

## Question 5 — What happens if the test job fails?

Because the `smoke` job uses `needs: test`, it will not run if the `test` job fails.

Without `needs: test`, both jobs could run independently. The Docker smoke test might run even when the unit tests have failed, wasting resources.

## Question 6 — Compare the first and second CI runs

**Result pending:** We need the durations of two GitHub Actions runs.

The second run may be faster if dependency or Docker build caching is enabled and reused. For example, cached Python dependencies can reduce download time.

We should compare the actual execution times to identify which step improved.

## Question 7 — Why use a commit SHA instead of `latest`?

A commit SHA identifies the exact version of the code used to build a Docker image.

The `latest` tag can change whenever a new image is published.

For example, Computer A downloads `latest` on Monday. On Tuesday, a developer publishes a new image using the same tag. Computer B downloads `latest` and receives different code.

Using commit SHA tags makes it easier to reproduce deployments and identify the code version.

## Question 8 — Why specify GitHub Actions permissions?

The workflow explicitly declares permissions to follow the principle of least privilege.

For example, the image publishing job needs `packages: write` to upload Docker images to GitHub Container Registry.

If a compromised workflow step had unrestricted write permissions, it could potentially modify repository contents, alter workflows, or publish malicious packages.

Limiting permissions reduces this risk.

## Question 9 — What happens when a pull request fails CI?

**Practical result pending:** We still need to perform the pull request experiment.

When a test fails, GitHub displays a failed check on the pull request.

If branch protection requires successful checks, the pull request cannot be merged normally until the problem is fixed.

This improves teamwork because every developer must pass the same automated checks instead of relying only on tests performed on individual laptops.

## Question 10 — What did the model promotion gate decide?

**Result pending:** We need to implement `promote.py` and evaluate the candidate models.

The promotion condition is:

`candidate_accuracy >= champion_accuracy + 2 × standard_error`

Where:

`standard_error = sqrt(p × (1 - p) / n)`

Here, `p` is the champion's accuracy and `n` is the number of evaluation examples.

If the candidate passes the threshold, it becomes the new `champion`. Otherwise, it receives the `challenger` alias.

We must record the candidate accuracies, champion accuracy, threshold, and actual decision after running the script.

## Question 11 — How can model promotion run automatically?

Two possible solutions are:

1. **Self-hosted GitHub Actions runner:** Install a runner on the machine that can access the local MLflow server. After training, the runner can execute `promote.py`.
2. **Hosted MLflow server:** Use a service such as DagsHub to host MLflow so a GitHub Actions runner can connect to it and execute the promotion script automatically.

Both approaches allow model promotion to run after training without manually launching the script.

## Question 12 — Draw the CI/CD pipeline

The intended pipeline is:

```text
Developer
   |
   | git push / pull request
   v
GitHub Actions (Automatic)
   |
   v
Ruff + Unit Tests
   |
   | PASS
   v
Docker Build
   |
   v
Docker Smoke Test
   |
   | PASS
   v
Push Image to GHCR (main only)
   |
   | Image tagged with commit SHA
   v
Human selects image for deployment
   |
   v
Docker Container Starts
   |
   | Loads model from MLflow
   v
MLflow Registry
   |
   | Model version selected by
   | champion alias
   v
Food-11 Prediction API
```

**Automatic steps:** Linting, unit tests, Docker build, smoke test, and image publishing once configured.

**Manual steps still required:** Deployment and model selection until automation is implemented.

**Code version:** Determined by the Git commit SHA used as the Docker image tag.

**Model version:** Determined by the `champion` alias in the MLflow Model Registry.
