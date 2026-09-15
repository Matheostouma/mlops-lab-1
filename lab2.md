# Lab 2 - Answers

**Question 1:** Look at pyproject.toml and uv.lock. What changed?

`pyproject.toml` gained four new dependencies (`mlflow`, `scikit-learn`, `torch`, `torchvision`), plus a `[[tool.uv.index]]` entry pointing to `https://download.pytorch.org/whl/cpu` and a `[tool.uv.sources]` block that pins `torch`/`torchvision` to that index instead of the default PyPI index. `uv.lock` gained the resolved dependency graph for all of these packages (116 packages total), and `torch`/`torchvision` are recorded with a `+cpu` version suffix and `source = { registry = "https://download.pytorch.org/whl/cpu" }`, confirming they resolve to the CPU-only wheels rather than the multi-GB CUDA build.

**Question 2:** What is `--backend-store-uri` used for? What is `--default-artifact-root` used for? What is the difference between the metadata mlflow stores and the artifacts it stores?

`--backend-store-uri` tells mlflow where to store run *metadata*: experiment/run records, params, metrics, tags, and run status. Here it's a SQLite database (`sqlite:///mlflow.db`). `--default-artifact-root` tells mlflow where to store run *artifacts*: arbitrary files produced by a run, such as the logged model, plots, or other outputs. Here that's the local `./mlruns` folder.

The difference: metadata is small, structured, queryable data (numbers and strings) that mlflow indexes so it can render charts, tables, and search/filter runs. Artifacts are opaque files/blobs (potentially large, like a serialized model) that mlflow just stores and serves by reference — the metadata store keeps a path/URI pointing to where the artifact lives.

**Question 3:** Why shouldn't `mlflow.db` and `mlruns/` be tracked by git, and why shouldn't they be tracked by dvc either?

They're generated local outputs, not source of truth — everything in them is reproducible by re-running the training script. Tracking them with git would mean committing a binary SQLite database and model binaries that change on every single run, producing constant, noisy, unreviewable diffs and bloating repo history. DVC is meant for versioning meaningful, reusable data artifacts (e.g. datasets that are inputs to a pipeline and need to be reproducible/shareable) — mlflow run outputs are experiment history, not pipeline input data, and mlflow's own tracking server (backed by a real database/artifact store in a team setting) is the right system of record for that, not a DVC-tracked folder.

**Question 4:** What happens the first time you call `set_experiment` with a name that doesn't exist yet? Check the mlflow UI.

mlflow automatically creates a new experiment with that name (assigning it a new experiment ID and a default artifact location under `mlruns/`), and the run gets logged into it. In the UI, a new "food11" experiment appears in the experiments list alongside "Default", and all subsequent runs land there.

**Question 5:** What is the difference between `mlflow.log_param` and `mlflow.log_metric`? Why does `log_metric` take a `step` argument and `log_param` doesn't?

`log_param` logs a fixed configuration value set once before training and never changed during the run (e.g. learning rate, batch size). `log_metric` logs a value produced during/after training that can evolve over the course of the run (e.g. loss, accuracy per epoch). `log_metric` takes a `step` argument because the same metric name can be logged multiple times over the run, and `step` (e.g. the epoch number) lets mlflow order those values and plot them as a time series/line chart. `log_param` has no `step` because a param is set once and doesn't have a notion of "evolving over time" — logging it twice with different values would be an error.

**Question 6:** Open the run in the mlflow UI. Find the params, the metric charts, and the logged model artifact. Where does the model artifact actually live on disk?

In the UI: the "Parameters" section on the run page shows `dataset`, `epochs`, `lr`, `batch_size`; the "Metrics" section shows `train_loss`, `val_loss`, `val_accuracy` as line charts (one point per epoch/step) plus a single `test_accuracy` value; the "Artifacts" tab shows a `model/` folder containing the logged PyTorch model (weights, conda/pip environment files, `MLmodel` metadata). On disk, since `--default-artifact-root ./mlruns` was set, the artifact files live under `mlruns/<experiment_id>/models/<model_id>/artifacts/` (mlflow's newer logged-models layout) inside the repo root — e.g. `mlruns/1/models/m-<hash>/artifacts/`.

**Question 7:** In the mlflow UI, open the `food11` experiment. Select these runs and click "Compare". Which learning rate gave the best `val_accuracy`? Is higher always better?

| lr | batch_size | best val_accuracy | test_accuracy |
|------|------------|--------------------|----------------|
| 0.01 | 32 | 0.1560 | 0.1679 |
| 0.001 | 32 | 0.5620 | 0.5429 |
| 0.0001 | 32 | 0.7217 | 0.7464 |
| 0.001 | 64 | 0.6223 | 0.6460 |

The lowest learning rate tested, `lr=0.0001`, gave the best `val_accuracy` (0.72). Higher is not better here — `lr=0.01` was far too large and the model barely learned (train_loss stayed near 2.3-3.4, val_accuracy stuck around 0.10-0.16), while `lr=0.001` did reasonably but noticeably worse than `lr=0.0001`. This makes sense for fine-tuning a pretrained model: large learning rates destroy the pretrained weights instead of gently adapting them.

**Question 8:** Use the parallel coordinates plot on the compare page to look at `lr`, `batch_size` and `val_accuracy` together. What pattern do you see?

`val_accuracy` decreases sharply as `lr` increases — the `lr=0.01` run is a clear outlier at the bottom, `lr=0.001` runs are in the middle, and `lr=0.0001` is at the top. Learning rate dominates the outcome; `batch_size` has a much smaller effect by comparison (going from batch_size 32 to 64 at the same `lr=0.001` moved val_accuracy from 0.56 to 0.62, a modest change next to the ~0.6 swing caused by learning rate alone).

**Question 9:** Sort the runs table by `val_accuracy` descending. Which run is the best one? Note its run ID, you'll need it in the next lab.

Best run: `lr=0.0001, batch_size=32, epochs=5`, val_accuracy=0.7217, test_accuracy=0.7464.
Run ID: `aa8f4f38176841329b89f4949f1e0ecf` (run name "gentle-sow-15").
