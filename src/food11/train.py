"""Train a ResNet18 classifier on Food-11 and track runs with mlflow."""

import argparse
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
NUM_CLASSES = 11

TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["processed", "mini"], default="mini")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def make_loader(root: Path, split: str, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = datasets.ImageFolder(root / split, transform=TRANSFORM)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def build_model() -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: optim.Optimizer | None = None,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            if is_train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def main() -> None:
    args = parse_args()

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("food11")

    dataset_dir = DATA_DIR / (
        "food11_processed" if args.dataset == "processed" else "food11_processed_mini"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = make_loader(dataset_dir, "training", args.batch_size, shuffle=True)
    val_loader = make_loader(dataset_dir, "validation", args.batch_size, shuffle=False)
    test_loader = make_loader(dataset_dir, "evaluation", args.batch_size, shuffle=False)

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    with mlflow.start_run():
        mlflow.log_params(
            {
                "dataset": args.dataset,
                "epochs": args.epochs,
                "lr": args.lr,
                "batch_size": args.batch_size,
            }
        )

        for epoch in range(args.epochs):
            train_loss, _ = run_epoch(model, train_loader, device, criterion, optimizer)
            val_loss, val_accuracy = run_epoch(model, val_loader, device, criterion)

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)

            print(
                f"epoch {epoch}: train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} val_accuracy={val_accuracy:.4f}"
            )

        test_loss, test_accuracy = run_epoch(model, test_loader, device, criterion)
        mlflow.log_metric("test_accuracy", test_accuracy)

        input_example, _ = next(iter(test_loader))
        mlflow.pytorch.log_model(
            model, "model", input_example=input_example[:1].cpu().numpy()
        )

        print(f"test_loss={test_loss:.4f} test_accuracy={test_accuracy:.4f}")


if __name__ == "__main__":
    main()
