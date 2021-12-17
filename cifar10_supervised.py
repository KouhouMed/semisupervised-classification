import os
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
from torch import nn
import torch
import argparse
import copy
import time


def main(args: argparse.Namespace):
    # classes in cifar10
    num_classes = 10

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Directories where results and models will be saved
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)

    # Data augmentation
    input_size = 32
    s = 1
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        ),
    }

    # Create datasets
    cifar10_datasets = {
        x: datasets.CIFAR10(
            root=args.data_dir,
            train=(x == "train"),
            download=True,
            transform=data_transforms[x],
        )
        for x in ["train", "test"]
    }

    dataloaders = {
        "test": DataLoader(
            cifar10_datasets["test"],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
        )
    }

    for label_perc in [0.01, 0.1, 0.5, 1]:
        # Set seed for reproducibility
        torch.manual_seed(args.seed)
        random.seed(args.seed)

        # Set loggers
        log_dir = os.path.join(
            args.log_dir,
            f"supervised_{label_perc}_{args.lr}_{args.batch_size}",
        )
        loggers = {
            x: SummaryWriter(os.path.join(log_dir, x))
            for x in ["train", "test"]
        }

        # Init model, optimizer and criterion
        model = models.resnet18(pretrained=False)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss()

        # Take subset of train set
        cifar10_subset_train = class_balanced_subset(
            cifar10_datasets["train"], label_perc
        )

        # Create Dataloaders
        dataloaders["train"] = DataLoader(
            cifar10_subset_train,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
        )

        print(f"Training on {label_perc * 100}% of train set")

        # Train and evaluate
        model = train_model(
            model,
            dataloaders,
            loggers,
            criterion,
            optimizer,
            args.num_epochs,
            device,
        )

        # Save the fine-tuned model
        model_path = os.path.join(
            args.save_dir,
            f"supervised_{label_perc}_{args.lr}_"
            f"{args.batch_size}_{args.num_epochs}.pt",
        )
        torch.save(model.state_dict(), model_path)


def train_model(
    model,
    dataloaders,
    loggers,
    criterion,
    optimizer,
    num_epochs,
    device,
):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "test"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)

                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = float(running_corrects) / len(
                dataloaders[phase].dataset
            )

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
            loggers[phase].add_scalar("Loss", epoch_loss, epoch)
            loggers[phase].add_scalar("Accuracy", epoch_acc, epoch)

            # deep copy the model
            if phase == "test" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(
        f"Training complete in {time_elapsed // 60:.0f}m"
        f" {time_elapsed % 60:.0f}s"
    )

    print(f"Best test Accuracy: {best_acc:4f}")

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def class_balanced_subset(dataset, size: float) -> Dataset:
    """Class-balanced subset of a dataset.

    Args:
        dataset: The whole dataset. Must have attributes ``data``
            and ``targets``.
        size (float): The size of the subset as a proportion of the dataset.
            Must be a float between 0.0 and 1.0.
    """
    n_classes = max(dataset.targets) + 1

    # Required samples per class
    n_samples = int(size * len(dataset) / n_classes)

    indices = []
    for k in range(n_classes):
        indices.extend(
            random.sample(
                list(np.where(np.asarray(dataset.targets) == k)[0]), n_samples
            )
        )
    return Subset(dataset, indices)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        help="Root directory where the dataset is stored",
        default="data",
        type=str,
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        help="Number of samples in mini-batch",
        default=32,
        type=int,
    )
    parser.add_argument(
        "--num_epochs",
        help="Number of training epochs for classifier",
        default=100,
        type=int,
    )
    parser.add_argument(
        "-lr",
        help="Learning rate used for training",
        default=1e-3,
        type=float,
    )
    parser.add_argument(
        "--log_dir",
        help="Directory where results are saved",
        default="runs",
        type=str,
    )
    parser.add_argument(
        "-s",
        "--seed",
        help="Seed for random number generator",
        default=40,
        type=int,
    )
    parser.add_argument(
        "--save_dir",
        help="Directory where trained models will be saved",
        default="models",
        type=str,
    )
    parser.add_argument(
        "--workers",
        help="Workers used for pytorch DataLoaders",
        default=4,
        type=int,
    )

    arguments = parser.parse_args()

    main(arguments)
