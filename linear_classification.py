import argparse
import os
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision import transforms, datasets, models
from models import VanillaAE, ResNetAE, LinearClassifier
import torch
from utils import train_model, inference
import random
from torch import nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from simclr import SimCLR


def main(args: argparse.Namespace) -> None:
    # Number of classes in cifar10
    num_classes = 10

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Directories where results and models will be saved
    os.makedirs(args.log_dir, exist_ok=True)

    # Datasets
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_set = datasets.CIFAR10(
        root=args.data_dir, transform=transform, download=True
    )
    test_set = datasets.CIFAR10(
        root=args.data_dir, train=False, transform=transform
    )

    # Dataloaders for feature inference
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size,
        num_workers=args.workers,
    )
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size,
        num_workers=args.workers,
    )

    # Init model
    model_ft, num_features = initialize_model(
            args.model_name, args.model, args.projection_dim
    )
    model_ft = model_ft.to(device)
    model_ft.eval()

    # Compute features of test set
    x_test, y_test = inference(
        test_loader, model_ft, device,
        is_denoising=(args.model_name in ["resnet_dae", "vanilla_dae"]),
    )

    # Compute features of complete train set
    x_train, y_train = inference(
        train_loader, model_ft, device,
        is_denoising=(args.model_name in ["resnet_dae", "vanilla_dae"]),
    )
    train_set = TensorDataset(
        torch.from_numpy(x_train), torch.from_numpy(y_train)
    )

    # Dataloaders for computed features
    feature_loaders = {
        "test": DataLoader(
            TensorDataset(
                torch.from_numpy(x_test), torch.from_numpy(y_test)
            ),
            batch_size=args.batch_size,
            shuffle=False,
        )
    }

    for label_perc in [0.01, 0.1, 0.5, 1]:
        # Set seed for reproducibility
        torch.manual_seed(args.seed)
        random.seed(args.seed)

        # Set loggers
        log_dir = os.path.join(
            args.log_dir,
            f"eval_{args.model_name}_{label_perc}_{args.lr}_{args.batch_size}",
        )
        loggers = {
            x: SummaryWriter(os.path.join(log_dir, x))
            for x in ["train", "test"]
        }

        # Init classifier, optimizer and criterion
        model = LinearClassifier(num_features, num_classes)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss()

        # Take subset of train set
        cifar10_subset_train = class_balanced_subset(train_set, label_perc)

        # Loader for train features
        feature_loaders["train"] = DataLoader(
            cifar10_subset_train,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
        )

        print(f"Training on {label_perc * 100}% of train set")

        # Train and evaluate
        train_model(
            model,
            feature_loaders,
            loggers,
            criterion,
            optimizer,
            args.num_epochs,
            device,
        )


def initialize_model(model_name, model_path, projection_dim):
    model_ft = None
    num_features = 768

    if model_name == "simclr":
        # Initialize ResNet
        encoder = models.resnet18(pretrained=False)
        n_features = encoder.fc.in_features  # get dimensions of fc layer

        # Initialize model
        model_ft = SimCLR(encoder, projection_dim, n_features)
        num_features = n_features
    elif model_name in ["resnet_ae", "resnet_dae"]:
        model_ft = ResNetAE()

    elif model_name in ["vanilla_ae", "vanilla_dae"]:
        model_ft = VanillaAE()

    model_ft.load_state_dict(torch.load(model_path))
    return model_ft, num_features


def class_balanced_subset(dataset: TensorDataset, size: float) -> Subset:
    """Class-balanced subset of a dataset.

    Args:
        dataset (TensorDataset): The whole dataset as a TensorDataset instance.
        size (float): The size of the subset as a proportion of the dataset.
            Must be a float between 0.0 and 1.0.
    """
    targets = dataset.tensors[1]
    n_classes = max(targets) + 1

    # Required samples per class
    n_samples = int(size * len(dataset) / n_classes)

    indices = []
    for k in range(n_classes):
        indices.extend(
            random.sample(
                list(np.where(np.asarray(targets) == k)[0]),
                n_samples,
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
        "model",
        help="Path to model that will be evaluated",
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
            "--projection_dim",
            help="in case of simclr, projection dim is needed",
            default=64,
            type=int,
    )
    parser.add_argument(
        "-s",
        "--seed",
        help="Seed for random number generator",
        default=40,
        type=int,
    )
    parser.add_argument(
        "--workers",
        help="Workers used for pytorch DataLoaders",
        default=4,
        type=int,
    )
    parser.add_argument(
        "--model_name",
        help="Method used for representation learning",
        choices=[
            "resnet_ae",
            "resnet_dae",
            "vanilla_ae",
            "vanilla_dae",
            "simclr",
        ],
        default="autoencoder",
        type=str,
    )
    arguments = parser.parse_args()

    main(arguments)
