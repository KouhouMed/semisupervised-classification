import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torch import nn
import torch
import argparse
from models import VanillaAE, ResNetAE
import copy
import time


def main(args: argparse.Namespace) -> None:
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set seed for reproducibility
    torch.manual_seed(args.seed)

    # Directories where results and models will be saved
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)

    # Set loggers
    log_dir = os.path.join(
        args.log_dir, f"{args.model_name}_{args.lr}_{args.batch_size}"
    )
    loggers = {
        x: SummaryWriter(os.path.join(log_dir, x)) for x in ["train", "test"]
    }

    # Init model, optimizer and criterion
    if args.model_name in ["resnet_dae", "resnet_ae"]:
        model = ResNetAE()
    else:
        model = VanillaAE()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()

    # Data augmentation and normalization for training
    # Normalization for testing
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

    print("Initializing Datasets and Dataloaders...")

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
    # Create Dataloaders
    dataloaders = {
        x: DataLoader(
            cifar10_datasets[x],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
        )
        for x in ["train", "test"]
    }

    print("Training...")

    # Train and evaluate
    model = train_model(
        model,
        dataloaders,
        loggers,
        criterion,
        optimizer,
        args.num_epochs,
        device,
        is_denoising=(args.model_name in ["vanilla_dae", "resnet_dae"]),
    )

    # Save the fine-tuned model
    model_path = os.path.join(
        args.save_dir,
        f"{args.model_name}_{args.lr}_{args.batch_size}_{args.num_epochs}.pt",
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
    is_denoising=False,
):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    min_loss = np.inf

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

            # Iterate over data.
            for inputs, _ in dataloaders[phase]:

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == "train"):
                    # If Denoising Autoencoder
                    if is_denoising:
                        noisy_inputs = inputs + 0.1 * torch.randn(inputs.shape)
                        noisy_inputs = torch.clip(noisy_inputs, min=0, max=1)
                        inputs = inputs.to(device)
                        noisy_inputs = noisy_inputs.to(device)
                        outputs = model(noisy_inputs)
                    else:
                        inputs = inputs.to(device)
                        outputs = model(inputs)

                    loss = criterion(outputs, inputs)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            print(f"{phase} Loss: {epoch_loss:.4f}")
            loggers[phase].add_scalar("Loss", epoch_loss, epoch)

            # deep copy the model
            if phase == "test" and epoch_loss < min_loss:
                min_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(
        f"Training complete in {time_elapsed // 60:.0f}m"
        f" {time_elapsed % 60:.0f}s"
    )

    print(f"Minimum test Loss: {min_loss:4f}")

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


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
        default=20,
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
    parser.add_argument(
        "--model_name",
        help="Method used for representation learning",
        choices=[
            "vanilla_ae",
            "vanilla_dae",
            "resnet_ae",
            "resnet_dae",
        ],
        default="vanilla_ae",
        type=str,
    )
    arguments = parser.parse_args()

    main(arguments)
