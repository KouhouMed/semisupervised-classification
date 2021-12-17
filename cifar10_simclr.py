from simclr import SimCLR
from simclr.modules import NT_Xent
import torch
from torch.utils.data import DataLoader
import os
from torchvision import transforms, datasets, models
from torch.utils.tensorboard import SummaryWriter
import argparse


def main(args: argparse.Namespace) -> None:
    # base model name for saving/logging
    model_name = f"simclr_{args.first_transform}_{args.second_transform}"

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set seed for reproducibility
    torch.manual_seed(args.seed)

    # Directories where results and models will be saved
    os.makedirs(args.save_dir, exist_ok=True)

    # Set logger
    logger = SummaryWriter(os.path.join(args.log_dir, model_name))

    # Initialize ResNet
    encoder = models.resnet18(pretrained=False)
    n_features = encoder.fc.in_features  # get dimensions of fc layer

    # Initialize model
    model = SimCLR(encoder, args.projection_dim, n_features)
    model = model.to(device)

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Criterion
    criterion = NT_Xent(args.batch_size, args.temperature, world_size=1)

    # Get transform combination
    first_transform = get_transform(args.first_transform)
    second_transform = get_transform(args.second_transform)
    simclr_transform = TransformsSimCLR(first_transform, second_transform)

    # Init dataset & dataloader
    train_dataset = datasets.CIFAR10(
        args.data_dir,
        download=True,
        transform=simclr_transform,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
    )

    # Start training
    args.global_step = 0
    for epoch in range(args.num_epochs):
        loss_epoch = train_simclr(
            args, train_loader, model, criterion, optimizer, logger, device
        )

        # save every 10 epochs
        # if (epoch + 1) % 10 == 0:
        #     model_path = os.path.join(
        #         args.save_dir, f"{model_name}_{epoch}.pt"
        #     )
        #     torch.save(model.state_dict(), model_path)

        logger.add_scalar("Loss/train", loss_epoch / len(train_loader), epoch)
        print(
            f"Epoch [{epoch}/{args.num_epochs}]\t"
            f"Loss: {loss_epoch / len(train_loader)}"
        )

    # End training
    model_path = os.path.join(
        args.save_dir, f"{model_name}_{args.num_epochs}.pt"
    )
    torch.save(model.state_dict(), model_path)


def train_simclr(
        args, train_loader, model, criterion, optimizer, writer, device
):
    loss_epoch = 0
    for step, ((x_i, x_j), _) in enumerate(train_loader):
        optimizer.zero_grad()
        x_i = x_i.to(device)
        x_j = x_j.to(device)

        # positive pair, with encoding
        h_i, h_j, z_i, z_j = model(x_i, x_j)

        loss = criterion(z_i, z_j)
        loss.backward()

        optimizer.step()

        if step % 50 == 0:
            print(f"Step [{step}/{len(train_loader)}]\t Loss: {loss.item()}")

        writer.add_scalar("Loss/train_epoch", loss.item(), args.global_step)
        loss_epoch += loss.item()
        args.global_step += 1
    return loss_epoch


def get_transform(transform_name, size=32):
    """A function to retrieve torchvision transforms for cifar10 by names.
    """
    s = 1
    color_jitter = transforms.ColorJitter(
        0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
    )
    gaussian_blur = transforms.GaussianBlur(kernel_size=3)
    if transform_name == "crop":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(size=size),
                transforms.RandomHorizontalFlip(),
            ]
        )
    elif transform_name == "color":
        return transforms.Compose(
            [
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
            ]
        )
    elif transform_name == "blur":
        return transforms.RandomApply([gaussian_blur], p=0.5)
    elif transform_name == "all":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(size=size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([gaussian_blur], p=0.5),
            ]
        )
    else:
        return None


class TransformsSimCLR:
    """A stochastic data augmentation module that transforms any given data
    example randomly resulting in two correlated views of the same example,
    denoted x ̃i and x ̃j, which we consider as a positive pair.
    """

    def __init__(self, first_transform, second_transform):
        if second_transform:
            self.train_transform = transforms.Compose(
                [
                    first_transform,
                    second_transform,
                    transforms.ToTensor(),
                ]
            )
        else:
            self.train_transform = transforms.Compose(
                [
                    first_transform,
                    transforms.ToTensor(),
                ]
            )

    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        help="Root directory where the dataset is stored",
        default="data",
        type=str,
    )
    parser.add_argument(
        "--first_transform",
        help="First transform to apply to images",
        choices=["crop", "color", "blur"],
        default="all",
        type=str,
    )
    parser.add_argument(
        "--second_transform",
        help="Second transform to compose with first transform",
        choices=["crop", "color", "blur"],
        default="",
        type=str,
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        help="Number of samples in mini-batch",
        default=256,
        type=int,
    )
    parser.add_argument(
        "--num_epochs",
        help="Number of training epochs for classifier",
        default=40,
        type=int,
    )
    parser.add_argument(
        "-lr",
        help="Learning rate used for training",
        default=3e-4,
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
        "--projection_dim",
        help="projection head of simclr",
        default=64,
        type=int,
    )
    parser.add_argument(
        "--temperature",
        help="Temperature for contrastive loss NT_Xent",
        default=0.5,
        type=float,
    )
    arguments = parser.parse_args()

    main(arguments)
