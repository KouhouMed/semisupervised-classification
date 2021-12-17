import numpy as np
from torch.utils.data import Dataset, Subset
import random
import time
import copy
import torch


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


@torch.no_grad()
def inference(loader, model, device, is_denoising=False):
    """inference function to map all input data $X$ to their latent
    representations $h$ that are used in linear evaluation.
    """
    feature_vector = []
    labels_vector = []
    for step, (x, y) in enumerate(loader):

        if is_denoising:
            x = x + 0.1 * torch.randn(x.shape)
            x = torch.clip(x, min=0, max=1)

        x = x.to(device)

        # get encoding
        h = model.encoder(x).reshape(x.size(0), -1)

        feature_vector.extend(h.cpu().detach().numpy())
        labels_vector.extend(y.numpy())

        if step % 20 == 0:
            print(f"Step [{step}/{len(loader)}]\t Computing features...")

    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    print("Features shape {}".format(feature_vector.shape))
    return feature_vector, labels_vector


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
