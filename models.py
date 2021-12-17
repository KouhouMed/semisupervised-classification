import torch.nn as nn
from torchvision import models


class VanillaAE(nn.Module):
    """Simple Autoencoder for 32 x 32 images."""

    def __init__(self):
        super(VanillaAE, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 3, stride=2, padding=1),
            # [batch, 12, 16, 16]
            nn.ReLU(inplace=True),
            nn.Conv2d(12, 24, 3, stride=2, padding=1),
            # [batch, 24, 8, 8]
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 48, 3, stride=2, padding=1),
            # [batch, 48, 4, 4]
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                48, 24, 3, stride=2, padding=1, output_padding=1
            ),
            # [batch, 24, 8, 8]
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                24, 12, 3, stride=2, padding=1, output_padding=1
            ),
            # [batch, 12, 16, 16]
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                12, 3, 3, stride=2, padding=1, output_padding=1
            ),
            # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class ResNetAE(nn.Module):
    """Autoencoder using ResNet18 as encoder for 32 x 32 images."""

    def __init__(self):
        super(ResNetAE, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.encoder = models.resnet18(num_classes=768)

        self.decoder = nn.Sequential(
            nn.Unflatten(1, (48, 4, 4)),
            Decoder(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class Decoder(nn.Module):
    """A convolutional module to up-sample (48, 4, 4) to (3, 32, 32) with
    Sigmoid output activation.
    """

    def __init__(self):
        super(Decoder, self).__init__()
        # Input size: [batch, 48, 4, 4]
        # Output size: [batch, 3, 32, 32]
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                48, 24, 3, stride=2, padding=1, output_padding=1
            ),
            # [batch, 24, 8, 8]
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                24, 12, 3, stride=2, padding=1, output_padding=1
            ),
            # [batch, 12, 16, 16]
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                12, 3, 3, stride=2, padding=1, output_padding=1
            ),
            # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(x)


class LinearClassifier(nn.Module):
    def __init__(self, n_features, n_classes):
        super(LinearClassifier, self).__init__()

        self.model = nn.Linear(n_features, n_classes)

    def forward(self, x):
        return self.model(x)
