import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim: int = 100, num_classes: int = 2, image_size: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.image_size = image_size

        self.label_emb = nn.Embedding(num_classes, latent_dim)

        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim * 2, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        label_input = self.label_emb(labels)
        combined = torch.cat([noise, label_input], dim=1)
        combined = combined.unsqueeze(-1).unsqueeze(-1)
        return self.model(combined)


class Discriminator(nn.Module):
    def __init__(self, num_classes: int = 2, image_size: int = 64):
        super().__init__()
        self.num_classes = num_classes
        self.image_size = image_size

        self.label_emb = nn.Embedding(num_classes, image_size * image_size)

        self.model = nn.Sequential(
            nn.Conv2d(4, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        batch_size = images.size(0)
        label_map = self.label_emb(labels)
        label_map = label_map.view(batch_size, 1, self.image_size, self.image_size)
        combined = torch.cat([images, label_map], dim=1)
        return self.model(combined).view(batch_size, -1)


def weights_init(m):
    classname = m.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
