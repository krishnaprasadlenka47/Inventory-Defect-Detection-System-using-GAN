import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from app.ml.gan.cgan import Generator, Discriminator, weights_init
from app.config import settings


def get_dataloader(data_dir: str, image_size: int, batch_size: int) -> DataLoader:
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)


def train(epochs: int, batch_size: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    os.makedirs(settings.checkpoint_dir, exist_ok=True)
    os.makedirs(settings.synthetic_dir, exist_ok=True)

    dataloader = get_dataloader(settings.real_dir, settings.image_size, batch_size)

    G = Generator(settings.latent_dim, settings.num_classes, settings.image_size).to(device)
    D = Discriminator(settings.num_classes, settings.image_size).to(device)

    G.apply(weights_init)
    D.apply(weights_init)

    criterion = nn.BCELoss()
    opt_G = optim.Adam(G.parameters(), lr=settings.learning_rate, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=settings.learning_rate, betas=(0.5, 0.999))

    fixed_noise = torch.randn(16, settings.latent_dim, device=device)
    fixed_labels = torch.tensor([0, 1] * 8, device=device)

    for epoch in range(1, epochs + 1):
        g_loss_total, d_loss_total = 0.0, 0.0

        for real_imgs, labels in dataloader:
            real_imgs = real_imgs.to(device)
            labels = labels.to(device)
            batch = real_imgs.size(0)

            real_target = torch.ones(batch, 1, device=device)
            fake_target = torch.zeros(batch, 1, device=device)

            D.zero_grad()
            d_real = D(real_imgs, labels)
            d_real_loss = criterion(d_real, real_target)

            noise = torch.randn(batch, settings.latent_dim, device=device)
            fake_imgs = G(noise, labels)
            d_fake = D(fake_imgs.detach(), labels)
            d_fake_loss = criterion(d_fake, fake_target)

            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            opt_D.step()

            G.zero_grad()
            g_out = D(fake_imgs, labels)
            g_loss = criterion(g_out, real_target)
            g_loss.backward()
            opt_G.step()

            g_loss_total += g_loss.item()
            d_loss_total += d_loss.item()

        avg_g = g_loss_total / len(dataloader)
        avg_d = d_loss_total / len(dataloader)
        print(f"Epoch [{epoch}/{epochs}]  G_loss: {avg_g:.4f}  D_loss: {avg_d:.4f}")

        if epoch % 10 == 0:
            with torch.no_grad():
                samples = G(fixed_noise, fixed_labels)
            save_image(
                samples,
                os.path.join(settings.synthetic_dir, f"epoch_{epoch:04d}.png"),
                normalize=True,
                nrow=4,
            )

    torch.save(G.state_dict(), os.path.join(settings.checkpoint_dir, "generator.pt"))
    torch.save(D.state_dict(), os.path.join(settings.checkpoint_dir, "discriminator.pt"))
    print("Training complete. Checkpoints saved.")
    return avg_g, avg_d


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=settings.epochs)
    parser.add_argument("--batch_size", type=int, default=settings.batch_size)
    args = parser.parse_args()
    train(args.epochs, args.batch_size)
