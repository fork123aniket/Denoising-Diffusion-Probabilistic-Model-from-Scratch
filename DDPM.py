import os
import copy
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
from torch import optim
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models import UNet, EMA
import logging

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, type="unconditional", device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device
        self.type = type

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def unconditional_sample(self, model, n, labels, cfg_scale=3):
        if self.type != "unconditional" and labels is None:
            raise ValueError('Labels must be passed to perform conditional sampling.')
        if self.type != "unconditional" and cfg_scale <= 0:
            raise ValueError('For conditional sampling, make sure the classifier-free guidance scale must be '
                             'greater than 0.')
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                if self.type == "unconditional":
                    predicted_noise = model(x, t)
                else:
                    predicted_noise = model(x, t, labels)
                    if cfg_scale > 0:
                        uncond_predicted_noise = model(x, t, None)
                        predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x


def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


def get_data(args):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(80),  # args.image_size + 1/4 *args.image_size
        torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader


def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)


def train(args):
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)
    model = UNet().to(device) if args.sampling_type == 'unconditional' else UNet(num_classes=args.num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, type=args.sampling_type, device=device)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)
    if args.sampling_type != "unconditional":
        ema = EMA(0.995)
        ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            if args.sampling_type == 'unconditional':
                predicted_noise = model(x_t, t)
            else:
                labels = labels.to(device)
                if np.random.random() < 0.1: labels = None
                predicted_noise = model(x_t, t, labels)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if args.sampling_type != 'unconditional':
                ema.step_ema(ema_model, model)

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        if epoch % 10 == 0:
            if args.sampling_type != "unconditional":
                labels = torch.arange(10).long().to(device)
                sampled_images = diffusion.sample(model, n=len(labels), labels=labels)
                ema_sampled_images = diffusion.sample(ema_model, n=len(labels), labels=labels)
                plot_images(sampled_images)
                save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
                save_images(ema_sampled_images, os.path.join("results", args.run_name, f"{epoch}_ema.jpg"))
                torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))
                torch.save(ema_model.state_dict(), os.path.join("models", args.run_name, f"ema_ckpt.pt"))
                torch.save(optimizer.state_dict(), os.path.join("models", args.run_name, f"optim.pt"))
            else:
                sampled_images = diffusion.sample(model, n=images.shape[0])
                save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
                torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))


def launch(sampling_type='unconditional'):
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.sampling_type = sampling_type
    args.run_name = "DDPM_"+args.sampling_type
    args.epochs = 500
    args.batch_size = 12
    args.image_size = 64
    args.num_classes = 10
    args.dataset_path = r"<path to Landscape dataset>"
    args.device = "cuda"
    args.lr = 3e-4
    train(args)


def generate_images(sampling_type='unconditional'):
    device = "cuda"
    if sampling_type != 'unconditional':
        model = UNet().to(device)
        model.load_state_dict(torch.load("<path to model checkpoint file>"))
        diffusion = Diffusion(img_size=64, device=device)
        x = diffusion.sample(model, 8, torch.Tensor([6] * 8).long().to(device), cfg_scale=0)
    else:
        model = UNet(num_classes=10).to(device)
        model.load_state_dict(torch.load("<path to model checkpoint file>"))
        diffusion = Diffusion(img_size=64, type=sampling_type, device=device)
        x = diffusion.sample(model, 8)
    plot_images(x)


if __name__ == '__main__':
    launch()
    generate_images()
    launch('conditional')
    generate_images('conditional')
