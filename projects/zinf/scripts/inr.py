#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from matplotlib import pyplot as plt
from torch import nn
from torchvision.transforms import Resize, Compose, ToTensor, Normalize

import mon

torch.manual_seed(0)

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]
data_dir     = current_dir
if data_dir.has_subdir("data"):
    data_dir = data_dir / "data"


# ----- Inputs -----
filename   = "a0952-kme_172_image_i_lr"
# filename   = "a0952-kme_172_depth_lr"
# filename   = "a0952-kme_172_edge_lr"
image_file = data_dir    / "inr" / f"{filename}.jpg"
output_dir = current_dir / f"run/inr/{filename}"
output_dir.mkdir(parents=True, exist_ok=True)

device     = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


# ----- Constants -----
window_size   = 1
down_size     = 256
lr    	      = 1e-5
losses        = {}
psnrs         = {}
outputs       = {}
refs          = {}
total_steps   = 500
summary_steps = 10
save_images   = True
font_size     = 10
line_width    = 2.0
fig_size      = (5, 4.5)
matplotlib.rc("font", **{
    # "family" : "normal",
    "size"   : font_size
})
plt.rcParams["figure.figsize"]    = [5, 4.5]
plt.rcParams["figure.autolayout"] = True


# ----- Utils -----
def get_image_tensor(path=image_file):
    image     = Image.open(path).convert("L")
    transform = Compose([
        Resize(down_size),
        ToTensor(),
        Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
    ])
    image     = transform(image)
    return image


def get_mgrid(dim=2):
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=down_size)])
    mgrid   = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid   = mgrid.reshape(-1, dim)
    return mgrid


def ff_embedding(image: torch.Tensor, B: torch.Tensor = None) -> torch.Tensor:
    if B is None:
        return image
    else:
        x_proj    = (2. * np.pi * image) @ B.T
        embedding = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], axis=-1)
        return embedding


def mse(image1, image2):
    image_array1 = np.array(image1)
    image_array2 = np.array(image2)
    # Calculate the squared difference
    squared_difference = (image_array1 - image_array2) ** 2
    # Calculate the mean squared difference
    return np.mean(squared_difference)


def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)


def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


# ----- Layer -----
class SigmoidLayer(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, *args, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.linear      = nn.Linear(in_channels, out_channels)
        self.act         = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.linear(x))


class TanhLayer(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, *args, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.linear      = nn.Linear(in_channels, out_channels)
        self.act         = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.linear(x))


class ReLULayer(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, *args, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.linear      = nn.Linear(in_channels, out_channels)
        self.act         = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.linear(x))


class SineLayer(nn.Module):

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        omega_0     : float = 30,
        is_first    : bool  = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.omega_0     = omega_0
        self.linear      = nn.Linear(in_channels, out_channels)
        self.is_first    = is_first
        self.init_weights()

    def init_weights(self):
        b = 1.0 / self.in_channels if self.is_first else np.sqrt(6.0 / self.in_channels) / self.omega_0
        with torch.no_grad():
            self.linear.weight.uniform_(-b, b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


class GaussLayer(nn.Module):

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        scale       : float = 10.0,
        *args, **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.scale       = scale
        self.linear      = nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(-(self.scale * self.linear(x)) ** 2)


class FINERLayer(nn.Module):

    def __init__(
        self,
        in_channels     : int,
        out_channels    : int,
        omega_0         : float = 30.0,
        first_bias_scale: float = 20.0,
        is_first        : bool  = False,
        scale_req_grad  : bool  = False,
    ):
        super().__init__()
        self.omega_0     = omega_0
        self.is_first    = is_first
        self.in_channels = in_channels
        self.linear      = nn.Linear(in_channels, out_channels)

        self.init_weights()
        self.scale_req_grad   = scale_req_grad
        self.first_bias_scale = first_bias_scale
        if self.first_bias_scale is not None:
            self.init_first_bias()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_channels, 1 / self.in_channels)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_channels) / self.omega_0,
                                             np.sqrt(6 / self.in_channels) / self.omega_0)

    def init_first_bias(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.bias.uniform_(-self.first_bias_scale, self.first_bias_scale)

    def generate_scale(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale_req_grad:
            scale = torch.abs(x) + 1
        else:
            with torch.no_grad():
                scale = torch.abs(x) + 1
        return scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        linear = self.linear(x)
        scale  = self.generate_scale(linear)
        return torch.sin(self.omega_0 * scale * linear)


class FINERLayer20(nn.Module):

    def __init__(
        self,
        in_channels     : int,
        out_channels    : int,
        omega_0         : float = 30.0,
        first_bias_scale: float = 20.0,
        is_first        : bool  = False,
        scale_req_grad  : bool  = False,
    ):
        super().__init__()
        self.omega_0     = omega_0
        self.is_first    = is_first
        self.in_channels = in_channels
        self.linear      = nn.Linear(in_channels, out_channels)

        self.init_weights()
        self.scale_req_grad   = scale_req_grad
        self.first_bias_scale = first_bias_scale
        if self.first_bias_scale is not None:
            self.init_first_bias()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_channels, 1 / self.in_channels)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_channels) / self.omega_0,
                                            np.sqrt(6 / self.in_channels) / self.omega_0)

    def init_first_bias(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.bias.uniform_(-self.first_bias_scale, self.first_bias_scale)

    def generate_scale(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale_req_grad:
            scale = torch.abs(x) + 1
        else:
            with torch.no_grad():
                scale = torch.abs(x) + 1
        return scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        linear = self.linear(x)
        scale  = self.generate_scale(linear)
        return torch.sin(self.omega_0 * scale * linear)


class INF(nn.Module):

    def __init__(
        self,
        hidden_features: int,
        hidden_layers  : int,
        out_features   : int,
        act_layer      : Any   = SineLayer,
        use_ff         : bool  = False,
        ff_scale       : float = 10.0,
    ):
        super().__init__()
        if use_ff:
            self.register_buffer("B", torch.randn((hidden_features, 2)) * ff_scale)
            in_features = hidden_features * 2
        else:
            self.B 	    = None
            in_features = 2

        net = [act_layer(in_features, hidden_features, is_first=True)]
        for _ in range(hidden_layers):
            net.append(act_layer(hidden_features, hidden_features))
        net.append(act_layer(hidden_features, out_features, is_first=False))

        self.net = nn.Sequential(*net)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        coords  = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input
        coords  = ff_embedding(coords, self.B)
        return self.net(coords)


# ----- Training -----
def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor, max_pixel_value: float = 1.0) -> float:
    """Calculate the Peak Signal-to-Noise Ratio (PSNR) between two images."""
    assert img1.shape == img2.shape, f"Input images must have the same dimensions: {img1.shape} vs {img2.shape}"
    mse  = F.mse_loss(img1, img2, reduction='mean').item()
    if mse == 0:
        return float("inf")  # Identical images
    psnr = 20 * torch.log10(torch.tensor(max_pixel_value)) - 10 * torch.log10(torch.tensor(mse))
    return psnr.item()


def train(model, total_steps, steps_til_summary):
    model.to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=3e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Input
    global image
    image   = get_image_tensor()
    pixels  = image.permute(1, 2, 0).view(-1, 1)
    coords  = get_mgrid(2)

    image   = image.to(device)
    pixels  = pixels.to(device)
    coords  = coords.to(device)

    #
    losses  = []
    psnrs   = []
    outputs = []
    refs    = []
    for step in range(total_steps):
        model.train()
        optimizer.zero_grad()
        output = model(coords)
        loss   = ((output - pixels) ** 2).mean()
        output = output.view(1, down_size, down_size).to(device)
        psnr   = calculate_psnr(output, image)
        losses.append(loss.item())
        psnrs.append(psnr)
        if not step % steps_til_summary or (step == total_steps - 1):
            print("Step %d, Loss %0.6f, PSNR %0.6f" % (step, loss, psnr))
        outputs.append(output.cpu().detach().squeeze(0).numpy())
        refs.append(image.cpu().view(1, down_size, down_size).detach().numpy())
        loss.backward()
        optimizer.step()

    return losses, psnrs, outputs


# ----- Model -----
#relu     = INF(256, 4, 1, ReLULayer)
#ff       = INF(256, 4, 1, ReLULayer, use_ff=True)

siren    = INF(256, 4, 1, SineLayer)
ff_siren = INF(256, 4, 1, SineLayer, use_ff=True)

finer    = INF(256, 4, 1, FINERLayer)
ff_finer = INF(256, 4, 1, FINERLayer, use_ff=True)


# ----- Run -----
#losses["relu"],     psnrs["relu"],     outputs["relu"],     = train(relu,     total_steps, summary_steps)
#losses["ff"],       psnrs["ff"],       outputs["ff"],       = train(ff,       total_steps, summary_steps)
losses["siren"],    psnrs["siren"],    outputs["siren"],    = train(siren,    total_steps, summary_steps)
losses["ff_siren"], psnrs["ff_siren"], outputs["ff_siren"], = train(ff_siren, total_steps, summary_steps)
losses["finer"],    psnrs["finer"],    outputs["finer"],    = train(finer,    total_steps, summary_steps)
losses["ff_finer"], psnrs["ff_finer"], outputs["ff_finer"], = train(ff_finer, total_steps, summary_steps)


# ----- Visualize -----
print("Final Loss:")
# plt.figure(figsize=fig_size)
plt.subplots(figsize=fig_size)
for n in losses:
    plt.plot(losses[n], label=n, linewidth=line_width)
    plt.legend(prop={"size": font_size})
    # print(f"{n}: {losses[n][-1]}")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.ylim(0.0, 0.2)
plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
plt.legend(bbox_to_anchor=(1, 1), loc=1, borderaxespad=0)
plt.savefig(f"{output_dir}/loss_{total_steps}.jpg", dpi=100, bbox_inches="tight")

plt.clf()
# plt.figure(figsize=fig_size)
plt.subplots(figsize=fig_size)
for n in losses:
  plt.plot(losses[n], label=n, linewidth=line_width)
  plt.legend(prop={"size": font_size})
  # print(f"{n}: {losses[n][-1]}")
plt.xlabel("Steps")
plt.ylabel("Loss")
# plt.ylim(0.0, 0.013)
# plt.ylim(0.0, 0.5)
plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
plt.legend(bbox_to_anchor=(1, 1), loc=1, borderaxespad=0)
# plt.savefig(f"{output_dir}/loss_{total_steps}_small.jpg", dpi=100, bbox_inches="tight")

print("Final PSNR:")
# plt.figure(figsize=fig_size)
plt.subplots(figsize=fig_size)
for n in psnrs:
  plt.plot(psnrs[n], label=n, linewidth=line_width)
  plt.legend(prop={"size": font_size})
  # Print best
  max_psnr  = max(psnrs[n])
  max_index = psnrs[n].index(max_psnr)
  print(f"{max_psnr}")
  # print(f"PSNR {n}: {max_psnr}")
  # print(f"PSNR {n}: {psnrs[n][-1]}")
plt.xlabel("Steps")
plt.ylabel("PSNR")
# plt.ylim(24, 28)
plt.ticklabel_format(axis="y", style="plain", scilimits=(0, 0))
plt.legend(bbox_to_anchor=(1, 1), loc=1, borderaxespad=0)
plt.savefig(f"{output_dir}/psnr_{total_steps}.jpg", dpi=100, bbox_inches="tight")

image = image.squeeze(0).cpu().detach().numpy()
image = np.around((image / 2 + 0.5) * 255).astype(np.uint8)
# plt.imshow(image, cmap="gray")
plt.title("groundtruth")


def draw_figure(image, text, save_path, cmap="gray"):
    font = {
        # 'family': 'serif',
        "color" : "white",
        "weight": "bold",
        "size"  : 28,
    }
    fig, axs       = plt.subplots(1, 1)
    dpi            = 100
    left, width    = 0, 1
    bottom, height = 0, 1
    right          = left   + width
    center_x       = left   + width  / 2
    center_y       = bottom + height / 2
    top            = bottom + height
    p              = plt.Rectangle((left, bottom), width, height, linewidth=0, fill=False, facecolor="none", edgecolor=None)
    p.set_transform(axs.transAxes)
    p.set_clip_on(True)

    axs.add_patch(p)
    axs.imshow(image, cmap=cmap)
    axs.title.set_text("")
    axs.text(right - 0.01, bottom + 0.01, text,
             horizontalalignment = "right",
             verticalalignment   = "bottom",
             transform           = axs.transAxes,
             fontdict            = font)
    axs.set_xticks([])
    axs.set_yticks([])
    axs.set_axis_off()
    # plt.show()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0)


if save_images:
    image_pil      = Image.fromarray(image)
    image_gradient = image_pil.filter(ImageFilter.FIND_EDGES)

    draw_figure(image,          "", f"{output_dir}/image_color.jpg",          cmap="viridis")
    draw_figure(image_gradient, "", f"{output_dir}/image_gradient_color.jpg", cmap="viridis")

    for i, n in enumerate(outputs):
        if n in ["ref"]:
            continue
        n_        = n.lower()
        max_psnr  = max(psnrs[n])
        max_index = psnrs[n].index(max_psnr)
        arr       = outputs[n][max_index]
        data8     = (255 * (np.clip(arr, -1, 1) + 1) / 2).astype(np.uint8)
        image     = Image.fromarray(data8)
        gradient  = image.filter(ImageFilter.FIND_EDGES)

        draw_figure(image, f"PSNR: {psnrs[n][-1]:.2f}", f"{output_dir}/{n_}_gray_{total_steps}.jpg")
        # (image, f"PSNR: {psnrs[n][-1]:.2f}", f"{output_dir}/{n_}_color_{total_steps}.jpg", cmap="viridis")

        # draw_figure(gradient, f"PSNR: {psnrs[n][-1]:.2f}", f"{output_dir}/{n_}_gradient_gray_{total_steps}.jpg")
        draw_figure(gradient, f"PSNR: {psnrs[n][-1]:.2f}", f"{output_dir}/{n_}_gradient_color_{total_steps}.jpg", cmap="viridis")
