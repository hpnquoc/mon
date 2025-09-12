#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements Zero-Restore model for zero-shot single image restoration.

References:
    - Paper: "Zero-shot Single Image Restoration through Controlled Perturbation
      of Koschmieder's Model," CVPR 2021.
    - Code: https://github.com/aupendu/zero-restore
"""

__all__ = [
    "ZeroRestoreDehaze",
    "ZeroRestoreLLE",
    "ZeroRestoreUE",
]

import abc
import random

import box
import torch

from mon.constants import MODELS
from mon.core import MLType, ModelMixin, nn, Path, Task

current_file = Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Module -----
class DoubleConv(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False, padding_mode="reflect"),
            nn.GroupNorm(num_channels=out_channels, num_groups=8, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False, padding_mode="reflect"),
            nn.GroupNorm(num_channels=out_channels, num_groups=8, affine=True),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.conv(input)


class InDoubleConv(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 9, stride=4, padding=4, bias=False, padding_mode="reflect"),
            nn.GroupNorm(num_channels=out_channels, num_groups=8, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False, padding_mode="reflect"),
            nn.GroupNorm(num_channels=out_channels, num_groups=8, affine=True),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.conv(input)


class InConv(nn.Module):
    
    def __init__(self, in_channels: int = 1, out_channels: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 7, stride=4, padding=3, bias=False, padding_mode="reflect"),
            nn.GroupNorm(num_channels=out_channels, num_groups=8, affine=True),
            nn.ReLU(inplace=True)
        )
        self.convf = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False, padding_mode="reflect"),
            nn.GroupNorm(num_channels=out_channels, num_groups=8, affine=True),
            nn.ReLU(inplace=False)
        )
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x    = input
        R    = x[:, 0:1, :, :]
        G    = x[:, 1:2, :, :]
        B    = x[:, 2:3, :, :]
        xR   = torch.unsqueeze(self.conv(R), 1)
        xG   = torch.unsqueeze(self.conv(G), 1)
        xB   = torch.unsqueeze(self.conv(B), 1)
        x    = torch.cat([xR, xG, xB], 1)
        x, _ = torch.min(x, dim=1)
        return self.convf(x)


class SKConv(nn.Module):
    
    def __init__(self, in_channels: int = 1, out_channels: int = 64, M: int = 4, L: int = 32):
        super().__init__()
        self.M     = M
        self.convs = nn.ModuleList([])
        in_conv    = InConv(in_channels, out_channels)
        for i in range(M):
            if i == 0:
                self.convs.append(in_conv)
            else:
                self.convs.append(
                    nn.Sequential(
                        nn.Upsample(scale_factor=1 / (2 ** i), mode="bilinear", align_corners=True),
                        in_conv,
                        nn.Upsample(scale_factor=2 ** i, mode="bilinear", align_corners=True)
                    )
                )
        self.fc  = nn.Linear(out_channels, L)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(nn.Linear(L, out_channels))
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x    = input
        feas = None
        for i, conv in enumerate(self.convs):
            fea = conv(x)
            fea = torch.unsqueeze(fea, 1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        fea_u = torch.sum(feas, dim=1)
        fea_s = fea_u.mean(-1).mean(-1)
        fea_z = self.fc(fea_s)
        
        attention_vectors = None
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z)
            vector = torch.unsqueeze(vector, 1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = torch.unsqueeze(attention_vectors, -1)
        attention_vectors = torch.unsqueeze(attention_vectors, -1)
        
        fea_v = (feas * attention_vectors).sum(dim=1)
        return fea_v


class Estimation(nn.Module):

    def __init__(self, num_channels: int = 64):
        super().__init__()
        self.num_channels = num_channels
        self.in_conv      = SKConv(1, num_channels, 3, 32)
        # Transmission Map
        self.conv_t1  = DoubleConv(num_channels, num_channels)
        self.conv_t2  = nn.Conv2d(num_channels, 1, 3, padding=1, stride=1, bias=False, padding_mode="reflect")
        self.up       = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        # Atmospheric Light
        self.conv_a1  = InDoubleConv(3, num_channels)
        self.conv_a2  = DoubleConv(num_channels, num_channels)
        self.maxpool  = nn.MaxPool2d(15, 7)
        self.pool     = nn.AdaptiveAvgPool2d(1)
        self.dense    = nn.Linear(num_channels, 3, bias=False)
        
        self.apply(self.init_weights)
        
    def init_weights(self, m: nn.Module):
        classname = m.__class__.__name__
        if classname.find("Conv2d") != -1:  # 0.02
            m.weight.data.normal_(0.0, 0.001)
        if classname.find("Linear") != -1:  # 0.02
            m.weight.data.normal_(0.0, 0.001)
            
    def forward(self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x     = input
        x_min = self.in_conv(x)
        trans = self.conv_t2(self.up(self.conv_t1(x_min)))
        trans = torch.sigmoid(trans) + 1e-12
        atm   = self.conv_a1(x)
        atm   = torch.mul(atm, x_min)
        atm   = self.pool(self.conv_a2(self.maxpool(atm)))
        atm   = atm.view(-1, self.num_channels)
        atm   = torch.sigmoid(self.dense(atm))
        return trans, atm


class EstimationLLIE(nn.Module):

    def __init__(self, num_channels: int = 64):
        super().__init__()
        self.num_channels = num_channels
        self.in_conv      = SKConv(1, num_channels, 3, 32)
        # Transmission Map
        self.conv_t1 = DoubleConv(num_channels, num_channels)
        self.conv_t2 = nn.Conv2d(num_channels, 1, 3, padding=1, stride=1, bias=False, padding_mode="reflect")
        self.up      = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        # Atmospheric Light
        self.conv_a1 = InDoubleConv(3, num_channels)
        self.conv_a2 = DoubleConv(num_channels, num_channels)
        self.conv_a3 = nn.Conv2d(num_channels, 1, 3, padding=1, stride=1, bias=False, padding_mode="reflect")
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x     = input
        x_min = self.in_conv(x)
        trans = self.conv_t2(self.up(self.conv_t1(x_min)))
        trans = torch.sigmoid(trans) + 1e-12
        atm   = self.conv_a1(x)
        atm   = torch.mul(atm, self.up(x_min))
        atm   = self.conv_a3(self.conv_a2(atm))
        atm   = torch.sigmoid(atm)
        return trans, atm
    

# ----- Models -----
class ZeroRestore(nn.Module, abc.ABC):
    """Zero-Restore model for image dehazing.
    
    References:
        - Paper: "Zero-shot Single Image Restoration through Controlled Perturbation
          of Koschmieder's Model," CVPR 2021.
        - Code: https://github.com/aupendu/zero-restore
    """
    
    def forward_once(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        trans, atm = self.model(image)
        atm        = torch.unsqueeze(torch.unsqueeze(atm, 2), 2)
        atm        = atm.expand_as(image)
        trans      = trans.expand_as(image)
        enhanced   = (image - (1 - trans.clone()) * atm) / trans
        return trans, atm, enhanced
        
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # Optimize
        self.model.load_state_dict(self.state_dict)
        self.model.train()
        optimizer = nn.Adam(self.parameters(), lr=1e-3, weight_decay=1e-2)
        
        for i in range(self.iters):
            optimizer.zero_grad()
            image_  = self.augment(image)
            
            # Forward 1
            trans1, atm1, enhanced1 = self.forward_once(image_)
            
            # Forward 2
            p_x     = 0.9
            image_x = image * p_x + (1 - p_x) * atm1
            trans_x, atm_x, enhanced_x = self.forward_once(image_x)
            
            # Loss
            o_tensor = torch.ones(enhanced1.shape).to(self.device)
            z_tensor = torch.zeros(enhanced1.shape).to(self.device)
            loss_t   = torch.sum((trans_x - p_x * trans1) ** 2)
            loss_a   = torch.sum((atm1 - atm_x) ** 2)
            loss_mx  =   torch.sum(torch.max(enhanced1, o_tensor)) + torch.sum(torch.max(enhanced_x, o_tensor)) - 2 * torch.sum(o_tensor)
            loss_mn  = - torch.sum(torch.min(enhanced1, z_tensor)) - torch.sum(torch.min(enhanced_x, z_tensor))
            loss_col = nn.ColorConstancyLoss()(enhanced1)
            loss_tv  = nn.TotalVariationLoss()(enhanced1)
            loss     = 0.001 * loss_tv + loss_t + loss_a + 0.001 * loss_mx + 0.001 * loss_mn + 1000 * loss_col
            
            loss.backward()
            optimizer.step()
        
        self.model.eval()
        trans, atm, enhanced = self.forward_once(image)
        
        return trans, atm, enhanced
        
    def augment(self, image: torch.Tensor) -> torch.Tensor:
        it = random.randint(0, 7)
        if it == 1:
            image = image.rot90(1, [2, 3])
        if it == 2:
            image = image.rot90(2, [2, 3])
        if it == 3:
            image = image.rot90(3, [2, 3])
        if it == 4:
            image = image.flip(2).rot90(1, [2, 3])
        if it == 5:
            image = image.flip(3).rot90(1, [2, 3])
        if it == 6:
            image = image.flip(2)
        if it == 7:
            image = image.flip(3)
        return image
    

@MODELS.register(name="zerorestore_dehaze", arch="zerorestore")
class ZeroRestoreDehaze(ZeroRestore, ModelMixin):
    """Zero-Restore model for image dehazing.
    
    References:
        - Paper: "Zero-shot Single Image Restoration through Controlled Perturbation
          of Koschmieder's Model," CVPR 2021.
        - Code: https://github.com/aupendu/zero-restore
    """
    
    arch     : str          = "zerorestore"
    name     : str          = "zerorestore_dehaze"
    tasks    : list[Task]   = [Task.DEHAZE]
    mltypes  : list[MLType] = [MLType.ZERO_SHOT]
    model_dir: Path         = current_dir
    zoo      : dict         = box.Box()
    
    def __init__(self, num_channels: int = 64, iters: int = 10000):
        super().__init__()
        self.iters      = iters
        self.model      = Estimation(num_channels=num_channels)
        self.state_dict = self.model.state_dict()


@MODELS.register(name="zerorestore_lle", arch="zerorestore")
class ZeroRestoreLLE(ZeroRestore, ModelMixin):
    """Zero-Restore model for low-light image enhancement.
    
    References:
        - Paper: "Zero-shot Single Image Restoration through Controlled Perturbation
          of Koschmieder's Model," CVPR 2021.
        - Code: https://github.com/aupendu/zero-restore
    """
    
    arch     : str          = "zerorestore"
    name     : str          = "zerorestore_lle"
    tasks    : list[Task]   = [Task.LLE]
    mltypes  : list[MLType] = [MLType.ZERO_SHOT]
    model_dir: Path         = current_dir
    zoo      : dict         = box.Box()
    
    def __init__(self, num_channels: int = 64, iters: int = 10000):
        super().__init__()
        self.iters      = iters
        self.model      = EstimationLLIE(num_channels=num_channels)
        self.state_dict = self.model.state_dict()


@MODELS.register(name="zerorestore_uie", arch="zerorestore")
class ZeroRestoreUE(ZeroRestore, ModelMixin):
    """Zero-Restore model for underwater image enhancement.
    
    References:
        - Paper: "Zero-shot Single Image Restoration through Controlled Perturbation
          of Koschmieder's Model," CVPR 2021.
        - Code: https://github.com/aupendu/zero-restore
    """
    
    arch     : str          = "zerorestore"
    name     : str          = "zerorestore_uie"
    tasks    : list[Task]   = [Task.UE]
    mltypes  : list[MLType] = [MLType.ZERO_SHOT]
    model_dir: Path         = current_dir
    zoo      : dict         = box.Box()
    
    def __init__(self, num_channels: int = 64, iters: int = 10000):
        super().__init__()
        self.iters      = iters
        self.model      = Estimation(num_channels=num_channels)
        self.state_dict = self.model.state_dict()
