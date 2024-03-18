import math, random
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch, torchvision
import tyro
from gsplat.project_gaussians import _ProjectGaussians
from gsplat.rasterize import _RasterizeGaussians
from PIL import Image
from torch import Tensor, optim

class GaussianProcessor:
    """
    Gaussian Processor 
    """
    def __init__(
        self, 
        gt_image: torch.Tensor, 
        batch_size: int = 1,
        num_points: int = 10000, 
        device: str = "cuda:0",
        init_guassians: bool = False):

        self.device = device
        self.batch_size = batch_size
        self.num_points = num_points

        BLOCK_X, BLOCK_Y = 16, 16
        fov_x = math.pi / 2.0
        self.H, self.W = gt_image.shape[1], gt_image.shape[2]
        self.focal = 0.5 * float(self.W) / math.tan(0.5 * fov_x)
        self.tile_bounds = (
            (self.W + BLOCK_X - 1) // BLOCK_X,
            (self.H + BLOCK_Y - 1) // BLOCK_Y,
            1,
        )
        self.img_size = torch.tensor([self.W, self.H, 1], device=self.device)
        self.block = torch.tensor([BLOCK_X, BLOCK_Y, 1], device=self.device)
        self.batch_size = gt_image.shape[0]

        self.viewmat = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 8.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            device=self.device,
        )
        self.background = torch.zeros(self.batch_size, 3, device=self.device)
        self.viewmat.requires_grad = False
        self.background.requires_grad = False
        print(f"Viewmat: {self.viewmat.shape}")
        print(f"background: {self.background.shape}")

        if init_guassians:
            self._init_gaussians()

    def _init_gaussians(self):
        """Random gaussians"""
        bd = 2

        self.means = bd * (torch.rand(self.batch_size, self.num_points, 3, device=self.device) - 0.5)
        self.scales = torch.rand(self.batch_size, self.num_points, 3, device=self.device)
        self.rgbs = torch.rand(self.batch_size, self.num_points, 3, device=self.device)

        u = torch.rand(self.batch_size,self.num_points, 1, device=self.device)
        v = torch.rand(self.batch_size,self.num_points, 1, device=self.device)
        w = torch.rand(self.batch_size,self.num_points, 1, device=self.device)

        self.quats = self.get_quats(u, v, w)
        self.opacities = torch.ones((self.batch_size, self.num_points, 1), device=self.device)

        self.means.requires_grad = True
        self.scales.requires_grad = True
        self.quats.requires_grad = True
        self.rgbs.requires_grad = True
        self.opacities.requires_grad = True

        print(f"Means: {self.means.shape}")
        print(f"Scales: {self.scales.shape}")
        print(f"Quats: {self.quats.shape}")
        print(f"RGBs: {self.rgbs.shape}")
        print(f"Opacities: {self.opacities.shape}")

    def get_quats(self, u, v, w):
        quats = torch.cat(
            [
                torch.sqrt(1.0 - u) * torch.sin(2.0 * math.pi * v),
                torch.sqrt(1.0 - u) * torch.cos(2.0 * math.pi * v),
                torch.sqrt(u) * torch.sin(2.0 * math.pi * w),
                torch.sqrt(u) * torch.cos(2.0 * math.pi * w),
            ],
            -1,
        )
        return quats

    def process(
        self,
        means,
        scales,
        u,v,w,
        rgbs,
        opacities
        ):

        if means is None:
            means = self.means
        if scales is None:
            scales = self.scales
        if rgbs is None:
            rgbs = self.rgbs
        if opacities is None:
            opacities = self.opacities
        if u is None or v is None or w is None:
            quats = self.quats
        elif u is not None and v is not None and w is not None:
            quats = self.get_quats(u, v, w)
        else:
            raise ValueError("Invalid quats")

        times = [0] * 2  # project, rasterize, backward
        start = time.time()
        xys, depths, radii, conics, num_tiles_hit, cov3d = _ProjectGaussians.apply(
            means.view(-1, 3),
            scales.view(-1, 3),
            1,
            quats.view(-1, 4),
            self.viewmat,
            self.viewmat,
            self.focal,
            self.focal,
            self.W / 2,
            self.H / 2,
            self.H,
            self.W,
            self.tile_bounds,
        )
        torch.cuda.synchronize()
        times[0] += time.time() - start

        xys = xys.view(self.batch_size, self.num_points, 2)
        depths = depths.view(self.batch_size, self.num_points)
        radii = radii.view(self.batch_size, self.num_points)
        conics = conics.view(self.batch_size, self.num_points, 3)
        num_tiles_hit = num_tiles_hit.view(self.batch_size, self.num_points)

        start = time.time()
        out_img = _RasterizeGaussians.apply(
            xys, # (B, N, 2)
            depths, # (B, N)
            radii, # (B, N)
            conics, # (B, N, 3)
            num_tiles_hit, # (B, N)
            torch.sigmoid(rgbs), # (B, N, 3)
            torch.sigmoid(opacities), # (B, N, 1)
            self.H,
            self.W,
            self.background, # (B, 3)
        ) # (B, H, W, 3)
        torch.cuda.synchronize()
        times[1] += time.time() - start
        return out_img