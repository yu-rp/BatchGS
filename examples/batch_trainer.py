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

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

set_seed(0)

class SimpleTrainer:
    """Trains random gaussians to fit an image."""

    def __init__(
        self,
        gt_image: Tensor,
        num_points: int = 2000,
    ):
        self.device = torch.device("cuda:0")
        self.gt_image = gt_image.to(device=self.device)
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
        # print(f"Batch size: {self.batch_size}")

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

        self.quats = torch.cat(
            [
                torch.sqrt(1.0 - u) * torch.sin(2.0 * math.pi * v),
                torch.sqrt(1.0 - u) * torch.cos(2.0 * math.pi * v),
                torch.sqrt(u) * torch.sin(2.0 * math.pi * w),
                torch.sqrt(u) * torch.cos(2.0 * math.pi * w),
            ],
            -1,
        )
        self.opacities = torch.ones((self.batch_size, self.num_points, 1), device=self.device)

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

        self.means.requires_grad = True
        self.scales.requires_grad = True
        self.quats.requires_grad = True
        self.rgbs.requires_grad = True
        self.opacities.requires_grad = True
        self.viewmat.requires_grad = False

        print(f"Means: {self.means.shape}")
        print(f"Scales: {self.scales.shape}")
        print(f"Quats: {self.quats.shape}")
        print(f"RGBs: {self.rgbs.shape}")
        print(f"Opacities: {self.opacities.shape}")
        print(f"Viewmat: {self.viewmat.shape}")

    def train(self, iterations: int = 1000, lr: float = 0.01, save_imgs: bool = False):
        optimizer = optim.Adam(
            [self.rgbs, self.means, self.scales, self.opacities, self.quats], lr
        )
        mse_loss = torch.nn.MSELoss()
        frames = []
        times = [0] * 3  # project, rasterize, backward
        for iter in range(iterations):
            start = time.time()
            xys, depths, radii, conics, num_tiles_hit, cov3d = _ProjectGaussians.apply(
                self.means.view(-1, 3),
                self.scales.view(-1, 3),
                1,
                self.quats.view(-1, 4),
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

            # print(f"xys: {xys.shape}")
            # print(f"depths: {depths.shape}")
            # print(f"radii: {radii.shape}")
            # print(f"conics: {conics.shape}")
            # print(f"num_tiles_hit: {num_tiles_hit.shape}")

            xys = xys.view(self.batch_size, self.num_points, 2)
            depths = depths.view(self.batch_size, self.num_points)
            radii = radii.view(self.batch_size, self.num_points)
            conics = conics.view(self.batch_size, self.num_points, 3)
            num_tiles_hit = num_tiles_hit.view(self.batch_size, self.num_points)

            # assert torch.dist(xys[0], xys[1]) == 0
            # assert torch.dist(depths[0].float(), depths[1].float()) == 0
            # assert torch.dist(radii[0].float(), radii[1].float()) == 0
            # assert torch.dist(conics[0], conics[1]) == 0
            # assert torch.dist(num_tiles_hit[0].float(), num_tiles_hit[1].float()) == 0
            # assert torch.dist(self.rgbs[0], self.rgbs[1]) == 0
            # assert torch.dist(self.opacities[0], self.opacities[1]) == 0

            start = time.time()
            out_img = _RasterizeGaussians.apply(
                xys, # (B, N, 2)
                depths, # (B, N)
                radii, # (B, N)
                conics, # (B, N, 3)
                num_tiles_hit, # (B, N)
                torch.sigmoid(self.rgbs), # (B, N, 3)
                torch.sigmoid(self.opacities), # (B, N, 1)
                self.H,
                self.W,
                self.background, # (B, 3)
            ) # (B, H, W, 3)
            torch.cuda.synchronize()
            times[1] += time.time() - start
            loss = mse_loss(out_img, self.gt_image) 
            optimizer.zero_grad()
            start = time.time()
            loss.backward()
            torch.cuda.synchronize()
            times[2] += time.time() - start
            optimizer.step()
            print(f"Iteration {iter + 1}/{iterations}, Loss: {loss.item()}")

            if save_imgs and iter % 5 == 0:
                imggrid = torchvision.utils.make_grid(
                    out_img[:4].permute(0,3,1,2).detach().cpu(), 
                    nrow=2)
                frames.append((
                    imggrid.permute(1,2,0).numpy() * 255).astype(np.uint8))
                # frames.append((
                #     out_img.detach().cpu().numpy() * 255).astype(np.uint8))
        if save_imgs:
            # save them as a gif with PIL
            frames = [Image.fromarray(frame) for frame in frames]
            out_dir = os.path.join(os.getcwd(), "renders")
            os.makedirs(out_dir, exist_ok=True)
            frames[0].save(
                f"{out_dir}/training.gif",
                save_all=True,
                append_images=frames[1:],
                optimize=False,
                duration=5,
                loop=0,
            )
        print(
            f"Total(s):\nProject: {times[0]:.3f}, Rasterize: {times[1]:.3f}, Backward: {times[2]:.3f}"
        )
        print(
            f"Per step(s):\nProject: {times[0]/iterations:.5f}, Rasterize: {times[1]/iterations:.5f}, Backward: {times[2]/iterations:.5f}"
        )


def image_path_to_tensor(image_path: Path):
    import torchvision.transforms as transforms

    img = Image.open(image_path)
    transform = transforms.ToTensor()
    img_tensor = transform(img).permute(1, 2, 0)[..., :3]
    return img_tensor


def main(
    height: int = 256,
    width: int = 256,
    num_points: int = 100000,
    save_imgs: bool = True,
    img_path: Optional[Path] = None,
    iterations: int = 10,
    lr: float = 0.01,
) -> None:
    img_path = [
        "/root/autodl-tmp/00017__000.png",
        "/root/autodl-tmp/00010__001.png",
        "/root/autodl-tmp/00010__001.png",
        "/root/autodl-tmp/00017__000.png",
    ]

    gt_image = [image_path_to_tensor(p) for p in img_path]
    gt_image = torch.stack(gt_image, dim=0)

    trainer = SimpleTrainer(gt_image=gt_image, num_points=num_points)
    # torch.save(trainer, "trainer.pt")

    trainer.train(
        iterations=iterations,
        lr=lr,
        save_imgs=save_imgs,
    )

if __name__ == "__main__":
    tyro.cli(main)

