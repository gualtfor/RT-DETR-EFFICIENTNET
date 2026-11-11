import torch
import numpy as np
import cv2
import PIL.Image

from typing import Any

import torchvision.transforms.v2 as T
from torchvision import tv_tensors

from engine.core import register


@register()
class CLAHE(T.Transform):

    def __init__(self, clip_limit=4.0, tile_grid_size=(8, 8), p=0.5):
        super().__init__()

        self.clip_limit = clip_limit
        self.tile_grid_size = tuple(tile_grid_size)
        self.p = p

        self.clahe_eng = cv2.createCLAHE(
            clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size
        )

    def transform(self, inpt: Any, params: dict) -> Any:
        if isinstance(
            inpt,
            (tv_tensors.BoundingBoxes, tv_tensors.Mask)
        ):
            return inpt
        
        if not isinstance(inpt, (torch.Tensor, PIL.Image.Image)):
            return inpt

        if torch.rand(1).item() > self.p:
            return inpt

        is_tensor = isinstance(inpt, torch.Tensor)

        if is_tensor:
            original_dtype = inpt.dtype
            original_device = inpt.device

            is_tv_image = isinstance(inpt, tv_tensors.Image)

            img_np = inpt.detach().cpu().numpy()

            if img_np.ndim == 3:
                img_np = np.moveaxis(img_np, 0, -1)

            original_max = float(img_np.max()) if img_np.size else 0.0

            if np.issubdtype(img_np.dtype, np.floating):
                if original_max <= 1.0:
                    img_np = img_np * 255.0

        else:
            img_np = np.array(inpt.convert("RGB"))

        if img_np.dtype != np.uint8:
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)

        if img_np.ndim == 2:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)

        elif img_np.shape[-1] == 1:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)

        elif img_np.shape[-1] == 4:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)

        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)

        l, a, b = cv2.split(lab)

        l_clahe = self.clahe_eng.apply(l)

        merged = cv2.merge((l_clahe, a, b))

        out_np = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

        if is_tensor:

            if torch.is_floating_point(inpt):
                if original_max <= 1.0:
                    out_np = out_np.astype(np.float32) / 255.0

            enhanced = torch.from_numpy(
                np.ascontiguousarray(np.moveaxis(out_np, -1, 0))
            ).to(device=original_device, dtype=original_dtype)

            if is_tv_image:
                enhanced = tv_tensors.Image(enhanced)

            return enhanced

        return PIL.Image.fromarray(out_np)
