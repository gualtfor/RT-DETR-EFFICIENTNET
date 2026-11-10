import torch
import numpy as np
import cv2
import PIL.Image
from typing import Any

import torchvision.transforms.v2 as T
from engine.core import register

@register()
class CLAHE(T.Transform):
    def __init__(self, clip_limit=4.0, tile_grid_size=(8, 8), p=0.5):
        super().__init__()
        self.clip_limit = clip_limit
        self.tile_grid_size = tuple(tile_grid_size)
        self.p = p
        self.clahe_eng = cv2.createCLAHE(
            clipLimit=self.clip_limit, 
            tileGridSize=self.tile_grid_size
        )

    def transform(self, inpt: Any, params: dict) -> Any:
        if torch.rand(1).item() > self.p:
            return inpt
                
        img_np = np.array(inpt.convert("RGB"))

        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l_clahe = self.clahe_eng.apply(l)
        merged = cv2.merge((l_clahe, a, b))
        out_np = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

        return PIL.Image.fromarray(out_np)
