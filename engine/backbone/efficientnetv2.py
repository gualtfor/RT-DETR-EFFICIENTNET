import torch.nn as nn
import timm
from ...core import register
from loguru import logger

__all__ = ['EfficientNetV2']

@register
class EfficientNetV2(nn.Module):
    def __init__(self, model_name='efficientnetv2_s', pretrained=True, out_indices=(2, 3, 4), freeze_at=-1):
        super().__init__()
        self.model = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices
        )
        self.out_channels = self.model.feature_info.channels()
        logger.info("The descripcion of the channels are: {}", self.out_channels)
        
        if freeze_at >= 0:
            self._freeze_layers(freeze_at)

    def forward(self, x):
        features = self.model(x)
        return features

    def _freeze_layers(self, freeze_at):
        counter = 0
        for child in self.model.children():
            counter += 1
            if counter <= freeze_at:
                for param in child.parameters():
                    param.requires_grad = False

# efficientnetv2_s efficientnetv2_m
