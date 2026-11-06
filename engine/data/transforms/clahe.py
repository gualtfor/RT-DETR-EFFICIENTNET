import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from engine.core import register

@register()
class CLAHE(object):
    def __init__(self, clip_limit=4.0, tile_grid_size=(8, 8), p=0.5):
        self.transform = A.Compose([
            A.CLAHE(
                clip_limit=clip_limit, 
                tile_grid_size=tuple(tile_grid_size), 
                p=p
            )
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

    def __call__(self, sample):
        image = sample['image']
        bboxes = sample.get('bboxes', [])
        labels = sample.get('labels', [])
        if len(bboxes) > 0:
            augmented = self.transform(image=image, bboxes=bboxes, labels=labels)
            sample['image'] = augmented['image']
            sample['bboxes'] = augmented['bboxes']
            sample['labels'] = augmented['labels']
        else:
            augmented = self.transform(image=image)
            sample['image'] = augmented['image']

        return sample

@register()
class RandomPhotometricDistortCLAHE(object):
    def __init__(self, p=0.5):
        self.transform = A.Compose([
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=p)
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

    def __call__(self, sample):
        image = sample['image']
        bboxes = sample.get('bboxes', [])
        labels = sample.get('labels', [])

        if len(bboxes) > 0:
            augmented = self.transform(image=image, bboxes=bboxes, labels=labels)
            sample['image'] = augmented['image']
            sample['bboxes'] = augmented['bboxes']
            sample['labels'] = augmented['labels']
        else:
            augmented = self.transform(image=image)
            sample['image'] = augmented['image']

        return sample