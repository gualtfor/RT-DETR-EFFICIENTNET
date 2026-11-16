# RT-DETRv4 with EfficientNetV2 for X-ray Threat Detection

**An experimental implementation of RT-DETRv4 integrating an EfficientNetV2 backbone and DINOv3 teacher model for detecting prohibited objects in X-ray security images.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c)](https://pytorch.org/)
[![Architecture](https://img.shields.io/badge/Architecture-RT--DETRv4-green)](https://github.com/AIMIA-PKU/RT-DETRv4)
[![Dataset](https://img.shields.io/badge/Dataset-PIDray-orange)](https://github.com/)

---

## Overview

This repository presents a modified implementation of **RT-DETRv4** for object detection in X-ray security images. The project explores the integration of an EfficientNetV2 backbone and DINOv3 as a teacher model to improve feature extraction and object detection performance.

The main objective is to investigate the application of modern deep learning architectures to the detection and localization of prohibited objects in security screening environments, particularly in scenarios involving overlapping and partially occluded objects.

The implementation includes the following modifications:

- **EfficientNetV2 backbone:** Replaces the original backbone with an EfficientNetV2 architecture to extract multi-scale visual features.
- **DINOv3 teacher model:** Integrates a Vision Foundation Model to support knowledge distillation during training.
- **CLAHE preprocessing:** Applies Contrast Limited Adaptive Histogram Equalization to improve local image contrast and enhance visual characteristics.
- **PIDray dataset:** Uses an X-ray security dataset containing 12 categories of prohibited objects.
- **Video inference:** Supports object detection and visualization on video sequences, displaying bounding boxes, predicted classes, and confidence scores.

This implementation is based on the original RT-DETRv4 framework, with modifications to the backbone, preprocessing pipeline, and experimental configuration.

## Architecture and Contributions

The proposed implementation combines three main components:

| Component | Implementation |
|---|---|
| Detection architecture | RT-DETRv4 |
| Feature extraction | EfficientNetV2-S |
| Teacher model | DINOv3 ViT-B/16 |
| Image preprocessing | CLAHE |
| Input resolution | 640 × 640 |
| Detection framework | PyTorch |
| Dataset | PIDray |

### EfficientNetV2 Backbone

EfficientNetV2-S is integrated as the feature extraction backbone. Features are extracted from three intermediate stages, providing multi-scale representations for the detection architecture.

The backbone produces feature maps with the following channel dimensions:

- Stage 1: 64 channels
- Stage 2: 160 channels
- Stage 3: 256 channels

These representations are passed to the detection encoder for subsequent object localization and classification.

### DINOv3 Teacher Model

DINOv3 ViT-B/16 is incorporated as a teacher model within the RT-DETRv4 knowledge distillation framework.

The objective is to transfer richer visual representations from a pretrained Vision Foundation Model to the detection architecture during training.

### CLAHE Preprocessing

Contrast Limited Adaptive Histogram Equalization (CLAHE) is incorporated into the preprocessing pipeline to improve local contrast in X-ray images.

Configuration:

- Clip limit: 4.0
- Tile grid size: 8 × 8
- Application probability: 0.5

This approach investigates whether local contrast enhancement can improve the representation of objects with low visual contrast and overlapping structures.

---

## Dataset: PIDray

The experiments were conducted using the PIDray dataset, a public X-ray security inspection dataset designed for prohibited-item detection.

The dataset contains 124,486 images and 12 object categories.

The evaluated categories include:

1. Baton
2. Pliers
3. Hammer
4. Powerbank
5. Scissors
6. Wrench
7. Gun
8. Bullet
9. Sprayer
10. Handcuffs
11. Knife
12. Lighter

The dataset provides different difficulty levels, including Easy, Hard, and Hidden subsets, allowing the evaluation of detection performance under different levels of visual complexity and object occlusion.

---

## Experimental Results

The proposed implementation was evaluated using the COCO evaluation protocol.

The primary evaluation metrics are:

- **AP@[0.50:0.95]:** Average Precision calculated across IoU thresholds from 0.50 to 0.95.
- **AP50:** Average Precision at an Intersection over Union threshold of 0.50.

### Quantitative Results

The following table presents the obtained results for the different PIDray subsets.

| PIDray subset | AP@[0.50:0.95] | AP50 |
|---|---:|---:|
| Easy | 73.3% | 81.9% |
| Hard | 60.8% | 77.1% |
| Hidden | 45.6% | 60.9% |

### Results Interpretation

The results demonstrate different detection performance depending on the difficulty of the evaluated images.

The Easy subset achieved the highest performance, with an AP of 73.3%. The Hard subset achieved an AP of 60.8%, indicating a reduction in detection performance as the visual complexity increases.

The Hidden subset presented the greatest challenge, achieving an AP of 45.6%. This reduction highlights the difficulty of detecting objects under occlusion and overlapping conditions, which remain important challenges in X-ray security inspection.

These results provide an experimental reference for evaluating the integration of EfficientNetV2, DINOv3, and CLAHE within the RT-DETRv4 framework.

*Note: These are experimental results obtained with this implementation on PIDray. They are not the original RT-DETRv4 results reported on COCO.*

---

## Model Complexity

The implemented detection model has the following reported computational characteristics:

| Metric | Value |
|---|---:|
| Trainable parameters | 28.09 M |
| GFLOPs | 65.04 |
| GMACs | 32.35 |
| Input resolution | 640 × 640 |

These values describe model complexity and should not be interpreted as measured inference latency or throughput.

---

## Video Demonstration

The following demonstration presents the model performing object detection on an X-ray video sequence.

The inference visualization includes:

- Bounding boxes around detected objects.
- Predicted object categories.
- Confidence scores for each detection.
- Frame-by-frame object detection.

### RT-DETRv4 + EfficientNetV2 inference

**Video:** `assets/torch_results.mp4`
<div align="center">
https://github.com/user-attachments/assets/8960bca6-b6df-4fa3-8b67-c80582da45da
</div>

</details>

The demonstration uses the trained checkpoint and the custom PyTorch inference implementation.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/gualtfor/RT-DETR-EFFICIENTNET.git

cd RT-DETR-EFFICIENTNET
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

For Google Colab, the repository also includes:

```bash
bash setup_colab.sh
```

Make sure the PyTorch and CUDA versions are compatible with your GPU environment.

## Inference

The implementation supports image and video inference using the trained PyTorch checkpoint.

Example:

```bash
python tools/inference/torch_inf.py \
    -c path/to/config.yml \
    -r path/to/model.pth \
    -i path/to/input_video.mp4 \
    -d cuda:0
```

For image inference:

```bash
python tools/inference/torch_inf.py \
    -c path/to/config.yml \
    -r path/to/model.pth \
    -i path/to/image.jpg \
    -d cuda:0
```

The inference implementation generates an annotated output with the detected objects, their corresponding class names, and confidence scores.

## Repository Structure

```text
RT-DETR-EFFICIENTNET/
│
├── configs/
│   └── ...                 # Model configurations
│
├── engine/
│   └── ...                 # Training and evaluation engine
│
├── tools/
│   ├── inference/
│   │   └── ...             # Image and video inference
│   └── ...
│
├── assets/
│   └── torch_results.mp4   # Custom inference demonstration
│
├── requirements.txt
├── setup_colab.sh
├── setup_dataset.py
├── train.py
└── README.md
```

---

## Acknowledgments

This project builds upon the contributions of researchers who have advanced real-time object detection, efficient convolutional neural networks, and self-supervised visual representation learning.

We sincerely acknowledge the original authors and research teams behind the following works:

### 1. RT-DETRv4

**Zijun Liao, Yian Zhao, Xin Shan, Yu Yan, Chang Liu, Lei Lu, Xiangyang Ji, and Jie Chen.**

*RT-DETRv4: Painlessly Furthering Real-Time Object Detection with Vision Foundation Models.*

We thank the RT-DETRv4 authors for developing and publicly releasing the detection framework that serves as the foundation of this project.

- [Official GitHub Repository](https://github.com/AIMIA-PKU/RT-DETRv4)
- [Research Paper](https://arxiv.org/abs/2510.25257)

### 2. EfficientNetV2

**Mingxing Tan and Quoc V. Le.**

*EfficientNetV2: Smaller Models and Faster Training.*

We acknowledge the original authors for introducing an efficient convolutional neural network architecture that provides the backbone used in this implementation.

- [Original Research Paper](https://proceedings.mlr.press/v139/tan21a.html)
- [Official Google Research Implementation](https://github.com/google/automl/tree/master/efficientnetv2)

### 3. DINOv3

**Oriane Siméoni, Huy V. Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, and collaborators at Meta AI Research.**

*DINOv3.*

We sincerely thank the DINOv3 research team for developing and releasing this Vision Foundation Model. Its visual representations provide the teacher model used in the knowledge distillation framework explored in this project.

- [Official GitHub Repository](https://github.com/facebookresearch/dinov3)
- [Research Paper](https://arxiv.org/abs/2508.10104)

### Final Acknowledgment

This repository would not have been possible without the open-source contributions and research advances made available by these research teams.

We recognize that the proposed implementation is an adaptation and experimental application of their original work. All original architectures, methods, and research contributions remain credited to their respective authors.

---

## Citation

If you use this implementation in academic research, please cite the original works:

**RT-DETRv4**

```bibtex
@article{liao2025rtdetrv4,
  title={RT-DETRv4: Painlessly Furthering Real-Time Object Detection with Vision Foundation Models},
  author={Liao, Zijun and Zhao, Yian and Shan, Xin and Yan, Yu and Liu, Chang and Lu, Lei and Ji, Xiangyang and Chen, Jie},
  journal={arXiv preprint arXiv:2510.25257},
  year={2025}
}
```

**EfficientNetV2**

```bibtex
@InProceedings{pmlr-v139-tan21a,
  title={EfficientNetV2: Smaller Models and Faster Training},
  author={Tan, Mingxing and Le, Quoc V.},
  booktitle={International Conference on Machine Learning},
  pages={10096--10106},
  year={2021}
}
```

**DINOv3**

```bibtex
@misc{simeoni2025dinov3,
  title={DINOv3},
  author={Siméoni, Oriane and Vo, Huy V. and Seitzer, Maximilian and others},
  year={2025},
  eprint={2508.10104},
  archivePrefix={arXiv}
}
```

---

## Author and Project

**Repository:** [RT-DETR-EFFICIENTNET](https://github.com/gualtfor/RT-DETR-EFFICIENTNET)

**Author:** gualtfor

This project is an independent experimental implementation focused on applying efficient object detection architectures to X-ray security inspection.
