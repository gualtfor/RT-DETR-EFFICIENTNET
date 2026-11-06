#!/usr/bin/env bash
set -euxo pipefail
if [ ! -d /content/RT-DETR-EFFICIENTNET ]; then
  git clone https://github.com/gualtfor/RT-DETR-EFFICIENTNET.git /content/RT-DETR-EFFICIENTNET
fi
pip -q install -r /content/RT-DETR-EFFICIENTNET/requirements.txt
if [ ! -d /content/dinov3 ]; then
  git clone https://github.com/facebookresearch/dinov3.git /content/dinov3
fi
