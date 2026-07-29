#!/usr/bin/env bash

set -e

ENV_NAME="pro@LGSSCL"

micromamba create -n "${ENV_NAME}" -y \
    -c pytorch \
    -c nvidia \
    -c conda-forge \
    python=3.12.13 \
    numpy=2.5.1 \
    pytorch=2.10.0 \
    torchvision=0.26.0 \
    mamba-ssm=2.3.0 \
    causal-conv1d=1.6.2.post1 \
    cuda-version=12.9 \
    triton=3.6.0

micromamba run -n "${ENV_NAME}" python -m pip install \
    scipy==1.18.0 \
    scikit-learn==1.9.0 \
    spectral==0.25 \
    packaging==26.2 \
    pandas==3.0.5 \
    matplotlib==3.11.1 \
    opencv-python==5.0.0.93 \
    einops==0.8.2 \
    timm==1.0.28 \
    hydra-core==1.3.4 \
    omegaconf==2.3.1 \
    loguru==0.7.3 \
    swanlab==0.9.1 \
    tifffile==2026.7.14

echo "Environment created. Activate it with: micromamba activate ${ENV_NAME}"
