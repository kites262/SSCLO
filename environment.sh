mamba create -n pro@MambaNO python=3.12 -y
mamba activate pro@MambaNO

mamba install \
  pytorch \
  mamba-ssm=2.3.0 \
  torchvision \
  -c pytorch -c nvidia

uv pip install \
    scikit-learn \
    spectral \
    packaging \
    pandas \
    matplotlib \
    opencv-python \
    einops \
    timm \
    hydra-core \
    omegaconf \
    loguru \
    swanlab
