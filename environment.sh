mamba create -n pro@MambaFNO python=3.12 -y
mamba activate pro@MambaFNO

mamba install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y
mamba install mamba-ssm

uv pip install \
    scikit-learn \
    spectral \
    packaging \
    pandas \
    matplotlib \
    opencv-python \
    einops \
    timm
