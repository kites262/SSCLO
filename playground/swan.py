import importlib.metadata as m

import swanlab

print(swanlab.__version__)


for r in m.requires("mamba-ssm"):
    print(r)
