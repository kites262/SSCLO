import json
import os
import shutil
from pathlib import Path

import numpy as np
from loguru import logger


def save_src_files(
    root_dir: str,
    paths: list[str],
    artifact_dir: str = "src",
) -> None:
    logger.debug(f"Saving {paths}")
    root = Path(root_dir)
    for rel_path in paths:
        src = root / rel_path
        dst = Path(artifact_dir) / rel_path

        if src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            logger.debug(f"Saved artifact file: {dst}")
        elif src.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            logger.debug(f"Saved artifact directory: {dst}")
        else:
            logger.warning(f"Artifact path {src} does not exist")


def _to_serializable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    return str(o)


def save_json_file(
    data: dict,
    path: str,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_to_serializable)
    logger.debug(f"Saved artifact JSON file: {path}")
