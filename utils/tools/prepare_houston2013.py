from __future__ import annotations

import re
import struct
import zipfile
from pathlib import Path

import numpy as np
import scipy.io as sio
import tifffile


RAW_DIR = Path("data/Houston/raw/2013_DFTC")
OUT_DIR = Path("data/Houston")

CASI_TIF = RAW_DIR / "2013_IEEE_GRSS_DF_Contest_CASI.tif"
TR_TXT = RAW_DIR / "2013_IEEE_GRSS_DF_Contest_Samples_TR.txt"
TR_ROI = RAW_DIR / "2013_IEEE_GRSS_DF_Contest_Samples_TR.roi"
VA_ZIP = RAW_DIR / "2013_IEEE_GRSS_DF_Contest_Samples_VA.zip"

HOUSTON_MAT = OUT_DIR / "Houston.mat"
HOUSTON_GT_MAT = OUT_DIR / "Houston_GT.mat"

CLASS_NAMES = [
    "grass_healthy",
    "grass_stressed",
    "grass_synthetic",
    "tree",
    "soil",
    "water",
    "residential",
    "commercial",
    "road",
    "highway",
    "railway",
    "parking_lot1",
    "parking_lot2",
    "tennis_court",
    "running_track",
]
CLASS_TO_LABEL = {name: idx + 1 for idx, name in enumerate(CLASS_NAMES)}


def ensure_paths() -> None:
    required = [CASI_TIF, TR_TXT, TR_ROI, VA_ZIP]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing Houston 2013 files:\n" + "\n".join(missing))
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def save_houston_cube() -> tuple[int, int]:
    cube = tifffile.imread(CASI_TIF)
    print("raw CASI shape:", cube.shape)

    if cube.ndim == 3 and cube.shape[0] < cube.shape[-1]:
        cube = np.transpose(cube, (1, 2, 0))

    sio.savemat(HOUSTON_MAT, {"Houston": cube})
    print("saved Houston cube:", HOUSTON_MAT, cube.shape)
    return int(cube.shape[0]), int(cube.shape[1])


def parse_tr_txt() -> dict[str, set[tuple[int, int]]]:
    rois: dict[str, set[tuple[int, int]]] = {}
    current_name: str | None = None
    current_points: set[tuple[int, int]] = set()

    for line in TR_TXT.read_text(errors="ignore").splitlines() + ["; ROI name: EOF"]:
        if line.startswith("; ROI name:"):
            if current_name is not None:
                rois[current_name] = current_points
            current_name = line.split(":", 1)[1].strip()
            current_points = set()
            continue

        if re.match(r"^\s*\d+\s+\d+\s+\d+", line):
            parts = line.split()
            x = int(parts[1])
            y = int(parts[2])
            current_points.add((x, y))

    rois.pop("EOF", None)
    return rois


def point_in_polygon(px: float, py: float, polygon: list[tuple[float, float]]) -> bool:
    inside = False
    num_vertices = len(polygon)

    for idx in range(num_vertices):
        x1, y1 = polygon[idx]
        x2, y2 = polygon[(idx + 1) % num_vertices]
        if (y1 > py) != (y2 > py):
            x_intersection = (x2 - x1) * (py - y1) / (y2 - y1) + x1
            if x_intersection > px:
                inside = not inside

    return inside


def rasterize_polygon(polygon: list[tuple[float, float]]) -> set[tuple[int, int]]:
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    x_min = int(np.floor(min(xs)))
    x_max = int(np.ceil(max(xs)))
    y_min = int(np.floor(min(ys)))
    y_max = int(np.ceil(max(ys)))

    pixels: set[tuple[int, int]] = set()
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            # ENVI ROI coordinates align well with pixel-center testing.
            if point_in_polygon(x - 0.5, y - 0.5, polygon):
                pixels.add((x, y))
    return pixels


def parse_roi_bytes(buffer: bytes) -> tuple[dict[str, set[tuple[int, int]]], int, int]:
    pos = 0
    magic = buffer[pos : pos + 4]
    pos += 4
    if magic != b"JoeB":
        raise ValueError(f"Unexpected ROI magic: {magic!r}")

    num_rois = struct.unpack(">I", buffer[pos : pos + 4])[0]
    pos += 4

    rois: dict[str, set[tuple[int, int]]] = {}
    width = height = -1

    for _ in range(num_rois):
        name_len_1, name_len_2 = struct.unpack(">II", buffer[pos : pos + 8])
        pos += 8
        if name_len_1 != name_len_2:
            raise ValueError("ROI name length header is inconsistent")

        name = buffer[pos : pos + name_len_1].decode("ascii")
        pos += ((name_len_1 + 3) // 4) * 4

        width, height, _zero, _npts, *_rest = struct.unpack(">7I", buffer[pos : pos + 28])
        _line_thickness = struct.unpack(">f", buffer[pos + 28 : pos + 32])[0]
        _display_size = struct.unpack(">I", buffer[pos + 32 : pos + 36])[0]
        pos += 36

        # ENVI ROI files reserve a fixed 128-byte block here.
        pos += 128

        _unused0, num_shapes, _unused1, _unused2 = struct.unpack(">4I", buffer[pos : pos + 16])
        pos += 16

        pixels: set[tuple[int, int]] = set()
        for _ in range(num_shapes):
            num_vertices, _shape_type = struct.unpack(">II", buffer[pos : pos + 8])
            pos += 8

            xs = [
                struct.unpack(">f", buffer[pos + 4 * idx : pos + 4 * (idx + 1)])[0]
                for idx in range(num_vertices)
            ]
            pos += 4 * num_vertices

            ys = [
                struct.unpack(">f", buffer[pos + 4 * idx : pos + 4 * (idx + 1)])[0]
                for idx in range(num_vertices)
            ]
            pos += 4 * num_vertices

            pixels.update(rasterize_polygon(list(zip(xs, ys))))

        rois[name] = pixels

    return rois, width, height


def load_va_roi() -> dict[str, set[tuple[int, int]]]:
    with zipfile.ZipFile(VA_ZIP) as archive:
        roi_members = [name for name in archive.namelist() if name.lower().endswith(".roi")]
        if len(roi_members) != 1:
            raise ValueError(f"Expected exactly one ROI in {VA_ZIP}, got {roi_members}")
        roi_bytes = archive.read(roi_members[0])

    va_rois, width, height = parse_roi_bytes(roi_bytes)
    print(f"parsed VA ROI: {len(va_rois)} classes, image size {height}x{width}")
    return va_rois


def build_sparse_gt(height: int, width: int) -> np.ndarray:
    tr_rois = parse_tr_txt()
    va_rois = load_va_roi()

    gt = np.zeros((height, width), dtype=np.uint8)

    for source_name, roi_dict in [("TR.txt", tr_rois), ("VA.roi", va_rois)]:
        for class_name, points in roi_dict.items():
            if class_name not in CLASS_TO_LABEL:
                raise KeyError(f"Unknown Houston class in {source_name}: {class_name}")

            label = CLASS_TO_LABEL[class_name]
            for x, y in points:
                # Source coordinates are 1-based.
                if not (1 <= x <= width and 1 <= y <= height):
                    raise ValueError(
                        f"Point {(x, y)} from {source_name}/{class_name} is outside image bounds "
                        f"{width}x{height}"
                    )
                gt[y - 1, x - 1] = label

    return gt


def main() -> None:
    ensure_paths()
    height, width = save_houston_cube()
    gt = build_sparse_gt(height=height, width=width)
    sio.savemat(HOUSTON_GT_MAT, {"Houston_GT": gt})

    labeled_pixels = int(np.count_nonzero(gt))
    unique_labels = np.unique(gt)
    print("saved Houston GT:", HOUSTON_GT_MAT, gt.shape)
    print("labeled pixels:", labeled_pixels)
    print("labels present:", unique_labels.tolist())


if __name__ == "__main__":
    main()
