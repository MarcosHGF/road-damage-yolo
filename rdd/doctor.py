"""Environment diagnostics: Python, PyTorch/CUDA, Ultralytics, disk, dataset."""
from __future__ import annotations

import platform
import shutil
import sys

from .constants import COUNTRY_ZIPS, Paths

CUDA_HELP = """\
PyTorch cannot see a GPU. The usual causes:
  1. A CPU-only PyTorch was installed (pip does this by default on Windows).
     Reinstall the CUDA build. Pick the command for your CUDA version at
     https://pytorch.org/get-started/locally/ , for example:
         pip uninstall -y torch torchvision
         pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
  2. The NVIDIA driver is missing or too old: run `nvidia-smi` in a terminal.
  3. No NVIDIA GPU (AMD/Intel/Apple): training falls back to the CPU
     (Apple Silicon can use `--device mps`).
"""


def run(paths: Paths) -> int:
    ok = True
    print(f"Python        : {sys.version.split()[0]} ({platform.system()} {platform.machine()})")
    if sys.version_info < (3, 10):
        ok = False
        print("  !! Python 3.10 or newer is required.")

    try:
        import torch

        print(f"PyTorch       : {torch.__version__}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                print(f"GPU {i}         : {p.name}, {p.total_memory / 1024**3:.1f} GB VRAM")
            print(f"CUDA (torch)  : {torch.version.cuda}")
        else:
            mps = getattr(torch.backends, "mps", None)
            if mps is not None and mps.is_available():
                print("GPU           : Apple MPS available (use `--device mps`)")
            else:
                print("GPU           : none visible to PyTorch")
                print("\n" + CUDA_HELP)
                print(f"nvidia-smi    : {'found' if shutil.which('nvidia-smi') else 'not found on PATH'}")
    except ImportError:
        ok = False
        print("PyTorch       : NOT INSTALLED (run `pip install -r requirements.txt`)")

    try:
        import ultralytics

        print(f"Ultralytics   : {ultralytics.__version__}")
    except ImportError:
        ok = False
        print("Ultralytics   : NOT INSTALLED (run `pip install -r requirements.txt`)")

    paths.data_dir.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(paths.data_dir).free / 1024**3
    print(f"Free disk     : {free_gb:.1f} GB in {paths.data_dir}")
    if free_gb < 8:
        print("  !! The default 3-country subset needs about 6 GB (archives + extracted + converted).")

    print(f"Raw data      : {_listing(paths.raw)}")
    print(f"Converted     : {_count(paths.converted / 'images')} images")
    print(f"YOLO dataset  : {'ready (' + str(paths.data_yaml) + ')' if paths.data_yaml.exists() else 'not built yet'}")
    print(f"Countries     : {', '.join(COUNTRY_ZIPS)}")
    return 0 if ok else 1


def _listing(raw) -> str:
    if not raw.exists():
        return "none"
    ready = [p.name for p in raw.iterdir() if p.is_dir() and (p / ".extracted_ok").exists()]
    if (raw / "SYNTHETIC_DEMO_DATA.txt").exists():
        return "synthetic demo data"
    return ", ".join(sorted(ready)) or "none"


def _count(path) -> int:
    return sum(1 for p in path.iterdir() if p.is_file()) if path.exists() else 0
