# Road damage detection with YOLO (RDD2022)

Detects four kinds of road damage in photos with a YOLO model trained on the
[RDD2022](https://github.com/sekilab/RoadDamageDetector) dataset:

| Class | RDD2022 code | Example |
|---|---|---|
| `longitudinal_crack` | D00 | crack running along the road |
| `transverse_crack` | D10 | crack running across the road |
| `alligator_crack` | D20 | interconnected cracking pattern |
| `pothole` | D40 | bowl-shaped hole in the surface |

The repository covers the whole workflow, from download to a trained model:
**download -> convert -> split -> train -> evaluate -> predict**, all behind one
command, `python -m rdd`. It runs locally on Windows, Linux and macOS.

> **About the data in this repository.** The full dataset is about 2 GB for the default
> three countries, and GitHub rejects files over 100 MB, so it is *downloaded by a script*
> instead of being stored here. `data/sample/` holds a small real-data sample that is
> committed so a fresh clone works immediately (see [Data](#data)). Nothing in this
> repository reports accuracy numbers: run the evaluation on your own machine to get them.

## Quick start

```bash
git clone <your-repo-url> road-damage-yolo
cd road-damage-yolo

python -m venv .venv
# Windows: .venv\Scripts\activate        Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

python -m rdd doctor        # checks Python, GPU, disk space
python -m rdd smoke-test    # ~1 minute on a CPU, no download: proves the install works
```

`smoke-test` builds a tiny **synthetic** dataset, trains a network for two epochs
and evaluates it. The resulting accuracy is meaningless on purpose; it only
confirms that every stage runs on your machine.

### NVIDIA GPU (important on Windows)

`pip install ultralytics` installs a **CPU-only PyTorch on Windows**, and training
then runs 20-50x slower without any error. Check with `python -m rdd doctor`. If it
says no GPU is visible, reinstall the CUDA build (choose the command for your
CUDA version at <https://pytorch.org/get-started/locally/>), for example:

```bash
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python -m rdd doctor
```

## Train on the real dataset

```bash
python -m rdd download      # Japan + Czech + India (~1.8 GB), resumable
python -m rdd convert       # Pascal VOC XML -> YOLO labels, prints class statistics
python -m rdd split         # stratified train/val/test + data/yolo_dataset/data.yaml
python -m rdd stats         # class/country tables and figures in reports/
python -m rdd train         # defaults in config.yaml: yolo11s, 100 epochs, 640 px, auto batch
python -m rdd evaluate      # overall + per-class + per-country metrics in reports/
python -m rdd predict --source path/to/photo.jpg
```

Useful variations:

```bash
python -m rdd train --model yolo11n.pt --batch 8      # small GPU (4 GB VRAM)
python -m rdd train --model yolo11m.pt --epochs 150   # bigger GPU (10 GB+)
python -m rdd train --epochs 5 --fraction 0.1         # quick dry run on 10% of the data
python -m rdd train --workers 0                       # Windows "paging file too small" errors
python -m rdd train --resume                          # continue an interrupted run
python -m rdd download --countries Japan Czech India Norway
```

The first `train` downloads the pretrained checkpoint (for example `yolo11s.pt`)
automatically. Results land in `runs/detect/rdd_yolo*/` (curves, confusion matrix,
`weights/best.pt`) and in `reports/` (`metrics.json`, `metrics.md`, figures).

If the download is blocked on your network, fetch the archives in a browser from the
links in [`rdd/constants.py`](rdd/constants.py) (`COUNTRY_ZIPS`), put them in
`data/downloads/` as `Japan.zip`, `Czech.zip`, `India.zip` (or `RDD2022_<Country>.zip`),
and run `python -m rdd download` again: it uses local archives first.

## Data

| Folder | Content | In git? |
|---|---|---|
| `data/downloads/`, `data/raw/` | official archives and extracted RDD2022 (Pascal VOC) | no |
| `data/converted/` | YOLO labels for the 4 classes + stats | no |
| `data/yolo_dataset/` | train/val/test folders and `data.yaml` (generated) | no |
| `data/sample/` | a few hundred real images with labels (<= 60 MB) | **yes** |

**Putting a real sample in your GitHub repository.** After the pipeline has run on the
real download once:

```bash
python -m rdd sample                 # picks ~300 class- and country-balanced images (<= 60 MB)
git add data/sample && git commit -m "Add RDD2022 sample (CC BY-SA 4.0)"
```

Anyone who clones the repository can then run `python -m rdd split --use-sample` and
`python -m rdd train` without downloading anything. `sample` refuses to run on
synthetic data, so fake images can never be published as a "sample of RDD2022". If you
need the whole dataset under version control, use Git LFS or a release asset instead of
plain git (GitHub blocks files over 100 MB, and LFS free storage is 1 GB).

**Licenses.** RDD2022 is CC BY-SA 4.0: keep the attribution in `data/sample/README.md`
and [`NOTICE.md`](NOTICE.md) and cite the paper. The code is MIT. Ultralytics is AGPL-3.0;
see `NOTICE.md` before any commercial use.

## Method notes

- **Four classes.** The challenge scores D00/D10/D20/D40. The dataset's other codes
  (D01, D11, D43, D44) are rare and ambiguous; they are dropped and reported by
  `convert` under "Ignored damage codes".
- **Damage-free images** are kept as background examples (default 15% of the number of
  damaged images, sampled per country with a fixed seed) to reduce false positives on
  clean asphalt. Images under a `train/` folder without an XML file count as damage-free.
  The official `test/` folders have no annotations, so they are never used.
- **Split.** 70/15/15, stratified by country and by damaged/damage-free so every split
  sees every country. Files are hard-linked rather than copied, which saves disk space
  (it falls back to copying where linking is impossible).
- **Possible leakage.** RDD2022 images come from vehicles driving along roads, so
  neighbouring image numbers can be near-duplicates. Whether that is the case depends on
  how the authors numbered the files, which this repository does not assume. If your
  test scores look suspiciously high, set `split.group_size` in `config.yaml` (for
  example 20) to keep consecutive image numbers in the same split and compare.
- **Per-class metrics** are mapped through Ultralytics' `ap_class_index`, so a class
  that is absent from a subset (for instance one country's test images) cannot shift
  the other classes' numbers onto the wrong names.
- **Augmentation.** Rotation and shear are small and vertical flips are off, since
  crack direction is what separates a longitudinal from a transverse crack.
- **Image size.** The default countries use 600-720 px images, so `imgsz: 640` is a
  good fit. Norway's images are 3650x2044: downscaling them to 640 px erases thin
  cracks, so train on them with a larger `--imgsz` (and more VRAM) or leave them out.
- **Reporting.** A fair write-up should give the per-class and per-country table from
  `reports/metrics.md`, not only the overall mAP50, and say which countries were used.

## Project layout

```
config.yaml            defaults for countries, conversion, split and training
rdd/
  constants.py         classes, country archive URLs, directory layout
  download.py          resumable download, zip verification, safe extraction
  convert.py           Pascal VOC XML -> YOLO txt (linear-time pairing)
  split.py             stratified split + data.yaml
  stats.py             dataset tables and figures
  sample.py            GitHub-sized real-data sample
  train.py             Ultralytics training wrapper
  evaluate.py          overall and per-country evaluation, markdown/JSON reports
  predict.py           inference on images/folders/videos
  demo_data.py         synthetic data in the RDD2022 layout (not the real dataset)
  smoke.py             end-to-end check on synthetic data
  doctor.py            environment diagnostics
  cli.py               `python -m rdd ...`
tests/                 pytest suite (download resume, conversion math, split, CLI, ...)
data/sample/           committed real-data sample (after you run `rdd sample`)
weights/               optional place to commit a final best.pt
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest -m "not slow"      # seconds
pytest                    # also trains a tiny model on the CPU (about a minute)
```

GitHub Actions (`.github/workflows/ci.yml`) runs the same on every push.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Training is very slow, `doctor` says no GPU | CPU-only PyTorch; see [NVIDIA GPU](#nvidia-gpu-important-on-windows) |
| `CUDA out of memory` | `--batch 8` (or 4), a smaller model (`yolo11n.pt`), or `--imgsz 512` |
| Windows: "The paging file is too small" | `--workers 0` or `--workers 2` |
| `HTTP 403` / blocked download | download the zips in a browser, see [Train on the real dataset](#train-on-the-real-dataset) |
| `data.yaml` points to the wrong folder after moving the repo | run `python -m rdd split` again (it rewrites the absolute path) |
| Want to start over | delete `data/converted`, `data/yolo_dataset` (and `runs/`), keep `data/raw` |

## Citation

If you use this work, cite the dataset:

```bibtex
@article{arya2022rdd2022,
  title   = {RDD2022: A multi-national image dataset for automatic Road Damage Detection},
  author  = {Arya, Deeksha and Maeda, Hiroya and Ghosh, Sanjay Kumar and Toshniwal, Durga and Sekimoto, Yoshihide},
  journal = {arXiv preprint arXiv:2209.08538},
  year    = {2022}
}
```
