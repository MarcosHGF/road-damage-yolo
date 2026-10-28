# data/

| Folder | Created by | Committed? |
|---|---|---|
| `downloads/` | `python -m rdd download` (zip archives) | no |
| `raw/` | `download` (extracted RDD2022, Pascal VOC XML) | no |
| `converted/` | `python -m rdd convert` (YOLO labels) | no |
| `yolo_dataset/` | `python -m rdd split` (train/val/test + `data.yaml`) | no |
| `sample/` | `python -m rdd sample` (a few hundred real images, <= 60 MB) | **yes** |

GitHub rejects files over 100 MB and asks repositories to stay far below 1 GB,
so the full dataset (about 2 GB for the default three countries) is downloaded
by the script rather than stored in the repository. `data/sample/` is the part
that is committed, so a fresh clone can train and evaluate right away:

    python -m rdd split --use-sample
    python -m rdd train --epochs 50
