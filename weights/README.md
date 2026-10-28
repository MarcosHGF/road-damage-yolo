# Trained weights

Put your final `best.pt` here if you want to version it (a YOLO11s checkpoint is
about 19 MB, well below GitHub's 100 MB per-file limit). Everything else named
`*.pt` is git-ignored.

Use it with:

    python -m rdd evaluate --weights weights/best.pt
    python -m rdd predict  --weights weights/best.pt --source path/to/photo.jpg
