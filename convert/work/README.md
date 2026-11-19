# Local conversion workspace

Place local weights, datasets and generated files in this directory. Everything
here except this README is excluded from Git and Docker build contexts.

```text
work/
├── best.pt                 # YOLOv8 checkpoint used by default
├── dataset/                # INT8 calibration images
│   ├── class_a/
│   │   ├── image001.jpg
│   │   └── ...
│   ├── class_b/
│   │   ├── image001.jpg
│   │   └── ...
│   └── ...
├── training-data/          # Optional local training/test data
├── output/                 # Generated ONNX, RKNN and JSON manifests
└── yolov8n.rknn            # Optional local example model
```

- Set `WEIGHTS_FILE` in `../docker/.env` to use a different checkpoint filename.
- The first-level directories under `dataset/` are treated as class names for
  balanced calibration sampling.
- FP16 conversion does not consume calibration images.
- Paths and filenames can be changed in `../docker/.env` without moving data.

The default configuration selects 200 images evenly across the class directories.
The image count must be divisible by the number of class directories. Run the
converter directly with `--per-class N` to set an explicit per-class quota.
