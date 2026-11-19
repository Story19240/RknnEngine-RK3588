"""
ONNX -> RKNN for Rockchip NPU boards (rk3566/rk3568/rk3588/...).

rknn-toolkit2 runs on an x86_64 Linux PC (not on the board, not on Windows):
    pip install -r requirements.txt        # rknn-toolkit2, numpy, opencv-python

Both toolkit generations are supported, picked automatically from the installed
version, because the .rknn must match the BOARD's librknnrt:
  - toolkit2 2.x   -> board librknnrt 2.x  (needs NPU driver >= 0.9.x)
  - toolkit2 1.5.0 -> board librknnrt 1.5.0 (driver 0.8.8 is fine, e.g. RK3588
                      boards on the old firmware reject 2.x models)

INT8 is strongly recommended for the NPU (best speed). Quantization needs
a few hundred representative images; see prepare_quant_dataset.py.

Usage:
    python prepare_quant_dataset.py --src /data/calib --num 200
    python to_rknn.py --onnx best.onnx --dataset dataset.txt --platform rk3588
    # -> best.rknn

FP16 (no calibration set needed, ~2x slower on NPU):
    python to_rknn.py --onnx best.onnx --quantized FP16 --platform rk3588
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
import sys

from rknn.api import RKNN


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_calibration_dataset(path: str) -> int:
    """Fail early when the dataset list references missing container paths."""
    image_paths = []
    with open(path, "r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            value = line.strip()
            if value:
                image_paths.append((line_number, value))

    if not image_paths:
        raise ValueError(f"calibration dataset is empty: {path}")

    missing = [
        (line_number, image_path)
        for line_number, image_path in image_paths
        if not os.path.isfile(image_path)
    ]
    if missing:
        examples = ", ".join(
            f"line {line_number}: {image_path}"
            for line_number, image_path in missing[:3]
        )
        raise ValueError(
            f"calibration dataset contains {len(missing)} missing image(s); "
            f"ensure the image directory is mounted in the container ({examples})"
        )
    return len(image_paths)


def toolkit_version() -> str:
    try:
        return importlib.metadata.version("rknn-toolkit2")
    except Exception:
        return "unknown"


def toolkit_major() -> int:
    try:
        return int(toolkit_version().split(".")[0])
    except Exception:
        return 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--platform", default="rk3568",
                    choices=["rk3566", "rk3568", "rk3562", "rk3588"])
    ap.add_argument("--dataset", default="dataset.txt",
                    help="image path list, only used for INT8")
    ap.add_argument("--quantized", choices=["INT8", "FP16"], default="INT8")
    ap.add_argument("--optimization-level", type=int, default=3, choices=[1, 2, 3],
                    help="1 = fewest graph rewrites (safest for old NPU drivers, "
                         "e.g. 0.8.8), 3 = most aggressive")
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--manifest",
        default=None,
        help="optional JSON file with conversion settings and checksums",
    )
    ap.add_argument("--mean", default="0,0,0",
                    help="per-channel mean, matches training normalization")
    ap.add_argument("--std", default="255,255,255",
                    help="per-channel std; 255,255,255 <=> divide-by-255")
    args = ap.parse_args()
    print(f"rknn-toolkit2 version: {toolkit_version()}")

    out = args.out or os.path.splitext(args.onnx)[0] + ".rknn"
    mean = [[float(x) for x in args.mean.split(",")]]
    std = [[float(x) for x in args.std.split(",")]]

    if args.quantized == "INT8":
        if not os.path.isfile(args.dataset):
            raise SystemExit(
                f"INT8 needs a calibration dataset file: {args.dataset}"
            )
        try:
            calibration_count = validate_calibration_dataset(args.dataset)
        except (OSError, UnicodeError, ValueError) as error:
            raise SystemExit(str(error)) from error
        print(f"[OK] calibration images checked: {calibration_count}")

    rknn = RKNN(verbose=True)

    cfg = dict(
        mean_values=mean,
        std_values=std,                       # baked into model: feed raw uint8 at runtime
        target_platform=args.platform,
        optimization_level=args.optimization_level,
    )
    if toolkit_major() >= 2:
        # toolkit2 2.x naming: per-channel INT8 / DFP fp16, plus quantized_method
        cfg["quantized_dtype"] = (
            "w8a8" if args.quantized == "INT8" else "w16a16i_dfp"
        )
        cfg["quantized_method"] = "channel"
    else:
        # toolkit2 1.x (e.g. 1.5.0): only asymmetric 8/16-bit quantization,
        # no quantized_method arg. 1.5.0 has no "fp16" dtype name -- the
        # 16-bit asymmetric mode is its FP16-equivalent.
        cfg["quantized_dtype"] = (
            "asymmetric_quantized-8" if args.quantized == "INT8"
            else "asymmetric_quantized-16"
        )
    try:
        ret = rknn.config(**cfg)
        if ret != 0:
            print(f"config failed: {ret}")
            return ret

        ret = rknn.load_onnx(model=args.onnx)
        if ret != 0:
            print(f"load_onnx failed: {ret}")
            return ret

        if args.quantized == "INT8":
            ret = rknn.build(do_quantization=True, dataset=args.dataset)
        else:
            ret = rknn.build(do_quantization=False)
        if ret != 0:
            print(f"build failed: {ret}")
            return ret

        ret = rknn.export_rknn(out)
        if ret != 0:
            print(f"export_rknn failed: {ret}")
            return ret
    finally:
        rknn.release()

    print(f"[OK] RKNN exported -> {out} (target={args.platform}, {args.quantized}, "
          f"opt_level={args.optimization_level}, toolkit2={toolkit_version()})")

    if args.manifest:
        manifest_path = os.path.abspath(args.manifest)
        manifest_dir = os.path.dirname(manifest_path)
        if manifest_dir:
            os.makedirs(manifest_dir, exist_ok=True)
        manifest = {
            "source_onnx": os.path.abspath(args.onnx),
            "source_onnx_sha256": sha256_file(args.onnx),
            "dataset": (
                os.path.abspath(args.dataset)
                if args.quantized == "INT8"
                else None
            ),
            "dataset_sha256": (
                sha256_file(args.dataset)
                if args.quantized == "INT8"
                else None
            ),
            "output_rknn": os.path.abspath(out),
            "output_rknn_sha256": sha256_file(out),
            "output_rknn_size": os.path.getsize(out),
            "target_platform": args.platform,
            "quantized": args.quantized,
            "quantized_dtype": cfg["quantized_dtype"],
            "optimization_level": args.optimization_level,
            "mean_values": mean,
            "std_values": std,
            "toolkit2_version": toolkit_version(),
        }
        with open(manifest_path, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        print(f"[OK] manifest -> {manifest_path}")


if __name__ == "__main__":
    sys.exit(main())
