"""Build a reproducible, class-balanced INT8 calibration image list."""
import argparse
import hashlib
import json
import os
import random
from collections import defaultdict
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_images(source: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in source.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ),
        key=lambda path: path.as_posix(),
    )


def class_name(source: Path, image: Path) -> str:
    relative = image.relative_to(source)
    return relative.parts[0] if len(relative.parts) > 1 else "__root__"


def select_balanced(
    source: Path,
    images: list[Path],
    total: int,
    per_class: int | None,
    seed: int,
) -> tuple[list[Path], dict[str, int]]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for image in images:
        grouped[class_name(source, image)].append(image)

    if not grouped:
        raise SystemExit(f"no images found under {source}")

    class_names = sorted(grouped)
    quota = per_class
    if quota is None:
        if total % len(class_names):
            raise SystemExit(
                f"--num {total} is not divisible by {len(class_names)} classes; "
                "use --per-class to set an explicit quota"
            )
        quota = total // len(class_names)

    selected: list[Path] = []
    counts: dict[str, int] = {}
    randomizer = random.Random(seed)
    for name in class_names:
        candidates = grouped[name]
        if len(candidates) < quota:
            raise SystemExit(
                f"class {name!r} has only {len(candidates)} images; "
                f"need {quota}"
            )
        randomizer.shuffle(candidates)
        selected.extend(candidates[:quota])
        counts[name] = quota

    randomizer.shuffle(selected)
    return selected, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="dir with calibration images")
    ap.add_argument("--num", type=int, default=200)
    ap.add_argument(
        "--per-class",
        type=int,
        default=None,
        help="images per class; defaults to --num divided by class count",
    )
    ap.add_argument("--out", default="dataset.txt")
    ap.add_argument(
        "--manifest",
        default=None,
        help="optional JSON manifest with selected paths and SHA-256 values",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    source = Path(args.src).resolve()
    if not source.is_dir():
        raise SystemExit(f"calibration directory not found: {source}")

    files = find_images(source)
    selected, counts = select_balanced(
        source,
        files,
        args.num,
        args.per_class,
        args.seed,
    )

    output = Path(args.out).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for path in selected:
            stream.write(path.as_posix() + "\n")

    print(f"[OK] wrote {len(selected)} images to {output}")
    print(
        "[OK] class counts: "
        + ", ".join(f"{name}={counts[name]}" for name in sorted(counts))
    )

    if args.manifest:
        manifest_path = Path(args.manifest).resolve()
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "source": source.as_posix(),
            "seed": args.seed,
            "total": len(selected),
            "per_class": args.per_class,
            "class_counts": counts,
            "dataset": output.as_posix(),
            "dataset_sha256": sha256_file(output),
            "images": [
                {
                    "path": path.as_posix(),
                    "class": class_name(source, path),
                    "sha256": sha256_file(path),
                }
                for path in selected
            ],
        }
        with manifest_path.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        print(f"[OK] manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
