"""Reference decoder for YOLOv8 slim-head reg/cls outputs.

This module intentionally uses only NumPy. It is the PC-side oracle for the
Android native decoder and can consume ONNX Runtime or RKNN raw outputs saved
as .npy/.npz files.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np


def load_array(path: str, key: str | None = None) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".npz":
        with np.load(path) as archive:
            if key is None:
                if len(archive.files) != 1:
                    raise ValueError(
                        f"{path} contains multiple arrays {archive.files}; specify --key"
                    )
                key = archive.files[0]
            if key not in archive:
                raise KeyError(f"{key!r} is not present in {path}")
            return np.asarray(archive[key])
    return np.asarray(np.load(path))


def channels_first(values: np.ndarray, channels: int, name: str) -> np.ndarray:
    """Return a [anchors, channels] view for NCHW or NHWC input."""
    if values.ndim != 3 or values.shape[0] != 1:
        raise ValueError(f"{name} must have shape [1, ...], got {values.shape}")
    if values.shape[1] == channels:
        return values[0].T
    if values.shape[2] == channels:
        return values[0]
    raise ValueError(
        f"cannot identify {name} channel axis {channels} in shape {values.shape}"
    )


def channels_first_for_anchors(
    values: np.ndarray,
    anchor_count: int,
    name: str,
) -> np.ndarray:
    """Return a [anchors, channels] view using the known anchor dimension."""
    if values.ndim != 3 or values.shape[0] != 1:
        raise ValueError(f"{name} must have shape [1, ...], got {values.shape}")
    if values.shape[2] == anchor_count:
        return values[0].T
    if values.shape[1] == anchor_count:
        return values[0]
    raise ValueError(
        f"cannot identify {name} anchor axis {anchor_count} in shape {values.shape}"
    )


def make_anchors(input_width: int, input_height: int, strides: list[int]) -> np.ndarray:
    anchors = []
    for stride in strides:
        height = input_height // stride
        width = input_width // stride
        y, x = np.meshgrid(
            np.arange(height, dtype=np.float32),
            np.arange(width, dtype=np.float32),
            indexing="ij",
        )
        centers = np.stack((x + 0.5, y + 0.5), axis=-1).reshape(-1, 2)
        anchors.append(
            np.concatenate(
                (
                    centers,
                    np.full((centers.shape[0], 1), stride, dtype=np.float32),
                ),
                axis=1,
            )
        )
    return np.concatenate(anchors, axis=0)


def sigmoid(values: np.ndarray) -> np.ndarray:
    output = np.empty_like(values, dtype=np.float32)
    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponent = np.exp(values[~positive])
    output[~positive] = exponent / (1.0 + exponent)
    return output


def decode_dfl(reg: np.ndarray) -> np.ndarray:
    """Decode [anchors, 64] logits into four grid-unit distances."""
    if reg.shape[1] % 4:
        raise ValueError(f"reg channel count must be divisible by 4, got {reg.shape[1]}")
    bins = reg.shape[1] // 4
    distributions = reg.reshape(reg.shape[0], 4, bins).astype(np.float64)
    distributions -= np.max(distributions, axis=2, keepdims=True)
    distributions = np.exp(distributions)
    distributions /= np.sum(distributions, axis=2, keepdims=True)
    weights = np.arange(bins, dtype=np.float64)
    return np.sum(distributions * weights, axis=2).astype(np.float32)


def box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.empty((0,), dtype=np.float32)
    top_left = np.maximum(box[:2], boxes[:, :2])
    bottom_right = np.minimum(box[2:], boxes[:, 2:])
    intersection = np.maximum(bottom_right - top_left, 0.0)
    intersection_area = intersection[:, 0] * intersection[:, 1]
    box_area = max(box[2] - box[0], 0.0) * max(box[3] - box[1], 0.0)
    boxes_area = np.maximum(boxes[:, 2] - boxes[:, 0], 0.0) * np.maximum(
        boxes[:, 3] - boxes[:, 1], 0.0
    )
    union = box_area + boxes_area - intersection_area
    return intersection_area / np.maximum(union, 1e-12)


def class_aware_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    iou_threshold: float,
    max_detections: int,
) -> list[int]:
    if boxes.size == 0:
        return []
    order = np.argsort(-scores, kind="stable")
    kept: list[int] = []
    while order.size:
        current = int(order[0])
        kept.append(current)
        if len(kept) >= max_detections:
            break
        remaining = order[1:]
        same_class = classes[remaining] == classes[current]
        overlaps = np.zeros(remaining.shape, dtype=np.float32)
        if np.any(same_class):
            overlaps[same_class] = box_iou(boxes[current], boxes[remaining[same_class]])
        order = remaining[overlaps <= iou_threshold]
    return kept


def decode(
    reg: np.ndarray,
    cls: np.ndarray,
    input_width: int,
    input_height: int,
    strides: list[int],
    confidence_threshold: float,
    iou_threshold: float,
    max_detections: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    boxes, confidences, class_ids = decode_anchor_outputs(
        reg,
        cls,
        input_width,
        input_height,
        strides,
    )

    selected = confidences > confidence_threshold
    boxes = boxes[selected]
    confidences = confidences[selected]
    class_ids = class_ids[selected]

    kept = class_aware_nms(
        boxes,
        confidences,
        class_ids,
        iou_threshold,
        max_detections,
    )
    return boxes[kept], confidences[kept], class_ids[kept]


def decode_anchor_outputs(
    reg: np.ndarray,
    cls: np.ndarray,
    input_width: int,
    input_height: int,
    strides: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reg_values = channels_first(reg, 64, "reg")
    cls_values = channels_first_for_anchors(cls, reg_values.shape[0], "cls")
    if reg_values.shape[0] != cls_values.shape[0]:
        raise ValueError(
            f"reg and cls anchor counts differ: {reg_values.shape[0]} and "
            f"{cls_values.shape[0]}"
        )

    anchors = make_anchors(input_width, input_height, strides)
    if anchors.shape[0] != reg_values.shape[0]:
        raise ValueError(
            f"anchor count mismatch: model has {reg_values.shape[0]}, "
            f"input {input_width}x{input_height} with strides {strides} "
            f"requires {anchors.shape[0]}"
        )

    distances = decode_dfl(reg_values)
    left, top, right, bottom = distances.T
    boxes = np.stack(
        (
            (anchors[:, 0] - left) * anchors[:, 2],
            (anchors[:, 1] - top) * anchors[:, 2],
            (anchors[:, 0] + right) * anchors[:, 2],
            (anchors[:, 1] + bottom) * anchors[:, 2],
        ),
        axis=1,
    )

    probabilities = sigmoid(cls_values)
    class_ids = np.argmax(probabilities, axis=1)
    confidences = probabilities[np.arange(probabilities.shape[0]), class_ids]
    return boxes, confidences, class_ids


def restore_source_coordinates(
    boxes: np.ndarray,
    source_width: int,
    source_height: int,
    input_width: int,
    input_height: int,
) -> tuple[np.ndarray, float, int, int]:
    scale = min(input_width / source_width, input_height / source_height)
    resized_width = max(1, math.floor(source_width * scale + 0.5))
    resized_height = max(1, math.floor(source_height * scale + 0.5))
    padding_left = (input_width - resized_width) // 2
    padding_top = (input_height - resized_height) // 2

    restored = boxes.copy()
    restored[:, 0::2] = (restored[:, 0::2] - padding_left) / scale
    restored[:, 1::2] = (restored[:, 1::2] - padding_top) / scale
    restored[:, 0::2] = np.clip(restored[:, 0::2], 0, source_width)
    restored[:, 1::2] = np.clip(restored[:, 1::2], 0, source_height)
    return restored, scale, padding_left, padding_top


def load_class_names(path: str | None) -> dict[int, str]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if isinstance(payload, list):
        return {index: str(name) for index, name in enumerate(payload)}
    if isinstance(payload, dict):
        return {int(index): str(name) for index, name in payload.items()}
    raise ValueError("class-name JSON must be a list or an index-to-name object")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reg", required=True, help="reg output .npy/.npz")
    parser.add_argument("--cls", required=True, help="cls output .npy/.npz")
    parser.add_argument("--reg-key", default=None)
    parser.add_argument("--cls-key", default=None)
    parser.add_argument("--input-width", type=int, default=640)
    parser.add_argument("--input-height", type=int, default=640)
    parser.add_argument("--strides", default="8,16,32")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--source-width", type=int, default=None)
    parser.add_argument("--source-height", type=int, default=None)
    parser.add_argument("--classes", default=None, help="JSON class-name list/map")
    parser.add_argument("--output", default=None, help="optional JSON result file")
    args = parser.parse_args()

    strides = [int(value) for value in args.strides.split(",")]
    reg = load_array(args.reg, args.reg_key)
    cls = load_array(args.cls, args.cls_key)
    boxes, confidences, class_ids = decode(
        reg,
        cls,
        args.input_width,
        args.input_height,
        strides,
        args.conf,
        args.iou,
        args.max_det,
    )

    source_width = args.source_width or args.input_width
    source_height = args.source_height or args.input_height
    boxes, scale, padding_left, padding_top = restore_source_coordinates(
        boxes,
        source_width,
        source_height,
        args.input_width,
        args.input_height,
    )
    class_names = load_class_names(args.classes)
    detections = [
        {
            "class_id": int(class_id),
            "class_name": class_names.get(int(class_id), str(int(class_id))),
            "confidence": float(confidence),
            "box": [float(value) for value in box],
        }
        for box, confidence, class_id in zip(boxes, confidences, class_ids)
    ]
    payload = {
        "source_width": source_width,
        "source_height": source_height,
        "input_width": args.input_width,
        "input_height": args.input_height,
        "letterbox": {
            "scale": scale,
            "padding_left": padding_left,
            "padding_top": padding_top,
        },
        "detections": detections,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
