"""Compare slim ONNX outputs and decoding with the original YOLOv8 head."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils.ops import non_max_suppression

from decode_slim import decode, decode_anchor_outputs


def load_test_image(path: str, imgsz: int) -> torch.Tensor:
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"failed to read image: {path}")

    letterbox = LetterBox(
        new_shape=(imgsz, imgsz),
        auto=False,
        scaleFill=False,
        scaleup=True,
        center=True,
        stride=32,
    )
    image = letterbox(image=image)
    if isinstance(image, tuple):
        image = image[0]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.ascontiguousarray(image.transpose(2, 0, 1))
    return torch.from_numpy(image).float().div_(255.0).unsqueeze(0)


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    converted = boxes.copy()
    converted[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    converted[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    converted[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    converted[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return converted


def match_detections(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> tuple[int, float, float]:
    if len(reference) == 0 and len(candidate) == 0:
        return 0, 0.0, 0.0
    if len(reference) != len(candidate):
        return min(len(reference), len(candidate)), float("inf"), float("inf")

    unmatched = set(range(len(candidate)))
    max_box_error = 0.0
    max_score_error = 0.0
    for reference_detection in reference:
        if not unmatched:
            break
        best_index = min(
            unmatched,
            key=lambda index: float(
                np.abs(candidate[index, :4] - reference_detection[:4]).max()
            ),
        )
        unmatched.remove(best_index)
        candidate_detection = candidate[best_index]
        if int(candidate_detection[5]) != int(reference_detection[5]):
            return min(len(reference), len(candidate)), float("inf"), float("inf")
        max_box_error = max(
            max_box_error,
            float(np.abs(candidate_detection[:4] - reference_detection[:4]).max()),
        )
        max_score_error = max(
            max_score_error,
            abs(float(candidate_detection[4]) - float(reference_detection[4])),
        )
    return len(reference), max_box_error, max_score_error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--image",
        default=None,
        help="optional real image; otherwise use a deterministic random tensor",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.image:
        sample = load_test_image(args.image, args.imgsz)
        input_source = str(Path(args.image).resolve())
    else:
        sample = torch.rand(
            1,
            3,
            args.imgsz,
            args.imgsz,
            dtype=torch.float32,
        )
        input_source = f"random:seed={args.seed}"

    model = YOLO(args.weights)
    model.model.eval()
    detector = model.model.model[-1]
    detector.export = True
    detector.format = "onnx"
    with torch.inference_mode():
        fused = model.model(sample)
    if isinstance(fused, (tuple, list)):
        fused = fused[0]
    fused = fused.detach().cpu().numpy()

    session = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    output_names = [value.name for value in session.get_outputs()]
    reg, cls = session.run(
        output_names,
        {session.get_inputs()[0].name: sample.numpy()},
    )

    num_classes = fused.shape[1] - 4
    reference_boxes = xywh_to_xyxy(fused[0, :4].T)
    reference_scores = fused[0, 4:].T
    reference_best_scores = reference_scores.max(axis=1)

    all_boxes, all_scores, _ = decode_anchor_outputs(
        reg,
        cls,
        args.imgsz,
        args.imgsz,
        [8, 16, 32],
    )
    decoded_boxes, decoded_scores, decoded_classes = decode(
        reg,
        cls,
        args.imgsz,
        args.imgsz,
        [8, 16, 32],
        args.conf,
        args.iou,
        300,
    )

    reference_nms = non_max_suppression(
        torch.from_numpy(fused),
        conf_thres=args.conf,
        iou_thres=args.iou,
        max_det=300,
        nc=num_classes,
    )[0].numpy()
    candidate_nms = np.concatenate(
        (
            decoded_boxes,
            decoded_scores[:, None],
            decoded_classes[:, None].astype(np.float32),
        ),
        axis=1,
    )

    matched, nms_box_error, nms_score_error = match_detections(
        reference_nms,
        candidate_nms,
    )
    report = {
        "weights": str(Path(args.weights).resolve()),
        "onnx": str(Path(args.onnx).resolve()),
        "input_source": input_source,
        "input_shape": list(sample.shape),
        "num_anchors": int(reg.shape[2]),
        "num_classes": num_classes,
        "anchor_box_max_abs_error": float(
            np.max(np.abs(reference_boxes - all_boxes))
        ),
        "anchor_score_max_abs_error": float(
            np.max(np.abs(reference_best_scores - all_scores))
        ),
        "nms_reference_count": int(len(reference_nms)),
        "nms_candidate_count": int(len(candidate_nms)),
        "nms_matched_count": int(matched),
        "nms_box_max_abs_error": nms_box_error,
        "nms_score_max_abs_error": nms_score_error,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
