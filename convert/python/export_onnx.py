"""Export a YOLOv8 checkpoint to ONNX for RKNN Toolkit2 conversion.

The default ``slim`` mode replaces the Detect head with a raw two-output head:

    output_reg = [1, 64, num_anchors]
    output_cls = [1, num_classes, num_anchors]

DFL, anchor generation, and classification sigmoid are intentionally left on
the CPU. This avoids the long anchor broadcasts that fail to submit on the
target RK3588's 0.8.8 NPU driver.

The legacy ``fused`` mode remains available for comparison. It exports the
normal Ultralytics Detect output and splits it into ``output_box`` and
``output_cls`` for debugging only.
"""
import argparse
import hashlib
import json
import os
import shutil
import tempfile

import onnx
from onnx import TensorProto, helper
import torch
from torch import nn
from ultralytics import YOLO


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SlimDetect(nn.Module):
    """Ultralytics YOLOv8 Detect head without DFL, anchors, or sigmoid."""

    def __init__(self, detect: nn.Module):
        super().__init__()
        self.detect = detect

    def forward(self, features):
        outputs = []
        for index in range(self.detect.nl):
            reg = self.detect.cv2[index](features[index])
            cls = self.detect.cv3[index](features[index])
            outputs.append(
                torch.cat((reg, cls), 1).view(reg.shape[0], self.detect.no, -1)
            )

        combined = torch.cat(outputs, 2)
        reg_channels = self.detect.reg_max * 4
        return combined[:, :reg_channels], combined[:, reg_channels:]


def replace_detect_head(model: YOLO) -> SlimDetect:
    detector = model.model.model[-1]
    required = ("cv2", "cv3", "nl", "nc", "no", "reg_max")
    missing = [name for name in required if not hasattr(detector, name)]
    if missing:
        raise SystemExit(
            "checkpoint does not expose the expected YOLOv8 Detect head; "
            f"missing attributes: {', '.join(missing)}"
        )

    slim = SlimDetect(detector)
    for attribute in ("f", "i", "type"):
        if hasattr(detector, attribute):
            setattr(slim, attribute, getattr(detector, attribute))
    model.model.model[-1] = slim
    return slim


def rename_onnx_outputs(onnx_path: str, names: list[str]) -> None:
    model = onnx.load(onnx_path)
    graph = model.graph
    if len(graph.output) != len(names):
        raise SystemExit(
            f"expected {len(names)} ONNX outputs, got {len(graph.output)}"
        )

    original_names = [value.name for value in graph.output]
    for node in graph.node:
        for index, output in enumerate(node.output):
            if output in original_names:
                node.output[index] = names[original_names.index(output)]
    for index, value in enumerate(graph.output):
        value.name = names[index]

    onnx.checker.check_model(model)
    onnx.save(model, onnx_path)


def tensor_shape(value_info) -> list[int]:
    return [dimension.dim_value for dimension in value_info.type.tensor_type.shape.dim]


def validate_slim_onnx(
    onnx_path: str,
    imgsz: int,
    num_classes: int,
    expected_anchors: int,
) -> dict:
    model = onnx.load(onnx_path)
    graph = model.graph
    onnx.checker.check_model(model)

    if len(graph.input) != 1 or len(graph.output) != 2:
        raise SystemExit(
            "slim ONNX must have one input and two outputs; "
            f"got {len(graph.input)} input(s), {len(graph.output)} output(s)"
        )

    input_shape = tensor_shape(graph.input[0])
    expected_input = [1, 3, imgsz, imgsz]
    if input_shape != expected_input:
        raise SystemExit(
            f"unexpected slim input shape {input_shape}; wanted {expected_input}"
        )

    output_names = [value.name for value in graph.output]
    output_shapes = {value.name: tensor_shape(value) for value in graph.output}
    expected_outputs = {
        "output_reg": [1, 64, expected_anchors],
        "output_cls": [1, num_classes, expected_anchors],
    }
    for name, expected_shape in expected_outputs.items():
        if name not in output_shapes:
            raise SystemExit(
                f"slim ONNX is missing output {name}; got {output_names}"
            )
        if output_shapes[name] != expected_shape:
            raise SystemExit(
                f"unexpected {name} shape {output_shapes[name]}; "
                f"wanted {expected_shape}"
            )

    softmax_count = sum(node.op_type == "Softmax" for node in graph.node)
    if softmax_count:
        raise SystemExit(
            f"slim ONNX still contains {softmax_count} Softmax node(s); "
            "the DFL softmax must remain on the CPU"
        )

    return {
        "input_shape": input_shape,
        "output_names": output_names,
        "output_shapes": output_shapes,
        "softmax_count": softmax_count,
    }


def verify_with_onnxruntime(
    onnx_path: str,
    model: YOLO,
    imgsz: int,
) -> dict[str, float]:
    try:
        import onnxruntime as ort
    except ImportError:
        print("[warn] onnxruntime is unavailable; skipped numeric verification")
        return {}

    torch.manual_seed(42)
    sample = torch.rand(1, 3, imgsz, imgsz, dtype=torch.float32)
    model.model.eval()
    with torch.inference_mode():
        reg, cls = model.model(sample)

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    reg_name, cls_name = [value.name for value in session.get_outputs()]
    actual_reg, actual_cls = session.run(
        [reg_name, cls_name],
        {session.get_inputs()[0].name: sample.numpy()},
    )

    reg_error = float(np_max_abs_error(reg.numpy(), actual_reg))
    cls_error = float(np_max_abs_error(cls.numpy(), actual_cls))
    print(
        f"[verify] ONNX Runtime max abs error: "
        f"reg={reg_error:.6e}, cls={cls_error:.6e}"
    )
    return {"reg_max_abs_error": reg_error, "cls_max_abs_error": cls_error}


def np_max_abs_error(expected, actual):
    import numpy as np

    return np.max(np.abs(expected - actual))


def split_fused_output(onnx_path: str, out_path: str) -> tuple[int, int]:
    """Split the fused [1, 4+nc, anchors] output into box + cls outputs.

    Inserts two ONNX Slice nodes that slice the fused tensor along axis 1,
    turns them into the two new graph outputs, and (best-effort) re-runs
    onnxsim so the graph stays clean. Returns (num_classes, num_anchors).
    """
    model = onnx.load(onnx_path)
    graph = model.graph

    if len(graph.output) != 1:
        raise SystemExit(
            f"expected 1 fused output to split, got {len(graph.output)}; "
            "is this already a split model?"
        )

    fused_vi = graph.output[0]
    fused_name = fused_vi.name
    shape = [d.dim_value for d in fused_vi.type.tensor_type.shape.dim]
    # Expected: [1, 4+nc, anchors], concrete because dynamic=False.
    if len(shape) != 3 or shape[0] != 1 or shape[1] < 5 or shape[2] <= 0:
        raise SystemExit(f"unexpected fused output shape {shape}; wanted [1,5+,anchors]")
    k = shape[1]
    anchors = shape[2]
    nc = k - 4

    box_name = "output_box"
    cls_name = "output_cls"

    # Slice opset>=10 takes 5 inputs: data, starts, ends, axes, steps (all 1-D).
    starts_box = helper.make_tensor("starts_box", TensorProto.INT64, [1], [0])
    ends_box   = helper.make_tensor("ends_box",   TensorProto.INT64, [1], [4])
    starts_cls = helper.make_tensor("starts_cls", TensorProto.INT64, [1], [4])
    ends_cls   = helper.make_tensor("ends_cls",   TensorProto.INT64, [1], [k])
    axes       = helper.make_tensor("slice_axes", TensorProto.INT64, [1], [1])
    steps      = helper.make_tensor("slice_steps", TensorProto.INT64, [1], [1])
    graph.initializer.extend(
        [starts_box, ends_box, starts_cls, ends_cls, axes, steps]
    )

    node_box = helper.make_node(
        "Slice",
        [fused_name, "starts_box", "ends_box", "slice_axes", "slice_steps"],
        [box_name],
        name="slice_box",
    )
    node_cls = helper.make_node(
        "Slice",
        [fused_name, "starts_cls", "ends_cls", "slice_axes", "slice_steps"],
        [cls_name],
        name="slice_cls",
    )
    graph.node.extend([node_box, node_cls])

    box_vi = helper.make_tensor_value_info(box_name, TensorProto.FLOAT, [1, 4, anchors])
    cls_vi = helper.make_tensor_value_info(cls_name, TensorProto.FLOAT, [1, nc, anchors])
    del graph.output[:]
    graph.output.extend([box_vi, cls_vi])

    onnx.checker.check_model(model)

    # Best-effort re-simplify: folds the (now trivial) fused/slice into a tidy
    # graph. Never fatal -- the un-simplified model is already valid and loadable.
    try:
        import onnxsim
        model, _ = onnxsim.simplify(model)
        onnx.checker.check_model(model)
    except Exception as e:
        print(f"[warn] onnxsim after split skipped: {e}")

    onnx.save(model, out_path)
    return nc, anchors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolov8n.pt", help="path to .pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--opset", type=int, default=12)
    ap.add_argument(
        "--simplify",
        type=int,
        default=0,
        choices=[0, 1],
        help="optional onnxsim pass; slim export does not require it",
    )
    ap.add_argument(
        "--head-mode",
        choices=["slim", "fused"],
        default="slim",
        help="slim outputs raw reg/cls; fused keeps the legacy split path",
    )
    ap.add_argument("--out", default="yolov8n.onnx")
    ap.add_argument(
        "--metadata-out",
        default=None,
        help="optional JSON file with model and output metadata",
    )
    ap.add_argument(
        "--verify",
        type=int,
        default=1,
        choices=[0, 1],
        help="compare PyTorch and ONNX Runtime outputs",
    )
    args = ap.parse_args()

    if not os.path.isfile(args.weights):
        raise SystemExit(f"weights not found: {args.weights}")
    if not os.path.isdir(os.path.dirname(os.path.abspath(args.out))):
        raise SystemExit(
            f"output directory does not exist: {os.path.dirname(os.path.abspath(args.out))}"
        )

    # Export from a writable copy of the weights: Ultralytics writes the ONNX
    # next to the checkpoint by default, but the checkpoint is often read-only.
    tmpdir = tempfile.mkdtemp(prefix="yolo_export_")
    local_w = os.path.join(tmpdir, os.path.basename(args.weights))
    shutil.copy(args.weights, local_w)
    model = YOLO(local_w)
    try:
        detector = model.model.model[-1]
        if args.head_mode == "slim":
            if not hasattr(detector, "reg_max"):
                raise SystemExit(
                    "slim-head export requires a YOLOv8 Detect head with reg_max"
                )
            num_classes = int(detector.nc)
            strides = [int(value) for value in detector.stride.tolist()]
            if any(args.imgsz % stride for stride in strides):
                raise SystemExit(
                    f"input size {args.imgsz} must be divisible by strides {strides}"
                )
            expected_anchors = sum(
                (args.imgsz // stride) ** 2 for stride in strides
            )
            replace_detect_head(model)
        else:
            num_classes = int(detector.nc)
            expected_anchors = 0

        path = model.export(
            format="onnx",
            imgsz=args.imgsz,
            opset=args.opset,
            simplify=bool(args.simplify),
            dynamic=False,
        )

        path = str(path)
        if os.path.abspath(path) != os.path.abspath(args.out):
            if os.path.exists(args.out):
                os.remove(args.out)
            shutil.move(path, args.out)

        if args.head_mode == "slim":
            rename_onnx_outputs(args.out, ["output_reg", "output_cls"])
            validation = validate_slim_onnx(
                args.out,
                args.imgsz,
                num_classes,
                expected_anchors,
            )
            verification = (
                verify_with_onnxruntime(args.out, model, args.imgsz)
                if args.verify
                else {}
            )
            metadata = {
                "source_weights": os.path.abspath(args.weights),
                "source_weights_sha256": sha256_file(args.weights),
                "onnx": os.path.abspath(args.out),
                "onnx_sha256": sha256_file(args.out),
                "head_mode": "slim",
                "imgsz": args.imgsz,
                "opset": args.opset,
                "strides": strides,
                "num_classes": num_classes,
                "class_names": model.names,
                **validation,
                **verification,
            }
            print(f"[OK] slim ONNX exported -> {args.out}")
            print(
                f"     output_reg {validation['output_shapes']['output_reg']} + "
                f"output_cls {validation['output_shapes']['output_cls']}"
            )
        else:
            num_classes, anchors = split_fused_output(args.out, args.out)
            metadata = {
                "source_weights": os.path.abspath(args.weights),
                "source_weights_sha256": sha256_file(args.weights),
                "onnx": os.path.abspath(args.out),
                "onnx_sha256": sha256_file(args.out),
                "head_mode": "fused",
                "imgsz": args.imgsz,
                "opset": args.opset,
                "num_classes": num_classes,
                "num_anchors": anchors,
                "class_names": model.names,
            }
            print(f"[OK] fused ONNX exported -> {args.out}")
            print(
                f"     split: output_box [1,4,{anchors}] + "
                f"output_cls [1,{num_classes},{anchors}]"
            )

        if args.metadata_out:
            metadata_path = os.path.abspath(args.metadata_out)
            with open(metadata_path, "w", encoding="utf-8") as stream:
                json.dump(metadata, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            print(f"[OK] metadata -> {metadata_path}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
