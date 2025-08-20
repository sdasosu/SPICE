# quanta/run.py
# Minimal, no-full-training quantization runner for SMP segmentation models.
# - PTQ: prepare_fx + calibrate a few batches (CPU) + convert_fx -> INT8
# - QAT: prepare_qat_fx (requires example_inputs) + short fine-tune + convert_fx -> INT8
# Shows: torchinfo.summary(depth=1), file size change (bytes), optional CPU latency, and quantized layer count.

import os
import sys
import argparse
from pathlib import Path
import time
import torch
import torch.nn as nn
from torchinfo import summary

# Project root (assumes this file lives at <repo>/quanta/run.py)
THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent
sys.path.insert(0, str(ROOT))

# Keep the SAME data source you already use
# If unavailable at import-time, we degrade gracefully to synthetic calibration
try:
    from data.data import get_loaders  # your project loader
    HAS_DATA = True
except Exception as _e:
    HAS_DATA = False

import segmentation_models_pytorch as smp
from quanta.quant_utils import (
    ptq_fx_convert,
    qat_fx_convert_short_train,
    save_torchscript_and_size,
    measure_cpu_latency,
    count_quantized_modules,
)

NUM_CLASSES = 5

def create_model(model_name: str, encoder_name: str=None, in_ch: int=3, num_classes: int=NUM_CLASSES,
                 pretrained: bool=False) -> nn.Module:
    m = model_name.lower()
    if m == "deeplabv3plus_resnet":
        return smp.DeepLabV3Plus(encoder_name=encoder_name or "resnet50",
                                 encoder_weights="imagenet" if pretrained else None,
                                 in_channels=in_ch, classes=num_classes, activation=None)
    if m == "deeplabv3plus_efficientnet":
        return smp.DeepLabV3Plus(encoder_name=encoder_name or "timm-efficientnet-b3",
                                 encoder_weights="imagenet" if pretrained else None,
                                 in_channels=in_ch, classes=num_classes, activation=None)
    if m == "unet_resnet":
        return smp.Unet(encoder_name=encoder_name or "resnet50",
                        encoder_weights="imagenet" if pretrained else None,
                        in_channels=in_ch, classes=num_classes, activation=None)
    if m == "unet_efficientnet":
        return smp.Unet(encoder_name=encoder_name or "timm-efficientnet-b3",
                        encoder_weights="imagenet" if pretrained else None,
                        in_channels=in_ch, classes=num_classes, activation=None)
    if m == "fpn_resnet":
        return smp.FPN(encoder_name=encoder_name or "resnet50",
                       encoder_weights="imagenet" if pretrained else None,
                       in_channels=in_ch, classes=num_classes, activation=None)
    if m == "fpn_efficientnet":
        return smp.FPN(encoder_name=encoder_name or "timm-efficientnet-b3",
                       encoder_weights="imagenet" if pretrained else None,
                       in_channels=in_ch, classes=num_classes, activation=None)
    raise ValueError(f"Unknown model: {model_name}")

def parse_args():
    p = argparse.ArgumentParser("Minimal quantization runner (no full training)")
    # Model
    p.add_argument("--model", required=True, choices=[
        "DeepLabV3Plus_resnet", "DeepLabV3Plus_efficientnet",
        "UNET_resnet", "UNET_efficientnet",
        "FPN_resnet", "FPN_efficientnet"
    ])
    p.add_argument("--encoder", type=str, default=None)
    p.add_argument("--in-ch", type=int, default=3)
    p.add_argument("--img-size", type=int, default=576)

    # Weights
    p.add_argument("--weights-pt", type=str, default=None, help="Path to FP32 .pt state_dict (preferred for quantization)")
    p.add_argument("--weights-script", type=str, default=None, help="Path to FP32 TorchScript .script (used only for baseline size/latency)")

    # Data for calibration / short QAT fine-tune
    p.add_argument("--data-root", type=str, default=str(ROOT / "data"))
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--calib-steps", type=int, default=50, help="PTQ: number of calibration batches")
    p.add_argument("--qat-steps", type=int, default=200, help="QAT: number of short training steps (very small)")

    # Method
    p.add_argument("--method", required=True, choices=["ptq", "qat"])

    # Outputs / reporting
    p.add_argument("--out-dir", type=str, default=str(ROOT / "checkpoints" / "quanta"))
    p.add_argument("--bench-cpu", action="store_true", help="Run a tiny CPU latency benchmark FP32 vs INT8")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def maybe_load_state_dict(model: nn.Module, weights_pt: str | None):
    if weights_pt and Path(weights_pt).is_file():
        print(f"[LOAD] Loading state_dict from: {weights_pt}")
        try:
            state = torch.load(weights_pt, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(weights_pt, map_location="cpu")
        model.load_state_dict(state, strict=False)
    return model

def main():
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Data (optional, but improves PTQ/QAT). If not available, we synthesize images/masks.
    if HAS_DATA:
        try:
            train_loader, val_loader, _ = get_loaders(
                data_root=args.data_root, batch_size=args.batch_size,
                num_workers=args.num_workers, img_size=args.img_size)
        except Exception as e:
            print(f"[WARN] get_loaders failed ({e}); falling back to synthetic calibration.")
            train_loader = val_loader = None
    else:
        train_loader = val_loader = None

    # Build FP32 model (CPU) and optionally load .pt state
    fp32_model = create_model(args.model, args.encoder, args.in_ch, NUM_CLASSES, pretrained=False).cpu()
    maybe_load_state_dict(fp32_model, args.weights_pt)

    example = torch.randn(1, args.in_ch, args.img_size, args.img_size)

    print("\n===== [SUMMARY] FP32 MODEL =====")
    try:
        summary(fp32_model, input_size=(1, args.in_ch, args.img_size, args.img_size),
                depth=1, col_names=("input_size", "output_size", "num_params"))
    except Exception as e:
        print(f"[WARN] torchinfo.summary failed (FP32): {e}")

    # Baseline sizes / latency
    fp32_script_path = out_dir / f"{args.model}_fp32.script"
    fp32_size = save_torchscript_and_size(fp32_model.eval(), example, fp32_script_path)
    print(f"[SIZE] FP32 TorchScript: {fp32_script_path.name} -> {fp32_size/1e6:.2f} MB")

    if args.weights_script and Path(args.weights_script).is_file():
        ws = Path(args.weights_script)
        print(f"[INFO] Provided FP32 script for reference: {ws} ({ws.stat().st_size/1e6:.2f} MB)")

    if args.method == "ptq":
        print("\n================= [PTQ] =================")
        int8_model = ptq_fx_convert(fp32_model, example, train_loader, calib_steps=args.calib_steps)
    else:
        print("\n================= [QAT] =================")
        int8_model = qat_fx_convert_short_train(fp32_model, example, train_loader, val_loader, steps=args.qat_steps)

    print("\n===== [SUMMARY] QUANTIZED MODEL (INT8 if convert succeeded) =====")
    try:
        summary(int8_model, input_size=(1, args.in_ch, args.img_size, args.img_size),
                depth=1, col_names=("input_size", "output_size", "num_params"))
    except Exception as e:
        print(f"[WARN] torchinfo.summary failed (quantized): {e}")

    # Save quantized script & compare sizes
    q_script_path = out_dir / f"{args.model}_{args.method}_int8.script"
    q_size = save_torchscript_and_size(int8_model.eval(), example, q_script_path)
    print(f"[SIZE] QUANT TorchScript: {q_script_path.name} -> {q_size/1e6:.2f} MB")
    print(f"[DELTA] Size change: {(fp32_size - q_size)/1e6:.2f} MB (positive = smaller after quant)")

    # Quantization coverage
    q_count = count_quantized_modules(int8_model)
    print(f"[COVERAGE] Quantized layers detected: {q_count}")

    if args.bench_cpu:
        fp32_lat = measure_cpu_latency(fp32_model.eval(), example)
        int8_lat = measure_cpu_latency(int8_model.eval(), example)
        print(f"[LATENCY CPU] FP32: {fp32_lat*1000:.2f} ms | INT8: {int8_lat*1000:.2f} ms | speedup x{fp32_lat/max(int8_lat,1e-6):.2f}")

    print("\n================= [DONE] =================")

if __name__ == "__main__":
    main()
