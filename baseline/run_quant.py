import argparse
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torchinfo import summary

THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent
sys.path.insert(0, str(ROOT))

import segmentation_models_pytorch as smp

from data.data import get_loaders
from experiments.train_test import inference
from quantization.quant_utils import (
    build_optim_and_sched,
    prepare_qat_minimal,
    ptq_minimal_calibrate_and_convert,
    sanity_train_one_epoch,
)
from quantization.train_test_quant import evaluate_one_epoch_val, train_validation_timed

NUM_CLASSES = 5


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def device_from_flag(cpu: bool) -> torch.device:
    return (
        torch.device("cpu")
        if cpu or not torch.cuda.is_available()
        else torch.device("cuda")
    )


def _auto_pick_gpu():
    if not torch.cuda.is_available():
        return
    free = []
    for i in range(torch.cuda.device_count()):
        try:
            mem_free, _ = torch.cuda.mem_get_info(i)
        except Exception:
            mem_free = 0
        free.append((mem_free, i))
    best = max(free)[1]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(best)


_auto_pick_gpu()


def create_model(
    model_name: str,
    encoder_name: str = None,
    in_ch: int = 3,
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
) -> nn.Module:
    m = model_name.lower()

    if m == "deeplabv3plus_resnet":
        return smp.DeepLabV3Plus(
            encoder_name=encoder_name or "resnet50",
            encoder_weights="imagenet" if pretrained else None,
            in_channels=in_ch,
            classes=num_classes,
            activation=None,
        )

    if m == "deeplabv3plus_efficientnet":
        return smp.DeepLabV3Plus(
            encoder_name=encoder_name or "timm-efficientnet-b3",
            encoder_weights="imagenet" if pretrained else None,
            in_channels=in_ch,
            classes=num_classes,
            activation=None,
        )

    if m == "unet_resnet":
        return smp.Unet(
            encoder_name=encoder_name or "resnet50",
            encoder_weights="imagenet" if pretrained else None,
            in_channels=in_ch,
            classes=num_classes,
            activation=None,
        )

    if m == "unet_efficientnet":
        return smp.Unet(
            encoder_name=encoder_name or "timm-efficientnet-b3",
            encoder_weights="imagenet" if pretrained else None,
            in_channels=in_ch,
            classes=num_classes,
            activation=None,
        )

    if m == "fpn_resnet":
        return smp.FPN(
            encoder_name=encoder_name or "resnet50",
            encoder_weights="imagenet" if pretrained else None,
            in_channels=in_ch,
            classes=num_classes,
            activation=None,
        )

    if m == "fpn_efficientnet":
        return smp.FPN(
            encoder_name=encoder_name or "timm-efficientnet-b3",
            encoder_weights="imagenet" if pretrained else None,
            in_channels=in_ch,
            classes=num_classes,
            activation=None,
        )

    raise ValueError(f"Unknown model: {model_name}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default=str(ROOT / "data"))
    p.add_argument("--img-size", type=int, default=576)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument(
        "--model",
        required=True,
        choices=[
            "DeepLabV3Plus_resnet",
            "DeepLabV3Plus_efficientnet",
            "UNET_resnet",
            "UNET_efficientnet",
            "FPN_resnet",
            "FPN_efficientnet",
        ],
    )
    p.add_argument("--encoder", type=str, default=None)
    p.add_argument("--in-ch", type=int, default=3)
    p.add_argument("--no-pretrained", action="store_true")

    # training lengths
    p.add_argument("--epochs-full", type=int, default=50)
    p.add_argument("--epochs-q", type=int, default=50)

    # optimization
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)

    # quantization
    p.add_argument("--quant-method", type=str, default="qat", choices=["qat", "ptq"])
    p.add_argument(
        "--sanity-thresh",
        type=float,
        default=0.0,
        help="Require quantized mIoU >= thresh to pass. Default=0.0 (just runs)",
    )
    p.add_argument(
        "--calib-iters", type=int, default=50, help="PTQ calibration steps (batches)"
    )
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = device_from_flag(args.cpu)

    train_loader, val_loader, test_loader = get_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
    )
    criterian = nn.CrossEntropyLoss()

    fp32_model = create_model(
        model_name=args.model,
        encoder_name=args.encoder,
        in_ch=args.in_ch,
        num_classes=NUM_CLASSES,
        pretrained=not args.no_pretrained,
    ).to(device)

    print("\n===== [SUMMARY] FP32 MODEL (pre-sanity) =====\n")
    try:
        summary(
            fp32_model,
            input_size=(1, args.in_ch, args.img_size, args.img_size),
            depth=1,
            col_names=("input_size", "output_size", "num_params"),
        )
    except Exception as e:
        print(f"[WARN] torchinfo.summary failed for FP32 model: {e}")
    print("\n")

    print("\n================= [SANITY] FP32 TRAIN (1 epoch) =================\n")
    print("[1] Train FP32 model for 1 epoch ...\n")
    opt_fp32, sch_fp32 = build_optim_and_sched(
        fp32_model, args.lr, args.weight_decay, epochs=1
    )

    base_miou, base_acc = sanity_train_one_epoch(
        model=fp32_model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterian=criterian,
        optimizer=opt_fp32,
        scheduler=sch_fp32,
        device=device,
        num_classes=NUM_CLASSES,
        model_name=args.model,
    )
    print(f"\n>>> [SANITY RESULT] FP32 mIoU={base_miou:.4f} | AvgAcc={base_acc:.4f}\n")

    print(f"\n[2] Prepare {args.quant_method.upper()} model ...\n")
    if args.quant_method == "qat":
        q_model = create_model(
            model_name=args.model,
            encoder_name=args.encoder,
            in_ch=args.in_ch,
            num_classes=NUM_CLASSES,
            pretrained=not args.no_pretrained,
        ).to(device)
        q_model = prepare_qat_minimal(q_model).to(device)

        print("\n===== [SUMMARY] QAT MODEL (pre-sanity) =====\n")
        try:
            summary(
                q_model,
                input_size=(1, args.in_ch, args.img_size, args.img_size),
                depth=1,
                col_names=("input_size", "output_size", "num_params"),
            )
        except Exception as e:
            print(f"[WARN] torchinfo.summary failed for QAT-prepared model: {e}")
        print("\n")

        opt_q, sch_q = build_optim_and_sched(
            q_model, args.lr * 0.5, args.weight_decay, epochs=1
        )
        q_miou, q_acc = sanity_train_one_epoch(
            model=q_model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterian=criterian,
            optimizer=opt_q,
            scheduler=sch_q,
            device=device,
            num_classes=NUM_CLASSES,
            model_name=args.model + "_QAT",
        )
        print(f"\n>>> [SANITY RESULT] QAT mIoU={q_miou:.4f} | AvgAcc={q_acc:.4f}\n")

    else:
        q_model = create_model(
            model_name=args.model,
            encoder_name=args.encoder,
            in_ch=args.in_ch,
            num_classes=NUM_CLASSES,
            pretrained=not args.no_pretrained,
        ).to(torch.device("cpu"))
        q_model = ptq_minimal_calibrate_and_convert(
            model=q_model, train_loader=train_loader, calib_iters=args.calib_iters
        )

        print("\n===== [SUMMARY] PTQ MODEL (int8, pre-eval) =====\n")
        try:
            summary(
                q_model,
                input_size=(1, args.in_ch, args.img_size, args.img_size),
                depth=1,
                col_names=("input_size", "output_size", "num_params"),
            )
        except Exception as e:
            print(f"[WARN] torchinfo.summary failed for PTQ (int8) model: {e}")
        print("\n")

        print("[2b] Evaluating PTQ (int8) model for 1 epoch on validation set ...\n")
        with torch.no_grad():
            q_miou, q_acc = evaluate_one_epoch_val(
                model=q_model,
                val_loader=val_loader,
                device=torch.device("cpu"),
                num_classes=NUM_CLASSES,
            )
        print(f"\n>>> [SANITY RESULT] PTQ mIoU={q_miou:.4f} | AvgAcc={q_acc:.4f}\n")

    passed = (q_miou is not None) and (q_miou >= args.sanity_thresh)
    print(
        f"[SANITY SUMMARY] FP32 mIoU={base_miou:.4f} | Quant mIoU={q_miou:.4f} | PASS={passed}\n"
    )
    if not passed:
        print("[SANITY] Sanity check failed — stopping as requested by threshold.\n")
        return

    print("\n================= [FULL FP32 TRAINING 50 EPOCHS] =================\n")
    print("===== [SUMMARY] FP32 MODEL (pre-full) =====\n")
    try:
        summary(
            fp32_model,
            input_size=(1, args.in_ch, args.img_size, args.img_size),
            depth=1,
            col_names=("input_size", "output_size", "num_params"),
        )
    except Exception as e:
        print(f"[WARN] torchinfo.summary failed for FP32 model (pre-FULL): {e}")
    print("\n")

    opt_full, sch_full = build_optim_and_sched(
        fp32_model, args.lr, args.weight_decay, args.epochs_full
    )
    best_rec_fp32 = train_validation_timed(
        model=fp32_model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterian=criterian,
        optimizer=opt_full,
        scheduler=sch_full,
        num_epoch=args.epochs_full,
        device=device,
        num_classes=NUM_CLASSES,
        model_name=args.model,
        tag="FULL",
    )
    print(
        f"\n[FULL RESULT] Best mIoU={best_rec_fp32['best_miou']:.4f} (avgAcc={best_rec_fp32['best_acc']:.4f}) at epoch {best_rec_fp32['best_epoch']}\n"
    )

    print("\n================= [QUANTIZED TRAINING 50 EPOCHS] =================\n")
    if args.quant_method == "qat":
        q_model = create_model(
            model_name=args.model,
            encoder_name=args.encoder,
            in_ch=args.in_ch,
            num_classes=NUM_CLASSES,
            pretrained=not args.no_pretrained,
        ).to(device)
        q_model = prepare_qat_minimal(q_model).to(device)

        print("===== [SUMMARY] QAT MODEL (pre-full) =====\n")
        try:
            summary(
                q_model,
                input_size=(1, args.in_ch, args.img_size, args.img_size),
                depth=1,
                col_names=("input_size", "output_size", "num_params"),
            )
        except Exception as e:
            print(
                f"[WARN] torchinfo.summary failed for QAT-prepared model (full run): {e}"
            )
        print("\n")

        opt_q_full, sch_q_full = build_optim_and_sched(
            q_model, args.lr, args.weight_decay, args.epochs_q
        )
        best_rec_q = train_validation_timed(
            model=q_model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterian=criterian,
            optimizer=opt_q_full,
            scheduler=sch_q_full,
            num_epoch=args.epochs_q,
            device=device,
            num_classes=NUM_CLASSES,
            model_name=args.model + "_QAT",
            tag="QAT",
        )

        try:
            q_model.eval().cpu()
            q_model = torch.ao.quantization.convert_fx(q_model)
            ckpt_dir = ROOT / "checkpoints" / "quantized" / (args.model + "_QAT")
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.jit.script(q_model).save(str(ckpt_dir / "model_qat_int8.script"))
            print(f"\n[QAT] Saved converted int8 TorchScript to {ckpt_dir}\n")
        except Exception as e:
            print(
                f"\n[QAT] convert_fx failed; keeping fake-quant model only. Reason: {e}\n"
            )

        print(
            f"\n[QAT RESULT] Best mIoU={best_rec_q['best_miou']:.4f} (avgAcc={best_rec_q['best_acc']:.4f}) at epoch {best_rec_q['best_epoch']}\n"
        )

    else:
        print(
            "[PTQ] Note: int8 PTQ models are not trainable. Performing QAT fine-tuning as a practical alternative.\n"
        )
        q_model_qat = create_model(
            model_name=args.model,
            encoder_name=args.encoder,
            in_ch=args.in_ch,
            num_classes=NUM_CLASSES,
            pretrained=not args.no_pretrained,
        ).to(device)
        q_model_qat = prepare_qat_minimal(q_model_qat).to(device)

        print("===== [SUMMARY] QAT MODEL (PTQ->QAT pre-full) =====\n")
        try:
            summary(
                q_model_qat,
                input_size=(1, args.in_ch, args.img_size, args.img_size),
                depth=1,
                col_names=("input_size", "output_size", "num_params"),
            )
        except Exception as e:
            print(
                f"[WARN] torchinfo.summary failed for QAT-prepared model (PTQ->QAT): {e}"
            )
        print("\n")

        opt_q_full, sch_q_full = build_optim_and_sched(
            q_model_qat, args.lr, args.weight_decay, args.epochs_q
        )
        best_rec_q = train_validation_timed(
            model=q_model_qat,
            train_loader=train_loader,
            val_loader=val_loader,
            criterian=criterian,
            optimizer=opt_q_full,
            scheduler=sch_q_full,
            num_epoch=args.epochs_q,
            device=device,
            num_classes=NUM_CLASSES,
            model_name=args.model + "_PTQ_QAT_FT",
            tag="QAT_after_PTQ",
        )
        print(
            f"\n[PTQ->QAT RESULT] Best mIoU={best_rec_q['best_miou']:.4f} (avgAcc={best_rec_q['best_acc']:.4f}) at epoch {best_rec_q['best_epoch']}\n"
        )

    print("\n================= [FINAL TEST (FP32 SCRIPT)] =================\n")
    ckpt_dir = ROOT / "checkpoints" / "fullsize" / args.model
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_pt = ckpt_dir / f"{args.model}.pt"
    ckpt_script = ckpt_dir / f"{args.model}.script"

    export_model = create_model(
        model_name=args.model,
        encoder_name=args.encoder,
        in_ch=args.in_ch,
        num_classes=NUM_CLASSES,
        pretrained=False,
    ).to(device)

    try:
        state = torch.load(ckpt_pt, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(ckpt_pt, map_location=device)
    export_model.load_state_dict(state)
    export_model.eval()

    example = torch.randn(1, args.in_ch, args.img_size, args.img_size, device=device)
    scripted = torch.jit.trace(export_model, example)
    scripted.save(str(ckpt_script))
    print(f"[INFO] TorchScript exported to: {ckpt_script}\n")

    model_for_eval = torch.jit.load(str(ckpt_script), map_location=device)
    print(f"[INFO] Loaded scripted model from: {ckpt_script}\n")
    inference(
        model=model_for_eval,
        test_loader=test_loader,
        device=device,
        num_classes=NUM_CLASSES,
    )
    print("\n================= [DONE] =================\n")


if __name__ == "__main__":
    main()
