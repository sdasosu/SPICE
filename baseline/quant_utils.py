from pathlib import Path

import torch
import torch.nn as nn


def build_optim_and_sched(
    model: nn.Module, lr: float, weight_decay: float, epochs: int
):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=max(epochs // 3, 1), gamma=0.1)
    return opt, sch


def sanity_train_one_epoch(
    model: nn.Module,
    train_loader,
    val_loader,
    criterian,
    optimizer,
    scheduler,
    device: torch.device,
    num_classes: int,
    model_name: str,
):
    import pytorch_lightning as pl

    from quantization.train_test_quant import SegLightningModuleTimed

    lit = SegLightningModuleTimed(
        model=model,
        criterian=criterian,
        num_classes=num_classes,
        model_name=model_name,
        save_dir=Path(__file__).resolve().parent.parent,
        external_optimizer=optimizer,
        external_scheduler=scheduler,
        tag="SANITY",
    )
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="gpu" if device.type == "cuda" else "cpu",
        devices=1,
        precision=32,
        log_every_n_steps=50,
        enable_progress_bar=False,
    )
    trainer.fit(lit, train_dataloaders=train_loader, val_dataloaders=val_loader)
    return float(lit.best_val_miou), float(lit.best_epoch_acc)


def prepare_qat_minimal(model: nn.Module) -> nn.Module:
    model.train()
    try:
        import torch.ao.quantization as tq
        import torch.ao.quantization.quantize_fx as quantize_fx

        qconfig = tq.get_default_qat_qconfig("fbgemm")
        qdict = {"": qconfig}
        prepared = quantize_fx.prepare_qat_fx(model, qdict)
        print("[QAT] FX prepare_qat_fx applied.")
        return prepared
    except Exception as e:
        print(f"[QAT] FX prepare failed; falling back to no-op (FP32). Reason: {e}")
        return model


@torch.no_grad()
def ptq_minimal_calibrate_and_convert(
    model: nn.Module, train_loader, calib_iters: int = 50
) -> nn.Module:
    model.eval().cpu()
    try:
        import torch.ao.quantization as tq
        import torch.ao.quantization.quantize_fx as quantize_fx

        qconfig = tq.get_default_qconfig("fbgemm")
        qdict = {"": qconfig}
        prepared = quantize_fx.prepare_fx(model, qdict)
        print(f"[PTQ] Running calibration for {calib_iters} batches ...")
        it = 0
        for batch in train_loader:
            images = (
                batch[0].cpu()
                if isinstance(batch, (list, tuple))
                else batch["image"].cpu()
            )
            _ = prepared(images)
            it += 1
            if it >= calib_iters:
                break
        converted = torch.ao.quantization.convert_fx(prepared)
        print("[PTQ] convert_fx complete (CPU int8).")
        return converted
    except Exception as e:
        print(
            f"[PTQ] FX PTQ failed; falling back to dynamic quant on Linear only. Reason: {e}"
        )
        try:
            qdyn = torch.quantization.quantize_dynamic(
                model.cpu(), {nn.Linear}, dtype=torch.qint8
            )
            print("[PTQ] Applied dynamic quantization (Linear).")
            return qdyn
        except Exception as e2:
            print(
                f"[PTQ] Dynamic quantization also failed; returning FP32 model. Reason: {e2}"
            )
            return model
