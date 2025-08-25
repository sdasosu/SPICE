import time
from pathlib import Path

import torch
import torch.nn as nn


def _get_engine():
    torch.backends.quantized.engine = "fbgemm"
    return torch.backends.quantized.engine


def _synthetic_loader(
    img_size: int = 576, in_ch: int = 3, steps: int = 64, batch: int = 4
):
    for _ in range(steps):
        x = torch.randn(batch, in_ch, img_size, img_size)
        y = torch.randint(0, 5, (batch, img_size, img_size), dtype=torch.long)
        yield x, y


@torch.no_grad()
def _calibrate(prepared: torch.nn.Module, train_loader, example, calib_steps: int = 50):
    prepared.eval()
    if train_loader is None:
        for x, _ in _synthetic_loader(
            img_size=example.shape[-1],
            in_ch=example.shape[1],
            steps=calib_steps,
            batch=example.shape[0],
        ):
            prepared(x.cpu())
        return
    it = 0
    for batch in train_loader:
        images = batch[0] if isinstance(batch, (list, tuple)) else batch["image"]
        prepared(images.cpu())
        it += 1
        if it >= calib_steps:
            break


def ptq_fx_convert(
    fp32_model: nn.Module,
    example_inputs: torch.Tensor,
    train_loader=None,
    calib_steps: int = 50,
) -> nn.Module:
    _get_engine()
    model = fp32_model.cpu().eval()
    try:
        import torch.ao.quantization as tq
        import torch.ao.quantization.quantize_fx as qfx

        qconfig = tq.get_default_qconfig("fbgemm")
        qdict = {"": qconfig}
        prepared = qfx.prepare_fx(model, qdict, example_inputs=example_inputs.cpu())
        print(f"[PTQ] Calibrating {calib_steps} batches ...")
        _calibrate(
            prepared, train_loader, example_inputs.cpu(), calib_steps=calib_steps
        )
        converted = tq.convert_fx(prepared)
        print("[PTQ] convert_fx complete (CPU int8).")
        return converted
    except Exception as e:
        print(f"[PTQ] FX PTQ failed: {e}")
        try:
            dyn = torch.quantization.quantize_dynamic(
                model, {nn.Linear}, dtype=torch.qint8
            )
            print("[PTQ] Fallback: dynamic quantization (Linear only).")
            return dyn
        except Exception as e2:
            print(f"[PTQ] Dynamic quant fallback failed: {e2}")
            return model


def qat_fx_convert_short_train(
    fp32_model: nn.Module,
    example_inputs: torch.Tensor,
    train_loader=None,
    val_loader=None,
    steps: int = 200,
    lr: float = 3e-4,
    wd: float = 1e-4,
) -> nn.Module:
    _get_engine()
    model = fp32_model.cpu()
    criterian = nn.CrossEntropyLoss()
    try:
        import torch.ao.quantization as tq
        import torch.ao.quantization.quantize_fx as qfx

        qconfig = tq.get_default_qat_qconfig("fbgemm")
        qdict = {"": qconfig}
        prepared = qfx.prepare_qat_fx(
            model.train(), qdict, example_inputs=example_inputs.cpu()
        )
        print(f"[QAT] prepared with example_inputs; short fine-tune steps={steps}")

        opt = torch.optim.AdamW(prepared.parameters(), lr=lr, weight_decay=wd)

        def _loader_or_synth():
            if train_loader is None:
                return _synthetic_loader(
                    img_size=example_inputs.shape[-1],
                    in_ch=example_inputs.shape[1],
                    steps=steps,
                    batch=example_inputs.shape[0],
                )
            for batch in train_loader:
                x = batch[0] if isinstance(batch, (list, tuple)) else batch["image"]
                y = batch[1] if isinstance(batch, (list, tuple)) else batch["mask"]
                yield x, y

        it = 0
        prepared.train()
        for x, y in _loader_or_synth():
            x = x.cpu()
            y = y.long().cpu()
            opt.zero_grad(set_to_none=True)
            out = prepared(x)
            if isinstance(out, dict):
                out = out.get("out", out.get("logits", out))
            loss = criterian(out, y)
            loss.backward()
            opt.step()
            it += 1
            if it >= steps:
                break

        prepared.eval()
        converted = tq.convert_fx(prepared)
        print("[QAT] convert_fx complete (CPU int8).")
        return converted
    except Exception as e:
        print(f"[QAT] FX QAT failed: {e}")
        return model.eval()


def save_torchscript_and_size(
    model: nn.Module, example: torch.Tensor, out_path: Path
) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        traced = torch.jit.trace(model, example.cpu())
        traced.save(str(out_path))
    sz = out_path.stat().st_size
    return sz


def measure_cpu_latency(
    model: nn.Module, example: torch.Tensor, warmup: int = 5, iters: int = 20
) -> float:
    model = model.eval().cpu()
    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(example.cpu())
        t0 = time.time()
        for _ in range(iters):
            _ = model(example.cpu())
        t1 = time.time()
    return (t1 - t0) / float(iters)


def count_quantized_modules(model: nn.Module) -> int:
    cnt = 0
    for m in model.modules():
        name = m.__class__.__name__
        if "Quantized" in name or "PackedParams" in name:
            cnt += 1
    return cnt
