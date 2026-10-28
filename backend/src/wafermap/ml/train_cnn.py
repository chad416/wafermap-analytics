"""Train the CNN on canonical 2x64x64 maps, select on validation, export to ONNX.

Protocol (same as the baseline)
-------------------------------
* Train on ``train``; choose the epoch with the best **validation macro-F1** (early stopping).
* Score ``test`` once with the selected weights; nothing is tuned on it.

Imbalance handling
------------------
* **Square-root sampling**: each epoch draws samples with class probability proportional to
  ``sqrt(n_c)`` instead of ``n_c``. Near-full goes from 0.09 % to ~1.5 % of draws while 'none'
  drops from 85 % to ~50 %, so minority classes are seen often without the model forgetting what
  a clean wafer looks like (full balancing over-predicts defects on a 'none'-dominated fab).
* **Label-preserving augmentation**: the 8 dihedral symmetries (rotations by 90 degrees and
  flips). Every WM-811K class is defined by radial / local shape, not by absolute orientation,
  so a rotated Scratch is still a Scratch. This multiplies the effective minority sample count.
* Mild label smoothing (0.05) because WM-811K labels are known to be noisy at class borders
  (Loc vs Edge-Loc, Center vs Loc).
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from joblib import Parallel, delayed
from torch import nn

from wafermap import wafer
from wafermap.constants import CANONICAL_SIZE, CLASSES
from wafermap.features.build import load_labelled
from wafermap.ml.cnn import WaferCNN
from wafermap.ml.metrics import evaluate, macro_f1, plot_confusion

log = logging.getLogger(__name__)

BATCH = 256
EPOCH_SAMPLES = 48_000
LR = 2e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 6
LABEL_SMOOTHING = 0.05


def _canon_chunk(blobs: list[bytes], rows: list[int], cols: list[int]) -> npt.NDArray[np.uint8]:
    return np.stack([wafer.canonical_u8(wafer.decode(b, r, c)) for b, r, c in zip(blobs, rows, cols, strict=True)])


def build_canonical_cache(processed_dir: Path, data: dict[str, np.ndarray]) -> npt.NDArray[np.uint8]:
    """uint8 canonical tensors for all labelled wafers (cached: ~1.4 GB, rebuilt if stale)."""
    path = processed_dir / "canonical_u8.npy"
    n = len(data["y"])
    if path.exists():
        arr = np.load(path, mmap_mode="r")
        if arr.shape == (n, 2, CANONICAL_SIZE, CANONICAL_SIZE):
            return np.asarray(arr)
    t0 = time.perf_counter()
    step = 4000
    parts = Parallel(n_jobs=-1)(
        delayed(_canon_chunk)(data["map_zlib"][i : i + step].tolist(), data["rows"][i : i + step].tolist(),
                              data["cols"][i : i + step].tolist())
        for i in range(0, n, step)
    )
    arr = np.concatenate(parts)
    np.save(path, arr)
    log.info("canonical cache built in %.1f s", time.perf_counter() - t0)
    return arr


def sqrt_sampling_weights(y: np.ndarray) -> npt.NDArray[np.float64]:
    counts = np.bincount(y, minlength=len(CLASSES)).astype(np.float64)
    per_class = np.sqrt(counts) / counts  # class mass sqrt(n_c), split evenly over its samples
    w = per_class[y]
    return w / w.sum()


def dihedral(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Apply an independent random dihedral transform to every sample in the batch."""
    k = torch.randint(0, 4, (x.shape[0],), generator=gen)
    flip = torch.randint(0, 2, (x.shape[0],), generator=gen).bool()
    out = x.clone()
    for r in range(4):
        sel = k == r
        if sel.any():
            out[sel] = torch.rot90(x[sel], r, dims=(2, 3))
    out[flip] = torch.flip(out[flip], dims=(3,))
    return out


def to_tensor(u8: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(u8)).float().div_(255.0)


@torch.no_grad()
def predict_logits(model: nn.Module, x_u8: np.ndarray, batch: int = 1024) -> npt.NDArray[np.float32]:
    model.eval()
    outs = [model(to_tensor(x_u8[i : i + batch])).numpy() for i in range(0, len(x_u8), batch)]
    return np.concatenate(outs)


def train_cnn(
    processed_dir: Path,
    model_dir: Path,
    reports_dir: Path,
    *,
    seed: int = 42,
    epochs: int = 30,
    threads: int = 0,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    if threads:
        torch.set_num_threads(threads)

    data = load_labelled(processed_dir)
    x_all = build_canonical_cache(processed_dir, data)
    y_all, split, lots = data["y"], data["split"], data["lot_name"]
    tr, va, te = (np.flatnonzero(split == s) for s in ("train", "val", "test"))
    p_train = sqrt_sampling_weights(y_all[tr])

    model = WaferCNN()
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = EPOCH_SAMPLES // BATCH
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=epochs * steps_per_epoch,
                                                pct_start=0.15)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    log.info("params=%d train=%d val=%d test=%d threads=%d", n_params, len(tr), len(va), len(te),
             torch.get_num_threads())

    history: list[dict[str, float]] = []
    best_f1, best_state, best_epoch, stale = -1.0, None, -1, 0
    t_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.perf_counter()
        draw = rng.choice(tr, size=steps_per_epoch * BATCH, replace=True, p=p_train)
        running = 0.0
        for s in range(steps_per_epoch):
            idx = np.sort(draw[s * BATCH : (s + 1) * BATCH])  # sorted -> faster memory access
            xb = dihedral(to_tensor(x_all[idx]), gen)
            yb = torch.from_numpy(y_all[idx])
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            sched.step()
            running += float(loss)
        val_pred = predict_logits(model, x_all[va]).argmax(1)
        f1 = macro_f1(y_all[va], val_pred)
        rec = {"epoch": epoch, "train_loss": round(running / steps_per_epoch, 4), "val_macro_f1": round(f1, 4),
               "seconds": round(time.perf_counter() - t0, 1)}
        history.append(rec)
        log.info("epoch %s", rec)
        if f1 > best_f1:
            best_f1, best_epoch, stale = f1, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= PATIENCE:
                log.info("early stop at epoch %d (best %d)", epoch, best_epoch)
                break
    train_seconds = time.perf_counter() - t_start
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()

    val_metrics = evaluate(y_all[va], predict_logits(model, x_all[va]).argmax(1))
    # --- the single, final look at the test split ---------------------------------------------
    test_logits = predict_logits(model, x_all[te])
    test_metrics = evaluate(y_all[te], test_logits.argmax(1), groups=lots[te])
    log.info("val macro-F1 %.4f | test macro-F1 %.4f", val_metrics["macro_f1"], test_metrics["macro_f1"])

    out_dir = model_dir / "cnn"
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "model.onnx"
    export_onnx(model, onnx_path)
    parity = onnx_parity(onnx_path, model, x_all[te[:512]])
    version = hashlib.sha256(onnx_path.read_bytes()).hexdigest()[:12]
    meta = {
        "name": "cnn",
        "kind": "onnx",
        "version": version,
        "classes": list(CLASSES),
        "input": {"shape": [2, CANONICAL_SIZE, CANONICAL_SIZE], "channels": ["die_coverage", "fail_coverage"],
                  "dtype": "float32", "scale": "uint8/255"},
        "params": n_params,
        "best_epoch": best_epoch,
        "val_macro_f1": val_metrics["macro_f1"],
        "test_macro_f1": test_metrics["macro_f1"],
        "onnx_max_abs_logit_diff": parity,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    config = {"batch": BATCH, "epoch_samples": EPOCH_SAMPLES, "lr": LR, "weight_decay": WEIGHT_DECAY,
              "max_epochs": epochs, "patience": PATIENCE, "label_smoothing": LABEL_SMOOTHING,
              "sampling": "sqrt class frequency", "augmentation": "random dihedral (8)", "seed": seed,
              "torch": torch.__version__, "threads": torch.get_num_threads(), "device": "cpu"}
    report = {"model": "cnn", "version": version, "params": n_params, "config": config,
              "best_epoch": best_epoch, "train_seconds": round(train_seconds, 1), "history": history,
              "val": val_metrics, "test": test_metrics, "onnx_max_abs_logit_diff": parity}
    metrics_dir = reports_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "cnn.json").write_text(json.dumps(report, indent=2) + "\n")
    plot_confusion(test_metrics["confusion_matrix"], reports_dir / "figures" / "confusion_cnn.png",
                   f"CNN - test split (n={test_metrics['n']:,}), macro-F1 {test_metrics['macro_f1']:.3f}")
    plot_history(history, best_epoch, reports_dir / "figures" / "cnn_training.png")
    return report


def export_onnx(model: nn.Module, path: Path) -> None:
    dummy = torch.zeros(1, 2, CANONICAL_SIZE, CANONICAL_SIZE)
    torch.onnx.export(model, (dummy,), str(path), input_names=["x"], output_names=["logits"],
                      dynamic_axes={"x": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17,
                      dynamo=False)


def onnx_parity(path: Path, model: nn.Module, x_u8: np.ndarray) -> float:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    ref = predict_logits(model, x_u8)
    got = sess.run(None, {"x": to_tensor(x_u8).numpy()})[0]
    return float(np.abs(ref - got).max())


def plot_history(history: list[dict[str, float]], best_epoch: int, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = [h["epoch"] for h in history]
    fig, ax1 = plt.subplots(figsize=(7, 3.6), dpi=150)
    ax1.plot(ep, [h["train_loss"] for h in history], color="#6b7280", label="train loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("train loss")
    ax2 = ax1.twinx()
    ax2.plot(ep, [h["val_macro_f1"] for h in history], color="#2563eb", marker="o", ms=3, label="val macro-F1")
    ax2.axvline(best_epoch, color="#2563eb", ls=":", lw=1)
    ax2.set_ylabel("validation macro-F1")
    fig.legend(loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    ax1.set_title(f"CNN training (selected epoch {best_epoch} by validation macro-F1)", fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
