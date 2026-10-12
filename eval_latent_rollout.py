#!/usr/bin/env python
"""Decode a latent-space rollout with the frozen pretrained decoder and score it like MODE=eval of
03_shear_flow_ddp.py (per-frame VRMSE on the held-out splits), so any latent model can be compared
with HNODE on equal terms.

Evaluate (8 GPUs; runs as one process too):
    srun -N1 -n8 -c7 --gpus-per-task=1 --gpu-bind=closest python -u eval_latent_rollout.py \
        eval LATENT_DIR PRED OUT_DIR

    LATENT_DIR  output of encode_latents.py ({split}.pt), or of pod_basis.py / ae_train.py encode
                (coeffs_{split}.pt)
    PRED        a .pt file {"valid": (N, 199, D), "test": (N, 199, D)}: predicted latents of frames
                1..199 for every trajectory, in LATENT_DIR's order (e.g. latent_baselines.py's pred.pt);
                or one of the built-in references
                "hnode" (the HNODE rollout stored by encode_latents.py) and
                "oracle" (the encoder's own latents of frames 1..199: the encoder/decoder error floor)
    OUT_DIR     receives vrmse.npz and summary.txt (same layout as MODE=eval)

    DECODER     (environment) "swin" (default: the pretrained HNODE checkpoint's decoder),
                "pod:R" (the first R modes of LATENT_DIR/basis.pt), or "ae:CHECKPOINT" (ae_train.py)

Compare evaluated models (CPU only):
    python eval_latent_rollout.py compare OUT.png NAME=OUT_DIR [NAME=OUT_DIR ...]

Environment: EVAL_SPLITS (default "valid,test"), MAX_TRAJ (per split, for testing; 0 = all).
"""
import os
import sys
import time

import numpy as np
import torch

import shear_flow_common as C

torch.set_default_dtype(torch.float64)
HORIZONS = [1, 10, 25, 50, 75, 100, 150, 199]
# Paper Fig. 4c HNODE curve, read off the published plot (approximate); same values as MODE=eval.
PAPER_T = [1, 10, 25, 50, 75, 100, 150, 199]
PAPER_V = [0.03, 0.03, 0.045, 0.08, 0.10, 0.105, 0.125, 0.13]


def load_predictions(pred, latent_dir, split, dim):
    path = os.path.join(latent_dir, f"{split}.pt")
    if not os.path.exists(path):  # pod_basis.py / ae_train.py layout
        path = os.path.join(latent_dir, f"coeffs_{split}.pt")
    lat = torch.load(path, weights_only=False)
    if "coeffs" in lat:
        lat["latents"] = lat.pop("coeffs")
    if pred == "hnode":
        p = lat["hnode"]
    elif pred == "oracle":
        p = lat["latents"][:, 1:, :dim]
    else:
        p = torch.load(pred, weights_only=False)[split]
    assert p.shape == (lat["latents"].shape[0], C.NUM_FRAMES - 1, dim), (p.shape, dim)
    return p, lat


def load_decoder(spec, latent_dir, device):
    """(decode: (T, D) latents -> (T, C, H, W) normalized frames, D)."""
    if spec == "swin":
        _, decoder, _ = C.load_models(device)
        return decoder, C.LATENT_DIM
    if spec.startswith("pod:"):
        r = int(spec.split(":")[1])
        b = torch.load(os.path.join(latent_dir, "basis.pt"), weights_only=False)
        mean = b["mean"].to(device).reshape(-1)
        modes = b["modes"][:, :r].to(device).double()
        H, W, NC = b["mean"].shape
        return (lambda a: (mean + a @ modes.T).reshape(-1, H, W, NC).permute(0, 3, 1, 2)), r
    if spec.startswith("ae:"):
        import ae_train

        model = ae_train.load_autoencoder(spec[3:], device)
        return (lambda a: ae_train.decode_float32(model, a)), model.latent_dim
    raise ValueError(f"DECODER must be swin, pod:R or ae:CHECKPOINT, got {spec!r}")


def evaluate(latent_dir, pred, out_dir):
    import torch.distributed as dist

    splits = os.environ.get("EVAL_SPLITS", "valid,test").split(",")
    max_traj = int(os.environ.get("MAX_TRAJ", 0))
    rank, local_rank, world_size = C.distributed_env()
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29502")
    user = os.environ.get("USER", "user")
    os.environ.setdefault("MIOPEN_USER_DB_PATH", f"/tmp/miopen-{user}-{rank}")
    os.environ.setdefault("MIOPEN_CUSTOM_CACHE_DIR", f"/tmp/miopen-{user}-{rank}")
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    device = C.pick_device(local_rank)
    log = (lambda *a: print(*a, flush=True)) if rank == 0 else (lambda *a: None)
    decoder, dim = load_decoder(os.environ.get("DECODER", "swin"), latent_dir, device)
    mean, std = C.frame_stats(device)

    results, t0 = [], time.time()
    for split in splits:
        preds, lat = load_predictions(pred, latent_dir, split, dim)
        well = C.well_split(split)
        n = min(len(well), max_traj) if max_traj else len(well)
        mine = list(range(rank, n, world_size))
        loader = torch.utils.data.DataLoader(mine, batch_size=None, num_workers=3, prefetch_factor=2,
                                             collate_fn=lambda i: (i, *C.full_trajectory(well, i)))
        with torch.no_grad():
            for k, (i, frames, scalars) in enumerate(loader):
                assert torch.equal(scalars, lat["scalars"][i]), (split, i)
                target_norm = ((frames.to(device) - mean) / std)[1:]  # frames 1..199
                recon_norm = torch.cat([decoder(c) for c in preds[i].to(device).split(50)])
                recon_norm = recon_norm.permute(0, 2, 3, 1)  # (199, H, W, C)
                mse_n = (recon_norm - target_norm).pow(2).mean(dim=(1, 2))  # (199, C)
                var_n = target_norm.var(dim=(1, 2))
                dup = lat["duplicate_of_train"][i] if lat["duplicate_of_train"] is not None else None
                results.append({
                    "split": split, "idx": i, "Re": float(scalars[0]), "Sc": float(scalars[1]),
                    "duplicate_of_train": dup,
                    "vrmse_well": (mse_n * std**2 / (var_n * std**2 + 1e-7)).sqrt().cpu().numpy(),
                    "vrmse_train": (mse_n / (var_n + 1e-3)).sqrt().cpu().numpy(),
                })
                if rank == 0 and (k % 10 == 0 or k == len(mine) - 1):
                    log(f"  {split} rank 0: {k + 1}/{len(mine)} ({time.time() - t0:.0f}s)")

    gathered = [None] * world_size
    dist.all_gather_object(gathered, results)
    if rank == 0:
        allres = sorted((r for part in gathered for r in part), key=lambda r: (r["split"], r["idx"]))
        os.makedirs(out_dir, exist_ok=True)
        vw = np.stack([r["vrmse_well"] for r in allres])  # (N, 199, C)
        vt = np.stack([r["vrmse_train"] for r in allres])
        dup = np.array([r["duplicate_of_train"] is not None for r in allres])
        np.savez(os.path.join(out_dir, "vrmse.npz"), vrmse_well=vw, vrmse_train=vt,
                 split=np.array([r["split"] for r in allres]), idx=np.array([r["idx"] for r in allres]),
                 Re=np.array([r["Re"] for r in allres]), Sc=np.array([r["Sc"] for r in allres]), duplicate=dup)
        summary = summarize(vw, vt, dup, np.array([r["Re"] for r in allres]))
        open(os.path.join(out_dir, "summary.txt"), "w").write(
            f"prediction: {pred}\nlatents: {latent_dir}\ndecoder: {os.environ.get('DECODER', 'swin')}\n{summary}\n")
        log(f"prediction: {pred}\nVRMSE (The Well definition) vs rollout horizon, averaged over trajectories "
            f"and fields:\n{summary}\nSaved {out_dir}/vrmse.npz, summary.txt ({time.time() - t0:.0f}s)")
    dist.barrier()
    dist.destroy_process_group()


def summarize(vw, vt, dup, re):
    """The MODE=eval table: VRMSE at selected horizons per group of trajectories."""
    groups = {"all held-out": np.ones(len(vw), bool), "unique (not in train)": ~dup, "duplicates of train": dup}
    for r in sorted(set(re)):
        groups[f"Re = {r:g}"] = re == r
    lines = [f"{'group':<24}{'n':>5}  " + "".join(f"{'t=' + str(t):>8}" for t in HORIZONS) + f"{'mean':>8}"]

    def row(name, curve, n):
        return f"{name:<24}{n:>5}  " + "".join(f"{curve[t - 1]:>8.4f}" for t in HORIZONS) + f"{curve.mean():>8.4f}"

    for name, mask in groups.items():
        if mask.sum():
            lines.append(row(name, vw[mask].mean(axis=(0, 2)), int(mask.sum())))
    lines.append(row("all (training-loss def.)", vt.mean(axis=(0, 2)), len(vt)))
    return "\n".join(lines)


def compare(out_png, entries):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    header = f"{'model':<16}{'group':<10}{'n':>5}  " + "".join(f"{'t=' + str(t):>8}" for t in HORIZONS) + f"{'mean':>8}"
    lines = [header]
    for entry in entries:
        name, path = entry.split("=", 1)
        d = np.load(os.path.join(path, "vrmse.npz"))
        vw, dup = d["vrmse_well"], d["duplicate"]
        for ax, (group, mask) in zip(axes, [("all", np.ones(len(vw), bool)), ("unique", ~dup)]):
            curve = vw[mask].mean(axis=(0, 2))
            ax.semilogy(np.arange(1, len(curve) + 1), curve, label=name)
            lines.append(f"{name:<16}{group:<10}{int(mask.sum()):>5}  "
                         + "".join(f"{curve[t - 1]:>8.4f}" for t in HORIZONS) + f"{curve.mean():>8.4f}")
    for ax, title in zip(axes, ["All held-out (valid + test)", "Unique held-out (not in train)"]):
        ax.semilogy(PAPER_T, PAPER_V, "k--o", ms=3, label="paper Fig. 4c HNODE (approx.)")
        ax.set_xlabel("Rollout prediction horizon (frames)")
        ax.set_title(title)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("VRMSE")
    plt.tight_layout()
    plt.savefig(out_png, dpi=130)
    print("\n".join(lines))
    print(f"Saved {out_png}")


if __name__ == "__main__":
    if len(sys.argv) >= 5 and sys.argv[1] == "eval":
        evaluate(*sys.argv[2:5])
    elif len(sys.argv) >= 4 and sys.argv[1] == "compare":
        compare(sys.argv[2], sys.argv[3:])
    else:
        sys.exit(__doc__)
