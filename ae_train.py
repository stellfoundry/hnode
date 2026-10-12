#!/usr/bin/env python
"""Stand-alone autoencoders for shear flow (two-stage: reconstruction only, no dynamics), whose frozen
latents the time steppers of latent_baselines.py are then trained on.

    srun -N1 -n8 -c7 --gpus-per-task=1 --gpu-bind=closest python -u ae_train.py train KIND OUT_DIR
    srun ... python -u ae_train.py encode OUT_DIR/ae.pt LATENT_DIR     (coeffs_{train,valid,test}.pt)
    (submit_ae.sh chains 2-hour training jobs, then encodes)

KIND:
    swin  SwinEncoder/SwinDecoder with the HNODE repo's shear-flow settings (latent 64, ~66 M parameters)
    cnn   convolutional autoencoder with periodic padding: 6 stride-2 stages 256x512 -> 4x8, then a
          linear map to the latent (and back)

Training (AdamW, warmup + cosine, steps with spiking loss skipped): random frames of the train split (normalized with the training stats), loss = the HNODE
training loss (VRMSE^2 per frame and channel, eps 1e-3). Optional SMOOTH > 0 adds
SMOOTH * ||e(t+1) - e(t)||^2 / ||e - mean(e)||^2 on consecutive frames (scale-free temporal smoothness).
float32. Data parallel with one flattened gradient all-reduce, like 03_shear_flow_ddp.py.

Writes OUT_DIR/: ae.pt (best on a fixed set of valid frames; what `encode` and the evaluation load),
ckpt.pt (resume state), log.txt, done (marker).

Environment (AE_ prefixes keep them apart from latent_baselines.py's knobs in chained jobs):
LATENT (64), AE_STEPS (40000), AE_BATCH (32 frame pairs, global), AE_LR (1e-4 swin, 3e-4 cnn), SMOOTH (0),
AE_EVAL_EVERY (500), CKPT_EVERY (200), TIME_LIMIT (6600 s), DDP_BACKEND (nccl).
"""
import json
import os
import sys
import time

import h5py
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

import shear_flow_common as C

H, W, NC = 256, 512, 4


# ---------------------------------------------------------------- models


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.body = nn.Sequential(nn.GroupNorm(8, c), nn.GELU(), nn.Conv2d(c, c, 3, padding=1, padding_mode="circular"),
                                  nn.GroupNorm(8, c), nn.GELU(), nn.Conv2d(c, c, 3, padding=1, padding_mode="circular"))

    def forward(self, x):
        return x + self.body(x)


class ConvAE(nn.Module):
    def __init__(self, latent_dim, chans=(32, 64, 128, 256, 256, 256)):
        super().__init__()
        self.latent_dim = latent_dim
        enc = [nn.Conv2d(NC, chans[0], 3, padding=1, padding_mode="circular")]
        for c_in, c_out in zip(chans, chans[1:] + (chans[-1],)):
            enc += [ResBlock(c_in), nn.Conv2d(c_in, c_out, 4, stride=2, padding=1, padding_mode="circular")]
        self.enc = nn.Sequential(*enc)
        self.bottom = (chans[-1], H >> len(chans), W >> len(chans))  # (256, 4, 8)
        n = int(np.prod(self.bottom))
        self.to_latent = nn.Linear(n, latent_dim)
        self.from_latent = nn.Linear(latent_dim, n)
        dec = []
        rev = chans[::-1]
        for c_in, c_out in zip((rev[0],) + rev, rev):
            dec += [nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(c_in, c_out, 3, padding=1, padding_mode="circular"), ResBlock(c_out)]
        dec += [nn.GroupNorm(8, rev[-1]), nn.GELU(), nn.Conv2d(rev[-1], NC, 3, padding=1, padding_mode="circular")]
        self.dec = nn.Sequential(*dec)

    def encode(self, x):
        return self.to_latent(self.enc(x).flatten(1))

    def decode(self, a):
        return self.dec(self.from_latent(a).view(-1, *self.bottom))


class SwinAE(nn.Module):
    """The HNODE repo's Swin encoder/decoder pair (same settings as the shear-flow checkpoint)."""

    def __init__(self, latent_dim):
        super().__init__()
        from swin_transformer_v2 import SwinDecoder, SwinEncoder

        self.latent_dim = latent_dim
        swin = dict(img_size=(H, W), patch_size=4, latent_dim=latent_dim, embed_dim=96, depths=(2, 2, 6, 2),
                    num_heads=(3, 6, 12, 24), window_size=(8, 8), mlp_ratio=4, drop=0.0, attn_drop=0.0,
                    drop_path_rate=0.0)
        self.encoder = SwinEncoder(in_chans=NC, final_channels=64, **swin)
        self.decoder = SwinDecoder(out_chans=NC, **swin)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, a):
        return self.decoder(a)


def build(kind, latent_dim):
    if kind == "swin":
        import swin_transformer_v2  # noqa: F401  (sets the default dtype to float64 on import)
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        return {"cnn": ConvAE, "swin": SwinAE}[kind](latent_dim)
    finally:
        torch.set_default_dtype(prev)


def load_autoencoder(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = build(ck["kind"], ck["latent_dim"])
    model.load_state_dict(ck["model"])
    return model.to(device).eval()


def decode_float32(model, a):
    """model.decode in float32 whatever the caller's default dtype (float64 in the HNODE scripts)."""
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        return model.decode(a.float()).double()
    finally:
        torch.set_default_dtype(prev)


# ---------------------------------------------------------------- data


class Frames(torch.utils.data.Dataset):
    """Random pairs of consecutive frames (t, t+1) of the train split, (2, C, H, W) float32, unnormalized."""

    def __init__(self, split, n_samples, seed):
        self.dir = os.path.join(C.WELL_DATA_PATH, split)
        self.items = C.split_index(split)
        self.n, self.seed = n_samples, seed
        self.files = {}

    def __len__(self):
        return self.n

    def read(self, i, t, n=2):
        name, traj = self.items[i]
        if name not in self.files:  # opened lazily, once per worker process
            self.files[name] = h5py.File(os.path.join(self.dir, name), "r")
        f = self.files[name]
        x = np.concatenate([f["t0_fields/tracer"][traj, t:t + n][..., None],
                            f["t0_fields/pressure"][traj, t:t + n][..., None],
                            f["t1_fields/velocity"][traj, t:t + n]], axis=-1)
        return torch.from_numpy(x).permute(0, 3, 1, 2)

    def __getitem__(self, k):
        g = np.random.default_rng([*self.seed, k])
        return self.read(int(g.integers(len(self.items))), int(g.integers(C.NUM_FRAMES - 1)))


def vrmse2(pred, target, eps=1e-3):
    """The HNODE training loss: mean over frames and channels of MSE / (variance + eps). (B, C, H, W)."""
    return ((pred - target).pow(2).mean((-2, -1)) / (target.var((-2, -1)) + eps)).mean()


# ---------------------------------------------------------------- train


def setup():
    rank, local_rank, world = C.distributed_env()
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29505")
    user = os.environ.get("USER", "user")
    os.environ.setdefault("MIOPEN_USER_DB_PATH", f"/tmp/miopen-{user}-{rank}")
    os.environ.setdefault("MIOPEN_CUSTOM_CACHE_DIR", f"/tmp/miopen-{user}-{rank}")
    dev = C.pick_device(local_rank)
    dist.init_process_group(os.environ.get("DDP_BACKEND", "nccl"), rank=rank, world_size=world)
    return rank, world, dev


def train(kind, out):
    latent = int(os.environ.get("LATENT", 64))
    steps = int(os.environ.get("AE_STEPS", 40000))
    batch = int(os.environ.get("AE_BATCH", 32))
    lr = float(os.environ.get("AE_LR", 1e-4 if kind == "swin" else 3e-4))
    smooth = float(os.environ.get("SMOOTH", 0))
    eval_every = int(os.environ.get("AE_EVAL_EVERY", 500))
    ckpt_every = int(os.environ.get("CKPT_EVERY", 200))
    time_limit = float(os.environ.get("TIME_LIMIT", 6600))
    t_start = time.time()
    rank, world, dev = setup()
    if os.path.exists(os.path.join(out, "done")):
        if rank == 0:
            print(f"{out}: already done", flush=True)
        dist.destroy_process_group()
        return
    assert batch % world == 0
    os.makedirs(out, exist_ok=True)
    logf = open(os.path.join(out, "log.txt"), "a") if rank == 0 else None

    def log(msg):
        if rank == 0:
            print(msg, flush=True)
            logf.write(msg + "\n"); logf.flush()

    mean, std = (t.float() for t in C.frame_stats(dev))
    norm = lambda x: (x.to(dev, non_blocking=True) - mean[:, None, None]) / std[:, None, None]
    torch.manual_seed(0)
    model = build(kind, latent).to(dev)
    torch.set_default_dtype(torch.float32)
    for p in model.parameters():
        dist.broadcast(p.data, 0)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    warmup = 500
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * (0.01 + 0.99 * 0.5 * (1 + np.cos(np.pi * min(s, steps) / steps))))
    state = {"step": 0, "best": float("inf"), "history": [], "t_train": 0.0}
    ckpt_path = os.path.join(out, "ckpt.pt")
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        state = ck["state"]
        log(f"resumed from step {state['step']}")
    else:
        log(f"{kind} autoencoder: latent {latent}, {sum(p.numel() for p in params):,} parameters, steps {steps}, "
            f"global batch {batch} frame pairs on {world} GPUs, lr {lr}, smooth {smooth}")

    # Fixed validation frames: 8 valid trajectories x frames 0, 50, 100, 150, 198 (+1 each).
    vds = Frames("valid", 0, (0,))
    vidx = np.linspace(0, len(vds.items) - 1, 8).astype(int)
    val = torch.cat([vds.read(int(i), t) for i in vidx for t in (0, 50, 100, 150, 198)])  # (80, C, H, W)

    def valid_loss():
        model.eval()
        mine = val[rank::world]
        with torch.no_grad():
            x = norm(mine)
            s = torch.stack([vrmse2(model.decode(model.encode(c)), c) * len(c) for c in x.split(8)]).sum()
        dist.all_reduce(s)
        model.train()
        return s.item() / len(val)

    per_rank = batch // world
    ds = Frames("train", (steps - state["step"]) * per_rank, seed=(rank, state["step"]))
    loader = torch.utils.data.DataLoader(ds, batch_size=per_rank, num_workers=6, pin_memory=True,
                                         prefetch_factor=4, persistent_workers=True)

    def save_ckpt():
        if rank == 0:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "state": state}, ckpt_path + ".tmp")
            os.replace(ckpt_path + ".tmp", ckpt_path)

    tl = time.time()
    t_data = 0.0
    for xb in loader:
        step = state["step"] + 1
        x = norm(xb).flatten(0, 1)  # (2b, C, H, W): frames t, t+1 interleaved
        torch.cuda.synchronize(); t_data += time.time() - tl
        tic = time.time()
        z = model.encode(x)
        loss_rec = vrmse2(model.decode(z), x)
        loss = loss_rec
        if smooth > 0:
            zz = z.view(-1, 2, latent)
            loss = loss + smooth * (zz[:, 1] - zz[:, 0]).pow(2).sum(-1).mean() / (z - z.mean(0)).pow(2).sum(-1).mean()
        opt.zero_grad(set_to_none=False)
        loss.backward()
        for p in params:
            if p.grad is None:  # parameters the forward pass does not use (some Swin buffers-as-parameters)
                p.grad = torch.zeros_like(p)
        flat = torch.cat([p.grad.reshape(-1) for p in params])
        dist.all_reduce(flat); flat /= world
        off = 0
        for p in params:
            p.grad.copy_(flat[off:off + p.numel()].view_as(p)); off += p.numel()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        # Skip steps whose loss spikes (frames with nearly no variance make VRMSE^2 huge); every rank agrees.
        tot = loss.detach().reshape(1).clone()
        dist.all_reduce(tot)
        lv = tot.item() / world
        recent = state.setdefault("recent", [])
        if not np.isfinite(lv) or (len(recent) >= 50 and lv > 10 * sorted(recent)[len(recent) // 2]):
            state["skipped"] = state.get("skipped", 0) + 1
            opt.zero_grad()
            log(f"step {step:6d} loss {lv:.3e}: spike, skipped ({state['skipped']} so far)")
        else:
            opt.step()
            recent.append(lv); del recent[:-200]
        sched.step()
        lr_now = sched.get_last_lr()[0]
        rec = loss_rec.detach().reshape(1)
        dist.all_reduce(rec)
        torch.cuda.synchronize()
        state["t_train"] += time.time() - tic
        state["step"] = step
        if step % 50 == 0 or step == 1:
            log(f"step {step:6d} recon VRMSE^2 {rec.item() / world:.4e} grad {gn.item():.2e} lr {lr_now:.1e} "
                f"| {state['t_train'] / max(step, 1):.2f}s/step compute, data wait {t_data / 50:.2f}s")
            t_data = 0.0
        if step % eval_every == 0 or step == steps:
            v = valid_loss()
            state["history"].append({"step": step, "train": rec.item() / world, "valid": v})
            better = v < state["best"]
            if better:
                state["best"] = v
                if rank == 0:
                    torch.save({"kind": kind, "latent_dim": latent, "model": model.state_dict(), "step": step,
                                "valid": v, "smooth": smooth}, os.path.join(out, "ae.pt"))
            log(f"  valid recon VRMSE^2 {v:.4e} (VRMSE ~ {v ** 0.5:.4f}){'  *' if better else ''}")
        if step % ckpt_every == 0 or step == steps:
            save_ckpt()
        stop = torch.tensor([float(time.time() - t_start > time_limit)], device=dev)
        dist.all_reduce(stop)
        if step >= steps or stop.item() > 0:
            break
        tl = time.time()
    if state["step"] < steps:
        save_ckpt()
        log(f"time limit: saved at step {state['step']}, exiting to resume")
    elif rank == 0:
        json.dump(state["history"], open(os.path.join(out, "history.json"), "w"), indent=1)
        open(os.path.join(out, "done"), "w").write(f"{state['step']}\n")
        log(f"done: best valid recon VRMSE^2 {state['best']:.4e}, {state['t_train'] * world / 3600:.1f} GPU-h")
    dist.barrier()
    dist.destroy_process_group()


# ---------------------------------------------------------------- encode


def encode(ckpt, out_dir):
    """Latents of every frame of every split, in the coeffs_{split}.pt layout of pod_basis.py."""
    rank, world, dev = setup()
    log = (lambda *a: print(*a, flush=True)) if rank == 0 else (lambda *a: None)
    model = load_autoencoder(ckpt, dev)
    torch.set_default_dtype(torch.float32)
    mean, std = (t.float() for t in C.frame_stats(dev))
    dup_path = os.environ.get("DUP_MAP", os.path.join(C.HERE, "..", "hnode_runs", "eval_full", "dup_map.json"))
    dup_map = json.load(open(dup_path)) if os.path.exists(dup_path) else {}
    t0 = time.time()
    for split in ["train", "valid", "test"]:
        ds = Frames(split, 0, (0,))
        mine = list(range(rank, len(ds.items), world))
        loader = torch.utils.data.DataLoader(mine, batch_size=None, num_workers=4, prefetch_factor=2,
                                             collate_fn=lambda i: (i, ds.read(i, 0, C.NUM_FRAMES)))
        res = {}
        with torch.no_grad():
            for k, (i, x) in enumerate(loader):
                x = (x.to(dev) - mean[:, None, None]) / std[:, None, None]
                res[i] = torch.cat([model.encode(c) for c in x.split(50)]).double().cpu()
                if k % 20 == 0:
                    log(f"  {split} rank 0: {k + 1}/{len(mine)} ({time.time() - t0:.0f}s)")
        parts = [None] * world
        dist.all_gather_object(parts, res)
        if rank == 0:
            allr = {i: v for p in parts for i, v in p.items()}
            scal = []
            for name, traj in ds.items:
                with h5py.File(os.path.join(ds.dir, name), "r") as f:
                    scal.append([float(f["scalars/Reynolds"][()]), float(f["scalars/Schmidt"][()])])
            scal = torch.tensor(np.array(scal, dtype=np.float32)).double()  # float32 values, as WellDataset
            meta = [dup_map.get(f"{split}:{i}", {}) for i in range(len(ds.items))]
            os.makedirs(out_dir, exist_ok=True)
            torch.save({"coeffs": torch.stack([allr[i] for i in range(len(ds.items))]), "scalars": scal,
                        "z": C.normalize_z(scal), "file": [f for f, _ in ds.items], "traj": [t for _, t in ds.items],
                        "duplicate_of_train": [m.get("duplicate_of_train") for m in meta] if split != "train" else None,
                        "autoencoder": ckpt}, os.path.join(out_dir, f"coeffs_{split}.pt"))
            log(f"saved {split}: {len(allr)} trajectories ({time.time() - t0:.0f}s)")
        dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "train":
        train(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 4 and sys.argv[1] == "encode":
        encode(sys.argv[2], sys.argv[3])
    else:
        sys.exit(__doc__)
