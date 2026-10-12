#!/usr/bin/env python
"""Spectral energy versus time for the latent-dimension study: kinetic-energy (and per-field) spectra of
the truth, the projection floor and the models of each row ROWS_DIR/{pod,fpod}R, decoded with the row's
POD or Fourier POD basis as eval_latent_rollout.py does with DECODER=pod:R; and the same for the two
64-dimensional Swin latent spaces (SWIN below), decoded with their Swin decoders.

Compute (8 GPUs; also runs as one process):
    srun -N1 -n8 -c7 --gpus-per-task=1 --gpu-bind=closest python -u spectral_energy.py compute OUT.npz [ROWS_DIR]
Plot (CPU):
    python spectral_energy.py plot OUT.npz [OUT_PREFIX]
    -> OUT_PREFIX_{pod,fpod,swin}_spectra.png (E(k) at frames 10, 50, 100, 199), _{pod,fpod,swin}_spectrogram.png
       (E(k, t), its ratio to the truth and the error spectrum over the truth), _{pod,fpod,swin}_bands.png (energy
       and error per band versus time), _vs_r.png (per band versus r, frames 100..199, with the Swin models at
       r = 64), the same for the held-out trajectories that do not duplicate a training one
       (OUT_PREFIX_unique_*), and _summary.txt. The swin figures hold both Swin latent spaces and, for reference,
       the Fourier POD HNODE at r = 64 and r = 1024.

Spectra. Fields are in physical units: the decoder's normalized output times the training std, plus the
mean. The domain is periodic, 0 <= x <= 1 by -1 <= y <= 1 (The Well's README; the HDF5 coordinates
both span [0, 1]), on 256 x 512 points, so the grid spacing is the same along x and y. With
f_hat = FFT2(f) / (256 * 512) and wavenumber k = |(n, m / 2)| (n, m: FFT indices along x, y; k in units
of 2 pi, k = 1 being one wavelength across x), the shell energy E_c(s) = sum of |f_hat_c|^2 over
s - 1/2 <= k < s + 1/2 adds up to the spatial mean of f_c^2 (Parseval). The kinetic-energy spectrum is
KE(s) = (E_vx(s) + E_vy(s)) / 2. The error spectrum Err_c(s) is the same sum of |f_hat_pred - f_hat_true|^2
(it adds up to the MSE). Err / E of the truth is 0 for a perfect prediction, 1 for a zero prediction and
2 for one with the right energy but random phases. Shell s is plotted at k = s; the bands (shells lo..hi-1)
are labelled with their k range (lo - 1/2 <= k < hi - 1/2). The plotted ratios (model / truth energy,
error / truth energy) are ratios of trajectory means, so they are energy weighted (dominated by the high-Re
trajectories); a field's Err summed over shells >= 1 over its truth E summed over shells >= 1 is its VRMSE^2,
up to eval's N - 1 variance and + 1e-7. The truth velocity is close to divergence free up to k ~ 64 but not
beyond (a property of the data), so read the k >= 63.5 band with care. The Swin decoders add a broadband
small-scale floor (4-pixel patch blockiness, with spikes on the patch lattice at k = 64, 90.5, 128, 181) that
is there without any dynamics (in the Swin AE projection, and from frame 1 on for the pretrained model), so
for them the k >= 15.5 bands mostly measure the decoder.

Models of each row: "oracle" (the basis's own coefficients of frames 1..199: the projection floor), the
HNODE runs selected as in summarize_latent_dim.py (SOAP, lr 1e-4, HNODE_DT = 1, trained to step 3000;
named "hnode <scale>") and the LSTM with the best learning rate (best.json). Swin groups: "swin" (the
two-stage Swin autoencoder of ae_train.py) has the same models, its HNODE runs at any number of steps
(named "hnode <steps / 1000>k"); "swin-e2e" (the pretrained end-to-end Swin-HNODE checkpoint) has only
"hnode" (its HNODE rollout from frames 0-1, stored by encode_latents.py): its encoder is trained on frames
0-1 alone, so its latents of later frames are no projection floor (decoded, VRMSE 0.9 at frame 10 and 8
at frame 199). OUT.npz:
    names, basis, r, run (M,)    model j (basis: pod, fpod, swin or swin-e2e)
    k (S,), frame (200,)         shell wavenumbers 0..S-1, frame index
    E      (M + 1, 4, 200, S)    mean over held-out trajectories of the shell energies (index 0: truth;
                                 fields tracer, pressure, vx, vy; NaN at frame 0 for the models)
    Err    (M, 4, 200, S)        mean error spectra
    E_unique, Err_unique         the same over the held-out trajectories that do not duplicate a training one
    band_KE (M + 1, N, 200, B), band_err (M, N, 200, B)   per trajectory, KE and KE error summed over
                                 the shells of each band in BANDS
    bands (B, 2), split (N,), idx (N,), duplicate (N,) (duplicate_of_train in the coefficient files)
Environment: BASES ("pod,fpod,swin,swin-e2e"; a Swin group is skipped when its latents are missing),
DIMS ("64,128,256,512,1024"; POD rows only), SERIES (regular expression on model names such as
"hnode floor:64"; default all), MAX_TRAJ (per split, for testing; 0 = all), EVAL_SPLITS ("valid,test"),
PLOT_HNODE (series drawn per r in the plots, default "hnode floor:64").
"""
import json
import os
import re
import sys
import time

import numpy as np
import torch

import shear_flow_common as C

torch.set_default_dtype(torch.float64)
H, W, NC = 256, 512, 4
FIELDS = ["tracer", "pressure", "vx", "vy"]
S = 182  # shells 0 .. 181 (the corner |(128, 128)| = 181.02)
BANDS = [(1, 4), (4, 16), (16, 64), (64, S)]  # shells lo <= s < hi; s = 0 is the spatial mean
HERE = os.path.dirname(os.path.abspath(__file__))
SWIN_X = 32  # x position of the (64-dimensional) Swin models in the versus-r figure
# The 64-dimensional Swin latent spaces, paths relative to ROWS_DIR/..: (latent directory, decoder checkpoint
# of ae_train.py or "swin" for the pretrained checkpoint's decoder, row of latent models trained on them).
SWIN = {"swin": ("ae/swin64/latents", "ae/swin64/ae.pt", "ae/swin64/row"),
        "swin-e2e": ("latent_export/heldout", "swin", None)}


def row_models(row, name_of):
    """[(name, run directory or "oracle")]: the oracle, the HNODE runs selected as in summarize_latent_dim.py
    (named name_of(history), skipped when it is None) and the LSTM of best.json."""
    found = [("oracle", "oracle")]
    for run in sorted(os.listdir(row)):
        h = os.path.join(row, run, "history.json")
        if not (run.startswith("hnode") and os.path.exists(h) and os.path.exists(os.path.join(row, run, "pred.pt"))):
            continue
        d = json.load(open(h))
        if d.get("optimizer") != "soap" or d.get("lr") != 1e-4 or d.get("hnode_dt") != 1.0 or name_of(d) is None:
            continue
        found.append((name_of(d), os.path.join(row, run)))
    best = os.path.join(row, "best.json")
    lstm = json.load(open(best)).get("lstm") if os.path.exists(best) else None
    if lstm and os.path.exists(os.path.join(lstm["dir"], "pred.pt")):
        found.append(("lstm", lstm["dir"]))
    return found


def find_models(rows_dir, bases, dims, series):
    """[(basis, r, name, run)] in the order of the output arrays; run is a run directory, "oracle" or "hnode"
    (C.LATENT_DIM is the Swin latent dimension, 64)."""
    models = []
    for basis in bases:
        if basis in SWIN:
            lat, _, row = SWIN[basis]
            if not os.path.isdir(os.path.join(rows_dir, "..", lat)) or (row and not os.path.isdir(os.path.join(rows_dir, "..", row))):
                continue
            found = ([("hnode", "hnode")] if row is None else
                     row_models(os.path.normpath(os.path.join(rows_dir, "..", row)), lambda d: f"hnode {d['history'][-1]['step'] // 1000}k"))
            models += [(basis, C.LATENT_DIM, name, run) for name, run in found if re.search(series, name)]
            continue
        for r in dims:
            row = os.path.join(rows_dir, f"{basis}{r}")
            if not os.path.isdir(row):
                continue
            found = row_models(row, lambda d: "hnode " + d.get("scale", "mode (Sept)") if d["history"][-1]["step"] == 3000 else None)
            models += [(basis, r, name, run) for name, run in found if re.search(series, name)]
    return models


def latent_file(rows_dir, basis, split):
    """The basis's latents of a split: pod_basis.py / ae_train.py's coeffs_{split}.pt, or encode_latents.py's
    {split}.pt with "latents" renamed "coeffs"."""
    if basis not in SWIN:
        return torch.load(os.path.join(rows_dir, "..", basis, "full", f"coeffs_{split}.pt"), weights_only=False)
    d = os.path.join(rows_dir, "..", SWIN[basis][0])
    path = os.path.join(d, f"{split}.pt")  # the order of eval_latent_rollout.load_predictions
    lat = torch.load(path if os.path.exists(path) else os.path.join(d, f"coeffs_{split}.pt"), weights_only=False)
    if "latents" in lat:
        lat["coeffs"] = lat.pop("latents")
    return lat


def shell_index(device):
    n = torch.fft.fftfreq(H, 1.0 / H, device=device)  # 0..127, -128..-1 (along x, length 1)
    m = torch.fft.fftfreq(W, 1.0 / W, device=device) / 2  # along y, length 2
    # shell s holds s - 1/2 <= k < s + 1/2 (rounding half up: k = 1/2, the longest wave along y, is in
    # shell 1, so shell 0 is the spatial mean alone)
    return (n[:, None] ** 2 + m[None, :] ** 2).sqrt().add(0.5).floor().long().reshape(-1)


def spectrum(f_hat, shells):
    """(T, C, H, W) complex -> (T, C, S) shell sums of |f_hat|^2."""
    p = f_hat.abs().pow(2).reshape(*f_hat.shape[:2], -1)
    return torch.zeros(*f_hat.shape[:2], S, device=p.device).index_add_(2, shells, p)


def compute(out_npz, rows_dir):
    import torch.distributed as dist

    from eval_latent_rollout import load_decoder

    bases = os.environ.get("BASES", "pod,fpod,swin,swin-e2e").split(",")
    dims = [int(r) for r in os.environ.get("DIMS", "64,128,256,512,1024").split(",")]
    splits = os.environ.get("EVAL_SPLITS", "valid,test").split(",")
    max_traj = int(os.environ.get("MAX_TRAJ", 0))
    rank, local_rank, world_size = C.distributed_env()
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29506")
    user = os.environ.get("USER", "user")  # one MIOpen database per rank for the Swin decoders' convolutions
    os.environ.setdefault("MIOPEN_USER_DB_PATH", f"/tmp/miopen-{user}-{rank}")
    os.environ.setdefault("MIOPEN_CUSTOM_CACHE_DIR", f"/tmp/miopen-{user}-{rank}")
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    device = C.pick_device(local_rank)
    log = (lambda *a: print(*a, flush=True)) if rank == 0 else (lambda *a: None)
    mean, std = (s[:, None, None] for s in C.frame_stats(device))
    shells = shell_index(device)

    rows_dir = os.path.abspath(rows_dir)
    models = find_models(rows_dir, bases, dims, os.environ.get("SERIES", ""))
    assert models, "no models matched"
    log(f"{len(models)} models:\n  " + "\n  ".join(f"{b}{r} {n}: {run}" for b, r, n, run in models))
    M, t0 = len(models), time.time()
    decode = {}  # basis -> ((T, r) latents -> (T, C, H, W) normalized frames), on the device
    for b in sorted({m[0] for m in models}):
        if b in SWIN:
            lat_dir, ckpt, _ = SWIN[b]
            spec = ckpt if ckpt == "swin" else "ae:" + os.path.join(rows_dir, "..", ckpt)
            decode[b] = load_decoder(spec, os.path.join(rows_dir, "..", lat_dir), device)[0]
            continue
        d = torch.load(os.path.join(rows_dir, "..", b, "full", "basis.pt"), weights_only=False)
        # mean (H*W*C,) normalized and modes (H*W*C, K) float64, in (H, W, C) order
        mu, modes = d["mean"].to(device).reshape(-1), d["modes"].to(device).double()
        decode[b] = lambda a, mu=mu, modes=modes: (mu + a @ modes[:, :a.shape[-1]].T).reshape(-1, H, W, NC).permute(0, 3, 1, 2)

    E = torch.zeros(2, M + 1, NC, C.NUM_FRAMES, S, device=device)  # sums over all, over unique trajectories
    Err = torch.zeros(2, M, NC, C.NUM_FRAMES, S, device=device)
    E[:, 1:, :, 0] = Err[:, :, :, 0] = float("nan")
    band_rows = []  # per trajectory: (split, idx, duplicate, band_KE (M + 1, 200, B), band_err (M, 200, B))

    def bands(x):  # (..., S) -> (..., B)
        return torch.stack([x[..., lo:hi].sum(-1) for lo, hi in BANDS], -1)

    for split in splits:
        well = C.well_split(split)
        n = min(len(well), max_traj) if max_traj else len(well)
        mine = list(range(rank, n, world_size))
        lat = {b: latent_file(rows_dir, b, split) for b in decode}
        order = C.split_index(split)
        for b in lat:  # coefficients (and the predictions made from them) in WellDataset's item order
            assert list(zip(lat[b]["file"], lat[b]["traj"])) == order, (split, b)
        preds = []  # (my trajectories, 199, r) per model, on the CPU
        for b, r, name, run in models:
            if run == "oracle":
                p = lat[b]["coeffs"][:, 1:, :r]
            elif run == "hnode":  # the pretrained end-to-end rollout stored with the latents (swin-e2e)
                p = lat[b]["hnode"]
            else:
                p = torch.load(os.path.join(run, "pred.pt"), weights_only=False)[split]
            assert p.shape == (len(well), C.NUM_FRAMES - 1, r), (b, r, name, p.shape)
            preds.append(p[mine].clone())
        loader = torch.utils.data.DataLoader(mine, batch_size=None, num_workers=3, prefetch_factor=2,
                                             collate_fn=lambda i: (i, *C.full_trajectory(well, i)))
        with torch.no_grad():
            for k, (i, frames, scalars) in enumerate(loader):
                assert i == mine[k], (split, i, k)
                for b in lat:
                    assert torch.equal(scalars, lat[b]["scalars"][i]), (split, i, b)
                dup = lat[next(iter(lat))]["duplicate_of_train"][i] is not None
                sets = [0] if dup else [0, 1]
                truth_hat = torch.fft.fft2(frames.to(device).permute(0, 3, 1, 2)) / (H * W)  # (200, C, H, W)
                e = spectrum(truth_hat, shells)
                E[sets, 0] += e.transpose(0, 1)
                bke, berr = [bands((e[:, 2] + e[:, 3]) / 2)], []
                for j, (b, r, name, run) in enumerate(models):
                    ej, errj = [], []
                    for c, a in zip(range(0, C.NUM_FRAMES - 1, 50), preds[j][k].to(device).split(50)):
                        x = decode[b](a)
                        x_hat = torch.fft.fft2(x * std + mean) / (H * W)
                        ej.append(spectrum(x_hat, shells))
                        errj.append(spectrum(x_hat - truth_hat[1 + c:1 + c + len(a)], shells))
                    ej, errj = torch.cat(ej), torch.cat(errj)
                    E[sets, j + 1, :, 1:] += ej.transpose(0, 1)
                    Err[sets, j, :, 1:] += errj.transpose(0, 1)
                    bke.append(torch.cat([torch.full((1, len(BANDS)), float("nan"), device=device),
                                          bands((ej[:, 2] + ej[:, 3]) / 2)]))
                    berr.append(torch.cat([torch.full((1, len(BANDS)), float("nan"), device=device),
                                           bands((errj[:, 2] + errj[:, 3]) / 2)]))
                band_rows.append((split, i, dup, torch.stack(bke).cpu().numpy(), torch.stack(berr).cpu().numpy()))
                if k % 5 == 0 or k == len(mine) - 1:
                    log(f"  {split} rank 0: {k + 1}/{len(mine)} ({time.time() - t0:.0f}s)")

    sums = [x.cpu() for x in (E, Err)]
    for x in sums:
        dist.all_reduce(x)  # NaN at frame 0 stays NaN
    gathered = [None] * world_size
    dist.all_gather_object(gathered, band_rows)
    if rank == 0:
        rows = sorted((r for part in gathered for r in part), key=lambda r: (r[0], r[1]))
        dup = np.array([r[2] for r in rows])
        N, Nu = len(rows), int((~dup).sum())
        os.makedirs(os.path.dirname(os.path.abspath(out_npz)), exist_ok=True)
        np.savez(out_npz, names=np.array([m[2] for m in models]), basis=np.array([m[0] for m in models]),
                 r=np.array([m[1] for m in models]), run=np.array([m[3] for m in models]),
                 k=np.arange(S), frame=np.arange(C.NUM_FRAMES), bands=np.array(BANDS),
                 E=sums[0][0].numpy() / N, Err=sums[1][0].numpy() / N,
                 E_unique=sums[0][1].numpy() / max(Nu, 1), Err_unique=sums[1][1].numpy() / max(Nu, 1),
                 band_KE=np.stack([r[3] for r in rows], 1), band_err=np.stack([r[4] for r in rows], 1),
                 split=np.array([r[0] for r in rows]), idx=np.array([r[1] for r in rows]), duplicate=dup)
        log(f"saved {out_npz}: {M} models, {N} trajectories ({Nu} not duplicates of train) ({time.time() - t0:.0f}s)")
    dist.barrier()
    dist.destroy_process_group()


def plot(npz, prefix):
    """Every figure for all held-out trajectories (PREFIX_*) and for those that do not duplicate a training
    trajectory (PREFIX_unique_*)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, TwoSlopeNorm

    d = dict(np.load(npz))
    names, basis, rs, k, frame = d["names"], d["basis"], d["r"], d["k"], d["frame"]
    hn = os.environ.get("PLOT_HNODE", "hnode floor:64")
    ks = slice(1, H // 2)  # shells 1..127 (128 lacks the +Nyquist row and column; beyond are corners)
    label = {"pod": "POD", "fpod": "Fourier POD", "swin": "Swin, 64 dimensions"}
    btxt = [f"{lo - 0.5:g} <= k < {hi - 0.5:g}" if hi < S else f"k >= {lo - 0.5:g}" for lo, hi in d["bands"]]
    style = {"oracle": dict(color="k", ls="--"), "hnode mode (Sept)": dict(color="tab:blue", marker="o", ls="--"),
             "hnode mode": dict(color="tab:orange", marker="o"), "hnode floor:64": dict(color="tab:green", marker="D"),
             "hnode shared": dict(color="tab:purple", marker="s"), "lstm": dict(color="tab:red", marker="^", ls=":")}
    # Swin models: (label, line style in the swin figures, marker at r = 64 in the versus-r figure)
    swin_style = {("swin", "oracle"): ("Swin AE projection", dict(color="0.5", ls="--", lw=1), dict(color="0.35", marker="x", ms=7)),
                  ("swin-e2e", "hnode"): ("Swin-HNODE (pretrained)", dict(color="navy", lw=1.6),
                                          dict(color="navy", marker="*", ms=13, mec="k")),
                  ("swin", "hnode 9k"): ("Swin AE + HNODE 9k", dict(color="tab:brown", lw=1.4), dict(color="tab:brown", marker="P", ms=8)),
                  ("swin", "hnode 3k"): ("Swin AE + HNODE 3k", dict(color="tab:cyan", lw=1.2), dict(color="tab:cyan", marker="p", ms=8, mec="k")),
                  ("swin", "lstm"): ("Swin AE + LSTM", dict(color="tab:pink", ls="-.", lw=1.2), dict(color="tab:pink", marker="v", ms=8))}
    ke_of = lambda e: (e[..., 2, :, :] + e[..., 3, :, :]) / 2  # (..., 200, S)
    find = lambda b, name, r=None: next((j for j in range(len(names)) if basis[j] == b and names[j] == name
                                         and (r is None or rs[j] == r)), None)
    cmap = lambda x: plt.get_cmap("viridis")(0.85 * x)  # r colours, without the faint yellow end
    present = [b for b in ["pod", "fpod"] if b in set(basis)]
    sets = []  # figure sets: (file key, title, curves [(label, model index, line style)], models in the table, note)
    for b in present:
        dims = sorted({int(r) for j, r in enumerate(rs) if basis[j] == b})
        col = {r: cmap(i / max(1, len(dims) - 1)) for i, r in enumerate(dims)}
        # Curves: projection at the smallest and largest r, the HNODE series at every r, LSTM at the smallest r.
        curves = [(f"projection r={r}", find(b, "oracle", r), dict(color="0.5", ls=ls, lw=1))
                  for r, ls in [(dims[0], "--"), (dims[-1], ":")] if find(b, "oracle", r) is not None]
        curves += [(f"HNODE r={r}", find(b, hn, r), dict(color=col[r], lw=1.4)) for r in dims if find(b, hn, r) is not None]
        curves += [(f"LSTM r={r}", find(b, "lstm", r), dict(color="tab:red", ls="-.", lw=1.2))
                   for r in dims[:1] if find(b, "lstm", r) is not None]
        sets.append((b, f"{label[b]} ({hn})", curves, [j for j in range(len(names)) if basis[j] == b], ""))
    # Swin: both latent spaces, with the Fourier POD HNODE at the smallest and largest r for reference.
    curves = [(lab, find(*key), st) for key, (lab, st, _) in swin_style.items()]
    curves += [(f"F-POD HNODE r={r}", find("fpod", hn, r), dict(color=cmap(c), lw=1.1, ls=(0, (4, 2))))
               for r, c in [(64, 0.0), (1024, 1.0)]]
    curves = [c for c in curves if c[1] is not None]
    if any(b in SWIN for b in basis):
        sets.append(("swin", label["swin"], curves, [j for j in range(len(names)) if basis[j] in SWIN],
                     f"\nAE: two-stage Swin autoencoder; pretrained: the end-to-end Swin-HNODE checkpoint; F-POD references: {hn}."
                     f"\nAbove k ~ 15 the Swin decoders add a broadband floor (already in the AE projection)."))
    lines = []
    for subset, sel in [("all", np.ones(len(d["duplicate"]), bool)), ("unique", ~d["duplicate"])]:
        sfx = "E" if subset == "all" else "E_unique"
        KE, ERR = ke_of(d[sfx]), ke_of(d["Err" if subset == "all" else "Err_unique"])  # (M + 1, 200, S), (M, 200, S)
        truth = KE[0]
        bke, berr = d["band_KE"][:, sel].mean(1), d["band_err"][:, sel].mean(1)  # (M + 1, 200, B), (M, 200, B)
        bt = bke[0]
        pre = prefix if subset == "all" else f"{prefix}_unique"
        who = f"{sel.sum()} held-out trajectories" + ("" if subset == "all" else " that are not duplicates of train")
        weighted = "ratios of trajectory-mean energies"
        for b, title, curves, members, note in sets:
            # 1. Kinetic-energy spectra at a few frames.
            fig, axes = plt.subplots(1, 4, figsize=(17, 4.4), sharey=True)
            for ax, t in zip(axes, [10, 50, 100, 199]):
                ax.loglog(k[ks], truth[t, ks], "k-", lw=2.2, label="truth")
                for lab, j, st in curves:
                    ax.loglog(k[ks], KE[j + 1, t, ks], label=lab, **st)
                ax.set(title=f"frame {t}", xlabel="wavenumber k (units of 2$\\pi$; Lx = 1)")
                ax.set_ylim(truth[1:, ks].min() * 0.1, truth[1:, ks].max() * 3)
                ax.grid(alpha=0.3, which="both")
            axes[0].set_ylabel("kinetic energy E(k)")
            axes[0].legend(fontsize=7)
            fig.suptitle(f"{title}: kinetic-energy spectrum, mean over {who}{note}")
            fig.tight_layout()
            fig.savefig(f"{pre}_{b}_spectra.png", dpi=120)
            plt.close(fig)

            # 2. Spectral energy versus time: E(k, t), its ratio to the truth and the error over the truth.
            cols = [("truth", None)] + [(lab, j) for lab, j, _ in curves]
            fig, axes = plt.subplots(3, len(cols), figsize=(2.6 * len(cols) + 1, 9), sharex=True, sharey=True,
                                     squeeze=False)
            kk = k[ks]
            norm0 = LogNorm(truth[1:, ks].min(), truth[1:, ks].max())
            m1 = m2 = None
            for c, (lab, j) in enumerate(cols):
                e = truth if j is None else KE[j + 1]
                m0 = axes[0, c].pcolormesh(frame[1:], kk, e[1:, ks].T, norm=norm0, cmap="magma", shading="nearest")
                axes[0, c].set_title(lab, fontsize=9)
                if j is None:
                    axes[1, c].axis("off"), axes[2, c].axis("off")
                    continue
                with np.errstate(divide="ignore", invalid="ignore"):
                    m1 = axes[1, c].pcolormesh(frame[1:], kk, np.log10(e[1:, ks] / truth[1:, ks]).T, cmap="RdBu_r",
                                               norm=TwoSlopeNorm(0, -1, 1), shading="nearest")
                    m2 = axes[2, c].pcolormesh(frame[1:], kk, np.log10(ERR[j, 1:, ks] / truth[1:, ks]).T, cmap="PuOr_r",
                                               norm=TwoSlopeNorm(0, -2, 2), shading="nearest")
            for a in axes.ravel():
                a.set_yscale("log")
            for a in axes[-1]:
                a.set_xlabel("frame")
            for a in axes[:, :2].ravel():
                a.set_ylabel("k")
            fig.colorbar(m0, ax=axes[0].tolist(), label="E(k, t)", fraction=0.02, extend="both")
            if m1 is not None:
                fig.colorbar(m1, ax=axes[1].tolist(), label="log10 (model / truth energy)", fraction=0.02, extend="both")
                fig.colorbar(m2, ax=axes[2].tolist(), label="log10 (error / truth energy)\n0: no skill", fraction=0.02,
                             extend="both")
            fig.suptitle(f"{title}: kinetic-energy spectrum versus time, mean over {who} (rows 2-3: {weighted}; "
                         f"colours clipped at the colour-bar ends){note}")
            fig.savefig(f"{pre}_{b}_spectrogram.png", dpi=110)
            plt.close(fig)

            # 3. Energy in wavenumber bands versus time, and the error over the truth per band.
            fig, axes = plt.subplots(2, len(btxt), figsize=(4.2 * len(btxt), 7), sharex=True)
            for c, txt in enumerate(btxt):
                axes[0, c].semilogy(frame, bt[:, c], "k-", lw=2.2, label="truth")
                for lab, j, st in curves:
                    axes[0, c].semilogy(frame, bke[j + 1, :, c], label=lab, **st)
                    axes[1, c].semilogy(frame, berr[j, :, c] / bt[:, c], label=lab, **st)
                axes[0, c].set_title(f"band {txt}")
                axes[1, c].axhline(1, color="k", lw=0.6)
                axes[1, c].set_ylim(bottom=max(axes[1, c].get_ylim()[0], 5e-6))
                axes[1, c].set_xlabel("frame")
                for a in axes[:, c]:
                    a.grid(alpha=0.3, which="both")
            axes[0, 0].set_ylabel("kinetic energy in band")
            axes[1, 0].set_ylabel("error / truth energy (1: no skill)")
            axes[0, 0].legend(fontsize=7)
            fig.suptitle(f"{title}: kinetic energy per wavenumber band versus time, mean over {who}{note}")
            fig.tight_layout()
            fig.savefig(f"{pre}_{b}_bands.png", dpi=120)
            plt.close(fig)

            # Table: energy ratio and error over the truth per band at a few frames, every model of the set.
            lines.append(f"\n{label[b]}, {who}: model/truth energy (error/truth energy) per band ({weighted})")
            lines.append(f"{'model':<26}{'frame':>6}" + "".join(f"{t:>24s}" for t in btxt))
            for j in members:
                tag = swin_style.get((basis[j], names[j]), (f"{basis[j]} {names[j]}",))[0] if basis[j] in SWIN else f"{names[j]} r={rs[j]}"
                for t in [10, 100, 199]:
                    lines.append(f"{tag:<26}{t:>6}" + "".join(
                        f"{bke[j + 1, t, c] / bt[t, c]:>13.4g} ({berr[j, t, c] / bt[t, c]:8.4g})" for c in range(len(btxt))))

        # 4. Versus r: per band, the error over the truth and the energy ratio, averaged over frames 100..199;
        # the Swin models (all r = 64) as points in their own slot left of r = 64 (a dotted line at the pretrained
        # Swin-HNODE's value).
        if not present:
            continue
        series = [s for s in style if s in set(names)]
        fig, axes = plt.subplots(2 * len(present), len(btxt), figsize=(4.2 * len(btxt), 3.2 * 2 * len(present)),
                                 sharex=True, squeeze=False)
        for i, b in enumerate(present):
            for s in series:
                js = sorted((rs[j], j) for j in range(len(names)) if basis[j] == b and names[j] == s)
                if not js:
                    continue
                x = [r for r, _ in js]
                err = np.array([(berr[j, 100:] / bt[100:]).mean(0) for _, j in js])  # (len(js), B)
                ene = np.array([(bke[j + 1, 100:] / bt[100:]).mean(0) for _, j in js])
                lab = "projection" if s == "oracle" else s
                for c in range(len(btxt)):
                    axes[2 * i, c].plot(x, err[:, c], label=lab, ms=4, **style[s])
                    axes[2 * i + 1, c].plot(x, ene[:, c], label=lab, ms=4, **style[s])
            for n, (key, (lab, _, st)) in enumerate(swin_style.items()):
                j = find(*key)
                if j is None:
                    continue
                err, ene = (berr[j, 100:] / bt[100:]).mean(0), (bke[j + 1, 100:] / bt[100:]).mean(0)
                for c in range(len(btxt)):
                    for a, v in [(axes[2 * i, c], err[c]), (axes[2 * i + 1, c], ene[c])]:
                        a.plot([SWIN_X * 2 ** (0.15 * (n - 2))], [v], ls="none", label=lab, **st)
                        if key == ("swin-e2e", "hnode"):
                            a.axhline(v, color=st["color"], ls=":", lw=0.8, label=f"{lab} level")
            for c, txt in enumerate(btxt):
                axes[2 * i, c].set_title(f"{label[b]}, band {txt}", fontsize=9)
                axes[2 * i, c].axhline(1, color="0.6", lw=0.6)
                axes[2 * i + 1, c].axhline(1, color="0.6", lw=0.6)
            axes[2 * i, 0].set_ylabel("error / truth energy")
            axes[2 * i + 1, 0].set_ylabel("model / truth energy")
        legend = {}  # label -> handle, over every basis
        for a in axes[::2, 0]:
            legend.update({lab: h for h, lab in zip(*a.get_legend_handles_labels()) if lab not in legend})
        swin = any(b in SWIN for b in basis)
        for a in axes.ravel():
            a.set_xscale("log", base=2)
            a.set_yscale("log")
            a.grid(alpha=0.3, which="both")
            if swin:
                ticks = [SWIN_X] + sorted({int(r) for j, r in enumerate(rs) if basis[j] in present})
                a.set_xticks(ticks, ["Swin\n(64)"] + [str(r) for r in ticks[1:]])
        for a in axes[-1]:
            a.set_xlabel("latent dimension r")
        fig.legend(legend.values(), legend.keys(), loc="center right", fontsize=8)
        fig.suptitle(f"Kinetic energy per band versus latent dimension, frames 100-199, {who} ({weighted}"
                     + ("; the Swin models, all 64-dimensional, at the left)" if swin else ")"))
        fig.tight_layout(rect=(0, 0, 0.84, 1))
        fig.savefig(f"{pre}_vs_r.png", dpi=120)
        plt.close(fig)

    print("\n".join(lines))
    open(f"{prefix}_summary.txt", "w").write("\n".join(lines) + "\n")
    print(f"saved {prefix}[_unique]_{{{','.join(s[0] for s in sets)}}}_{{spectra,spectrogram,bands}}.png, "
          + (f"{prefix}[_unique]_vs_r.png, " if present else "") + f"{prefix}_summary.txt")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "compute":
        compute(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else os.path.join(HERE, "..", "hnode_runs", "rows"))
    elif len(sys.argv) >= 3 and sys.argv[1] == "plot":
        plot(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else os.path.splitext(sys.argv[2])[0])
    else:
        sys.exit(__doc__)
