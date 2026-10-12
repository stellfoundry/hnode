#!/usr/bin/env python
"""POD basis for shear flow: a frozen linear encoder/decoder shared by HNODE and the latent baselines.

    srun -N1 -n8 -c7 --gpus-per-task=1 --gpu-bind=closest python -u pod_basis.py OUT_DIR
    (or sbatch run_pod_basis.sh)

Frames are normalized per channel with the training stats (as for the Swin model) and flattened in
(H, W, C) order: 256 x 512 x 4 = 524,288 values. Each rank holds 1/world_size of the rows (H) of
every frame, read straight from The Well's HDF5 files.

1. Basis (method of snapshots): train frames 0, STRIDE, 2*STRIDE, ... of every trajectory form X
   (n snapshots x d pixels). G = (X - mean) (X - mean)^T is summed over ranks; its top K eigenpairs
   (randomized subspace iteration) give the modes Phi = (X - mean)^T V / sigma. The same is done on
   the snapshots at twice the stride (a submatrix of G) to check that the subsampling is fine enough.
2. Coefficients a = Phi^T (x - mean) of every frame of every split (K modes; any r <= K is a slice).
3. Projection floor: VRMSE (The Well's definition, as in MODE=eval) of the rank-r reconstruction of
   every valid/test frame, for r in EVAL_R, with both bases.

Outputs in OUT_DIR:
    basis.pt             mean (H, W, C) float64, modes (H*W*C, K) float32, sigma (K), sigma_half (K)
    coeffs_{split}.pt    coeffs (N, 200, K) float64, scalars, z, file, traj, duplicate_of_train
    projection_floor.npz vrmse_well (2 bases, len(EVAL_R), N, 200, C), r, split, idx, Re, Sc, duplicate
    summary.txt, pod_floor.png

BASIS=fourier: Fourier POD instead of step 1. The flow is periodic and statistically homogeneous along
H (x), so the DFT along H decouples the covariance by wavenumber k = 0..128: for each k, POD over the
W x C = 2,048 values (a 2,048 x 2,048 Hermitian eigenproblem). Each complex mode psi(w, c) e^{2 pi i k x/H}
gives two real orthonormal modes (its real and imaginary parts, i.e. cos and sin phases; one for k = 0
and k = H/2), each carrying the eigenvalue. The K real modes with the largest eigenvalues are kept, in
that order, with no cos/sin pair split across the ranks in EVAL_R. The output has the same layout as POD
(basis.pt also lists each mode's (k, j, part)), so coefficients, latent_baselines.py and DECODER=pod:R
work unchanged. For a given rank, POD's projection error on the training snapshots is the smallest
possible, but on held-out frames Fourier POD can do better: every training frame counts at all H shifts
(8 training trajectories: held-out VRMSE 0.77 vs 1.28 at r = 64). It may also make the dynamics simpler
(advection along x is a phase rotation of each (cos, sin) pair).

Environment: STRIDE (4), K (1024), EVAL_R ("32,64,128,256,512,1024"), MAX_TRAJ (per split, for
testing; 0 = all), PROJECT_SPLITS ("train,valid,test"), DUP_MAP, BASIS ("pod" or "fourier").
"""
import math
import json
import os
import sys
import time

import h5py
import numpy as np
import torch
import torch.distributed as dist

import shear_flow_common as C

torch.set_default_dtype(torch.float64)
H, W, NC = 256, 512, 4


class RowSlices(torch.utils.data.Dataset):
    """Frames `frames` of trajectory items[i], rows `rows` only: (T, rows, W, C) float32, and [Re, Sc]."""

    def __init__(self, split, items, rows, frames):
        self.dir = os.path.join(C.WELL_DATA_PATH, split)
        self.items, self.rows, self.frames = items, rows, frames
        self.files = {}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        name, traj = self.items[i]
        if name not in self.files:  # opened lazily, once per worker process
            self.files[name] = h5py.File(os.path.join(self.dir, name), "r")
        f, r = self.files[name], self.rows
        x = np.concatenate([f["t0_fields/tracer"][traj, self.frames, r][..., None],
                            f["t0_fields/pressure"][traj, self.frames, r][..., None],
                            f["t1_fields/velocity"][traj, self.frames, r]], axis=-1)
        scalars = torch.tensor([f["scalars/Reynolds"][()], f["scalars/Schmidt"][()]]).double()  # float32 values
        return i, torch.from_numpy(x), scalars


def loader(ds):
    return torch.utils.data.DataLoader(ds, batch_size=None, shuffle=False, num_workers=4, prefetch_factor=2)


def top_eigenpairs(G, k, iters=4, oversample=128, seed=0):
    """Top-k eigenpairs of a symmetric PSD matrix by randomized subspace iteration.
    Matrix products on G's device; QR and the small eigenproblem on the CPU (hipSOLVER-free)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    Q = torch.randn(G.shape[0], k + oversample, generator=g)
    for _ in range(iters + 1):
        Q, _ = torch.linalg.qr((G @ Q.to(G.device)).cpu())
    B = Q.T @ (G @ Q.to(G.device)).cpu()
    lam, Wv = torch.linalg.eigh((B + B.T) / 2)
    lam, Wv = lam.flip(0)[:k], Wv.flip(1)[:, :k]
    V = Q @ Wv
    res = ((G @ V.to(G.device)).cpu() - V * lam).norm(dim=0) / lam  # relative eigen-residual per pair
    return lam, V, res


def fourier_basis(X, rows, half, K, bounds, rank, world_size, device, log, t0, chunk=200):
    """Fourier POD from the centered snapshots X (n, rows x W x C) of this rank's rows of H.
    Returns Phi (d_rank, K), Phi_h, sigma (K), sigma_h, modes [(k, j, part)], modes_h."""
    n, D, nk = X.shape[0], W * NC, H // 2 + 1
    xs = torch.arange(rows.start, rows.stop, device=device, dtype=torch.float64)
    theta = 2 * math.pi * xs[:, None] * torch.arange(nk, device=device, dtype=torch.float64)[None, :] / H
    dft = torch.polar(torch.full_like(theta, H ** -0.5), -theta)  # (rows, nk): e^{-i theta} / sqrt(H)
    mine = list(range(rank, nk, world_size))  # wavenumbers whose covariance this rank accumulates
    cov = torch.zeros(2, len(mine), D, D, dtype=torch.complex128, device=device)  # all, half snapshots
    in_half = torch.zeros(n, dtype=torch.bool, device=device); in_half[half] = True
    Xr = X.view(n, -1, D)
    for s in range(0, n, chunk):
        F = torch.einsum("trd,rk->tkd", Xr[s:s + chunk].to(torch.complex128), dft)  # this rank's rows
        F = F.contiguous()
        dist.all_reduce(F)  # full DFT along H
        Fm = F[:, mine].transpose(0, 1)  # (len(mine), t, D)
        cov[0] += Fm.transpose(1, 2) @ Fm.conj()  # sum_t f f^H
        Fh = Fm[:, in_half[s:s + chunk]]
        cov[1] += Fh.transpose(1, 2) @ Fh.conj()
        if (s // chunk) % 50 == 0:
            log(f"  Fourier covariance: {min(s + chunk, n)}/{n} snapshots ({time.time() - t0:.0f}s)")
    del F, Fm, Fh

    # Eigenpairs per wavenumber on the CPU (k = 0 and H/2 are real: real eigenvectors).
    lam = torch.zeros(2, nk, D, dtype=torch.float64, device=device)
    psi = {}
    for b in range(2):
        for i, k in enumerate(mine):
            M = cov[b, i].cpu()
            if k in (0, H // 2):
                l, v = torch.linalg.eigh(M.real)
                v = v.to(torch.complex128)
            else:
                l, v = torch.linalg.eigh(M)
            lam[b, k] = l.flip(0).to(device)
            psi[b, k] = v.flip(1)  # columns: eigenvectors, descending
    del cov
    torch.cuda.empty_cache()
    dist.all_reduce(lam)
    log(f"Fourier POD eigenpairs: {nk} wavenumbers x {D} ({time.time() - t0:.0f}s)")

    def select(l):
        """Real modes [(k, j, part)] by descending eigenvalue; no (c, s) pair across a bound."""
        top = torch.topk(l.reshape(-1), min(l.numel(), 2 * K + 4096)).indices.tolist()
        cand = [(i // D, i % D) for i in top]
        used, out, i = set(), [], 0
        while len(out) < K:
            while cand[i] in used:
                i += 1
            k, j = cand[i]
            if k in (0, H // 2):
                out.append((k, j, "r")); used.add((k, j)); continue
            nxt = min([b for b in bounds if b > len(out)] + [K])
            if len(out) + 2 > nxt:  # the pair would straddle a bound: take the best unused single first
                single = next((c for c in cand[i:] if c not in used and c[0] in (0, H // 2)), None)
                if single is not None:
                    out.append((*single, "r")); used.add(single); continue
                log(f"  no single mode left to keep a pair whole at r = {nxt}; the pair straddles it")
            out += [(k, j, "c"), (k, j, "s")]; used.add((k, j))
        return out[:K]

    results = []
    for b in range(2):
        modes = select(lam[b])
        pairs = sorted({(k, j) for k, j, _ in modes})
        index = {p: m for m, p in enumerate(pairs)}
        P = torch.zeros(len(pairs), D, dtype=torch.complex128, device=device)  # psi of the selected (k, j)
        for m, (k, j) in enumerate(pairs):
            if (b, k) in psi:
                P[m] = psi[b, k][:, j].to(device)
        dist.all_reduce(P)
        ks = torch.tensor([k for k, _, _ in modes], device=device, dtype=torch.float64)
        th = 2 * math.pi * xs[:, None] * ks[None, :] / H  # (rows, K)
        cos, sin = th.cos(), th.sin()
        Pm = P[[index[(k, j)] for k, j, _ in modes]]  # (K, D)
        re, im = Pm.real.T, Pm.imag.T  # (D, K)
        part = [p for _, _, p in modes]
        is_s = torch.tensor([p == "s" for p in part], device=device)
        scale = torch.tensor([H ** -0.5 if p == "r" else (2 / H) ** 0.5 for p in part], device=device)
        # c: Re(psi e^{i theta}) = re cos - im sin; s: Im(psi e^{i theta}) = re sin + im cos; r: psi cos
        a = torch.where(is_s, sin, cos)[:, None, :] * re[None]  # (rows, D, K)
        bterm = torch.where(is_s, cos, -sin)[:, None, :] * im[None]
        Phi = ((a + bterm) * scale).reshape(-1, K)
        lam_sel = torch.stack([lam[b, k, j] for k, j, _ in modes])
        results.append((Phi, lam_sel.clamp_min(0).sqrt(), modes))
    (Phi, sigma, modes), (Phi_h, sigma_h, modes_h) = results
    n_pairs = sum(p == "c" for _, _, p in modes)
    log(f"Fourier POD: {K} real modes = {n_pairs} cos/sin pairs + {K - 2 * n_pairs} single; "
        f"wavenumbers used: {len({k for k, _, _ in modes})}, max k {max(k for k, _, _ in modes)}")
    return Phi, Phi_h, sigma, sigma_h, modes, modes_h


def main():
    out_dir = sys.argv[1]
    stride = int(os.environ.get("STRIDE", 4))
    K = int(os.environ.get("K", 1024))
    eval_r = [int(r) for r in os.environ.get("EVAL_R", "32,64,128,256,512,1024").split(",")]
    max_traj = int(os.environ.get("MAX_TRAJ", 0))
    project_splits = os.environ.get("PROJECT_SPLITS", "train,valid,test").split(",")
    dup_path = os.environ.get("DUP_MAP", os.path.join(C.HERE, "..", "hnode_runs", "eval_full", "dup_map.json"))
    kind = os.environ.get("BASIS", "pod")
    assert kind in ("pod", "fourier"), kind

    rank, local_rank, world_size = C.distributed_env()
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29503")
    device = C.pick_device(local_rank)
    dist.init_process_group("nccl" if device.type == "cuda" else "gloo", rank=rank, world_size=world_size)
    log = (lambda *a: print(*a, flush=True)) if rank == 0 else (lambda *a: None)
    rows = np.array_split(np.arange(H), world_size)[rank]
    rows = slice(int(rows[0]), int(rows[-1]) + 1)
    mean_c, std_c = C.frame_stats(device)
    eval_r = [r for r in eval_r if r <= K]
    t0 = time.time()

    def items(split):
        it = C.split_index(split)
        return it[:max_traj] if max_traj else it

    # 1. Basis from subsampled train frames.
    train_items = items("train")
    frames_b = slice(0, C.NUM_FRAMES, stride)
    n_per = len(range(0, C.NUM_FRAMES, stride))
    X = None
    for i, x, _ in loader(RowSlices("train", train_items, rows, frames_b)):
        x = ((x.to(device).double() - mean_c) / std_c).reshape(n_per, -1)
        if X is None:
            X = torch.empty(len(train_items) * n_per, x.shape[1], device=device)
        X[i * n_per:(i + 1) * n_per] = x
        if i % 100 == 0:
            log(f"  basis snapshots: {i + 1}/{len(train_items)} trajectories ({time.time() - t0:.0f}s)")
    mean = X.mean(0)
    X -= mean
    n = X.shape[0]
    K = min(K, n - 1)
    eval_r = [r for r in eval_r if r <= K]
    log(f"snapshots: {n} (every {stride}th frame of {len(train_items)} trajectories), "
        f"{X.shape[1]} values per rank, K = {K} ({time.time() - t0:.0f}s)")
    half = torch.tensor([j for j in range(n) if (j % n_per) * stride % (2 * stride) == 0], device=device)
    modes = modes_h = None
    if kind == "fourier":
        Phi, Phi_h, sigma, sigma_h, modes, modes_h = fourier_basis(
            X, rows, half, K, sorted(set(eval_r) | {K}), rank, world_size, device, log, t0)
        del X
    else:
        Phi, Phi_h, sigma, sigma_h = pod_modes(X, half, K, rank, device, log, t0, stride)
        del X
    torch.cuda.empty_cache()
    gram = Phi.T @ Phi
    dist.all_reduce(gram)
    log(f"modes: max |Phi^T Phi - I| = {(gram - torch.eye(K, device=device)).abs().max():.1e}")
    gram_h = Phi_h.T @ Phi_h
    dist.all_reduce(gram_h)
    log(f"modes (half snapshots): max |Phi^T Phi - I| = {(gram_h - torch.eye(gram_h.shape[0], device=device)).abs().max():.1e}")
    save_and_project(out_dir, kind, modes, modes_h, mean, Phi, Phi_h, sigma, sigma_h, stride, n, K, eval_r,
                     project_splits, items, rows, dup_path, mean_c, std_c, rank, world_size, device, log, t0)


def pod_modes(X, half, K, rank, device, log, t0, stride):
    """POD modes by the method of snapshots (X centered, rows of this rank); also on half the snapshots."""
    n = X.shape[0]
    G = X @ X.T
    dist.all_reduce(G)
    log(f"Gram matrix {tuple(G.shape)} reduced ({time.time() - t0:.0f}s)")

    # Eigenpairs on rank 0, broadcast so every rank builds its rows of the same modes.
    Kh = min(K, len(half) - 1)
    lam = torch.empty(K, device=device); V = torch.empty(n, K, device=device)
    lam_h = torch.empty(Kh, device=device); V_h = torch.empty(len(half), Kh, device=device)
    if rank == 0:
        l, v, res = top_eigenpairs(G, K)
        lam.copy_(l); V.copy_(v)
        log(f"eigenpairs (stride {stride}): max relative residual {res.max():.1e} "
            f"(top 64: {res[:64].max():.1e}) ({time.time() - t0:.0f}s)")
        l, v, res = top_eigenpairs(G.index_select(0, half).index_select(1, half), Kh)
        lam_h.copy_(l); V_h.copy_(v)
        log(f"eigenpairs (stride {2 * stride}, {len(half)} snapshots): max relative residual {res.max():.1e}")
    del G
    torch.cuda.empty_cache()
    for t in (lam, V, lam_h, V_h):
        dist.broadcast(t, 0)
    sigma, sigma_h = lam.clamp_min(0).sqrt(), lam_h.clamp_min(0).sqrt()
    Phi = X.T @ (V / sigma)             # (d_rank, K)
    Phi_h = X[half].T @ (V_h / sigma_h)
    return Phi, Phi_h, sigma, sigma_h


def save_and_project(out_dir, kind, modes, modes_h, mean, Phi, Phi_h, sigma, sigma_h, stride, n, K, eval_r,
                     project_splits, items, rows, dup_path, mean_c, std_c, rank, world_size, device, log, t0):
    # Save the basis: gather each rank's rows of mean and modes on rank 0.
    parts_mean = [torch.empty(0)] * world_size
    parts_phi = [torch.empty(0)] * world_size
    dist.all_gather_object(parts_mean, mean.reshape(-1, W, NC).cpu())
    dist.all_gather_object(parts_phi, Phi.float().reshape(-1, W, NC, K).cpu())
    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)
        torch.save({"mean": torch.cat(parts_mean), "modes": torch.cat(parts_phi).reshape(-1, K),
                    "sigma": sigma.cpu(), "sigma_half": sigma_h.cpu(), "stride": stride, "kind": kind,
                    "fourier_modes": modes, "fourier_modes_half": modes_h,
                    "n_snapshots": n, "layout": "(H, W, C) flattened; frames normalized by stats.npy"},
                   os.path.join(out_dir, "basis.pt"))
        log(f"saved basis.pt ({time.time() - t0:.0f}s)")
    del parts_mean, parts_phi

    # 2-3. Coefficients of every frame; projection floor on valid/test.
    dup_map = json.load(open(dup_path)) if os.path.exists(dup_path) else {}
    floor = {}
    for split in project_splits:
        its = items(split)
        do_floor = split != "train"
        coeffs = torch.empty(len(its), C.NUM_FRAMES, K) if rank == 0 else None
        scal = torch.empty(len(its), 2)
        if do_floor:
            vr = np.empty((2, len(eval_r), len(its), C.NUM_FRAMES, NC))
        for i, x, scalars in loader(RowSlices(split, its, rows, slice(None))):
            x = ((x.to(device).double() - mean_c) / std_c).reshape(C.NUM_FRAMES, -1)  # (T, d_rank)
            y = x - mean
            a = torch.cat([y @ Phi, y @ Phi_h], dim=1)
            dist.all_reduce(a)
            a, a_h = a[:, :K], a[:, K:]
            if rank == 0:
                coeffs[i] = a.cpu()
            scal[i] = scalars
            if do_floor:
                xc = x.reshape(C.NUM_FRAMES, -1, NC)
                stats = [xc.sum(1), xc.pow(2).sum(1)]  # for the per-frame, per-channel target variance
                for b, (P, A) in enumerate([(Phi, a), (Phi_h, a_h)]):
                    for j, r in enumerate(eval_r):
                        rr = min(r, P.shape[1])
                        res = (y - A[:, :rr] @ P[:, :rr].T).reshape(C.NUM_FRAMES, -1, NC)
                        stats.append(res.pow(2).sum(1))
                s = torch.stack(stats)  # (2 + 2*len(eval_r), T, C)
                dist.all_reduce(s)
                npx = H * W
                var_n = (s[1] - s[0].pow(2) / npx) / (npx - 1)  # unbiased, as torch.var
                mse_n = s[2:].reshape(2, len(eval_r), C.NUM_FRAMES, NC) / npx
                vr[:, :, i] = (mse_n * std_c**2 / (var_n * std_c**2 + 1e-7)).sqrt().cpu().numpy()
            if i % 100 == 0:
                log(f"  {split}: {i + 1}/{len(its)} trajectories ({time.time() - t0:.0f}s)")
        if rank == 0:
            meta = [dup_map.get(f"{split}:{i}", {}) for i in range(len(its))]
            torch.save({"coeffs": coeffs, "scalars": scal, "z": C.normalize_z(scal),
                        "file": [f for f, _ in its], "traj": [t for _, t in its],
                        "duplicate_of_train": [m.get("duplicate_of_train") for m in meta] if do_floor else None},
                       os.path.join(out_dir, f"coeffs_{split}.pt"))
            log(f"saved coeffs_{split}.pt {tuple(coeffs.shape)} ({time.time() - t0:.0f}s)")
            if do_floor:
                floor[split] = (vr, scal.numpy(), [m.get("duplicate_of_train") is not None for m in meta])
        del coeffs

    if rank == 0 and floor:
        report(out_dir, floor, eval_r, sigma.cpu().numpy(), stride, "Fourier POD" if kind == "fourier" else "POD")
    dist.barrier()
    dist.destroy_process_group()


def report(out_dir, floor, eval_r, sigma, stride, name="POD"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    splits = sorted(floor)
    vr = np.concatenate([floor[s][0] for s in splits], axis=2)  # (2, R, N, T, C)
    sc = np.concatenate([floor[s][1] for s in splits])
    dup = np.concatenate([np.array(floor[s][2], bool) for s in splits])
    np.savez(os.path.join(out_dir, "projection_floor.npz"), vrmse_well=vr, r=np.array(eval_r),
             split=np.concatenate([[s] * len(floor[s][1]) for s in splits]),
             idx=np.concatenate([np.arange(len(floor[s][1])) for s in splits]),
             Re=sc[:, 0], Sc=sc[:, 1], duplicate=dup)
    v = vr[:, :, :, 1:]  # frames 1..199, as in MODE=eval
    fields = ["tracer", "pressure", "vx", "vy"]
    res = sorted(set(sc[:, 0]))
    lines = [f"{name} projection floor: VRMSE (The Well definition) of the rank-r reconstruction, frames 1-199,",
             f"{len(sc)} held-out trajectories ({', '.join(splits)}). Basis: every {stride}th train frame "
             f"(second row per r: every {2 * stride}th).",
             f"{'r':>5} {'basis':>6} {'mean':>8} {'unique':>8} " + "".join(f"{f:>9}" for f in fields)
             + "".join(f"{'Re=' + format(r, 'g'):>9}" for r in res)]
    for j, r in enumerate(eval_r):
        for b, lbl in enumerate([f"/{stride}", f"/{2 * stride}"]):
            m = v[b, j]
            lines.append(f"{r:>5} {lbl:>6} {m.mean():>8.4f} {m[~dup].mean():>8.4f} "
                         + "".join(f"{m[..., c].mean():>9.4f}" for c in range(NC))
                         + "".join(f"{m[sc[:, 0] == rv].mean():>9.4f}" for rv in res))
    lines.append("Reference: Swin-HNODE rollout (pretrained checkpoint), mean VRMSE over frames 1-199: 0.0424")
    summary = "\n".join(lines)
    open(os.path.join(out_dir, "summary.txt"), "w").write(summary + "\n")
    print(summary, flush=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].semilogy(np.arange(1, len(sigma) + 1), sigma**2 / sigma[0] ** 2)
    axes[0].set_xlabel("mode"); axes[0].set_ylabel("eigenvalue / first"); axes[0].set_title(f"{name} spectrum")
    for c in range(NC):
        axes[1].loglog(eval_r, [v[0, j, ..., c].mean() for j in range(len(eval_r))], "o-", label=fields[c])
    axes[1].loglog(eval_r, [v[0, j].mean() for j in range(len(eval_r))], "k^-", label="mean")
    axes[1].loglog(eval_r, [v[1, j].mean() for j in range(len(eval_r))], "k:", label=f"mean (every {2 * stride}th)")
    axes[1].axhline(0.0424, color="tab:red", ls="--", label="Swin-HNODE rollout")
    axes[1].set_xlabel("r (modes)"); axes[1].set_ylabel("VRMSE"); axes[1].set_title("Projection floor per field")
    axes[1].legend(fontsize=8)
    for rv in res:
        m = sc[:, 0] == rv
        axes[2].loglog(eval_r, [v[0, j][m].mean() for j in range(len(eval_r))], "o-", label=f"Re = {rv:g}")
    axes[2].axhline(0.0424, color="tab:red", ls="--", label="Swin-HNODE rollout")
    axes[2].set_xlabel("r (modes)"); axes[2].set_title("Projection floor per Reynolds number")
    axes[2].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "pod_floor.png"), dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
