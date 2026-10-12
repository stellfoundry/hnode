#!/usr/bin/env python
"""Time-stepper comparison in a frozen latent space of shear flow: HNODE vs first/second-order neural
ODEs, LSTM, Koopman and parametric DMD, all trained the same way on the same latents.

    python latent_baselines.py MODEL OUT_DIR      MODEL: hnode | node1 | node2 | lstm | koopman | dmd
    srun -n 8 ... python latent_baselines.py hnode OUT_DIR   (data-parallel; the global batch is kept)
    (submit_latent_row.sh trains a whole row: every model, chained 2-hour jobs)

LATENT_DIR holds coeffs_{train,valid,test}.pt with "coeffs" (N, 200, D), "z", "scalars", ... as written
by pod_basis.py (POD coefficients) or ae_train.py encode (autoencoder latents). The first R of the D
coordinates are used.

Every model gets the latents (scaled per coordinate by their training std) of two consecutive frames
t, t+1 and the conditioning z, and predicts frames t+1 .. t+h. The loss is the MSE of the unscaled
latents (per-coordinate weights std_i^2, normalized to sum 1); for POD this is the pixel MSE in
normalized units minus the fixed projection error. Training uses a horizon curriculum (h = 10, 50,
then 199 from frame 0) with the same steps, global batch and optimizer for every model. The model
kept is the best 199-frame rollout on the valid split.

Writes OUT_DIR/MODEL/: model.pt, pred.pt {"valid", "test": (N, 199, R) unscaled latents of frames
1..199 rolled out from frames 0-1}, history.json (done marker), log.txt, ckpt.pt (resume state).
Decode and score with eval_latent_rollout.py.

Optimizer: OPT "adam" (default) or "soap" (the paper's HNODE optimizer: SOAP, precondition_frequency 10,
weight decay 1e-4 on matrices; default for hnode). LR defaults to 1e-3 (adam) or 1e-4 (soap); results go to
OUT_DIR/MODEL{TAG}, so learning-rate sweeps use TAG (e.g. TAG=-lr3e-4). A step whose loss is not finite or
more than SPIKE (10) times the median of the last 50 losses at the same horizon is skipped.

Environment: LATENT_DIR (POD basis dir), R (128), STEPS (3000), BATCH (16, global), OPT, LR, TAG (""),
SPIKE (10), EVAL_EVERY (250), CKPT_EVERY (50), TIME_LIMIT (seconds; save and exit before it, default 6600),
SEED (0), SCALE ("mode": every coordinate scaled to unit std, default; "shared": all divided by the largest
std, so low-energy coordinates stay small; "floor:K": unit std for the K largest coordinates and the K-th std
for the rest, i.e. "mode" when R <= K), MICRO (trajectories per forward pass per GPU, gradients accumulated; 0 = all), HNODE_SOLVER ("dense" or "block"), HNODE_DT (time per frame for HNODE, default 1),
HNODE_DETACH ("1" default: detach the final Newton step's Jacobian; same gradient, far less memory),
HNODE_RELTOL ("1" default: Newton tolerance 1e-12 * max(1, max|state|)), HNODE_MAX_ITERS (solver default 100).
"""
import json
import math
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.nn as nn

import shear_flow_common as C

torch.set_default_dtype(torch.float64)
LATENT_DIR = os.environ.get("LATENT_DIR", os.path.join(C.HERE, "..", "hnode_runs", "pod", "full"))
R = int(os.environ.get("R", 128))
STEPS = int(os.environ.get("STEPS", 3000))
BATCH = int(os.environ.get("BATCH", 16))
MICRO = int(os.environ.get("MICRO", 0))  # trajectories per forward pass on a GPU (0: the whole local batch)
SPIKE = float(os.environ.get("SPIKE", 10))
SCALE = os.environ.get("SCALE", "mode")  # "mode": each coordinate to unit std; "shared": one factor
HNODE_DT = float(os.environ.get("HNODE_DT", 1.0))  # time per frame for HNODE
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", 250))
CKPT_EVERY = int(os.environ.get("CKPT_EVERY", 50))
TIME_LIMIT = float(os.environ.get("TIME_LIMIT", 6600))
T = C.NUM_FRAMES
CURRICULUM = [(0.2, 10), (0.5, 50), (1.0, T - 1)]  # (fraction of steps, horizon)


def load_split(split, scale=None):
    d = torch.load(os.path.join(LATENT_DIR, f"coeffs_{split}.pt"), weights_only=False)
    a = d["coeffs"][:, :, :R]
    if scale is None:
        scale = a.reshape(-1, R).std(0)
        if SCALE == "shared":  # one factor for all coordinates (the largest std): small modes stay small
            scale = scale.max().expand(R).clone()
        elif SCALE.startswith("floor:"):  # unit std for the K largest coordinates, the K-th std for the rest
            k = min(int(SCALE.split(":")[1]), R)  # (so "mode" for R <= K; beyond K small modes stay small)
            scale = scale.clamp(min=scale.sort(descending=True).values[k - 1].item())
    return a / scale, d["z"], scale


# ---------------------------------------------------------------- models
# rollout(x0, x1, z, h): scaled coefficients of frames t, t+1 (B, R) and z (B, 2) -> frames t+1..t+h (B, h, R)


class HNODE(nn.Module):
    """The paper's model at the shear-flow settings (5 cells per window), with time measured in frames
    (HNODE_DT = 1, like the neural ODEs) by default. The checkpoint's dt = 0.01 per frame suits jointly
    trained latents, whose scale the encoder chooses; on fixed latents (per-frame change ~0.3) it asks the
    tanh network for accelerations of order 1e3."""

    def __init__(self):
        super().__init__()
        from hybrid_node_solver import HybridNODESolver
        from neural_networks import CrossAttnTransformer

        self.net = CrossAttnTransformer(problem_dim=R, latent_dim=64 * 4, num_blocks=3, num_heads=4, z_dim=2,
                                        dropout_rate=0, activation=nn.Tanh(), use_self_attention_z_block=True)
        # Start from zero acceleration, as the other models start from a zero right-hand side. (With
        # dt = 0.01 the time scale did this implicitly: a random O(1) output barely moves the state.)
        nn.init.zeros_(self.net.final_projection.weight); nn.init.zeros_(self.net.final_projection.bias)
        self.cells = 5
        self.dt = HNODE_DT
        self.solver = HybridNODESolver(self.net, self.dt * self.cells, self.cells + 1, R, 1,
                                       backprop_type="matrix_free", use_manual_jacobian=True,
                                       use_sparse_solver=False, use_cpu_solver=False, verbose=False,
                                       device=torch.device("cuda"),
                                       linear_solver=os.environ.get("HNODE_SOLVER", "dense"))
        # Detaching the converged Newton step's Jacobian leaves the gradient unchanged (it multiplies a
        # residual below the tolerance) and drops the graph through the network Jacobians: 6.7 -> 0.4 GiB
        # per trajectory at R = 64 (profile_latent_dim.py).
        self.solver.detach_final_jacobian = os.environ.get("HNODE_DETACH", "1") == "1"
        # Newton tolerance relative to the state size (rounding floor) and an optional iteration cap.
        self.solver.relative_tol = os.environ.get("HNODE_RELTOL", "1") == "1"
        self.solver.max_iters = int(os.environ.get("HNODE_MAX_ITERS", self.solver.max_iters))

    def rollout(self, x0, x1, z, h):
        self.solver.num_training_rollouts = math.ceil(h / self.cells)
        _, u = self.solver.evolve_batch({"J_init": (x1 - x0) / self.dt, "u_init": (x0 + x1) / 2.0, "z": z})
        return u[:, :h]


def mlp(d_in, d_out, width=512, depth=4):
    layers, d = [], d_in
    for _ in range(depth - 1):
        layers += [nn.Linear(d, width), nn.GELU()]
        d = width
    layers.append(nn.Linear(d, d_out))
    nn.init.zeros_(layers[-1].weight); nn.init.zeros_(layers[-1].bias)  # start from a zero right-hand side
    return nn.Sequential(*layers)


def rk4(f, y, dt=1.0):
    k1 = f(y); k2 = f(y + dt / 2 * k1); k3 = f(y + dt / 2 * k2); k4 = f(y + dt * k3)
    return y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


class NODE1(nn.Module):
    """dx/dtau = f(x, z), one RK4 step per frame (tau in frames)."""

    def __init__(self):
        super().__init__()
        self.f = mlp(R + 2, R)

    def rollout(self, x0, x1, z, h):
        x, out = x1, [x1]
        for _ in range(h - 1):
            x = rk4(lambda y: self.f(torch.cat([y, z], -1)), x)
            out.append(x)
        return torch.stack(out, 1)


class NODE2(nn.Module):
    """Second order like HNODE, explicit: dx/dtau = v, dv/dtau = f(x, v, z); v0 = x1 - x0."""

    def __init__(self):
        super().__init__()
        self.f = mlp(2 * R + 2, R)

    def rollout(self, x0, x1, z, h):
        def rhs(s):
            x, v = s.chunk(2, -1)
            return torch.cat([v, self.f(torch.cat([x, v, z], -1))], -1)

        s, out = torch.cat([x1, x1 - x0], -1), [x1]
        for _ in range(h - 1):
            s = rk4(rhs, s)
            out.append(s[:, :R])
        return torch.stack(out, 1)


class LSTM(nn.Module):
    """2-layer LSTM on [x_t, z], predicting x_{t+1} - x_t; warmed up on frames t and t+1."""

    def __init__(self, hidden=512):
        super().__init__()
        self.cells = nn.ModuleList([nn.LSTMCell(R + 2, hidden), nn.LSTMCell(hidden, hidden)])
        self.head = nn.Linear(hidden, R)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)  # start as x_{t+1} = x_t

    def step(self, x, z, state):
        inp, new = torch.cat([x, z], -1), []
        for cell, s in zip(self.cells, state):
            hc = cell(inp, s)
            new.append(hc)
            inp = hc[0]
        return x + self.head(inp), new

    def rollout(self, x0, x1, z, h):
        state = [None, None]
        _, state = self.step(x0, z, state)
        x, out = x1, [x1]
        for _ in range(h - 1):
            x, state = self.step(x, z, state)
            out.append(x)
        return torch.stack(out, 1)


class Koopman(nn.Module):
    """Lift g = [x, psi(x)], advance linearly g_{k+1} = K(z) g_k with K(z) = exp(A(z)),
    A = S - S^T - N N^T - eps*I (eigenvalues of K inside the unit disk). The state x is part of g."""

    def __init__(self, extra=128):
        super().__init__()
        self.psi = mlp(R, extra, width=256, depth=3)
        self.d = d = R + extra
        self.S = nn.Parameter(torch.zeros(3, d, d))  # A(z) = A0 + z1 A1 + z2 A2, per factor
        self.N = nn.Parameter(0.01 * torch.randn(3, d, d))

    def K(self, z):
        w = torch.cat([torch.ones_like(z[:, :1]), z], -1)  # (B, 3)
        S = torch.einsum("bk,kij->bij", w, self.S)
        N = torch.einsum("bk,kij->bij", w, self.N)
        A = S - S.transpose(1, 2) - N @ N.transpose(1, 2) / self.d - 1e-4 * torch.eye(self.d, device=z.device)
        return torch.linalg.matrix_exp(A)

    def rollout(self, x0, x1, z, h):
        K = self.K(z)
        g, out = torch.cat([x1, self.psi(x1)], -1), [x1]
        for _ in range(h - 1):
            g = (K @ g.unsqueeze(-1)).squeeze(-1)
            out.append(g[:, :R])
        return torch.stack(out, 1)


class DMD(nn.Module):
    """Parametric DMD, closed form: one affine map x_{k+1} = K x_k + c per (Re, Sc), ridge least squares
    on all training transitions of that parameter pair."""

    def __init__(self, ridge=1e-3):
        super().__init__()
        self.ridge = ridge
        self.maps = {}

    def fit(self, x, z):
        for key in {tuple(v.tolist()) for v in z}:
            m = (z == torch.tensor(key, device=z.device)).all(-1)
            X = x[m, :-1].reshape(-1, R)
            Y = x[m, 1:].reshape(-1, R)
            Xa = torch.cat([X, torch.ones_like(X[:, :1])], -1)
            Kc = torch.linalg.solve((Xa.T @ Xa + self.ridge * torch.eye(R + 1, device=x.device)).cpu(),
                                    (Xa.T @ Y).cpu()).to(x.device)
            self.maps[key] = Kc

    def rollout(self, x0, x1, z, h):
        Kc = torch.stack([self.maps[tuple(v.tolist())] for v in z])  # (B, R+1, R)
        x, out = x1, [x1]
        for _ in range(h - 1):
            x = (torch.cat([x, torch.ones_like(x[:, :1])], -1).unsqueeze(1) @ Kc).squeeze(1)
            out.append(x)
        return torch.stack(out, 1)


MODELS = {"hnode": HNODE, "node1": NODE1, "node2": NODE2, "lstm": LSTM, "koopman": Koopman, "dmd": DMD}


# ---------------------------------------------------------------- training


def loss_fn(pred, target, w):
    return ((pred - target).pow(2) * w).sum(-1).mean()


@torch.no_grad()
def full_rollout(model, x, z, chunk=16):
    """Frames 1..199 from frames 0-1, in chunks of trajectories."""
    if len(x) == 0:
        return x[:, 1:]
    return torch.cat([model.rollout(x[i:i + chunk, 0], x[i:i + chunk, 1], z[i:i + chunk], T - 1)
                      for i in range(0, len(x), chunk)])


def main():
    name, out_root = sys.argv[1], sys.argv[2]
    opt_name = os.environ.get("OPT", "soap" if name == "hnode" else "adam")
    lr = float(os.environ.get("LR", 1e-4 if opt_name == "soap" else 1e-3))
    out = os.path.join(out_root, name + os.environ.get("TAG", ""))
    rank, local_rank, world = C.distributed_env()
    if name == "dmd":
        rank, world = 0, 1
    dev = C.pick_device(local_rank)
    if world > 1:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29504")
        dist.init_process_group(os.environ.get("DDP_BACKEND", "nccl"), rank=rank, world_size=world)
    assert BATCH % world == 0, (BATCH, world)
    if os.path.exists(os.path.join(out, "history.json")):
        if rank == 0:
            print(f"{name}: already done ({out}/history.json)", flush=True)
        return
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(int(os.environ.get("SEED", 0)))
    t_start = time.time()
    logf = open(os.path.join(out, "log.txt"), "a") if rank == 0 else None

    def log(msg):
        if rank == 0:
            print(msg, flush=True)
            logf.write(msg + "\n"); logf.flush()

    def allreduce(t):
        if world > 1:
            dist.all_reduce(t)
        return t

    xtr, ztr, scale = load_split("train")
    xva, zva, _ = load_split("valid", scale)
    xte, zte, _ = load_split("test", scale)
    xtr, ztr, xva, zva, xte, zte, scale = (t.to(dev) for t in (xtr, ztr, xva, zva, xte, zte, scale))
    # Loss weights: the unscaled latent MSE normalized by the total variance, whatever the scaling.
    var = scale.pow(2) if SCALE == "mode" else (xtr * scale).reshape(-1, R).var(0)
    w = scale.pow(2) / var.sum()
    model = MODELS[name]().to(dev)
    if world > 1:  # identical initial weights on every rank
        for p in model.parameters():
            dist.broadcast(p.data, 0)
    n_params = sum(p.numel() for p in model.parameters())

    def rollout_split(x, z):
        """199-frame rollouts of a split, trajectories spread over ranks; full result on rank 0."""
        mine = torch.arange(rank, len(x), world)
        p = full_rollout(model, x[mine], z[mine]).cpu()
        if world == 1:
            return p
        parts = [None] * world
        dist.all_gather_object(parts, p)
        full = torch.empty(len(x), T - 1, R)
        for r, part in enumerate(parts):
            full[r::world] = part
        return full

    def valid_loss():
        model.eval()
        mine = torch.arange(rank, len(xva), world)
        p = full_rollout(model, xva[mine], zva[mine])
        s = allreduce(((p - xva[mine, 1:]).pow(2) * w).sum(-1).mean(-1).sum().reshape(1))
        model.train()
        return s.item() / len(xva)

    ckpt_path = os.path.join(out, "ckpt.pt")
    state = {"step": 0, "history": [], "best": float("inf"), "t_train": 0.0}
    if name == "dmd":
        t0 = time.time(); model.fit(xtr, ztr); state["t_train"] = time.time() - t0
        state["best"] = valid_loss()
        log(f"dmd: R={R}, fitted in {state['t_train']:.1f}s, valid loss {state['best']:.4e}")
        state["history"].append({"step": 0, "valid": state["best"]})
    else:
        if opt_name == "soap":
            from soap import SOAP

            groups = [{"params": [p for p in model.parameters() if p.ndim >= 2], "weight_decay": 1e-4},
                      {"params": [p for p in model.parameters() if p.ndim < 2], "weight_decay": 0.0}]
            opt = SOAP(groups, precondition_frequency=10, lr=lr)
        else:
            opt = torch.optim.Adam(model.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, eta_min=lr / 100)
        g = torch.Generator(device="cpu").manual_seed(0)  # same on every rank: same global batch
        if os.path.exists(ckpt_path):
            ck = torch.load(ckpt_path, weights_only=False, map_location=dev)
            model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
            g.set_state(ck["gen"].cpu()); state = ck["state"]
            log(f"resumed from step {state['step']}")
        else:
            log(f"{name}: R={R}, {n_params:,} parameters, train {tuple(xtr.shape)}, steps {STEPS}, "
                f"global batch {BATCH} on {world} GPU(s), {opt_name} lr {lr}, latents {LATENT_DIR}")

        def save_ckpt():
            if rank == 0:
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                            "gen": g.get_state(), "state": state}, ckpt_path + ".tmp")
                os.replace(ckpt_path + ".tmp", ckpt_path)

        params = [p for p in model.parameters() if p.requires_grad]
        step_times = []
        while state["step"] < STEPS:
            step = state["step"] + 1
            h = next(hh for frac, hh in CURRICULUM if step <= frac * STEPS)
            idx = torch.randint(len(xtr), (BATCH,), generator=g)
            t0 = torch.randint(T - h, (BATCH,), generator=g) if h < T - 1 else torch.zeros(BATCH, dtype=torch.long)
            idx, t0 = idx[rank::world], t0[rank::world]
            traj = xtr[idx[:, None], t0[:, None] + torch.arange(h + 1)]  # (b, h+1, R): frames t .. t+h
            tic = time.time()
            opt.zero_grad()
            loss = 0.0
            micro = MICRO or len(idx)
            for s in range(0, len(idx), micro):  # gradient accumulation over micro-batches of trajectories
                part = slice(s, s + micro)
                pred = model.rollout(traj[part, 0], traj[part, 1], ztr[idx[part]], h)
                part_loss = loss_fn(pred, traj[part, 1:], w) * (len(pred) / len(idx))
                part_loss.backward()
                loss = loss + part_loss.detach()
                del pred, part_loss
            if world > 1:
                flat = torch.cat([p.grad.reshape(-1) for p in params])
                dist.all_reduce(flat); flat /= world
                off = 0
                for p in params:
                    p.grad.copy_(flat[off:off + p.numel()].view_as(p)); off += p.numel()
                loss = allreduce(loss.detach().reshape(1)) / world
            gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
            recent = state.setdefault("recent", {}).setdefault(str(h), [])
            lv = loss.item()  # identical on every rank
            spike = not math.isfinite(lv) or (len(recent) >= 20 and lv > SPIKE * sorted(recent)[len(recent) // 2])
            if spike:
                state["skipped"] = state.get("skipped", 0) + 1
                opt.zero_grad()
                log(f"step {step:5d} h {h:3d} loss {lv:.4e}: spike, skipped ({state['skipped']} so far)")
            else:
                opt.step()
                recent.append(lv); del recent[:-50]
            sched.step()
            torch.cuda.synchronize()
            dt_step = time.time() - tic
            state["t_train"] += dt_step; step_times.append(dt_step)
            state["step"] = step
            if step % 50 == 0 or step == 1 or STEPS <= 50:
                log(f"step {step:5d} h {h:3d} loss {loss.item():.4e} grad {gn.item():.2e} "
                    f"({sum(step_times[-50:]) / len(step_times[-50:]):.2f}s/step, "
                    f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB)")
            if step % EVAL_EVERY == 0 or step == STEPS:
                v = valid_loss()
                state["history"].append({"step": step, "h": h, "train": loss.item(), "valid": v,
                                         "time": state["t_train"]})
                # Keep only full-horizon checkpoints: shorter horizons say little about 199 frames.
                better = h == T - 1 and v < state["best"]
                if better:
                    state["best"] = v
                    if rank == 0:
                        torch.save(model.state_dict(), os.path.join(out, "model.pt"))
                log(f"  valid loss (199-frame rollout) {v:.4e}{'  *' if better else ''}")
            if step % CKPT_EVERY == 0 or step == STEPS:
                save_ckpt()
            # Stop before the job's time limit (all ranks agree); the next job resumes.
            stop = torch.tensor([float(time.time() - t_start > TIME_LIMIT)], device=dev)
            if allreduce(stop).item() > 0 and step < STEPS:
                save_ckpt()
                log(f"time limit: saved at step {step}, exiting to resume")
                if world > 1:
                    dist.barrier(); dist.destroy_process_group()
                return
        if rank == 0 and not os.path.exists(os.path.join(out, "model.pt")):  # no finite full-horizon validation
            torch.save(model.state_dict(), os.path.join(out, "model.pt"))
        if world > 1:
            dist.barrier()
        model.load_state_dict(torch.load(os.path.join(out, "model.pt"), map_location=dev))
    model.eval()

    tic = time.time()
    pred = {"valid": rollout_split(xva, zva), "test": rollout_split(xte, zte)}
    torch.cuda.synchronize()
    t_pred = (time.time() - tic) * world / (len(xva) + len(xte))  # GPU-seconds per trajectory
    if rank == 0:
        sc = scale.cpu()
        torch.save({k: v * sc for k, v in pred.items()}, os.path.join(out, "pred.pt"))
        rel = {k: loss_fn(pred[k], x[:, 1:].cpu(), w.cpu()).item() for k, x in [("valid", xva), ("test", xte)]}
        summary = {"model": name, "tag": os.environ.get("TAG", ""), "optimizer": opt_name, "lr": lr,
                   "skipped_steps": state.get("skipped", 0), "R": R, "scale": SCALE,
                   "hnode_dt": HNODE_DT if name == "hnode" else None, "latent_dir": LATENT_DIR, "parameters": n_params, "gpus": world,
                   "train_gpu_seconds": state["t_train"] * world, "rollout_gpu_seconds_per_trajectory": t_pred,
                   "best_valid": state["best"], "latent_loss": rel, "history": state["history"]}
        json.dump(summary, open(os.path.join(out, "history.json"), "w"), indent=1)
        log(f"done: latent loss valid {rel['valid']:.4e} test {rel['test']:.4e}, "
            f"train {state['t_train'] * world / 3600:.2f} GPU-h, rollout {t_pred:.3f} GPU-s/trajectory")
    if world > 1:
        dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
