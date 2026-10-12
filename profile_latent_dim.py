#!/usr/bin/env python
"""Cost of HNODE as the latent dimension F grows, on shear-flow POD coefficients.

    python profile_latent_dim.py F CONFIG OUT.json      (one case per process)
    python profile_latent_dim.py summary DIR            (table and plots from DIR/*.json)
    (sbatch run_profile_latent_dim.sh runs every case on one node, 8 at a time)

CONFIG is "+"-joined options:
    dense | block        linear solver (dense is the default solver)
    chunkN               network Jacobian in chunks of N rows (CrossAttnTransformer.jacobian_chunk_size)
    detach               detach the final Newton step's Jacobian (HybridNODESolver.detach_final_jacobian)
    gradcheck            also compare the gradient with and without "detach" (same inputs)

Setup: the first F POD modes of shear-flow frames 0-1 (scaled by each mode's training std),
conditioning z, N_theta = CrossAttnTransformer(problem_dim=F, width 256, 3 blocks, 4 heads) with its
default random initialization, and the shear-flow time setup (dt = 0.01 per frame, 40 windows x 5
cells, batch 1). Costs per Newton iteration do not depend on training; iteration counts do, so they
are reported but are only indicative.

Measured: a 40-window rollout without gradients (time, Newton iterations, time in the network and
its Jacobian, time in the linear solves, peak memory), and a training step (40-window rollout with
the implicit-function gradient, then backward; time and peak memory). Out of memory is recorded.

Inputs: INPUTS (default hnode_runs/latent_dim_profile/inputs.pt).
"""
import contextlib
import io
import json
import os
import re
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
torch.set_default_dtype(torch.float64)
INPUTS = os.environ.get("INPUTS", os.path.join(HERE, "..", "hnode_runs", "latent_dim_profile", "inputs.pt"))
DT, CELLS, WINDOWS = 0.01, 5, 40


def sync():
    torch.cuda.synchronize()


class Timer:
    """Wrap a callable; accumulate its synchronized wall time."""

    def __init__(self, fn):
        self.fn, self.total, self.calls = fn, 0.0, 0

    def __call__(self, *a, **k):
        sync(); t = time.perf_counter()
        out = self.fn(*a, **k)
        sync(); self.total += time.perf_counter() - t; self.calls += 1
        return out


def build(F, opts, dev):
    from hybrid_node_solver import HybridNODESolver
    from neural_networks import CrossAttnTransformer

    torch.manual_seed(0)
    net = CrossAttnTransformer(problem_dim=F, latent_dim=256, num_blocks=3, num_heads=4, z_dim=2, dropout_rate=0,
                               activation=torch.nn.Tanh(), use_self_attention_z_block=True).to(dev)
    chunk = next((int(o[5:]) for o in opts if o.startswith("chunk")), None)
    net.jacobian_chunk_size = chunk
    solver = HybridNODESolver(net, DT * CELLS, CELLS + 1, F, WINDOWS, backprop_type="matrix_free",
                              use_manual_jacobian=True, use_sparse_solver=False, verbose=True, device=dev,
                              linear_solver="block" if "block" in opts else "dense")
    solver.store_last_jacobian = False
    solver.detach_final_jacobian = "detach" in opts
    return net, solver


def inputs(F, dev):
    d = torch.load(INPUTS, weights_only=False)
    x = (d["x01"][:1, :, :F] / d["std"][:F]).to(dev)  # (1, 2, F): frames 0-1 of one trajectory
    return {"J_init": (x[:, 1] - x[:, 0]) / DT, "u_init": (x[:, 0] + x[:, 1]) / 2, "z": d["z"][:1].to(dev)}


def peak_gib():
    return torch.cuda.max_memory_allocated() / 2**30


def run_case(F, config, out_path):
    opts = config.split("+")
    dev = torch.device("cuda")
    res = {"F": F, "config": config, "unknowns_per_window": F * (2 * CELLS + 3), "gpu": torch.cuda.get_device_name()}
    t0 = time.perf_counter()
    try:
        net, solver = build(F, opts, dev)
        res["build_s"] = time.perf_counter() - t0
        b = inputs(F, dev)
        net_timer, solve_timer = Timer(net.forward), Timer(solver._solve)
        net.forward, solver._solve = net_timer, solve_timer

        # Warm-up (kernel selection, MIOpen/hipBLAS initialization): one window.
        with contextlib.redirect_stdout(io.StringIO()):
            solver.rollout_batch(b, num_rollouts=1)
        net_timer.total = solve_timer.total = 0.0; net_timer.calls = solve_timer.calls = 0

        # Rollout without gradients.
        torch.cuda.reset_peak_memory_stats(); sync()
        log = io.StringIO(); t = time.perf_counter()
        with contextlib.redirect_stdout(log):
            _, u = solver.rollout_batch(b)
        sync()
        iters = [int(m) for m in re.findall(r"Converged after (\d+) iterations", log.getvalue())]
        res["rollout"] = {"s": time.perf_counter() - t, "network_s": net_timer.total, "solve_s": solve_timer.total,
                          "network_calls": net_timer.calls, "newton_iters": iters,
                          "not_converged": log.getvalue().count("Did not converge"),
                          "finite": bool(torch.isfinite(u).all()), "peak_gib": peak_gib()}
        print(f"F={F} {config} rollout: {res['rollout']['s']:.1f}s, network {net_timer.total:.1f}s, "
              f"solves {solve_timer.total:.1f}s, iters {iters[:5]}..., peak {res['rollout']['peak_gib']:.1f} GiB", flush=True)

        # Training step: rollout with the implicit-function gradient, then backward.
        solver.verbose = False
        net.zero_grad(set_to_none=True)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); sync()
        t = time.perf_counter()
        try:
            _, u = solver.evolve_batch(b)
            sync(); t_fwd = time.perf_counter() - t
            u.square().mean().backward()
            sync()
            g = torch.cat([p.grad.reshape(-1) for p in net.parameters() if p.grad is not None])
            res["train"] = {"s": time.perf_counter() - t, "forward_s": t_fwd, "peak_gib": peak_gib(),
                            "grad_norm": g.norm().item()}
            print(f"F={F} {config} train step: {res['train']['s']:.1f}s, peak {res['train']['peak_gib']:.1f} GiB", flush=True)
        except torch.OutOfMemoryError as e:
            res["train"] = {"oom": True, "peak_gib": peak_gib(), "error": str(e)[:200]}
            print(f"F={F} {config} train step: out of memory", flush=True)

        if "gradcheck" in opts:
            # The same gradient with the final Jacobian kept vs detached.
            grads = {}
            for flag in (False, True):
                solver.detach_final_jacobian = flag
                net.zero_grad(set_to_none=True); torch.cuda.empty_cache()
                _, u = solver.evolve_batch(b)
                u.square().mean().backward()
                grads[flag] = torch.cat([p.grad.reshape(-1) for p in net.parameters() if p.grad is not None]).clone()
            rel = ((grads[True] - grads[False]).norm() / grads[False].norm()).item()
            res["gradcheck_rel_diff"] = rel
            print(f"F={F} gradient, detached vs kept final Jacobian: relative difference {rel:.2e}", flush=True)
    except torch.OutOfMemoryError as e:
        res["oom"] = True; res["error"] = str(e)[:300]
        print(f"F={F} {config}: out of memory ({res['error'][:120]})", flush=True)
    except Exception as e:  # recorded, not fatal: other cases continue
        res["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        print(f"F={F} {config}: {res['error']}", flush=True)
    res["total_s"] = time.perf_counter() - t0
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    json.dump(res, open(out_path, "w"), indent=1)


def summary(d):
    import glob

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, "*.json")))]
    rows.sort(key=lambda r: (r["config"], r["F"]))

    def cell(r, phase, key, fmt):
        if phase not in r:  # failed before this phase (a later gradcheck OOM keeps earlier results)
            return "OOM" if r.get("oom") else "error" if "error" in r else "—"
        p = r.get(phase, {})
        if p.get("oom"):
            return "OOM"
        return fmt.format(p[key]) if key in p else "—"

    lines = [f"{'config':<22}{'F':>6}{'unknowns':>10} | {'rollout s':>10}{'network s':>10}{'solve s':>9}"
             f"{'iters/win':>10}{'peak GiB':>9} | {'train s':>8}{'peak GiB':>9}"]
    for r in rows:
        it = r.get("rollout", {}).get("newton_iters", [])
        lines.append(f"{r['config']:<22}{r['F']:>6}{r['unknowns_per_window']:>10} | "
                     f"{cell(r, 'rollout', 's', '{:.1f}'):>10}{cell(r, 'rollout', 'network_s', '{:.1f}'):>10}"
                     f"{cell(r, 'rollout', 'solve_s', '{:.1f}'):>9}{(f'{sum(it) / len(it):.1f}' if it else '—'):>10}"
                     f"{cell(r, 'rollout', 'peak_gib', '{:.1f}'):>9} | {cell(r, 'train', 's', '{:.1f}'):>8}"
                     f"{cell(r, 'train', 'peak_gib', '{:.1f}'):>9}"
                     + (f"   error: {r['error'][:60]}" if "error" in r and not r.get("oom") else "")
                     + ("   (gradcheck reference with kept Jacobian: OOM)" if r.get("oom") and "train" in r else ""))
    checks = [f"gradient check F={r['F']}: detached vs kept final Jacobian, relative difference {r['gradcheck_rel_diff']:.2e}"
              for r in rows if "gradcheck_rel_diff" in r]
    text = "\n".join(lines + [""] + checks)
    open(os.path.join(d, "summary.txt"), "w").write(text + "\n")
    print(text)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    for config in sorted({r["config"].replace("+gradcheck", "") for r in rows}):
        rs = [r for r in rows if r["config"].replace("+gradcheck", "") == config]
        for ax, (phase, key) in zip(axes, [("rollout", "s"), ("train", "s"), ("train", "peak_gib")]):
            pts = sorted((r["F"], r[phase][key]) for r in rs if key in r.get(phase, {}) and not r[phase].get("oom"))
            if pts:
                ax.loglog(*zip(*pts), "o-", label=config)
    for ax, title in zip(axes, ["Rollout, 40 windows (s)", "Training step (s)", "Training step peak memory (GiB)"]):
        ax.set_xlabel("latent dimension F"); ax.set_title(title); ax.legend(fontsize=8)
    axes[2].axhline(64, color="k", ls=":", lw=1)
    plt.tight_layout(); plt.savefig(os.path.join(d, "latent_dim_profile.png"), dpi=130)


if __name__ == "__main__":
    if sys.argv[1] == "summary":
        summary(sys.argv[2])
    else:
        run_case(int(sys.argv[1]), sys.argv[2], sys.argv[3])
