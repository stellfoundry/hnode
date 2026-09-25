"""Checks for HybridNODESolver(linear_solver="block").

1. linear_solver="dense" (the default) is bit-identical to the committed hybrid_node_solver.py.
2. "block" matches "dense": rollouts and parameter gradients, and Newton iteration counts.
3. "block" handles a size that "dense" cannot construct (F=256, M=40 on a 64 GB GPU).
4. Timing of a shear-flow-sized rollout and training step in both modes.

    python test_block_mode.py
"""
import contextlib
import importlib.util
import io
import os
import re
import subprocess
import sys
import tempfile
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from hybrid_node_solver import HybridNODESolver  # noqa: E402
from neural_networks import CrossAttnTransformer  # noqa: E402

torch.set_default_dtype(torch.float64)
dev = torch.device("cuda")


def committed_solver_class():
    """HybridNODESolver from the last commit (git HEAD), loaded as a separate module."""
    src = subprocess.run(["git", "-C", HERE, "show", "HEAD:hybrid_node_solver.py"],
                         capture_output=True, text=True, check=True).stdout
    path = os.path.join(tempfile.mkdtemp(), "hns_head.py")
    open(path, "w").write(src)
    spec = importlib.util.spec_from_file_location("hns_head", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.HybridNODESolver


def network(F, pretrained):
    torch.manual_seed(0)
    m = CrossAttnTransformer(problem_dim=F, latent_dim=256, num_blocks=3, num_heads=4, z_dim=2,
                             dropout_rate=0, activation=torch.nn.Tanh(), use_self_attention_z_block=True)
    if pretrained:
        ck = torch.load(os.path.join(HERE, "shear_flow_redo_part_2.pt"), map_location="cpu", weights_only=False)
        m.load_state_dict(ck["nn_rhs_model"])
    return m.to(dev)


def batch(F, B, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    return {"J_init": torch.randn(B, F, device=dev, generator=g),
            "u_init": torch.randn(B, F, device=dev, generator=g),
            "z": torch.randn(B, 2, device=dev, generator=g)}


def run(solver_cls, net, F, M, windows, b, **kw):
    """Rollout, loss gradient w.r.t. the network, and Newton iteration counts."""
    s = solver_cls(net, 0.01 * M, M + 1, F, windows, use_sparse_solver=False, verbose=True, device=dev, **kw)
    net.zero_grad()
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        _, u = s.evolve_batch(b)
        u.square().mean().backward()
    grads = torch.cat([p.grad.reshape(-1) for p in net.parameters() if p.grad is not None])
    iters = [int(x) for x in re.findall(r"Converged after (\d+) iterations", out.getvalue())]
    return u.detach(), grads.clone(), iters


def main():
    F, M, windows = 64, 5, 4
    net = network(F, pretrained=True)
    b = batch(F, 2)

    print("1. default (dense) vs committed version")
    u_new, g_new, it_new = run(HybridNODESolver, net, F, M, windows, b)
    u_old, g_old, it_old = run(committed_solver_class(), net, F, M, windows, b)
    print(f"   rollout bit-identical: {torch.equal(u_new, u_old)}   gradients bit-identical: {torch.equal(g_new, g_old)}"
          f"   Newton iterations equal: {it_new == it_old}")

    print("2. block vs dense")
    for (FF, MM, pretrained) in [(64, 5, True), (64, 20, True), (128, 10, False)]:
        net_i = net if pretrained else network(FF, False)
        bb = batch(FF, 2, seed=1)
        u_d, g_d, it_d = run(HybridNODESolver, net_i, FF, MM, windows, bb)
        u_b, g_b, it_b = run(HybridNODESolver, net_i, FF, MM, windows, bb, linear_solver="block")
        du = ((u_b - u_d).norm() / u_d.norm()).item()
        dg = ((g_b - g_d).norm() / g_d.norm()).item()
        print(f"   F={FF:<4} M={MM:<3} rollout rel diff {du:.1e}   gradient rel diff {dg:.1e}"
              f"   Newton iterations dense {it_d} block {it_b}")

    print("3. F=256, M=40 (one window, batch 1)")
    net256 = network(256, False)
    b256 = batch(256, 1)
    for mode in ["dense", "block"]:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        try:
            t0 = time.perf_counter()
            with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
                s = HybridNODESolver(net256, 0.4, 41, 256, 1, use_sparse_solver=False, device=dev, linear_solver=mode)
                _, u = s.rollout_batch(b256)
            torch.cuda.synchronize()
            print(f"   {mode:<5}: OK in {time.perf_counter() - t0:.1f}s, peak GPU memory "
                  f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB, finite: {torch.isfinite(u).all().item()}")
        except Exception as e:
            print(f"   {mode:<5}: {type(e).__name__}: {str(e)[:90]}")
        s = None

    print("4. timing, shear-flow size (F=64, 200 cells = 40 windows x 5, batch 1, pretrained network)")
    b1 = batch(64, 1, seed=2)
    for MM in [5, 10]:
        for mode in ["dense", "block"]:
            s = HybridNODESolver(net, 0.01 * MM, MM + 1, 64, 200 // MM, use_sparse_solver=False, device=dev,
                                 linear_solver=mode)
            with contextlib.redirect_stdout(io.StringIO()):
                s.rollout_batch(b1)  # warm-up
                torch.cuda.synchronize(); t0 = time.perf_counter()
                s.rollout_batch(b1); torch.cuda.synchronize(); t_roll = time.perf_counter() - t0
                net.zero_grad(); torch.cuda.reset_peak_memory_stats(); t0 = time.perf_counter()
                _, u = s.evolve_batch(b1); u.square().mean().backward(); torch.cuda.synchronize()
                t_train = time.perf_counter() - t0
            print(f"   M={MM:<3} {mode:<5}: rollout {t_roll:5.2f}s   training step {t_train:5.2f}s   "
                  f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
