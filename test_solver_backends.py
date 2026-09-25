"""Correctness and speed harness for solver_backends.py on real HNODE per-window Jacobians.

Cases are built with HybridNODESolver itself (its A matrix plus the network's Jacobian blocks), so
the matrices have exactly the structure the Newton solve sees. Each backend is compared with a
float64 reference solve (CPU LAPACK for n <= 6000, GPU MAGMA above).

    python test_solver_backends.py [--out DIR] [--cases name,name,...]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hybrid_node_solver import HybridNODESolver  # noqa: E402
from neural_networks import CrossAttnTransformer  # noqa: E402
from solver_backends import TimeBlockSolver, dense_solve, scipy_sparse_solve  # noqa: E402

torch.set_default_dtype(torch.float64)
dev = torch.device("cuda")

SHEAR_CKPT = "shear_flow_redo_part_2.pt"
FAIL_MATRIX = os.environ.get("FAIL_MATRIX", "")  # optional saved {"J", "H"} (the ROCm false-singular case)
# name: (latent dim F, cells per window M, batch B, weights)
CASES = {
    "sf_F64_M5": (64, 5, 8, "pretrained"),   # shear flow as trained
    "sf_F64_M20": (64, 20, 4, "pretrained"),
    "sf_F64_M40": (64, 40, 2, "pretrained"),
    "F128_M10": (128, 10, 2, "random"),
    "F256_M10": (256, 10, 2, "random"),
    "F256_M40": (256, 40, 1, "random"),
}


def make_network(F, weights):
    m = CrossAttnTransformer(problem_dim=F, latent_dim=256, num_blocks=3, num_heads=4, z_dim=2,
                             dropout_rate=0, activation=torch.nn.Tanh(), use_self_attention_z_block=True)
    if weights == "pretrained":
        ck = torch.load(SHEAR_CKPT, map_location="cpu", weights_only=False)
        m.load_state_dict(ck["nn_rhs_model"])
    return m.to(dev).eval()


def make_case(F, M, B, net):
    """Jacobian and residual at a perturbed Newton starting point, exactly as _timestep builds them."""
    s = HybridNODESolver(net, 0.01 * M, M + 1, F, 1, use_sparse_solver=False, device=dev)
    g = torch.Generator(device=dev).manual_seed(0)
    J_init = torch.randn(B, F, device=dev, generator=g)
    u_init = torch.randn(B, F, device=dev, generator=g)
    z = torch.randn(B, 2, device=dev, generator=g)
    with torch.no_grad():
        params = s._init_params(J_init, u_init)
        params = params + 0.1 * torch.randn(params.shape, device=dev, generator=g)
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):
            J, H = s._compute_J_and_residual(params, J_init, u_init, z)
    return J.contiguous(), H.contiguous()


def reference(J, rhs):
    n = J.shape[-1]
    if n <= 6000:
        return torch.from_numpy(np.linalg.solve(J.cpu().numpy(), rhs.cpu().numpy()[..., None])[..., 0]).to(dev)
    prev = torch.backends.cuda.preferred_linalg_library()
    torch.backends.cuda.preferred_linalg_library("magma")
    try:
        return torch.linalg.solve(J, rhs.unsqueeze(-1)).squeeze(-1)
    finally:
        torch.backends.cuda.preferred_linalg_library(prev)


def timed(fn, repeats=3):
    fn()  # warm-up
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        t0 = time.perf_counter(); out = fn(); torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        peak = torch.cuda.max_memory_allocated() - base
    return out, float(np.median(times)), peak


def errors(x, x_ref, J, rhs):
    rel = ((x - x_ref).norm() / x_ref.norm()).item()
    res = ((J @ x.unsqueeze(-1)).squeeze(-1) - rhs).norm().item() / rhs.norm().item()
    return rel, res


def gradient_check():
    """Block solver vs dense solve: gradients of sum(x^2) w.r.t. J and rhs (small case, CPU float64).

    dL/dJ is compared on J's nonzero pattern only: the dense solve also returns gradients for
    structurally-zero entries, which the block solver never reads. In HNODE those entries are
    constant zeros (independent of the network weights), so their gradients are never used.
    """
    F, M = 8, 4
    net = make_network(F, "random")
    J, H = make_case(F, M, 2, net)
    J, H = J.cpu(), H.cpu()
    blk = TimeBlockSolver(F, M + 1, torch.device("cpu"))
    grads = []
    for solve in (lambda A, b: torch.linalg.solve(A, b.unsqueeze(-1)).squeeze(-1), blk.solve):
        A = J.clone().requires_grad_(); b = H.clone().requires_grad_()
        solve(A, b).square().sum().backward()
        grads.append((A.grad, b.grad))
    mask = J != 0
    gd, gb = grads[0][0][mask], grads[1][0][mask]
    dJ = ((gd - gb).norm() / gd.norm()).item()
    db = ((grads[0][1] - grads[1][1]).norm() / grads[0][1].norm()).item()
    return dJ, db


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="solver_backends_results")
    ap.add_argument("--cases", default=",".join(CASES))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    print(f"torch {torch.__version__}, device {torch.cuda.get_device_name(0)}", flush=True)

    rows = []
    nets = {}
    todo = [(name, *CASES[name]) for name in args.cases.split(",") if name in CASES]
    if FAIL_MATRIX and os.path.exists(FAIL_MATRIX):
        todo.insert(0, ("cyl_rocm_fail", 64, 4, 1, "saved"))
    for name, F, M, B, weights in todo:
        if weights == "saved":
            d = torch.load(FAIL_MATRIX)
            J, H = d["J"].to(dev), d["H"].to(dev)
        else:
            key = (F, weights)
            if key not in nets:
                nets[key] = make_network(F, weights)
            try:
                J, H = make_case(F, M, B, nets[key])
            except torch.OutOfMemoryError:
                print(f"{name}: out of GPU memory while building the Jacobian; skipped", flush=True)
                torch.cuda.empty_cache()
                continue
        n = J.shape[-1]
        blk = TimeBlockSolver(F, M + 1, dev)
        structure_violation = blk.check_structure(J)
        x_ref = reference(J, H)
        row = {"case": name, "F": F, "M": M, "B": B, "n": n,
               "nnz_frac": (J[0] != 0).double().mean().item(), "structure_violation": structure_violation}

        x, t, peak = timed(lambda: dense_solve(J, H)[0])
        row["dense_gpu"] = dict(zip(["rel_err", "residual"], errors(x, x_ref, J, H)), time_s=t, peak_gib=peak / 2**30,
                                info_nonzero=int((dense_solve(J, H)[1] != 0).sum()))

        x, tim = scipy_sparse_solve(J, H, return_timing=True)
        _, t, _ = timed(lambda: scipy_sparse_solve(J, H))
        row["scipy_cpu"] = dict(zip(["rel_err", "residual"], errors(x, x_ref, J, H)), time_s=t, **tim)

        x, t, peak = timed(lambda: blk.solve(J, H))
        row["block_gpu"] = dict(zip(["rel_err", "residual"], errors(x, x_ref, J, H)), time_s=t, peak_gib=peak / 2**30,
                                failed_block_solves=blk.failed_block_solves)
        rows.append(row)
        print(json.dumps(row), flush=True)
        del J, H, x, x_ref
        torch.cuda.empty_cache()

    dJ, db = gradient_check()
    print(f"\ngradient check (block solver vs dense, F=8, M=4): rel diff dL/dJ (on J's nonzeros) {dJ:.1e}, dL/drhs {db:.1e}")

    hdr = f"{'case':<14}{'n':>7}{'nnz%':>6}{'struct':>8} | {'dense GPU':>22} | {'SciPy CPU':>30} | {'block GPU':>22}"
    print("\n" + hdr + "\n" + "-" * len(hdr))
    for r in rows:
        d, s, b = r["dense_gpu"], r["scipy_cpu"], r["block_gpu"]
        print(f"{r['case']:<14}{r['n']:>7}{100 * r['nnz_frac']:>6.1f}{r['structure_violation']:>8.0e} | "
              f"{d['time_s']:>8.4f}s err {d['rel_err']:>7.1e} | "
              f"{s['time_s']:>8.3f}s (LU {s['factor_solve_s']:.3f}s) err {s['rel_err']:>7.1e} | "
              f"{b['time_s']:>8.4f}s err {b['rel_err']:>7.1e}")
    json.dump({"cases": rows, "gradient_check": {"dJ": dJ, "drhs": db}}, open(os.path.join(args.out, "results.json"), "w"), indent=1)
    print(f"\nsaved {args.out}/results.json")


if __name__ == "__main__":
    main()
