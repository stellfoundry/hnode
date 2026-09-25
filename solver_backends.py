"""Alternative linear-solve backends for HybridNODESolver's per-window Newton systems.

Experimental: nothing here is imported by hybrid_node_solver.py. See test_solver_backends.py.

All backends solve J x = rhs for a batch of per-window Jacobians J (B, n, n) with
n = F * (2 * N_t + 1), where F = num_fields, M = N_t - 1 cells per window, and unknowns and
equations are laid out per field (field-major) as in HybridNODESolver:

    columns (per field): J_0 .. J_M | u_0 .. u_{M-1} | lambda_1 | lambda_2
    rows    (per field): node rows 0..M | cell rows 0..M-1 | J(0) = J_init | lambda_1 = u_init
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch


def dense_solve(J, rhs):
    """Reference GPU/CPU path: one batched dense LU (what HybridNODESolver uses today)."""
    x, info = torch.linalg.solve_ex(J, rhs.unsqueeze(-1))
    return x.squeeze(-1), info


def scipy_sparse_solve(J, rhs, return_timing=False):
    """CPU baseline: SuperLU (scipy.sparse.linalg.splu) on each batch element's nonzeros.

    Not differentiable (goes through NumPy). Returns x on J's device.
    """
    import time

    t0 = time.perf_counter()
    J_np = J.detach().cpu().numpy()
    rhs_np = rhs.detach().cpu().numpy()
    mats = [sp.csc_matrix(J_np[b]) for b in range(J_np.shape[0])]
    t1 = time.perf_counter()
    xs = np.stack([spla.splu(m).solve(rhs_np[b]) for b, m in enumerate(mats)])
    t2 = time.perf_counter()
    x = torch.from_numpy(xs).to(J.device)
    if return_timing:
        return x, {"convert_s": t1 - t0, "factor_solve_s": t2 - t1, "nnz": int(np.mean([m.nnz for m in mats]))}
    return x


class TimeBlockSolver:
    """Exact solve that marches through the window cell by cell (block forward substitution).

    Grouping unknowns and equations by time:
        slab "init":  unknowns (lambda_1, J_0)       <- rows (lambda_1 = u_init, J(0) = J_init)
        slab k:       unknowns (u_k, J_{k+1})        <- rows (node row k, cell row k),  k = 0..M-1
        slab "final": unknowns lambda_2              <- row  node row M
    makes the matrix block lower-triangular: rows of slab k touch only slabs k, k-1, k-2 (and
    init for k <= 1), and the final row touches slabs M-1, M-2. So x follows from M + 2 small
    dense solves (2F x 2F for the cell slabs) instead of one n x n solve. Pure torch ops, so it is
    differentiable and runs on any GPU. check_structure() verifies the block pattern on a given J.
    """

    def __init__(self, num_fields, N_t, device):
        F, M, P = num_fields, N_t - 1, 2 * N_t + 1
        fields = torch.arange(F, device=device) * P
        at = lambda local: fields + local  # noqa: E731  (row/column index of `local` for every field)
        slabs = [{"name": "init", "rows": torch.cat([at(2 * N_t), at(2 * N_t - 1)]),
                  "cols": torch.cat([at(2 * N_t - 1), at(0)]), "deps": []}]
        for k in range(M):
            deps = [0] if k <= 1 else []
            deps += [1 + j for j in (k - 2, k - 1) if j >= 0]
            slabs.append({"name": f"cell{k}", "rows": torch.cat([at(k), at(N_t + k)]),
                          "cols": torch.cat([at(N_t + k), at(k + 1)]), "deps": deps})
        slabs.append({"name": "final", "rows": at(M), "cols": at(2 * N_t),
                      "deps": [1 + j for j in (M - 2, M - 1) if j >= 0]})
        self.slabs = slabs
        self.n = F * P
        perm = torch.cat([s["cols"] for s in slabs])
        assert torch.equal(perm.sort().values, torch.arange(self.n, device=device)), "columns not a permutation"
        assert torch.equal(torch.cat([s["rows"] for s in slabs]).sort().values,
                           torch.arange(self.n, device=device)), "rows not a permutation"
        self.inv_perm = torch.empty_like(perm)
        self.inv_perm[perm] = torch.arange(self.n, device=device)
        self.failed_block_solves = 0

    @staticmethod
    def _block(J, rows, cols):
        return J[:, rows[:, None], cols[None, :]]

    def check_structure(self, J, tol=0.0):
        """Largest |entry| of J outside the assumed block pattern (should be 0)."""
        worst = 0.0
        for i, s in enumerate(self.slabs):
            allowed = set(s["deps"]) | {i}
            for j, t in enumerate(self.slabs):
                if j not in allowed:
                    worst = max(worst, self._block(J, s["rows"], t["cols"]).abs().max().item())
        return worst

    def solve(self, J, rhs):
        Y = []
        for s in self.slabs:
            r = rhs[:, s["rows"]]
            for d in s["deps"]:
                r = r - (self._block(J, s["rows"], self.slabs[d]["cols"]) @ Y[d].unsqueeze(-1)).squeeze(-1)
            D = self._block(J, s["rows"], s["cols"])
            y, info = torch.linalg.solve_ex(D, r.unsqueeze(-1))
            bad = info != 0
            if bad.any():  # same ROCm false-"singular" workaround as HybridNODESolver._dense_solve
                self.failed_block_solves += int(bad.sum())
                y = torch.linalg.solve(D.cpu(), r.unsqueeze(-1).cpu()).to(J.device)
            Y.append(y.squeeze(-1))
        return torch.cat(Y, dim=1)[:, self.inv_perm]
