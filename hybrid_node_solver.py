import torch
import numpy as np
import skfem
import time
import os
import warnings

# sparse_solve is an optional cuDSS-backed extension for GPU-sparse linear solves.
# It is not required for the examples in this repository (all use use_sparse_solver=False).
# If you have cuDSS available and wish to enable it, build the extension separately
# and set use_sparse_solver=True in HybridNODESolver.
try:
    from sparse_solve import sparse_linear_solve
except ImportError:
    sparse_linear_solve = None


class HybridNODESolver:
    def __init__(
        self,
        nn_rhs_model,
        delta_t,
        N_t,
        num_fields,
        num_training_rollouts,
        max_iters=100,
        tol=1e-12,
        verbose=False,
        backprop_type="matrix_free",
        use_manual_jacobian=True,
        use_sparse_solver=True,
        use_cpu_solver=False,
        device=None,
        linear_solver="dense",
    ):
        """
        Initialize the HybridNODESolver.

        Args:
            nn_rhs_model: Neural network model for the right-hand side.
            delta_t: Time step size.
            N_t: Number of time steps (+1).
            num_fields: Number of fields in the system.
            num_training_rollouts: Number of training rollouts.
            max_iters: Maximum number of iterations for convergence.
            tol: Tolerance for convergence.
            verbose: Whether to print verbose output.
            backprop_type: Type of backpropagation method to use ("matrix_free" or "schur").
            use_manual_jacobian: Whether to use manual Jacobian construction.
            use_sparse_solver: Whether to use a sparse solver for linear systems.
            use_cpu_solver: Whether to use CPU solver for sparse linear systems.
            device: Device to run computations on (default is CUDA if available).
            linear_solver: How the dense-path Newton systems are solved. "dense" (default):
                build the full Jacobian and use one batched LU. "block": never build the
                full n x n matrices; keep the per-field A block and the network's per-cell
                Jacobian blocks, and solve by marching cell by cell (the Jacobian is block
                lower-triangular in time). Same result to round-off; much less memory for
                large num_fields or N_t. Requires matrix_free backprop, the manual Jacobian
                and use_sparse_solver=False; self.A, self.A_inv and self.exp_M_p0 do not
                exist in block mode (use apply_A for A @ params).

        """
        self.nn_rhs_model = nn_rhs_model
        self.delta_t = delta_t
        self.N_t = N_t
        self.num_fields = num_fields
        self.num_training_rollouts = num_training_rollouts
        self.max_iters = max_iters
        self.tol = tol
        self.verbose = verbose
        self.backprop_type = backprop_type
        self.use_manual_jacobian = use_manual_jacobian
        self.use_sparse_solver = use_sparse_solver
        self.use_cpu_solver = use_cpu_solver
        self.linear_solver = linear_solver

        if self.linear_solver not in ("dense", "block"):
            raise ValueError(f"linear_solver must be 'dense' or 'block', got {linear_solver!r}")
        if self.linear_solver == "block":
            assert (
                self.backprop_type == "matrix_free"
                and self.use_manual_jacobian
                and not self.use_sparse_solver
            ), "linear_solver='block' requires matrix_free backprop, the manual Jacobian and use_sparse_solver=False"

        if self.use_manual_jacobian:
            assert (
                self.backprop_type == "matrix_free"
            ), "Manual Jacobian construction is only supported with matrix-free backpropagation"
        if self.use_sparse_solver:
            assert (
                self.use_manual_jacobian
            ), "Manual Jacobian construction is required for sparse solver"
            assert (
                self.backprop_type == "matrix_free"
            ), "Sparse solver is only supported with matrix-free backpropagation"

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        if self.linear_solver == "block":
            self._build_block_system()
        else:
            self.A, self.M_p0, self.exp_M_p0 = self._build_A_and_M_p0()

            # Cache A_inv for quick initialization of params during solve
            self.A_inv = torch.linalg.inv(self.A)

        if self.backprop_type == "schur":
            # Compute permutation indices
            (
                self.permutation_indices_without_lambdas,
                self.permutation_indices,
            ) = self._compute_permutation_indices()
            # Compute matrices need for Schur complement
            self.B = self._compute_B()
            self.C = self._compute_C()
            self.D = self._compute_D()

    def _evaluate_sparse_function(self, basis, weights_exponent=0.5, operation=None):
        """
        Evaluate a sparse function using the given basis.

        Args:
            basis: The basis for evaluation.
            weights_exponent: Exponent for weights computation.
            operation: Operation to perform ("grad" for gradient, None for value).

        Returns:
            A sparse tensor representation of the evaluated function.
        """
        weights = basis.dx.reshape(-1) ** weights_exponent

        rows = []
        cols = []
        vals = []

        for i in range(basis.N):
            x = np.zeros(basis.N)
            x[i] = 1.0
            # Evaluate the basis function at quadrature points and multiply by weights
            if operation == "grad":
                eval_result = basis.interpolate(x).grad.reshape(-1) * weights
            else:
                eval_result = basis.interpolate(x).value.reshape(-1) * weights

            nonzero_indices = np.nonzero(eval_result)[0]
            if len(nonzero_indices) > 0:
                rows.extend([i] * len(nonzero_indices))
                cols.extend(nonzero_indices)
                vals.extend(eval_result[nonzero_indices])

        sparse_tensor = torch.sparse_coo_tensor(
            torch.tensor([rows, cols]),
            torch.tensor(vals),
            size=(basis.N, len(weights)),
        ).coalesce()

        return sparse_tensor

    def _sparse_zeros_like(self, sparse_tensor):
        """
        Create a sparse tensor of zeros with the same size and sparsity pattern as the input.

        Args:
            sparse_tensor: The sparse tensor to mimic.

        Returns:
            A sparse tensor of zeros.
        """
        return torch.sparse_coo_tensor(
            sparse_tensor.indices(),
            torch.zeros(sparse_tensor._nnz(), device=sparse_tensor.device),
            sparse_tensor.size(),
        )

    def _sparse_zeros(self, size):
        """
        Create a sparse tensor of zeros with the given size.

        Args:
            size: The size of the sparse tensor.

        Returns:
            A sparse tensor of zeros.
        """
        return torch.sparse_coo_tensor(
            torch.empty((2, 0), dtype=torch.long), torch.empty(0), size
        )

    def _sparse_kron(self, A, B):
        """
        Compute Kronecker product of two sparse COO matrices without densification.

        For matrices A (m x n) and B (p x q), computes C = A ⊗ B of size (mp x nq).
        Each element A[i,j] is multiplied by the entire matrix B and placed at
        the appropriate block position in the result.

        Args:
            A: torch.sparse_coo_tensor of shape (m, n)
            B: torch.sparse_coo_tensor of shape (p, q)

        Returns:
            C: torch.sparse_coo_tensor of shape (m*p, n*q) representing A ⊗ B
        """
        # Input validation
        if not (A.is_sparse and B.is_sparse):
            raise ValueError("Both inputs must be sparse tensors")

        if A.layout != torch.sparse_coo or B.layout != torch.sparse_coo:
            raise ValueError("Both inputs must be in COO format")

        # Ensure inputs are coalesced (combines duplicate indices)
        A = A.coalesce()
        B = B.coalesce()

        # Get dimensions
        m, n = A.shape
        p, q = B.shape

        # Handle empty matrices
        if A._nnz() == 0 or B._nnz() == 0:
            return torch.sparse_coo_tensor(
                torch.empty((2, 0), dtype=torch.long, device=A.device),
                torch.empty(
                    (0,), dtype=torch.result_type(A.dtype, B.dtype), device=A.device
                ),
                (m * p, n * q),
            )

        # Get indices and values
        A_indices = A.indices()  # 2 × nnz_A
        A_values = A.values()  # nnz_A
        B_indices = B.indices()  # 2 × nnz_B
        B_values = B.values()  # nnz_B

        nnz_A = A_values.shape[0]
        nnz_B = B_values.shape[0]

        # For each non-zero element A[i,j] and each non-zero element B[k,l],
        # we create an element at position (i*p + k, j*q + l) with value A[i,j] * B[k,l]

        # Create all combinations using broadcasting
        # Shape: nnz_A × nnz_B for each component
        A_row_expanded = A_indices[0].unsqueeze(1).expand(-1, nnz_B)  # nnz_A × nnz_B
        A_col_expanded = A_indices[1].unsqueeze(1).expand(-1, nnz_B)  # nnz_A × nnz_B
        A_val_expanded = A_values.unsqueeze(1).expand(-1, nnz_B)  # nnz_A × nnz_B

        B_row_expanded = B_indices[0].unsqueeze(0).expand(nnz_A, -1)  # nnz_A × nnz_B
        B_col_expanded = B_indices[1].unsqueeze(0).expand(nnz_A, -1)  # nnz_A × nnz_B
        B_val_expanded = B_values.unsqueeze(0).expand(nnz_A, -1)  # nnz_A × nnz_B

        # Compute result indices
        # Row: i*p + k, Col: j*q + l
        result_rows = A_row_expanded * p + B_row_expanded  # nnz_A × nnz_B
        result_cols = A_col_expanded * q + B_col_expanded  # nnz_A × nnz_B

        # Flatten to get final indices
        result_indices = torch.stack(
            [result_rows.flatten(), result_cols.flatten()]
        )  # 2 × (nnz_A * nnz_B)

        # Compute result values
        result_values = (A_val_expanded * B_val_expanded).flatten()  # nnz_A * nnz_B

        # Create result sparse tensor
        C = torch.sparse_coo_tensor(
            result_indices,
            result_values,
            (m * p, n * q),
            dtype=torch.float64,
            device=A.device,
        )

        # Coalesce to combine any duplicate indices (shouldn't happen, but good practice)
        return C.coalesce()

    def _build_A_per_field(self):
        """
        Build the per-field block of A (size 2*N_t+1) and the P0 mass matrix M_p0.

        A itself is kron(I_num_fields, A_per_field).

        Returns:
            A tuple (A_per_field, M_p0) of dense float64 matrices on the CPU.
        """
        with torch.no_grad():
            mesh = skfem.MeshLine(np.linspace(0.0, self.delta_t, self.N_t))
            p0_element = skfem.ElementLineP0()
            rt_element = skfem.ElementLineP1()
            p0_basis = skfem.Basis(mesh, p0_element)
            rt_basis = skfem.Basis(mesh, rt_element)

            p0_eval = self._evaluate_sparse_function(p0_basis)
            rt_eval = self._evaluate_sparse_function(rt_basis)
            rt_div_eval = self._evaluate_sparse_function(rt_basis, operation="grad")

            indices_C1 = torch.tensor([[0, 0]])
            values_C1 = torch.tensor([1.0])
            size_C1 = (self.N_t, 1)
            C1_sparse = torch.sparse_coo_tensor(indices_C1.t(), values_C1, size_C1)

            indices_C2 = torch.tensor([[self.N_t - 1, 0]])
            values_C2 = torch.tensor([-1.0])
            size_C2 = (self.N_t, 1)
            C2_sparse = torch.sparse_coo_tensor(indices_C2.t(), values_C2, size_C2)

            M_p0 = p0_eval @ p0_eval.T
            M_rt = rt_eval @ rt_eval.T
            G = p0_eval @ rt_div_eval.T

            self.M_rt = M_rt
            self.G = G

            first_row_block = torch.cat([M_rt, G.T, C1_sparse, C2_sparse], dim=1)
            second_row_block = torch.cat(
                [
                    G,
                    self._sparse_zeros_like(M_p0),
                    self._sparse_zeros((self.N_t - 1, 2)),
                ],
                dim=1,
            )
            third_row_block = torch.cat(
                [
                    C1_sparse.T,
                    self._sparse_zeros((1, self.N_t - 1)),
                    self._sparse_zeros((1, 2)),
                ],
                dim=1,
            )
            fourth_row_block = torch.cat(
                [
                    C2_sparse.T,
                    self._sparse_zeros((1, self.N_t - 1)),
                    self._sparse_zeros((1, 2)),
                ],
                dim=1,
            )

            A = torch.cat(
                [
                    first_row_block,
                    second_row_block,
                    third_row_block,
                    fourth_row_block,
                ],
                dim=0,
            )

            # Even for very large systems, storing *one* dense version of each of these should be fine (for ODEs)
            A = A.to_dense().double()
            M_p0 = M_p0.to_dense().double()

            A[-1, :] = 0.0
            A[-1, -2] = 1.0

            return A, M_p0

    def _build_A_and_M_p0(self):
        """
        Build the A matrix and M_p0 matrix used for rollouts.

        Returns:
            A tuple (A, M_p0) of matrices.
        """
        with torch.no_grad():
            A, M_p0 = self._build_A_per_field()

            A = torch.kron(torch.eye(self.num_fields, device=A.device), A)

            N = A.shape[0] // self.num_fields
            exp_M_p0 = torch.zeros_like(A)
            for field_ndx in range(self.num_fields):
                exp_M_p0[
                    field_ndx * N
                    + self.N_t : field_ndx * N
                    + self.N_t
                    + (self.N_t - 1),
                    field_ndx * N
                    + self.N_t : field_ndx * N
                    + self.N_t
                    + (self.N_t - 1),
                ] = M_p0

            self.A_sparse = A.to_sparse_coo().to(self.device)
            self.exp_M_p0_sparse = exp_M_p0.to_sparse_coo().to(self.device)

            return A.to(self.device), M_p0.to(self.device), exp_M_p0.to(self.device)

    def _build_block_system(self):
        """
        Set up linear_solver="block": the per-field A block, the cell sizes, and the slab layout.

        Grouping unknowns and equations by time makes the window's Jacobian block
        lower-triangular:
            slab "init":  unknowns (lambda_1, J_0)   <- rows (lambda_1 = u_init, J(0) = J_init)
            slab k:       unknowns (u_k, J_{k+1})    <- rows (node row k, cell row k), k = 0..M-1
            slab "final": unknown  lambda_2          <- row  node row M
        Rows of slab k touch only slabs k, k-1 and k-2 (and "init" for k <= 1); the final row
        touches the last two cell slabs. Within a slab, unknowns and rows are ordered
        (local index, field), so the A part of every block is kron(A_per_field[rows, cols], I_F).
        """
        A_pf, M_p0 = self._build_A_per_field()
        off_diagonal = M_p0 - torch.diag(M_p0.diagonal())
        assert off_diagonal.abs().max() == 0, "block solver assumes a diagonal P0 mass matrix"
        F, N_t = self.num_fields, self.N_t
        M, P = N_t - 1, 2 * N_t + 1
        dev = self.device

        self.A_per_field = A_pf.to(dev)
        self.A_per_field_inv = torch.linalg.inv(A_pf).to(dev)  # small (P x P), inverted on the CPU
        self.M_p0 = M_p0.to(dev)
        self.cell_sizes = M_p0.diagonal().to(dev)  # h_k: M_p0 is diag(h)

        # Local (per-field) row and column indices of every slab, and its dependencies.
        slabs = [{"rows": [2 * N_t, 2 * N_t - 1], "cols": [2 * N_t - 1, 0], "deps": []}]
        for k in range(M):
            deps = [0] if k <= 1 else []
            deps += [1 + j for j in (k - 2, k - 1) if j >= 0]
            slabs.append({"rows": [k, N_t + k], "cols": [N_t + k, k + 1], "deps": deps})
        slabs.append({"rows": [M], "cols": [2 * N_t], "deps": [1 + j for j in (M - 2, M - 1) if j >= 0]})

        eye_F = torch.eye(F, dtype=torch.float64)
        fields = torch.arange(F) * P

        def a_block(rows, cols):
            return torch.kron(A_pf[rows][:, cols], eye_F).to(dev)

        def global_index(locals_):
            return torch.cat([fields + q for q in locals_]).to(dev)

        for s in slabs:
            s["row_index"] = global_index(s["rows"])
            s["col_index"] = global_index(s["cols"])
            s["A_diag"] = a_block(s["rows"], s["cols"])
            s["A_deps"] = {d: a_block(s["rows"], slabs[d]["cols"]) for d in s["deps"]}
        perm = torch.cat([s["col_index"] for s in slabs])
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(perm.numel(), device=dev)
        self._slabs = slabs
        self._slab_inv_perm = inv_perm

    def apply_A(self, params):
        """A @ params for params of shape (batch, n), in either linear_solver mode."""
        if self.linear_solver == "block":
            batch_size = params.shape[0]
            x = params.reshape(batch_size, self.num_fields, -1)
            return torch.einsum("ij,bfj->bfi", self.A_per_field, x).reshape(batch_size, -1)
        return (self.A @ params.unsqueeze(-1)).squeeze(-1)

    def _compute_block_J_and_residual(self, params, J_init, u_init, z=None):
        """linear_solver="block" version of _compute_J_and_residual: returns the Jacobian as
        per-slab blocks (a _BlockJacobian) instead of a dense (batch, n, n) tensor."""
        batch_size = params.shape[0]
        params_reshaped = params.reshape(batch_size, self.num_fields, -1).transpose(
            -2, -1
        )
        J_hat = params_reshaped[:, : self.N_t, :]
        u_hat = params_reshaped[:, self.N_t : -2, :]

        t_start = time.time()
        (u_hat_jacobian, z_jacobians), u_hat_forcing = self.nn_rhs_model(
            J_hat, u_hat, z
        )
        t_end = time.time()
        if not hasattr(self, "nn_time"):
            self.nn_time = 0.0
        self.nn_time += t_end - t_start

        linear_part = self.apply_A(params)
        nonlinear_part = self.M_p0 @ u_hat_forcing
        nonlinear_part = (
            torch.cat(
                [
                    torch.zeros_like(J_hat),
                    nonlinear_part,
                    torch.zeros(batch_size, 2, self.num_fields, device=self.device),
                ],
                dim=1,
            )
            .transpose(-2, -1)
            .reshape(batch_size, -1)
        )
        residual = linear_part + nonlinear_part
        rhs = torch.cat(
            [
                torch.zeros(
                    batch_size,
                    params_reshaped.shape[1] - 2,
                    self.num_fields,
                    device=self.device,
                ),
                J_init.unsqueeze(1),
                u_init.unsqueeze(1),
            ],
            dim=1,
        )
        residual = residual - rhs.transpose(-2, -1).reshape(batch_size, -1)

        # Network Jacobian blocks enter only the cell rows, scaled by the cell size:
        #   cell row k: + h_k dN_k/du_k (cols u_k), + h_k dN_k/dJ_{k+1} (cols J_{k+1}),
        #               + h_k dN_k/dJ_k (cols J_k, held by the previous slab).
        h = self.cell_sizes.view(1, -1, 1, 1)
        d_u = h * u_hat_jacobian  # (batch, M, F, F): [out field, in field]
        d_left = h * z_jacobians[:, :, :, 0, :]
        d_right = h * z_jacobians[:, :, :, 1, :]
        return _BlockJacobian(self, d_u, d_left, d_right), residual

    def _solve(self, J, H):
        """Newton step solve J x = H, dispatching on linear_solver."""
        if self.linear_solver == "block":
            return J.solve(H)
        return self._dense_solve(J, H.unsqueeze(-1)).squeeze(-1)

    def _build_indexers(self, batch_size):
        N = self.N_t + (self.N_t - 1) + 2

        if not self.use_sparse_solver:
            # Dense path: only the flat scatter indices are used. Build them with
            # broadcasting, in the same (batch, row field, col field, cell) order
            # as the loops below; the loops launch several small GPU ops per
            # (batch, field, field) and take hours for batches of ~600 with 64
            # fields.
            F, n_t, FN = self.num_fields, self.N_t, self.num_fields * N
            b = torch.arange(batch_size, device=self.device).view(-1, 1, 1, 1)
            rf = torch.arange(F, device=self.device).view(1, -1, 1, 1)
            cf = torch.arange(F, device=self.device).view(1, 1, -1, 1)
            c = torch.arange(n_t - 1, device=self.device).view(1, 1, 1, -1)
            row = b * FN * FN + (rf * N + n_t + c) * FN
            self.u_hat_u_hat_indices = (row + cf * N + n_t + c).reshape(-1)
            self.u_hat_J_hat_left_indices = (row + cf * N + c).reshape(-1)
            self.u_hat_J_hat_right_indices = (row + cf * N + 1 + c).reshape(-1)
            return

        index_matrix = torch.arange(
            batch_size * self.num_fields * N * self.num_fields * N, device=self.device
        ).reshape(batch_size, self.num_fields * N, self.num_fields * N)

        u_hat_u_hat_indices = []
        u_hat_u_hat_rows = []
        u_hat_u_hat_cols = []
        for batch_ndx in range(batch_size):
            for row_field_ndx in range(self.num_fields):
                for col_field_ndx in range(self.num_fields):
                    row_start = row_field_ndx * N
                    col_start = col_field_ndx * N
                    u_hat_u_hat_indices.append(
                        index_matrix[
                            batch_ndx,
                            row_start
                            + self.N_t : row_start
                            + self.N_t
                            + (self.N_t - 1),
                            col_start
                            + self.N_t : col_start
                            + self.N_t
                            + (self.N_t - 1),
                        ].diag()
                    )
                    rows = batch_ndx * N * self.num_fields + torch.arange(
                        row_start + self.N_t, row_start + self.N_t + (self.N_t - 1)
                    )
                    cols = batch_ndx * N * self.num_fields + torch.arange(
                        col_start + self.N_t, col_start + self.N_t + (self.N_t - 1)
                    )
                    u_hat_u_hat_rows.append(rows)
                    u_hat_u_hat_cols.append(cols)
        self.u_hat_u_hat_indices = torch.cat(u_hat_u_hat_indices)
        self.u_hat_u_hat_coo_indices = torch.stack(
            [torch.cat(u_hat_u_hat_rows), torch.cat(u_hat_u_hat_cols)], dim=0
        ).to(self.device)

        u_hat_J_hat_left_indices = []
        u_hat_J_hat_right_indices = []
        u_hat_J_hat_rows = []
        u_hat_J_hat_left_cols = []
        u_hat_J_hat_right_cols = []
        for batch_ndx in range(batch_size):
            for row_field_ndx in range(self.num_fields):
                for col_field_ndx in range(self.num_fields):
                    row_start = row_field_ndx * N
                    col_start = col_field_ndx * N
                    u_hat_indices = torch.arange(
                        row_start + self.N_t,
                        row_start + self.N_t + (self.N_t - 1),
                        device=self.device,
                    )
                    rows = batch_ndx * N * self.num_fields + u_hat_indices
                    u_hat_J_hat_rows.append(rows)
                    J_hat_left_indices = torch.arange(
                        col_start, col_start + self.N_t - 1, device=self.device
                    )
                    cols = batch_ndx * N * self.num_fields + J_hat_left_indices
                    u_hat_J_hat_left_cols.append(cols)
                    J_hat_right_indices = torch.arange(
                        col_start + 1, col_start + self.N_t, device=self.device
                    )
                    cols = batch_ndx * N * self.num_fields + J_hat_right_indices
                    u_hat_J_hat_right_cols.append(cols)
                    u_hat_J_hat_left_indices.append(
                        index_matrix[batch_ndx, u_hat_indices, J_hat_left_indices]
                    )
                    u_hat_J_hat_right_indices.append(
                        index_matrix[batch_ndx, u_hat_indices, J_hat_right_indices]
                    )
        self.u_hat_J_hat_left_indices = torch.cat(u_hat_J_hat_left_indices)
        self.u_hat_J_hat_right_indices = torch.cat(u_hat_J_hat_right_indices)
        self.u_hat_J_hat_left_coo_indices = torch.stack(
            [torch.cat(u_hat_J_hat_rows), torch.cat(u_hat_J_hat_left_cols)], dim=0
        )
        self.u_hat_J_hat_right_coo_indices = torch.stack(
            [torch.cat(u_hat_J_hat_rows), torch.cat(u_hat_J_hat_right_cols)], dim=0
        )

        self.combined_coo_indices = torch.cat(
            [
                self.u_hat_u_hat_coo_indices,
                self.u_hat_J_hat_left_coo_indices,
                self.u_hat_J_hat_right_coo_indices,
            ],
            dim=1,
        )

        self.batched_A_sparse = self._sparse_kron(
            torch.eye(batch_size, device=self.device).to_sparse_coo(), self.A_sparse
        )
        self.batched_exp_M_p0_sparse = self._sparse_kron(
            torch.eye(batch_size, device=self.device).to_sparse_coo(),
            self.exp_M_p0_sparse,
        )

    def _init_params(self, J_init, u_init):
        batch_size = J_init.shape[0]
        rhs = torch.zeros(
            batch_size,
            self.N_t + (self.N_t - 1) + 2,
            self.num_fields,
            device=self.device,
            dtype=torch.float64,
        )
        rhs[:, -2, :] = J_init
        rhs[:, -1, :] = u_init
        if self.linear_solver == "block":
            # A = kron(I, A_per_field): solve field by field.
            return (self.A_per_field_inv @ rhs.transpose(-2, -1).unsqueeze(-1)).reshape(
                batch_size, -1
            )
        return (self.A_inv @ rhs.transpose(-2, -1).reshape(batch_size, -1, 1)).squeeze(
            -1
        )

    def _compute_residual(self, params, J_init, u_init):
        """
        Compute the residual of the system.

        Args:
            params: Parameters of the system.
            J_init: Initial J values.
            u_init: Initial u values.

        Returns:
            A tuple (residual, residual) of the computed residual.
        """
        params_reshaped = params.reshape(self.num_fields, -1).transpose(-2, -1)
        J_hat = params_reshaped[: self.N_t, :]
        u_hat = params_reshaped[self.N_t : -2, :]

        linear_part = self.A @ params
        if hasattr(self.nn_rhs_model, "forward_original"):
            forward_func = self.nn_rhs_model.forward_original
        else:
            # try to use the default forward method, though this might give an error
            forward_func = self.nn_rhs_model.forward
        nonlinear_part = self.M_p0 @ forward_func(
            J_hat,
            u_hat,
        )
        nonlinear_part = (
            torch.cat(
                [
                    torch.zeros_like(J_hat),
                    nonlinear_part,
                    torch.zeros(2, self.num_fields, device=self.device),
                ]
            )
            .transpose(-2, -1)
            .reshape(-1)
        )
        residual = linear_part + nonlinear_part
        residual = residual.squeeze(-1)
        rhs = torch.zeros_like(params_reshaped)
        rhs[-2, :] = J_init
        rhs[-1, :] = u_init
        residual = residual - rhs.transpose(-2, -1).reshape(-1)
        return residual, residual

    def _compute_J_and_residual(self, params, J_init, u_init, z=None, is_final=False):
        """
        Compute the Jacobian and residual of the system.

        Args:
            params: Parameters of the system.
            J_init: Initial J values.
            u_init: Initial u values.

        Returns:
            A tuple (Jacobian, residual).
        """
        if self.linear_solver == "block":
            return self._compute_block_J_and_residual(params, J_init, u_init, z)
        batch_size = params.shape[0]
        params_reshaped = params.reshape(batch_size, self.num_fields, -1).transpose(
            -2, -1
        )
        J_hat = params_reshaped[:, : self.N_t, :]
        u_hat = params_reshaped[:, self.N_t : -2, :]

        t_start = time.time()
        (u_hat_jacobian, z_jacobians), u_hat_forcing = self.nn_rhs_model(
            J_hat, u_hat, z
        )
        t_end = time.time()
        if not hasattr(self, "nn_time"):
            self.nn_time = 0.0
        self.nn_time += t_end - t_start

        # self.u_hat_forcing = u_hat_forcing.clone().detach().cpu()

        linear_part = (self.A @ params.unsqueeze(-1)).squeeze(-1)
        nonlinear_part = self.M_p0 @ u_hat_forcing
        nonlinear_part = (
            torch.cat(
                [
                    torch.zeros_like(J_hat),
                    nonlinear_part,
                    torch.zeros(batch_size, 2, self.num_fields, device=self.device),
                ],
                dim=1,
            )
            .transpose(-2, -1)
            .reshape(batch_size, -1)
        )
        residual = linear_part + nonlinear_part

        rhs = torch.cat(
            [
                torch.zeros(
                    batch_size,
                    params_reshaped.shape[1] - 2,
                    self.num_fields,
                    device=self.device,
                ),
                J_init.unsqueeze(1),
                u_init.unsqueeze(1),
            ],
            dim=1,
        )

        residual = residual - rhs.transpose(-2, -1).reshape(batch_size, -1)

        if not hasattr(self, "batch_size") or self.batch_size != batch_size:
            print("rebuilding indexers... ")
            self.batch_size = batch_size
            self._build_indexers(batch_size)
            print("done rebuilding indexers")

        N = self.N_t + (self.N_t - 1) + 2
        if not self.use_sparse_solver:
            target_Jacobian = torch.zeros(
                batch_size * self.num_fields * N * self.num_fields * N,
                device=self.device,
                dtype=torch.float64,
            ).contiguous()
            target_Jacobian = torch.scatter_add(
                target_Jacobian,
                0,
                self.u_hat_u_hat_indices,
                u_hat_jacobian.permute(0, 2, 3, 1).contiguous().reshape(-1),
            )
            target_Jacobian = torch.scatter_add(
                target_Jacobian,
                0,
                self.u_hat_J_hat_left_indices,
                z_jacobians.permute(3, 0, 2, 4, 1)[0].contiguous().reshape(-1),
            )
            target_Jacobian = torch.scatter_add(
                target_Jacobian,
                0,
                self.u_hat_J_hat_right_indices,
                z_jacobians.permute(3, 0, 2, 4, 1)[1].contiguous().reshape(-1),
            )
            target_Jacobian = target_Jacobian.contiguous()
            J = self.A + torch.matmul(
                self.exp_M_p0,
                target_Jacobian.reshape(
                    batch_size, self.num_fields * N, self.num_fields * N
                ),
            )
            neural_jac = torch.matmul(
                self.exp_M_p0,
                target_Jacobian.reshape(
                    batch_size, self.num_fields * N, self.num_fields * N
                ),
            )
            if is_final and hasattr(self, "neural_jacs"):
                self.neural_jacs.append(neural_jac)
        else:
            J = torch.sparse_coo_tensor(
                self.combined_coo_indices,
                torch.cat(
                    [
                        u_hat_jacobian.permute(0, 2, 3, 1).reshape(-1),
                        z_jacobians.permute(3, 0, 2, 4, 1)[0].reshape(-1),
                        z_jacobians.permute(3, 0, 2, 4, 1)[1].reshape(-1),
                    ]
                ),
                size=(
                    batch_size * self.num_fields * N,
                    batch_size * self.num_fields * N,
                ),
                device=self.device,
            )
            J = self.batched_A_sparse + self.batched_exp_M_p0_sparse @ J
            J = J.coalesce()

        return J, residual

    def _dense_solve(self, J, rhs):
        """
        Solve the batched dense system J x = rhs, re-solving failed batch elements on CPU.

        Some GPU backends (observed with hipSOLVER on ROCm) spuriously report
        well-conditioned matrices as singular or return non-finite solutions.

        Args:
            J: Batched matrices of shape (batch, n, n).
            rhs: Right-hand sides of shape (batch, n, 1).

        Returns:
            The solution x of shape (batch, n, 1).
        """
        x, info = torch.linalg.solve_ex(J, rhs)
        bad = (info != 0) | ~torch.isfinite(x).flatten(1).all(dim=-1)
        if bad.any():
            if not hasattr(self, "dense_solve_fallbacks"):
                self.dense_solve_fallbacks = 0
            self.dense_solve_fallbacks += int(bad.sum())
            print(
                f"Dense solve failed on {J.device} for batch elements "
                f"{bad.nonzero().flatten().tolist()} (info={info[bad].tolist()}); "
                f"re-solving on CPU (total fallbacks: {self.dense_solve_fallbacks})"
            )
            # Rebuild x from scratch: keeping the failed solve in the autograd
            # graph would leak NaNs into the gradients even after overwriting it.
            good = ~bad
            x = torch.zeros_like(rhs)
            if good.any():
                x[good] = torch.linalg.solve(J[good], rhs[good])
            x[bad] = torch.linalg.solve(J[bad].cpu(), rhs[bad].cpu()).to(J.device)
        return x

    def _timestep(self, J_init, u_init, z=None, use_cpu_solver=False):
        """
        Perform a single timestep of the system.

        Args:
            J_init: Initial J values.
            u_init: Initial u values.

        Returns:
            A tuple (params, (lhs, rhs)) of the computed parameters and linear system.
        """
        # Infer batch size from input
        batch_size = J_init.shape[0]

        params = self._init_params(J_init, u_init)
        if self.nn_rhs_model.dropout_rate > 0:
            seed = int(time.time())
        eta = torch.ones(batch_size, device=self.device)
        with torch.no_grad():
            for i in range(self.max_iters):
                if self.nn_rhs_model.dropout_rate > 0:
                    torch.manual_seed(seed)
                if self.use_manual_jacobian:
                    J, H = self._compute_J_and_residual(params, J_init, u_init, z)
                else:
                    J, H = torch.vmap(
                        torch.func.jacrev(self._compute_residual, has_aux=True),
                        randomness="different",
                    )(params, J_init, u_init)
                if (H.abs() < self.tol).all():
                    break
                if i > 0:
                    improved = (
                        H.square().mean(dim=-1).sqrt()
                        < old_H.square().mean(dim=-1).sqrt()
                    )
                    eta = torch.where(
                        improved,
                        1.0,
                        0.5 * eta,
                    )
                    params = torch.where(improved.unsqueeze(-1), params, old_params)
                if self.use_sparse_solver:
                    t_start = time.time()
                    delta_params = -sparse_linear_solve(
                        J, H.reshape(-1), batch_size, use_cpu_solver
                    ).reshape(batch_size, -1)
                    t_end = time.time()
                    if not hasattr(self, "sparse_solve_time"):
                        self.sparse_solve_time = 0.0
                    self.sparse_solve_time += t_end - t_start
                else:
                    t_start = time.time()
                    delta_params = -self._solve(J, H)
                    t_end = time.time()
                    if not hasattr(self, "dense_solve_time"):
                        self.dense_solve_time = 0.0
                    self.dense_solve_time += t_end - t_start
                old_params = params.clone()
                if i == 0:
                    old_H = H.clone()
                else:
                    old_H = torch.where(improved.unsqueeze(-1), H, old_H)
                params = params + eta.unsqueeze(-1) * delta_params

        if i == self.max_iters - 1:
            print(
                f"Did not converge after {self.max_iters} iterations: {H.abs().max().item()}"
            )
            if self.use_sparse_solver and not use_cpu_solver:
                # try again with the CPU solver, this happens sometimes with CuDSS
                return self._timestep(J_init, u_init, z=z, use_cpu_solver=True)
        else:
            if self.verbose:
                print(f"Converged after {i+1} iterations: {H.abs().max().item()}")
        if self.nn_rhs_model.dropout_rate > 0:
            torch.manual_seed(seed)
        if self.use_manual_jacobian:
            J, H = self._compute_J_and_residual(
                params, J_init, u_init, z, is_final=True
            )
        else:
            J, H = torch.vmap(
                torch.func.jacrev(self._compute_residual, has_aux=True),
                randomness="different",
            )(params, J_init, u_init)

        # Host copy of the last Jacobian (batch * (F*(2*N_t+1))^2 * 8 bytes per
        # window); callers that never read self.J can set
        # store_last_jacobian = False to skip it.
        # (Not stored in block mode, where J is a set of per-slab blocks.)
        if getattr(self, "store_last_jacobian", True) and self.linear_solver == "dense":
            self.J = J.detach().cpu()

        if self.backprop_type == "schur":
            lhs = J
            rhs = J @ params.unsqueeze(-1) - H.unsqueeze(-1)
            return params, (lhs, rhs)
        else:
            if self.use_sparse_solver:
                delta_params = -sparse_linear_solve(
                    J, H.reshape(-1), batch_size, use_cpu_solver
                ).reshape(batch_size, -1)
                params = params + eta.unsqueeze(-1) * delta_params
            else:
                delta_params = -self._solve(J, H)
                # params = torch.linalg.solve(
                #     J, J @ params.unsqueeze(-1) - H.unsqueeze(-1)
                # ).squeeze(-1)
                params = params + eta.unsqueeze(-1) * delta_params

        return params, (None, None)

    def evolve_batch(self, batch):
        """
        Evolve a batch of initial conditions through time.

        Args:
            batch: Batch of data containing J_init and u_init.

        Returns:
            Tuple of (J_hat_rollout, u_hat_rollout)
        """
        if isinstance(batch, torch.Tensor):
            u_init = batch
            J_init = torch.zeros_like(u_init)
            batch_size = u_init.shape[0]
            z = None
        else:
            batch_size = batch["J_init"].shape[0]
            J_init = batch["J_init"]
            u_init = batch["u_init"]
            z = batch.get("z", None)

        J_hat_list = []
        u_hat_list = []
        lambda_1_list = []
        lambda_2_list = []
        lhs_list = []
        rhs_list = []

        # if hasattr(self, "include_mortars") and self.include_mortars:
        #     u_hat_list.append(u_init.unsqueeze(1))

        if hasattr(self, "model_list"):
            self.nn_rhs_model = self.model_list[0]

        params, (lhs, rhs) = self._timestep(J_init, u_init, z, self.use_cpu_solver)
        params = params.reshape(batch_size, self.num_fields, -1).transpose(-2, -1)
        J_hat = params[:, : self.N_t, :]
        u_hat = params[:, self.N_t : -2, :]
        lambda_1 = params[:, -2, :]
        lambda_2 = params[:, -1, :]
        u_next = lambda_2

        J_hat_list.append(J_hat[:, :-1, :])
        u_hat_list.append(u_hat)
        lambda_1_list.append(lambda_1)
        if hasattr(self, "include_mortars") and self.include_mortars:
            u_hat_list.append(lambda_2.unsqueeze(1))
        lambda_2_list.append(lambda_2)
        lhs_list.append(lhs)
        rhs_list.append(rhs)

        for i in range(1, self.num_training_rollouts):
            J_init = J_hat[:, -1, :]
            u_init = u_next
            if hasattr(self, "model_list"):
                self.nn_rhs_model = self.model_list[i]
            params, (lhs, rhs) = self._timestep(J_init, u_init, z, self.use_cpu_solver)
            params = params.reshape(batch_size, self.num_fields, -1).transpose(-2, -1)
            J_hat = params[:, : self.N_t, :]
            u_hat = params[:, self.N_t : -2, :]
            lambda_1 = params[:, -2, :]
            lambda_2 = params[:, -1, :]
            u_next = lambda_2

            J_hat_list.append(J_hat[:, :-1, :])
            u_hat_list.append(u_hat)
            if hasattr(self, "include_mortars") and self.include_mortars:
                u_hat_list.append(lambda_2.unsqueeze(1))
            lambda_1_list.append(lambda_1)
            lambda_2_list.append(lambda_2)
            lhs_list.append(lhs)
            rhs_list.append(rhs)

        self.last_J_hat = J_hat[:, -1, :]
        self.last_u_hat = u_next
        self.lambda_2_list = lambda_2_list

        if self.backprop_type == "schur":
            J_hat_rollout_schur, u_hat_rollout_schur = self._schur_complement_solve(
                batch, lhs_list, rhs_list
            )
            return J_hat_rollout_schur, u_hat_rollout_schur
        else:
            J_hat_rollout = torch.cat(J_hat_list, dim=1)
            u_hat_rollout = torch.cat(u_hat_list, dim=1)

        return J_hat_rollout, u_hat_rollout

    def rollout_batch(self, batch, num_rollouts=None):
        """
        Perform a forward-only rollout of the batch without saving information for backpropagation.
        This is more efficient for validation or inference where gradients are not needed.

        Args:
            batch: Batch of data containing J_init and u_init.
            num_rollouts: Optional number of rollouts to perform.
                          Defaults to self.num_training_rollouts if None.

        Returns:
            The evolved *J_hat_rollout, u_hat_rollout)
        """
        with torch.no_grad():
            # Store original values
            original_backprop_type = self.backprop_type
            original_num_training_rollouts = self.num_training_rollouts

            # Set temporary values
            self.backprop_type = "matrix_free"
            if num_rollouts is not None:
                self.num_training_rollouts = num_rollouts

            try:
                # Use the existing evolve_batch function
                J_hat_rollout, u_hat_rollout = self.evolve_batch(batch)
                return J_hat_rollout, u_hat_rollout
            finally:
                # Restore original values (even if an exception occurs)
                self.backprop_type = original_backprop_type
                self.num_training_rollouts = original_num_training_rollouts

    def _compute_permutation_indices(self):
        """
        Compute permutation indices for the system.

        Returns:
            A tuple (permutation_indices_without_lambdas, permutation_indices) of indices.
        """
        # first, find the indices of the lambdas
        lambda_indices = torch.tensor(
            [
                [
                    field_ndx * (self.N_t + (self.N_t - 1) + 2) - 2,
                    field_ndx * (self.N_t + (self.N_t - 1) + 2) - 1,
                ]
                for field_ndx in range(1, self.num_fields + 1)
            ],
            dtype=torch.long,
        ).flatten()
        # next, deinterleave the indices: lambda_1s are first, then the lambda_2s
        tmp = lambda_indices.clone()
        (
            tmp[: tmp.shape[0] // 2],
            tmp[tmp.shape[0] // 2 :],
        ) = (
            lambda_indices[::2],
            lambda_indices[1::2],
        )
        lambda_indices = tmp
        # build up permutation indices:
        permutation_indices = [
            i for i in range(self.A.shape[0]) if i not in lambda_indices
        ]
        permutation_indices_without_lambdas = torch.tensor(
            permutation_indices, dtype=torch.long
        )
        permutation_indices = torch.cat(
            [
                permutation_indices_without_lambdas,
                lambda_indices,
            ]
        )
        return permutation_indices_without_lambdas.to(
            self.device
        ), permutation_indices.to(self.device)

    def _compute_B(self):
        """
        Compute the B matrix for the Schur complement method.

        Returns:
            The computed B matrix.
        """
        B = self.A[self.permutation_indices, :][:, self.permutation_indices]
        B = B[: self.num_fields * (self.N_t + (self.N_t - 1)), :][
            :, self.num_fields * (self.N_t + (self.N_t - 1)) :
        ]
        B = torch.kron(torch.eye(self.num_training_rollouts, device=self.device), B)
        return B.to(torch.float64)

    def _compute_C(self):
        """
        Compute the C matrix for the Schur complement method.

        Returns:
            The computed C matrix.
        """
        J_hat_init_indices = torch.zeros(
            self.N_t + (self.N_t - 1) + 2,
            self.num_fields,
            dtype=torch.long,
            device=self.device,
        )
        J_hat_init_indices[0, :] = 1
        J_hat_init_indices = (
            J_hat_init_indices.T.reshape(-1)[self.permutation_indices]
            .nonzero()
            .squeeze()
        )
        J_hat_last_indices = torch.zeros(
            self.N_t + (self.N_t - 1) + 2,
            self.num_fields,
            dtype=torch.long,
            device=self.device,
        )
        J_hat_last_indices[self.N_t - 1, :] = 1
        J_hat_last_indices = (
            J_hat_last_indices.T.reshape(-1)[self.permutation_indices]
            .nonzero()
            .squeeze()
        )

        row_indices_list = []
        col_indices_list = []
        values_list = []
        for rollout_ndx in range(self.num_training_rollouts):
            row_offset = rollout_ndx * 2 * self.num_fields
            row_indices = row_offset + torch.arange(self.num_fields, device=self.device)
            if rollout_ndx == 0:
                col_indices = J_hat_init_indices
                values = torch.ones_like(col_indices)
                row_indices_list.append(row_indices)
                col_indices_list.append(col_indices)
                values_list.append(values)
                continue
            col_offset_current = (
                self.num_fields * (self.N_t + (self.N_t - 1)) * rollout_ndx
            )
            col_offset_last = (
                self.num_fields * (self.N_t + (self.N_t - 1)) * (rollout_ndx - 1)
            )
            col_indices = col_offset_last + J_hat_last_indices
            values = -torch.ones_like(col_indices)
            row_indices_list.append(row_indices)
            col_indices_list.append(col_indices)
            values_list.append(values)
            col_indices = col_offset_current + J_hat_init_indices
            values = torch.ones_like(col_indices)
            row_indices_list.append(row_indices)
            col_indices_list.append(col_indices)
            values = torch.ones_like(col_indices)
            values_list.append(values)
        row_indices = torch.cat(row_indices_list)
        if self.num_fields == 1:
            col_indices = torch.stack(col_indices_list)
        else:
            col_indices = torch.cat(col_indices_list)
        if self.num_fields == 1:
            values = torch.stack(values_list)
        else:
            values = torch.cat(values_list)
        C = torch.sparse_coo_tensor(
            torch.stack([row_indices, col_indices]),
            values,
            size=(
                self.num_training_rollouts * 2 * self.num_fields,
                self.num_training_rollouts
                * self.num_fields
                * (self.N_t + (self.N_t - 1)),
            ),
        ).to(self.device)
        return C.to_dense().to(torch.float64)

    def _compute_D(self):
        """
        Compute the D matrix for the Schur complement method.

        Returns:
            The computed D matrix.
        """
        lambda_1_indices = torch.arange(self.num_fields)
        lambda_2_indices = torch.arange(self.num_fields) + self.num_fields
        row_indices_list = []
        col_indices_list = []
        values_list = []
        for rollout_ndx in range(self.num_training_rollouts):
            row_offset = rollout_ndx * 2 * self.num_fields
            row_indices = row_offset + lambda_2_indices
            if rollout_ndx == 0:
                col_indices = lambda_1_indices
                values = torch.ones_like(col_indices)
                row_indices_list.append(row_indices)
                col_indices_list.append(col_indices)
                values_list.append(values)
                continue
            col_offset_current = 2 * self.num_fields * rollout_ndx
            col_offset_last = 2 * self.num_fields * (rollout_ndx - 1)
            col_indices = col_offset_last + lambda_2_indices
            values = -torch.ones_like(col_indices)
            row_indices_list.append(row_indices)
            col_indices_list.append(col_indices)
            values_list.append(values)
            col_indices = col_offset_current + lambda_1_indices
            values = torch.ones_like(col_indices)
            row_indices_list.append(row_indices)
            col_indices_list.append(col_indices)
            values_list.append(values)
        row_indices = torch.cat(row_indices_list)
        col_indices = torch.cat(col_indices_list)
        values = torch.cat(values_list)
        D = torch.sparse_coo_tensor(
            torch.stack([row_indices, col_indices]),
            values,
            size=(
                self.num_training_rollouts * 2 * self.num_fields,
                self.num_training_rollouts * 2 * self.num_fields,
            ),
        ).to(self.device)
        return D.to_dense().to(torch.float64)

    def _schur_complement_solve(self, batch, lhs_list, rhs_list):
        """
        Solve the system using the Schur complement method.

        Args:
            batch: Batch of data containing J_init and u_init.
            lhs_list: List of left-hand sides of the linear systems.
            rhs_list: List of right-hand sides of the linear systems.

        Returns:
            The computed u_hat_rollout.
        """
        # Infer batch size from input
        batch_size = batch["J_init"].shape[0]

        self.lhs_list = lhs_list

        lhs = torch.stack(lhs_list).transpose(0, 1)[
            ..., self.permutation_indices_without_lambdas, :
        ][
            ..., self.permutation_indices_without_lambdas
        ]  # batch_ndx, time_ndx, matrix
        rhs = (
            torch.stack(rhs_list)
            .transpose(0, 1)[..., self.permutation_indices_without_lambdas, :]
            .squeeze(-1)
        )
        u = rhs.reshape(batch_size, -1, 1)
        inits = torch.cat([batch["J_init"], batch["u_init"]], dim=-1)
        v = torch.zeros(batch_size, self.num_training_rollouts * 2 * self.num_fields)
        v[:, : inits.shape[1]] = inits
        v = v.to(self.device)
        v = v.unsqueeze(-1)
        A_inv = torch.linalg.inv(lhs)
        A_inv = torch.vmap(lambda X: torch.block_diag(*X))(A_inv)
        S = self.D - self.C @ A_inv @ self.B
        S_inv = torch.linalg.inv(S)
        y = S_inv @ (v - self.C @ A_inv @ u)
        x = A_inv @ (u - self.B @ y)
        J_hat_rollout = (
            x.squeeze(-1)
            .reshape(batch_size, self.num_training_rollouts, self.num_fields, -1)[
                :, :, :, : self.N_t - 1
            ]
            .transpose(-2, -1)
        ).reshape(batch_size, -1, self.num_fields)
        u_hat_rollout = (
            x.squeeze(-1)
            .reshape(batch_size, self.num_training_rollouts, self.num_fields, -1)[
                :, :, :, self.N_t :
            ]
            .transpose(-2, -1)
        ).reshape(batch_size, -1, self.num_fields)
        return J_hat_rollout, u_hat_rollout


class _BlockJacobian:
    """A window's Newton Jacobian for linear_solver="block", kept as per-slab blocks.

    The constant part comes from HybridNODESolver._build_block_system (A blocks); the network
    part is the per-cell Jacobians scaled by the cell size (h_k dN_k/du_k, dN_k/dJ_k,
    dN_k/dJ_{k+1}), each (batch, M, F, F). solve() does block forward substitution in time;
    it is built from differentiable torch ops, so the final Newton step backpropagates as in
    the dense path.
    """

    def __init__(self, solver, d_u, d_left, d_right):
        self.solver = solver
        self.d_u = d_u
        self.d_left = d_left
        self.d_right = d_right

    def _cell_rows_block(self, left, right):
        """(batch, 2F, 2F) block that is zero in the node rows and [left | right] in the cell rows."""
        top = torch.zeros(
            left.shape[0], left.shape[1], 2 * left.shape[2], dtype=left.dtype, device=left.device
        )
        return torch.cat([top, torch.cat([left, right], dim=-1)], dim=1)

    def solve(self, H):
        s = self.solver
        slabs = s._slabs
        batch_size = H.shape[0]
        n_slabs = len(slabs)
        Y = []
        for i, slab in enumerate(slabs):
            is_cell = 1 <= i <= n_slabs - 2
            k = i - 1  # cell index of a cell slab
            r = H[:, slab["row_index"]]
            for d in slab["deps"]:
                block = slab["A_deps"][d]
                if is_cell and d == i - 1:
                    # The previous slab holds J_k (its second column group).
                    zeros = torch.zeros_like(self.d_left[:, k])
                    block = block + self._cell_rows_block(zeros, self.d_left[:, k])
                r = r - (block @ Y[d].unsqueeze(-1)).squeeze(-1)
            D = slab["A_diag"]
            if is_cell:
                D = D + self._cell_rows_block(self.d_u[:, k], self.d_right[:, k])
            else:
                D = D.expand(batch_size, -1, -1)
            Y.append(s._dense_solve(D, r.unsqueeze(-1)).squeeze(-1))
        return torch.cat(Y, dim=1)[:, s._slab_inv_perm]
