# HNODE: High-Level Overview

Notes on the algorithm and this codebase.
Paper: *A Hybridizable Neural Time Integrator for Stable Autoregressive Forecasting*,
Kinch, Hu, Huang, Hansen, Meltzer, Hamlin, Sirajuddin, Cyr, Trask. [arXiv:2604.21101](https://arxiv.org/abs/2604.21101)

## Notation

Some letters mean two things; context (and this list) tells them apart.

| Symbol | Meaning |
|---|---|
| u, J | State and velocity (J = du/dt) |
| N_θ | The neural network (latent acceleration) |
| N | Number of rollout windows per training sample (the code's `N`) |
| M | Cells per window (`N_t − 1`), except in **H(p) = A·p + M·N_θ(p) − b**, where M is the cell mass matrix |
| h | Cell size (the time step) |
| λ₁, λ₂ | u at a window's start and end (traces) |
| Jac | Jacobian of H (written Jac to avoid confusion with J) |
| F | Latent dimension = number of fields (64 for shear flow) |
| B | Batch size |

"Rollout" and "window" mean the same thing: one Newton-solved time interval.

## The idea in one paragraph

Autoregressive neural models of chaotic systems usually either blow up during long rollouts
or get exploding gradients when trained over long horizons. HNODE fixes both by placing the
neural network inside a classical mixed finite-element time integrator. The network only
supplies the local dynamics. The finite-element structure supplies the stability: discrete
energy is conserved for Hamiltonian systems and decays when dissipation is added, and
gradients have a bound that does not depend on the number of rollouts.

## The pieces

### 1. The model equation
The system is written as second order:

    ü = N(u, u̇; θ),   u(0) = u₀,   u̇(0) = v₀

The velocity becomes its own unknown, `J = u̇`, which gives two first-order equations.
**N is the neural network.** It predicts an acceleration, not a velocity.

### 2. Time finite elements
Time is cut into N rollout windows, and each window into M sub-intervals.
- `u`: piecewise constant (P0), one value per sub-interval.
- `J`: continuous piecewise linear (P1), with values at the nodes.

This pairing makes d/dt map exactly from J's space into u's space (the 1D "de Rham"
property). That exactness is what drives the energy proof, through a summation-by-parts identity.

Weak form on window i (paper's sign convention):

    (J, v) + (u, v') = λ_{i,i+1} v(t_{i+1}) − λ_{i,i} v(t_i)
    (J', q) − (N[u, J; θ], q) = 0
    J(t_i)   = J_{i−1}(t_i)
    λ_{i,i}  = λ_{i−1,i}

### 3. Shooting and hybridization (the "H" in HNODE)
Each window is solved on its own, and its end state is the next window's start. This is
shooting. Two Lagrange multipliers, λ (the trace of u at each window edge), glue the windows
together, as in mortar methods. The paper describes this as the first extension of mortar
methods to initial-value problems.

### 4. A local transformer as N
A cell's output depends only on its own `u` and the two `J` values at its edges, plus any
conditioning such as log(Re) or voltage. This keeps the Jacobian sparse and the method
faithful to the finite-element structure.

### 5. Training by implicit differentiation
Each window is a nonlinear system `H(Y; Y_prev, θ) = 0`, solved with Newton's method.
Gradients come from the implicit function theorem, not from unrolling the solver iterations.
The key theorem (paper, Theorem 4): the gradient bound depends on the cell size and the total
time, but does not grow with the number of rollouts N. Ordinary backpropagation through time
has no such guarantee.

### 6. Latent space for big problems
For PDE data, a Swin-V2 encoder and decoder map fields to a 64-dimensional latent vector
(the paper calls them latent tokens), and HNODE evolves that vector. The encoder, decoder and HNODE are trained end to end.

## How the code implements it

### Unknowns per window (per field)
`N_t` values of J at the nodes, `N_t − 1` cell values of u, then λ₁ and λ₂:
`2·N_t + 1` unknowns in total. For Lorenz (`N_t = 11`, 3 fields) that is a dense 69×69
system per batch item.

The per-field system built in `_build_A_per_field` (four blocks of rows; `_build_A_and_M_p0`
then forms the full A = kron(I_F, ·) in dense mode):

    block 1 (N_t rows):     M_rt·J + Gᵀ·u + C1·λ₁ + C2·λ₂ = 0    (J = du/dt weakly; λ are the u traces)
    block 2 (N_t − 1 rows): G·J + M_p0·N(J, u)          = 0    (dJ/dt = −N; the sign is flipped from the paper)
    block 3 (1 row):        J(0)                         = J_init
    block 4 (1 row):        λ₁                           = u_init   (A[-1,:] is overwritten to impose this)

The output of each window is `u_next = λ₂` and `J_next = J(Δt)`. These become the next
window's `u_init` and `J_init` (see `evolve_batch`).

### Forward pass: Newton solve per window (`_timestep`)
1. Build the residual `H = A·params + M·N_θ(params) − b`.
2. Build the Jacobian `A + M·∂N_θ/∂params`. The network computes its own input-Jacobian with
   `vmap(jacrev)`, and the result is block-sparse because each cell depends only on its
   neighbours. In dense mode, `_build_indexers` scatters those blocks into the full matrix; in
   block mode (`linear_solver="block"`) they are kept as per-cell blocks.
3. Run damped Newton, halving the step if the residual grows, until the residual is below `1e-12`
   (optionally relative to the state size: `relative_tol`, see "Large latent dimensions").

### Backward pass
- `backprop_type="matrix_free"` (the default in the examples): the Newton loop runs under
  `no_grad`, then one final Newton step runs with autograd on. Since the residual is about
  zero at convergence, the gradient of that step equals the implicit-function-theorem
  gradient, `−Jac⁻¹ ∂H/∂θ`. Gradients still flow across windows through `J_init` and `u_init`.
- `backprop_type="schur"`: all windows are solved together as one linear system, using a
  Schur complement over the λ's (`_compute_B/C/D`, `_schur_complement_solve`).

### Data and loss
`ode_datasets.py` samples the trajectory on an interleaved grid: even indices are nodes
(`J`), odd indices are cell midpoints (`u`). The loss is MSE between the predicted per-cell
values `u_hat` across all windows and the midpoint ground truth. States are standardized,
and time is rescaled by `t_scale` so that J has unit variance.

## Paper → code map

| Paper concept | Code |
|---|---|
| The weak form: finite-element matrices for (J, v), (u, v′) and (J′, q) | `_build_A_per_field`, which builds `M_rt`, `G`, `M_p0` with scikit-fem |
| λ multipliers and continuity between windows | The last two unknowns per field; `u_next = lambda_2`, `J_init = J_hat[:, -1]` in `evolve_batch` |
| Local transformer N | `CrossAttnTransformer.forward` in `neural_networks.py`: 3 tokens (J_left, u, J_right), Jacobian from `vmap(jacrev)` |
| Newton solve of H = 0 | `_timestep` (damped Newton) and `_compute_J_and_residual` |
| Implicit-differentiation gradient | The final Newton step taken with autograd on, after a `no_grad` loop |
| All windows solved together via Schur complement | `backprop_type="schur"` |

## Files

| File | Role |
|---|---|
| `hybrid_node_solver.py` | Core: the finite-element matrices, Newton solve (dense or block linear solve), rollout, and Schur-complement backprop |
| `neural_networks.py` | `CrossAttnTransformer`, the per-cell N_θ and its Jacobian |
| `ode_datasets.py`, `pca_datasets.py` | Data sampled on the staggered node/midpoint grid |
| `01_lorenz.py` | Simplest case: the 3-D state is used directly (start here) |
| `02_flow_past_cylinder.py` | State is 64 PCA modes, conditioned on `z = log(Re)` |
| `03_shear_flow.py`, `04_plasma_gun.ipynb` | A Swin-V2 encoder/decoder maps fields to a latent space; HNODE runs there, trained end to end |
| `03_shear_flow_ddp.py`, `run_shear_flow_ddp.sh` | Data-parallel shear-flow training on Frontier, reading The Well HDF5 files; `MODE=grad_check` and `MODE=eval` check modes |
| `run_shear_flow_repro.sh` | Training from scratch in 24-hour jobs (`-p extended`) that resume from the last checkpoint; see "Training from scratch vs the reference checkpoint" |
| `soap.py` | The SOAP optimizer; on ROCm its GPU QR/eigh results are checked and recomputed on the CPU when wrong (see "Platform note: SOAP's QR on ROCm") |
| `solver_backends.py`, `test_solver_backends.py`, `test_block_mode.py` | Linear-solver comparison harness and the tests for `linear_solver="block"` |
| `shear_flow_common.py` | Helpers shared by the latent-space scripts below (devices, frame statistics, The Well splits, loading the pretrained models) |
| `pod_basis.py`, `run_pod_basis.sh` | POD basis of shear flow (1,024 modes, distributed method of snapshots), the coefficients of every frame and the projection floor (`hnode_runs/pod/full/`) |
| `ae_train.py` | Two-stage autoencoders (`cnn` or `swin`) trained on frames, and `encode` to write their latents |
| `encode_latents.py` | Latents of every frame from the frozen pretrained Swin encoder, plus the reference HNODE rollout |
| `latent_baselines.py` | Time steppers in a frozen latent space: `hnode`, `node1`, `node2`, `lstm`, `koopman`, `dmd`, all trained the same way; see "Large latent dimensions" for the HNODE settings |
| `eval_latent_rollout.py` | Decodes any latent rollout (`DECODER=swin`, `pod:R` or `ae:CKPT`) and scores it like `MODE=eval`; `compare` plots several |
| `run_latent_job.sh`, `submit_latent_row.sh`, `submit_ae.sh` | Slurm launchers: one "row" = every time stepper on one latent space, then the evaluation |
| `profile_latent_dim.py`, `run_profile_latent_dim.sh` | Time and memory of HNODE vs latent dimension F (64–1024) and solver options; see "Large latent dimensions" |
| `summarize_latent_dim.py` | VRMSE versus latent dimension for every row and HNODE scaling (`hnode_runs/rows/latent_dim_vrmse.png`) |
| `spectral_energy.py` | Kinetic-energy spectra versus time of every row and the Swin models (`compute` on 8 GPUs, then `plot`); see "Does a larger latent dimension help HNODE?" |

## Headline results (from the paper)
- **Lorenz:** stable for about 10,000 Lyapunov times, and it reproduces the lobe-switching
  statistics (the histogram at the end of `01_lorenz.py`).
- **Flow past a cylinder:** 9 simulations with Re between 60 and 800; lift and drag stay
  stable indefinitely across the parameter family.
- **Shear flow (The Well benchmark):** 2–3× lower VRMSE than PhysiX, with 65× fewer
  parameters (69M against 4.5B).
- **Fusion MITL (plasma gun):** about 9,000× faster than particle-in-cell, trained on only
  12 simulations.

> These paper details came from a summarizing web fetch, not a full read of the PDF.
> Check exact equations and constants against the paper.

---

# Deep dive (shear-flow example)

## Overall picture

```
 frame 0, frame 1 ──SwinEncoder──► L0, L1  (64 each)
                                    │
                 u_init = (L0+L1)/2 ,  J_init = (L1−L0)/dt ,  z = (log Re, log Sc)
                                    │
                                    ▼
 ┌──────── WINDOW k  (delta_t = 0.05 = 5 frames, 5 cells, h = 0.01 = 1 frame) ───────┐
 │   λ₁ │  u_0  │  u_1  │  u_2  │  u_3  │  u_4  │ λ₂       unknowns p: 64 × 13 = 832 │
 │     J_0     J_1     J_2     J_3     J_4     J_5                                   │
 │   Solve  H(p) = A·p + M·N_θ(p) − b = 0   with damped Newton (~3–7 iterations)     │
 │   hand-off: λ₂ → next λ₁ ,  J_5 → next J_0                                        │
 └───────────────────────────────────────────────────────────────────────────────────┘
                    │  × 40 windows
                    ▼
  u_hat_rollout: 200 cells × 64   (cell k ↔ frame k+1)
                    │  pick 12 random cells ──SwinDecoder──► 12 frames
                    ▼
              VRMSE vs real frames = loss
                    │
  loss.backward(): decoder ← 40 windows (implicit gradients) ← encoder
```

## The variables

| Symbol | Meaning (shear flow) | Discretization | Decoded / supervised? |
|---|---|---|---|
| **u** | Latent state: the encoder's 64-number code for a frame | Piecewise constant (P0), one 64-vector per cell | Yes: cell k is decoded to frame k+1 |
| **J** | Latent velocity du/dt (not the physical fluid velocity) | Piecewise linear (P1), one 64-vector per node | No, internal only |
| **λ₁** | u at the window's start (trace) | One per field per window | No; pinned to `u_init` |
| **λ₂** | u at the window's end (trace) | One per field per window | No; becomes the next window's `u_init` |
| **N_θ** | Latent acceleration dJ/dt per cell | One 64-vector per cell | The learned dynamics |

- **The hat** (`u_hat`, `J_hat`) means the discrete finite-element coefficients of u(t) and J(t).
  `u_hat` is `(B, 5, 64)` inside one window and `u_hat_rollout` is `(B, 200, 64)` over all windows.
  In Lorenz and the cylinder flow, `batch["u_hat"]` holds the training targets.
- **L0 and L1** are `encoder(frame 0)` and `encoder(frame 1)`. They are inputs, not unknowns.
  Their average and difference give u and du/dt at t = 0.5 frames, where the first window starts.
  The first predicted cell, u_0, is centred on frame 1, so u_0 ≈ L1, but the solver computes u_0 rather than setting it.
- **Why λ exists:** u is discontinuous, so it has no value at a window edge. Integrating
  `(J, v) = (du/dt, v)` by parts gives boundary terms, and λ₁ and λ₂ stand in for them
  (Lagrange multipliers, also called traces or mortars). The discrete equations give:

      u_0 = λ₁ + h·(J_0/3 + J_1/6)        λ₂ = u_4 + h·(J_4/6 + J_5/3)

  The edge value equals the nearest cell value plus or minus the integral of J over half a cell.

## Per-window system: H(p) = A·p + M·N_θ(p) − b = 0

Per field, `p = [J_0…J_5, u_0…u_4, λ₁, λ₂]` (13 numbers). The 13 equations are:

| Rows | Equation | Meaning |
|---|---|---|
| 6 (J rows) | `(J, v) + (u, v') − λ-terms = 0` | J = du/dt |
| 5 (u rows) | `J_{c+1} − J_c + h·N_θ(u_c, J_c, J_{c+1}) = 0` | J changes according to the network |
| 1 | `J_0 − J_init = 0` | Initial velocity |
| 1 | `λ₁ − u_init = 0` | Initial state |

### Shapes (one window, batch size B, shear flow, dense mode)

| Symbol | Kind | Math shape | Code shape | Contents |
|---|---|---|---|---|
| p | vector | 832 | `params` `(B, 832)` | All unknowns |
| A | matrix | 832 × 832 | `self.A` | `kron(I₆₄, A₁₃ₓ₁₃)`, fixed |
| N_θ(p) | vector | 5 × 64 | `u_hat_forcing` `(B, 5, 64)` | Network output per cell |
| M | matrix | 5 × 5 | `self.M_p0`, diagonal = h | Cell mass matrix (full-size version: `self.exp_M_p0`, 832 × 832) |
| M·N_θ(p) | vector | 832 after zero padding | `(B, 832)` | Nonzero only in the u rows |
| b | vector | 832 | `rhs` `(B, 13, 64)` → `(B, 832)` | `J_init` and `u_init` in the last rows |
| H | vector | 832 | `residual` `(B, 832)` | Equation errors; zero at the solution |
| Jac | matrix | 832 × 832 | `(B, 832, 832)` | A + M·∂N/∂p, about 9% nonzero |
| ∂N_c/∂u_c | 64×64 block per cell | `u_hat_jacobian` `(B, 5, 64, 64)` | Scattered into Jac |
| ∂N_c/∂(J_c, J_{c+1}) | two 64×64 blocks per cell | `z_jacobians` `(B, 5, 64, 2, 64)` | Scattered into Jac |
| Δp | vector | 832 | `delta_params` `(B, 832)` | Newton correction |

## What is known and what is unknown

**A, M, b and N_θ are all known. Only p is unknown.** Newton adjusts p until H(p) = 0.

| Symbol | Known or unknown? | Where it comes from |
|---|---|---|
| A | known, fixed | Built once from the window size (`delta_t`, `N_t`) |
| M | known, fixed | The cell size h = 0.01 on its diagonal |
| b | known for this window | `J_init` and `u_init`, from the encoder or the previous window |
| N_θ | known function | The network, with its weights frozen during the solve |
| **p** | **unknown** | The 832 numbers being solved for |

The weights θ change only during training, when the optimizer updates them from the loss after
the whole rollout.

### The same idea with one unknown
A toy equation with the same structure: `H(p) = 2·p + 0.1·p² − 3 = 0`. Here `2·p` plays A·p,
`0.1·p²` plays M·N_θ(p), and `3` plays b. The guess ignores the nonlinear part (`2p = 3`, the
equivalent of `A⁻¹b`), then each round updates p ← p − H/H′:

| Round | p | Error H |
|---|---|---|
| 0 | 1.500000 | 0.225 |
| 1 | 1.402174 | 0.00096 |
| 2 | 1.401754 | 1.8e-8 |
| 3 | 1.401754 | 4e-16 |

With 832 unknowns: p and H become vectors of 832, the slope becomes the 832 × 832 Jacobian,
and "divide by the slope" becomes solving `Jacobian · Δp = H`.

### In plain words
Think of 832 dials and 832 rules. One rule uses the network, which reads the same dials being
set, so the dials can't be filled in one at a time. Instead: **guess** (everything keeps its
speed), **check** how wrong each rule is (H), **work out** how each dial affects each error
(the Jacobian), **turn all the dials at once** by the computed amount, and if that made things
worse, undo it and try a half-size turn. The errors collapse quickly (0.1, 0.01, 0.0001, …),
so 3–7 rounds suffice. Then λ₂ is passed to the next window, like a relay.

## Where A·p + M·N_θ(p) − b = 0 comes from

It is the **finite element (Galerkin) method applied in time** to the model equations.

**Continuous equations** on one window [t₀, t₁]:

    du/dt = J,    dJ/dt = N_θ(u, J),    J(t₀) = J_init,    u(t₀) = u_init

**Approximation:** u is piecewise constant (5 cell values) and J is piecewise linear (6 node
values). A piecewise-constant u has no derivative inside a cell and jumps at the edges, so the
equations are required to hold **on average against test functions** (the weak form):
multiply by a test function, integrate, and demand zero. Using as many test functions as
unknowns gives a square system.

1. **J = du/dt**, tested with piecewise-linear v and integrated by parts:

       ∫ J·v dt = [u·v] from t₀ to t₁ − ∫ u·(dv/dt) dt

   The derivative moves off the discontinuous u. The boundary term needs u at t₀ and t₁,
   which a piecewise-constant u doesn't have, so they become new unknowns: **λ₁ and λ₂**.
   → 6 linear rows, part of **A·p**.
2. **dJ/dt = N_θ**, tested with one indicator function per cell:

       ∫ over cell c of dJ/dt dt = J_{c+1} − J_c        (exact: J is linear on the cell)
       ∫ over cell c of N_θ dt   = h · N_θ(u_c, J_c, J_{c+1})

   → 5 rows: the J part goes into **A·p**, the network part becomes **M·N_θ(p)**.
3. **Initial conditions:** J₀ = J_init and λ₁ = u_init → 2 rows; the known values form **b**.

Stacking 13 rows × 64 channels gives the 832 equations. The code uses the opposite sign in
step 2 (J′ = −N), which only flips what the network learns.

### The results behind it

| Result | What it guarantees here |
|---|---|
| Galerkin method / weak formulation | A square system (832 equations, 832 unknowns) that approximates the true equations better as cells shrink |
| Compatible spaces (1D de Rham property: the derivative of a piecewise-linear function is piecewise constant) | Equation 2 is exact cell by cell; this is why u is P0 and J is P1 |
| Summation by parts (paper, Lemma 1) → discrete energy conservation (paper, Theorem 2) | Sums telescope across cells and windows; for Hamiltonian dynamics the discrete energy ½J² + V(u) is preserved exactly, and with dissipation it only decreases. **This is the source of long-rollout stability** |
| Lagrange multipliers / hybridization (mortar methods) | λ₁, λ₂ enforce continuity weakly, so each window is solved on its own and handed to the next (shooting) |
| Newton's method (Newton–Kantorovich theorem) | With an invertible Jacobian and a close guess, the error roughly squares each round, hence 3–7 rounds |
| Implicit function theorem | H(p, θ) = 0 defines p(θ), with dp/dθ = −Jac⁻¹ ∂H/∂θ: the final Newton step with autograd. Paper Theorem 4 bounds it independently of the number of windows |

In one sentence: the equation is the model written in weak (integrated) form on simple
piecewise functions. Compatible spaces make it exact and energy-preserving, λ glues the
windows, Newton solves it, and the implicit function theorem gives its gradients.

## Newton solve per window (`_timestep`)

```python
p = A⁻¹ b                                     # guess: constant-velocity motion (N_θ = 0)
with torch.no_grad():
    for i in range(100):
        Jac, H = _compute_J_and_residual(p)   # network forward + per-cell Jacobians (vmap(jacrev))
        if max|H| < 1e-12: break
        if i > 0:                             # per batch element: reject a step that made H worse
            improved = RMS(H) < RMS(old_H)
            eta = 1 if improved else eta/2
            p   = p if improved else old_p
        Δp = −solve(Jac, H)                   # 832×832 dense LU (_dense_solve), or the block solve
        old_p = p
        p = p + eta·Δp
Jac, H = _compute_J_and_residual(p)           # one more step WITH autograd:
p = p + eta · (−solve(Jac, H))                # attaches the implicit gradient −Jac⁻¹ ∂H/∂θ
```

- **Cost** is dominated by the network Jacobian: 64 backward passes through the transformer per cell.
  The 832×832 solve is cheap on a GPU.
- **Measured iteration counts:** Lorenz (trained) 5–6, cylinder flow (trained) 6–7, shear flow
  (pretrained) 4–5 (3 with an untrained network). The residual reaches about 1e-16 each time.
- **Quirks** (from reading the code, harmless in practice):
  1. A rejected step reuses Δp computed at the rejected point, applied at `old_p` with half the step size.
     This is not a textbook backtracking line search.
  2. The final step is scaled by eta, so the gradient is too if eta < 1 at convergence. Usually eta = 1.

## Window size (time subdivision)

```python
delta_t = dt * num_time_steps_per_data_sample / N    # window length
N_t     = num_time_steps_per_data_sample // N + 1    # nodes per window
h       = delta_t / (N_t - 1)                        # cell size
```

| Example | Window `delta_t` | Cells per window | Cell size h | Windows in training (`N`) | Unknowns per window |
|---|---|---|---|---|---|
| Lorenz | 0.1 (scaled to 0.745) | 10 | 0.01 | 11 | 3 × 23 = 69 |
| Cylinder flow | 0.4 = 4 data steps (scaled to 0.409) | 4 | 0.1 | 8 | 64 × 11 = 704 |
| Shear flow | 0.05 = 5 frames | 5 | 0.01 = 1 frame | 40 | 64 × 13 = 832 |

"Scaled" means divided by the dataset's `t_scale` normalization; the cell sizes h are in
physical (unscaled) time units.

The window size is baked into `A` when the solver is built; to use a different one, rebuild the
solver (no retraining needed, see below). `num_rollouts` changes only the number of windows. This
is unrelated to the Swin `window_size=(8, 8)`, which is a spatial attention window.

### Cells per window affect performance, not accuracy

**With the cell size h fixed, the number of cells per window changes only speed and memory, not
the solution.** Where two windows meet, the last J-row of one window plus the first J-row of the
next add up to exactly the interior row of one long window:

    window k, last row:     (h/6)J₄ + (h/3)J₅ + u₄ − λ₂ = 0
    window k+1, first row:  (h/3)J₀′ + (h/6)J₁′ − u₀′ + λ₁′ = 0
    with λ₁′ = λ₂, J₀′ = J₅: (h/6)J₄ + (2h/3)J₅ + (h/6)J₁′ + u₄ − u₀′ = 0   ← the interior row

So chaining windows through λ solves the same global system as a single window over all cells, and
because time-stepping is causal, solving window by window is exact.

Checked numerically (shear flow, same network and start state, 200 cells):

| Cells per window | max \|u − u(5 cells)\| | Relative gradient difference |
|---|---|---|
| 1 | 1.1e-14 | 3.3e-15 |
| 10 | 8.9e-15 | 2.9e-15 |
| 40 | 2.0e-14 | 3.8e-15 |

| You change… | Cell size h | Effect |
|---|---|---|
| Cells per window (`N_t − 1`), keeping h | same | Performance only |
| Window length, keeping cells per window | changes | Accuracy: h is the real time step; also changes the cell ↔ frame mapping (needs code changes and retraining) |

In `03_shear_flow.py`, h = `dt` whatever `N` is, so **`N` is purely a performance knob** (it must
divide 200). Caveats: Newton must converge to the same root (a very stiff trained model might
converge more slowly or fail in long windows; that is speed or robustness, not accuracy), and
round-off differences of about 1e-14 can grow over very long chaotic rollouts, like running on
different hardware.

**Measured cost** (shear flow, 200 cells, one MI210, untrained network, solver only):

| Cells per window | Windows | Unknowns per window | Newton iterations | Rollout (s) | Training step (s) | Peak memory (GiB) |
|---|---|---|---|---|---|---|
| 1 | 200 | 320 | 3.0 | 14.4 | 18.4 | 6.6 |
| 2 | 100 | 448 | 3.0 | 8.1 | 9.7 | 6.7 |
| 5 (current) | 40 | 832 | 3.0 | 4.0 | 4.2 | 6.7 |
| 10 | 20 | 1,472 | 3.0 | 2.3 | 2.5 | 7.1 |
| 20 | 10 | 2,752 | 3.0 | 2.2 | 3.1 | 7.7 |
| 40 | 5 | 5,312 | 3.4 | 2.5 | 2.7 | 9.4 |

Small windows mean many sequential solves too small to keep the GPU busy; large windows mean
fewer solves, but each dense solve costs n³ and memory grows as n².

**How to choose:**
1. Keep h equal to the data spacing (1 frame).
2. Pick cells per window for speed: it must divide the training horizon (for 200 frames: 5, 8,
   10, 20, 25, 40, …), in dense mode the Jacobian (unknowns² × 8 bytes × batch) must fit in memory, and
   Newton must still converge in a few iterations.
3. Re-measure with the trained checkpoint: a stiffer trained network may need more Newton
   iterations in long windows.
4. A checkpoint trained with one window size can be rolled out with another: rebuild the solver
   with the new `N`.

For shear flow, `N = 20` (10 cells per window) looks best: 1.4–1.7× faster per training step
(1.7× in this untrained sweep, 1.4× with the pretrained network; see "Linear solvers"), with the
same results.

## N_θ: local cross-attention transformer (`CrossAttnTransformer.forward`)

> **N_θ takes one cell's local state and returns that cell's acceleration.**

```
Input  (for one cell c)                     Output
─────────────────────────                   ───────────────────────
u_c        state in this cell      (64)     N_c  = dJ/dt in this cell   (64)
J_c        velocity at left edge   (64)  →        (latent acceleration)
J_{c+1}    velocity at right edge  (64)
z          conditioning (Re, Sc)    (2)
```

- The same network, with the same weights, runs on every cell independently (5 cells per window,
  200 per training sample).
- For Newton, the code also returns N_θ's derivatives with respect to its three vector inputs
  (three 64×64 matrices per cell), computed automatically with `vmap(jacrev)`.
- Its output enters the cell equation **J_{c+1} − J_c + h·N_c = 0** (the code's sign convention),
  so N_θ decides how the velocity changes from one node to the next.

Per window, as called by the solver (B = batch size):

| | Name | Shape |
|---|---|---|
| Input | `J_hat` (all 6 nodes) | (B, 6, 64) |
| | `u_hat` (all 5 cells) | (B, 5, 64) |
| | `z` | (B, 2) |
| Output | `u_hat_forcing` (N_c for each cell) | (B, 5, 64) |
| | `u_hat_jacobian` (∂N/∂u) | (B, 5, 64, 64) |
| | `z_jacobians` (∂N/∂J_left, ∂N/∂J_right) | (B, 5, 64, 2, 64) |

### Inside N_θ

For each cell c, **one query token attends to that cell's own context tokens only**:

```
query:        Emb_u(u_c)
keys/values:  Emb_u(u_c) + t(0),  Emb_J(J_c) + t(−1),  Emb_J(J_{c+1}) + t(+1),  [2 z-tokens from (Re, Sc)]
kv = SelfAttnBlock(kv)                    # once
3 × { q += CrossAttn(norm(q), kv);  q += FFN(norm(q)) }    # latent 256, 4 heads, FFN 4×, tanh, DyT norm
N_c = Linear(256 → 64)(q)
```

- Cells never attend to each other, which keeps the Jacobian block-sparse (three 64×64 blocks per cell).
- `t(·)` is a learned embedding of −1, 0, +1 that tags tokens as left, centre or right.
- The weights are shared across cells, windows and trajectories. The conditioning z enters as extra key tokens.
- Note: the `use_self_attention_z_block` constructor flag is never read (`neural_networks.py:169`).
  The z self-attention block is always created and used whenever z is given.

### Where the conditioning parameters go

Conditioning (Re, Sc, voltage, …) goes **only** into `CrossAttnTransformer`. The solver passes it
through untouched; the encoder and decoder never see it.

```
03_shear_flow.py:338   z = (log10(Re, Sc) − z_mean) / z_std          # (B, 2), normalized
      ↓  latent_batch["z"]
HybridNODESolver       evolve_batch → _timestep → _compute_J_and_residual(..., z)
      ↓                (unchanged, every Newton iteration, every window)
CrossAttnTransformer.forward(J_hat, u_hat, z)
   :266   copied to every cell
   :284   z_embedding: Linear(z_dim → z_dim × 256), giving z_dim tokens
   :285   added to the key/value tokens: [u_c, J_c, J_c+1] + z-tokens
   :287   self-attention block mixes state and z tokens
          3 cross-attention blocks: query u_c attends to them
```

- z enters as **extra context tokens** in attention: not concatenated onto the state, and not a
  Newton unknown.
- The same z is used for every cell, window and Newton iteration of a trajectory.
- The Jacobian ignores z (`jacrev(argnums=(0, 1))` differentiates only u and J).
- A, M, b, the encoder and the decoder don't depend on z: the latent space is shared across
  parameter values, and only the **dynamics** change with z.

| Example | z | z tokens |
|---|---|---|
| Lorenz | none | 0 |
| Cylinder flow | normalized log Re | 1 |
| Shear flow | normalized (log₁₀ Re, log₁₀ Sc) | 2 |
| Plasma gun | applied voltage (per the paper; not traced in the code) | — |

**For a new problem:** set `CrossAttnTransformer(..., z_dim=<number of parameters>)`, put a
normalized `(B, z_dim)` tensor in `latent_batch["z"]` (log-scale parameters that span orders of
magnitude), or omit z and use `z_dim=None`. Leave `use_u_init_embedding` off: that path uses a
`u_init_embedding` layer the constructor never creates.

## Using a different autoencoder

The solver never sees the Swin networks: it exchanges only **latent vectors** with them
(`evolve_batch` takes `J_init`, `u_init`, `z` and returns `u_hat`). Any encoder/decoder pair that
maps the data to a fixed-size vector and back plugs in: CNN, U-Net encoder, graph network for
meshes, point-cloud or Fourier autoencoder. The repo already shows two kinds: fixed linear PCA
(`02_flow_past_cylinder`) and a jointly trained Swin autoencoder (`03_shear_flow`).

**Requirements:**
1. **The latent is a flat vector of size F.** The solver treats each latent dimension as a field
   (`num_fields = latent_dim`). A spatial latent (C × h × w) must be flattened, so F = C·h·w.
2. **F stays modest.** Per window with M cells: unknowns = F·(2M+3); the dense Jacobian costs
   (F·(2M+3))² memory and cubic solve time; the network Jacobian needs F backward passes per
   cell (the dominant cost); ∂N/∂u is F × F per cell. F = 64 is comfortable, a few hundred is
   probably workable with smaller windows. Much larger F (e.g. 1024, about 13,000 unknowns per
   window at M = 5) needs `linear_solver="block"` (the repo's cuDSS sparse path is NVIDIA-only;
   see "Linear solvers on Frontier") and a cheaper way to compute the network Jacobian. This is
   estimated from the scaling, not tested.
3. **Differentiable, if trained jointly:** gradients flow from the decoder, through all windows,
   into the encoder.
4. **Two starting frames** with known spacing, for `J_init = (L1 − L0)/dt` and
   `u_init = (L0 + L1)/2`.

**Design choices:**
- **Joint vs frozen.** Joint training lets the encoder learn latents that move smoothly in time,
  which is easier for N_θ and for Newton. A frozen pretrained autoencoder is simpler, but its
  latents may move erratically. PCA sits in between.
- **Latent scale.** Keep latents of order 1: N_θ uses tanh, and Newton uses an absolute tolerance
  of 1e-12.
- **Precision.** The solver needs float64; the autoencoder may not. Running it in float32 and
  casting the latents could save a lot (untested).
- **What the guarantees cover.** Energy conservation and bounded gradients apply to the **latent**
  dynamics. The decoder can still produce physically inconsistent fields; physical conservation
  would have to be built into the decoder or the loss.

| Piece in `03_shear_flow.py` | Change |
|---|---|
| `SwinEncoder` / `SwinDecoder` | Your encoder (data → F) and decoder (F → data) |
| `latent_dim` | Your F: sets `problem_dim` of N_θ and `num_fields` of the solver |
| `rearrange(... "B T H W C -> (B T) C H W")` | Reshapes matching your data layout |
| `normalize`, `stats.npy`, `vrmse_loss` dimensions | Your normalization and loss |
| `z` and `z_dim` | Your conditioning parameters, or none |
| `HybridNODESolver`, `CrossAttnTransformer` | No change |

## Why only 12 random frames are decoded

Each decoded frame is a full 256×512×4 Swin-decoder pass whose activations must be stored for
backprop, so decoding all 200 frames is too expensive. The 12 frames are drawn uniformly from
1–199, so their average VRMSE is an **unbiased estimate** of the full-trajectory loss (minibatching
along time). The full 40-window rollout still runs every time, so late targets send gradients
back through every window to the encoder. Frame 0 is excluded because it is only an input.
This is my reading of the code; the script doesn't say why.

## Training vs inference

**Learned:** network weights only: N_θ, plus the encoder and decoder for shear flow and the plasma gun.
**Solved each window:** u, J, λ (they are states, not parameters).
**Fixed:** A, M, the normalization stats and the PCA basis (cylinder flow).

Training is already a full autoregressive rollout with no teacher forcing. Data enters only as the
initial condition and the loss targets.

| | Training | Inference (`rollout_batch`) |
|---|---|---|
| Windows | `N` (40 for shear flow) | Any number (1000 in the cylinder example) |
| Newton solve and network Jacobian | Yes | Yes: Newton still needs the Jacobian |
| Final gradient step | Carries autograd | Runs under `no_grad` |
| Decoded frames | 12 random | All |

Inference needs the trained weights, **the same cell size h** (= `delta_t / (N_t − 1)`; the
number of cells per window may differ, see "Cells per window affect performance, not accuracy"),
**the same normalization stats**, an initial condition and z. With the wrong stats the model fails: a shortened Lorenz
dataset gave an MSE of 1.86 instead of 4.6e-5.

## Verified numbers (runs on Frontier)

- **Lorenz** (`lorenz_prod_day3.pt`: 133,000 steps, final training loss 1.5e-6): an 11-window
  rollout on a random batch of 4 gives MSE 4.6e-5.
- **Cylinder flow:** all 9 Reynolds numbers roll out 1000 windows. The largest final |u| is
  5.4–6.3, against 10.4 in the data.
- **Shear flow:** see "Accuracy vs the paper" (224 held-out trajectories) and "Step timing" below.

## Platform note: AMD (ROCm) false "singular" linear solves

### Symptom
Running `02_flow_past_cylinder.py` on Frontier crashed partway through the 1000-window evaluation:

```
torch._C._LinAlgError: torch.linalg.solve: (Batch element 0):
The solver failed because the input matrix is singular.
```

It was raised by the dense Newton solve in `_timestep` (then `hybrid_node_solver.py:699`).
Re = 60, 80, 100, 150 and 200 each completed 1000 windows; **Re = 300 failed at window 164, on
the second Newton iteration**. The rollout itself was healthy: earlier windows converged in 6–7
iterations and the state stayed bounded (max |u| ≈ 5.6, against 10.4 in the data).

### The matrix was not singular
The failing 704 × 704 Jacobian (64 modes × 11 unknowns per window) was saved and tested:

| Check | Result |
|---|---|
| Finite / NaN | All finite, no NaN |
| Condition number | ≈ 104 (smallest singular value 0.09): well conditioned |
| CPU `solve` | Solved, residual 7.8e-16 |
| GPU, MAGMA backend | Solved, residual 6.7e-16 |
| GPU, default backend (hipSOLVER; PyTorch calls it "cusolver") | **`info = 98`, NaN output**, so it raises "singular" |
| Same backend, `lu_factor_ex` alone | `info = 0`, all pivots equal to 1: the factorization is fine |
| Same backend, `lu_factor` + `lu_solve` | Solved, residual 6.7e-16 |

### Cause
A numerical bug in ROCm's combined solve path (torch 2.9.1 + ROCm 6.4): for certain matrices it
falsely reports a zero pivot at row 98. It is not the model, the checkpoint or the data;
`repro.pt` reproduced the authors' output exactly (`t_scale = 0.9770`, 53712 samples), and their
notebook ran on NVIDIA/CUDA. It is rare (about 9 in 60,000 solves), but the dense solve had no
fallback, so one false report ended the whole run. Every failure reported `info = 98`, which
suggests it is tied to the structure of these matrices rather than being random.

### Options considered
1. `torch.backends.cuda.preferred_linalg_library("magma")`: works, about 13% slower, and changes
   the backend globally.
2. `lu_factor` + `lu_solve`: worked on the failing matrix, but was not tested end to end.
3. **Chosen:** `solve_ex` on the GPU, falling back to the CPU only for the failed batch items.

### The fix: `_dense_solve` in `hybrid_node_solver.py`
Used at both dense solve sites (the Newton loop and the final differentiable step):

```python
x, info = torch.linalg.solve_ex(J, rhs)
bad = (info != 0) | ~torch.isfinite(x).flatten(1).all(dim=-1)
if bad.any():
    print("Dense solve failed on ... re-solving on CPU (total fallbacks: N)")
    x = torch.zeros_like(rhs)                                 # rebuild from scratch
    x[good] = torch.linalg.solve(J[good], rhs[good])          # GPU
    x[bad]  = torch.linalg.solve(J[bad].cpu(), rhs[bad].cpu()).to(J.device)  # CPU
```

The solution is **rebuilt from scratch** rather than patched. A first version only overwrote the
bad items, and it leaked **NaN into the gradients**: the failed solve stayed in the autograd graph,
and its backward pass multiplied zero by the NaN solution. The counter is kept in
`solver.dense_solve_fallbacks`.

### Verification
- All 9 Reynolds numbers complete 1000 windows on the default GPU backend, with 9 CPU fallbacks
  in total (1 at Re = 300, 8 at Re = 400).
- No slowdown: 91–124 s per Reynolds number, the same as before the fix.
- Values and gradients through the fallback match a CPU reference to about 1e-14, both for a batch
  with a failed element and for a clean batch.

### Notes
- The same bug can hit the Lorenz and shear-flow examples on Frontier; the fix covers them.
- `_schur_complement_solve` (`backprop_type="schur"`) uses `torch.linalg.inv` through the same
  backend and is not covered. No example uses that mode.

## Platform note: SOAP's QR on ROCm

Training the shear-flow model **from scratch** on Frontier went NaN twice, at steps 714 and 731
(jobs 5555394 and 5557663). Resuming from the pretrained checkpoint never showed it.

### Symptom
- From about step 150, the parameter check every 50 steps found the 8 ranks' weights differing by up
  to 100% ("drifted (rel. 1.00e+00)"), although their gradients are identical after the all-reduce.
- The NaN appeared right after a SOAP preconditioner update (every 10 steps), then spread to every
  rank. Newton solves on rank 0 were fine up to that step. Every later solve failed, steps slowed
  from 5 s to 100 s, and rank 0 wrote a 1.6 GB log.

### Cause
SOAP refreshes each preconditioner's eigenbasis with `torch.linalg.qr` (and sometimes
`torch.linalg.eigh`) in float32 on the GPU. On ROCm, hipSOLVER's QR can return a Q that is far
from orthogonal:

| Input (from the saved SOAP state, step 400) | \|QᵀQ − I\| |
|---|---|
| GPU, float32 (as SOAP runs it) | **0.43**, the same on three repeats |
| CPU, float32 | 7e-7 |
| CPU, float64 | 1e-15 |

- The matrix is 225×225 and rank-deficient, with entries up to 5.6e-14. 225 is the row count of
  the Swin relative-position-bias tables (15×15), which get almost no gradient early in training.
- Whether the GPU QR fails depends on the exact bits of the input: the same matrix computed on the
  GPU instead of the CPU factors correctly. So the 8 ranks got different bases, which is the drift,
  and a corrupt basis eventually blew the weights up.
- The reference was trained on an NVIDIA H200, where this does not happen.

### Fix: `SOAP._checked` in `soap.py`
Every GPU QR and eigh result is checked (|QᵀQ − I| < 1e-3 and finite). If the check fails, it is
recomputed on the CPU in float64 and the log gets one line:
`SOAP: QR on cuda:0 not orthogonal (...); recomputed on CPU in float64 (total fallbacks: N)`.
Nothing changes when the GPU result is fine.

### Verification
Job 5558706, from scratch, 2,600 steps:
- About 250 fallbacks per rank, all on the same 225×225 matrix, nearly one per preconditioner
  update.
- No NaN and no skipped steps.
- Rank drift stayed at round-off level, at most 9.4e-5 over 53 checks, where the failing runs
  reached 1.0.
- The loss followed the reference (next section).

The same bug family as the false "singular" solves above: ROCm's dense factorizations can
fail silently on inputs the CPU handles, so results that feed training should be checked.

## Linear solvers on Frontier: dense, sparse and block

### What exists
| Candidate | Status on Frontier (torch 2.9.1 + ROCm 6.4) |
|---|---|
| Dense batched LU (`_dense_solve`) | The default; works, with the CPU fallback for the ROCm bug |
| Repo's sparse path (`use_sparse_solver=True`, `sparse_solve` extension) | NVIDIA-only (cuDSS); the extension is not present, so it would crash at the first solve |
| `torch.sparse.spsolve` | "not supported in ROCm build" |
| CuPy sparse | Not installed |
| SciPy `splu` (SuperLU, CPU) | Works, CPU only |
| rocSOLVER / hipSOLVER / rocSPARSE, SuperLU_DIST modules | Present, but need a compiled extension |

### The block solver (`linear_solver="block"`)
After eliminating the known λ₁ = u_init and J₀ = J_init, each window's Jacobian is **block
lower-triangular in time**: the equations of cell k involve only the unknowns of cells k, k−1 and
k−2. Grouping unknowns as slabs (u_k, J_{k+1}) turns the Newton solve into a forward march: one
2F × 2F dense solve per cell instead of one n × n solve. Its determinant is the product of the small
blocks' determinants, so it is solvable exactly when the full system is; each block uses LU with
pivoting, and errors propagate like an implicit time-stepper, as in the dense solve.

`HybridNODESolver(..., linear_solver="block")` never builds `A`, `A_inv`, `exp_M_p0` or the dense
Jacobian: it keeps the per-field block `A_per_field` (A = kron(I_F, A_per_field)) and the network's
per-cell Jacobian blocks (`_BlockJacobian`). The starting guess uses the small per-field inverse.
`apply_A(params)` gives A @ params in either mode. The default `"dense"` path is unchanged.

### Results
Standalone harness (`test_solver_backends.py`; real per-window Jacobians; one MI210; per solve call
for a batch of 1–8 windows; results in `hnode_runs/solver_backends/`):

| Case | Unknowns | Nonzeros | Dense GPU | SciPy CPU | Block GPU | Block vs dense |
|---|---|---|---|---|---|---|
| Cylinder matrix that trips the ROCm bug | 704 | 10.2% | NaN (bug) | exact | exact | — |
| F = 64, M = 5 (shear flow) | 832 | 9.2% | 0.043 s | 0.11 s | 0.0085 s | 5.1× |
| F = 64, M = 20 | 2,752 | 3.3% | 0.122 s | 0.37 s | 0.029 s | 4.3× |
| F = 64, M = 40 | 5,312 | 1.8% | 0.144 s | 0.49 s | 0.058 s | 2.5× |
| F = 128, M = 10 | 2,944 | 5.8% | 0.067 s | 0.42 s | 0.025 s | 2.7× |
| F = 256, M = 10 | 5,888 | 5.7% | 0.164 s | 26 s | 0.068 s | 2.4× |

All backends match a float64 reference to about 1e-15; the block pattern holds exactly; block vs
dense gradients agree to about 1e-15 (for J on its nonzeros, and for the right-hand side). Run the
harness one case per process: many cases in one process eventually trigger a hipBLAS failure.

Solver option end to end (`test_block_mode.py`):
- **Default mode is bit-identical** to the previous version (rollouts, gradients, Newton counts).
- **Block vs dense:** rollouts agree to about 1e-17 and gradients to about 1e-15, with identical
  Newton iteration counts (F = 64/M = 5 and 20, F = 128/M = 10).
- **F = 256, M = 40 (21,248 unknowns per window):** the dense constructor fails (hipBLAS error on
  `A_inv`); block mode solves a window in 0.6 s using 1.5 GiB.
- **Shear-flow size (200 cells, batch 1):**

| Cells per window | Mode | Rollout | Training step | Peak memory |
|---|---|---|---|---|
| 5 | dense | 3.74 s | 4.44 s | 6.88 GiB |
| 5 | block | 4.01 s | 4.91 s | 6.57 GiB |
| 10 | dense | 2.70 s | 3.09 s | 7.16 GiB |
| 10 | block | 2.48 s | 2.99 s | 6.49 GiB |

### Memory: where the solver matters
The dense path's extra memory is its n × n matrices; at shear-flow size they are small:

| Size | n | Fixed dense matrices (A, A_inv, exp_M_p0) | Per-window Jacobians kept for backprop | Dense-only total (est.) |
|---|---|---|---|---|
| F = 64, M = 5 | 832 | 17 MB | 40 × ~11 MB | ~0.4 GB (measured saving 0.31 GiB) |
| F = 64, M = 10 | 1,472 | 52 MB | 20 × ~35 MB | ~0.7 GB (measured saving 0.67 GiB) |
| F = 256, M = 10 | 5,888 | 0.8 GB | 20 × ~0.55 GB | ~12 GB |
| F = 256, M = 40 | 21,248 | 10.8 GB | 5 × ~7.2 GB | ~47 GB (does not fit) |

At F = 64 about 95% of the memory is the network Jacobian (`vmap(jacrev(N_θ))`, F backward passes
per cell) and its double-backward graph, which the solver choice does not touch. To cut memory at
this size, compute the Jacobian in chunks of cells, avoid keeping the whole double-backward graph,
or run the autoencoder in float32.
(The double-backward graph is now optional: `detach_final_jacobian`, which cuts training memory
from 6.7 to 0.4 GiB per trajectory at F = 64 with the same gradient; see "Large latent dimensions".)

### Recommendation
- Keep `linear_solver="dense"` for shear flow and the gx project at their current sizes.
- Use `"block"` for large latent dimensions or long windows, and as a workaround if the dense solve
  keeps tripping the ROCm bug.
- The bigger free speed-up for shear flow is **10 cells per window instead of 5** (`N = 20`):
  1.4–1.6× faster training steps with the pretrained network in either mode (4.44 → 3.09 s dense,
  4.91 → 2.99 s block), with identical results.

## Data-parallel training for shear flow (`03_shear_flow_ddp.py`)

A separate script, `03_shear_flow_ddp.py`, trains the shear-flow model with one process per GPU.
The original `03_shear_flow.py` is unchanged. Launch script: `run_shear_flow_ddp.sh`
(1 node, 8 GPUs, `srun`).

### The two groups of parameters
The script trains three networks, which fall into two groups:

| Group | Modules | How they're called |
|---|---|---|
| Autoencoder | `encoder`, `decoder` (Swin-V2, `torch.compile`d, most of the 69,032,704 parameters) | Ordinary forward calls, once per step |
| Dynamics | `nn_rhs_model` (N_θ, 3,206,728 parameters) | Inside the solver: a few (3–7) Newton iterations × 40 windows per step, through `vmap(jacrev)`, with a double backward in the final step |

### Why not wrap the models in `DistributedDataParallel`
- **Many forwards per backward.** The solver calls N_θ about 40 times with autograd on (the final
  Newton step of each window), plus the `no_grad` iterations, before a single `loss.backward()`.
  DDP's reducer expects one forward per backward.
- **The solver reads attributes off the model.** `_timestep` reads `nn_rhs_model.dropout_rate`, and
  `_compute_residual` checks for `forward_original`. A DDP wrapper hides those attributes.
- **torch.func transforms.** `vmap(jacrev(...))` over the module's parameters is fragile under DDP's
  forward hooks.

### Strategy: one manual gradient all-reduce for all three models
After `loss.backward()`, the gradients of all three models are flattened into one buffer,
all-reduced, divided by the world size, and copied back:

```python
params = [p for p in all_params if p.grad is not None]
flat = _flatten_dense_tensors([p.grad for p in params])
dist.all_reduce(flat); flat.div_(world_size)
# copy back into each p.grad
```

This is mathematically the same as DDP: each rank's loss is a mean, so the averaged gradient is
the gradient of the global mean. It needs no changes to the solver or the models, and it works with
`torch.compile`, `jacrev`, the double backward and the CPU fallback in `_dense_solve`. The cost is
one float64 all-reduce of about 0.55 GB per step, not overlapped with backward; that is expected
to be small next to the Newton solves, and the script prints the time per step.

### Other pieces
| Piece | What it does |
|---|---|
| Rank info | From the `srun` step (`SLURM_PROCID`, `SLURM_LOCALID`, `SLURM_STEP_NUM_TASKS`), or torchrun variables if set; one process runs as a single rank |
| Device | `--gpus-per-task=1` gives each rank one visible GPU; otherwise the local rank picks the GPU |
| Backend | `nccl` (RCCL on ROCm); `DDP_BACKEND=gloo` lets ranks share one GPU for testing |
| Same starting weights | Every rank loads the checkpoint (so SOAP's state matches too), then rank 0's parameters are broadcast |
| Data | `DistributedSampler` (with `set_epoch`), so each rank takes the same number of steps; per-rank seeds give different random frames |
| Global batch | `batch_size × world_size`; the learning rate is kept at 1e-4 |
| Safety checks | Every step, ranks confirm they all-reduce the same set of gradients; a step whose averaged gradient is not finite is skipped on every rank (the log names the ranks with a non-finite loss); every 50 steps, parameter checksums are compared and rank 0's weights re-broadcast if they drifted; checkpoints with non-finite weights are never saved, and the previous one is kept as `SAVE_PATH.prev` |
| Rank 0 only | Logging, test loss, checkpoint saves and the evaluation figures (`shear_flow_ddp_loss.png`, `shear_flow_ddp_rollout.png`, `shear_flow_ddp_vrmse.png`) |
| MIOpen cache | Per rank in `/tmp`, to avoid lock contention on a shared database |
| Data source | `DATA_FORMAT=well` (default): The Well HDF5 files under `WELL_DATA_PATH`; `DATA_FORMAT=npy`: the local per-frame folders |
| Test knobs | `MODE` (`train`, `grad_check`, `eval`), `CHECKPOINT_PATH` (`none` = from scratch), `SAVE_PATH`, `NUM_EPOCHS`, `MAX_STEPS`, `RUN_EVAL=0` |
| Long-run knobs | `TOTAL_STEPS` (stop once the loss history, carried by a resumed checkpoint, reaches it), `TIME_LIMIT` (save and stop cleanly after this many seconds), `TEST_EVERY` (rank-0 test loss interval, default 25), `SOLVER_VERBOSE=0` (drop the per-solve "Converged after ..." lines); a resumed run continues the epoch count |
| Batch and learning rate | `BATCH_PER_RANK` (trajectories per GPU, default 1), `LR` (SOAP learning rate, default 1e-4; logged at start-up). A resumed run keeps the learning rate stored in its checkpoint's optimizer state, whatever `LR` says |
| Multi-node (`run_shear_flow_repro.sh`) | `sbatch -N <nodes>`, `RANKS_PER_NODE` (default 8), `RCCL_PLUGIN_DIR` (aws-ofi-rccl for RCCL over Slingshot; with more than one node the script also sets `NCCL_SOCKET_IFNAME=hsn0`) |

### Running
```bash
sbatch run_shear_flow_ddp.sh                          # 1 node × 8 GPUs, account fus184
MAX_STEPS=5 RUN_EVAL=0 sbatch run_shear_flow_ddp.sh   # short smoke test
```
Or, inside an allocation:
`srun -N1 -n8 -c7 --gpus-per-task=1 --gpu-bind=closest python -u 03_shear_flow_ddp.py`

Check modes (same script, same model and checkpoint setup; training is skipped):
```bash
MODE=grad_check sbatch run_shear_flow_ddp.sh     # 8-GPU averaged gradient vs 1-GPU batch-8 gradient
MODE=eval EVAL_OUT=<dir> DUP_MAP=<dup_map.json> sbatch run_shear_flow_ddp.sh   # held-out evaluation
```
`MODE=eval` rolls out all 200 frames of every valid and test trajectory (`EVAL_SPLITS`), computes
per-frame VRMSE, and writes `vrmse.npz`, `summary.txt` and `vrmse_vs_horizon.png`.

**Launcher fix.** `sbatch` copies the submitting shell's environment. When the conda env was
already active there, `module load python` put the base interpreter first on `PATH` and
`conda activate` did nothing, so `python` had no torch. `run_shear_flow_ddp.sh` now runs
`$CONDA_PREFIX/bin/python` explicitly and checks that torch imports before launching.

### Data: The Well HDF5 files, read directly

**Location:** `/lustre/orion/world-shared/fus184/jyc/the_well/datasets/shear_flow/`, downloaded with
`the-well-download --base-path <dir> --dataset shear_flow --split train` (and `--split valid`, `--split test`).

| | |
|---|---|
| Files | One HDF5 file per (Re, Sc) pair and split: 28 in `data/train` (32 trajectories each, 896 total), 28 in `data/valid` and 28 in `data/test` (4 each, 112 total per split); about 16.8 GB per train file, 58.7 GB per held-out split |
| Per trajectory | 200 frames on a 256 × 512 grid (x × y), periodic in both directions |
| Fields | `t0_fields/tracer`, `t0_fields/pressure` (trajectories, 200, 256, 512) and `t1_fields/velocity` (…, 2), float32; uncompressed and stored contiguously |
| Scalars | `scalars/Reynolds`, `scalars/Schmidt` (float32) |

**Verified mapping to the repo's frame format** (exact, zero difference):
- Channels: `[tracer, pressure, velocity_x, velocity_y]`, with no transpose. At frame 0 channel 2
  equals the tracer, because the initial vx equals the tracer field; later frames separate them.
- Local `traj_0` (train and test) is trajectory 0 of `shear_flow_Reynolds_1e4_Schmidt_1e-1.hdf5`,
  which comes first in sorted file order.
- Frames are numbered 0–199 (the 201 files in a local `traj_N` folder include `scalars.npy`).
- Local `stats.npy` has the same mean and std as The Well's `stats.yaml`, which is rounded to
  5 digits; the script keeps `stats.npy` (full precision, what the checkpoint was trained with).

**The Well API vs a custom reader** (measured):

| | The Well API (`WellDataset`) | Custom h5py reader |
|---|---|---|
| How it gets a sample (frames 0, 1 + 12 random) | Reads only contiguous windows, so `full_trajectory_mode` loads all 200 frames; 14 are kept | Reads just the 14 frames |
| Data read per sample | 400 MiB | 28 MiB |
| Time per sample | 0.47–0.58 s | 0.66 s |
| Code | About 15 lines of wrapper | About 50 lines |
| Extras | Hugging Face streaming, normalization, filters, maintained upstream | None |

Speed is similar because large contiguous reads are fast on Lustre. The API reads 14× more data
(about 3.2 GB per step across 8 ranks, and about 3 GB of prefetch memory per rank): fine on one
node, but it grows with node count. **Chosen: the API wrapper**; the custom reader is the
fallback if I/O becomes a bottleneck at scale.

**Implementation:** `WellShearFlowDataset` in `03_shear_flow_ddp.py` wraps
`WellDataset(path=<split dir>, n_steps_input=2, full_trajectory_mode=True, max_rollout_steps=10_000,
return_grid=False, boundary_return_type=None, use_normalization=False)`, then keeps the 2 start
frames and 12 random frames in the loader worker, so only about 28 MB per sample reaches
training. It returns the same batch as `SubsamplingDataset`, and both classes have
`full_trajectory(idx)` for the evaluation rollout.
- `max_rollout_steps` must be raised: the default (100) silently truncates trajectories.
- Scalars are kept as float32 (Sc = 0.10000000149…), like the local `scalars.npy`; converting to an
  exact 0.1 would slightly change the conditioning input the checkpoint was trained with.

**Train / held-out overlap (important for any accuracy number):** 88 of 112 valid and 89 of 112
test trajectories are byte-identical copies of training trajectories (matched by hashing tracer
frames 10 and 150; 6 spot-checked over all 200 frames and all fields). Only 47 of the 224 held-out
trajectories are unseen. For example, train trajectory 0 equals test trajectory 0 (the local
`traj_0` in both folders). The map is in `hnode_runs/eval_full/dup_map.json` (built by
`hnode_runs/eval_full/make_dup_map.py`). The cause is unknown; these are the raw downloaded files.

**Verification:** with the same seed, `WellShearFlowDataset` and `SubsamplingDataset` return
identical samples (all four fields, dtype and values) for train and test, and identical full
200-frame trajectories. The loader delivers about 0.22 s per sample with 4 workers, after a 25 s
startup.

### Test status

| Date | Where | Result |
|---|---|---|
| 2026-09-23 | Login node, 1 MI210, 1 rank NCCL, local `.npy` data | Init, broadcast, a full training step (40 windows, 3 Newton iterations each, residual about 1e-16), all-reduce and `optimizer.step()` ran; stopped in the test evaluation because only `test/traj_0` exists locally |
| 2026-09-24 | Login node, 1 MI210, 1 rank NCCL, The Well data, from scratch, 2 steps | **Complete:** 896/112 trajectories; step 0 took 88 s (compilation plus the first test evaluation), step 1 took 4.7 s; parameter sync check, checkpoint save (5.1 GB) and both evaluation figures all worked |
| 2026-09-24 | Compute node, 8 GPUs, job 5539007 | Failed after 7 s: `python` was the base interpreter (no torch). Fixed in `run_shear_flow_ddp.sh` (see "Running") |
| 2026-09-24 | 8 GPUs, job 5540785, 5 steps from the pretrained checkpoint | **Passed:** 8 ranks over RCCL; loss 2.7–4.3e-3 (checkpoint's recent median 2.5e-3); parameters in sync; checkpoint saved |
| 2026-09-24 | 8 GPUs, job 5540926, 30 steps with a per-step timing breakdown | **Passed;** timing below |
| 2026-09-24 | 8 GPUs, job 5543764, `MODE=grad_check` | **Passed:** per-sample losses equal to 3e-17; averaged gradient vs single-GPU batch-8 gradient relative difference **1.25e-14** |
| 2026-09-24 | 8 GPUs, job 5543765, `MODE=eval` (224 held-out trajectories) | **Completed in 6 min;** results in "Accuracy vs the paper" below |
| 2026-09-27 | 8 GPUs, job 5555394, from scratch (24-h extended job) | **Failed:** NaN at step 714; cancelled after 13.9 h (see "Platform note: SOAP's QR on ROCm") |
| 2026-09-28 | 8 GPUs, job 5557663, from scratch, with the non-finite-gradient guard | **Failed:** NaN at step 731 despite the guard, because the NaN came from SOAP's update, not the gradient |
| 2026-09-28 | 8 GPUs, job 5558706, from scratch, with the SOAP fix | **Passed:** 2,600 steps in 3 h 41 min, loss matching the reference (below); stopped on purpose |
| 2026-09-28 | 8 GPUs, job 5560440, 1/2/4/8 trajectories per GPU, 13 steps each | 1 and 2 **passed** (19.1 and 33.3 GiB); 4 failed on the first backward ("HIP error: no kernel image is available"); 8 out of memory |
| 2026-09-28 | 2 nodes, job 5563103, 552 MB float64 all-reduce | Sockets 435 ms; **aws-ofi-rccl 33 ms** |
| 2026-09-28 | 8 GPUs, job 5560751, batch 16 (2 per GPU), lr 1.4e-4, from scratch | **Passed:** 2,248 steps in 3 h 52 min (below) |
| 2026-09-29 | 4 nodes × 8 GCDs, job 5564781, batch 64 (2 per GPU), lr 2.83e-4, from scratch | **Passed:** 922 steps in 1 h 52 min; first multi-node run (below) |
| 2026-09-29 | 2 nodes × 8 GCDs, job 5565107, batch 32 (2 per GPU), lr 2e-4, from scratch | **Passed:** 920 steps in 1 h 58 min (below) |

RCCL refuses two ranks on one GPU ("Duplicate GPU detected"), so multi-rank NCCL can only be
tested on a compute node. With 8 ranks an epoch is 112 steps per rank, so 10 epochs is about
1,120 steps.

### Step timing (8 GPUs, batch 8, job 5540926)

| Part | Typical step |
|---|---|
| Data wait | 0.00 s (the loader keeps up) |
| Forward (encoder, 40 Newton-solved windows, decoder) | 3.85 s |
| Backward | 0.82 s |
| Gradient all-reduce | 0.02 s (slowest rank about 0.1 s) |
| Optimizer | 0.20 s |
| **Total** | **about 4.9 s** |

Occasional slow steps:
- **One-time recompilation** of the Swin encoder on step 2 (about 70 s): `torch.compile` guards on
  an attention-mask cache that fills on the first call.
- **Every 10th step, +6–7 s:** SOAP's preconditioner update (`precondition_frequency=10`), about
  12% on average.
- **Every 25th step, +44 s on rank 0:** the test evaluation, while the other 7 ranks wait (seen as a
  44 s all-reduce on the next step). Up to about 35% overhead if it stays that slow; worth fixing
  by spreading it over ranks or running it less often.

One epoch (896 trajectories, 112 steps) takes roughly 11–14 minutes. The paper trained for 8 days on
one H200 with batch 8 (Table 1), about 4.8 s per step over ~144,000 steps; one Frontier node runs
at about the same speed, so full training from scratch would take roughly 8–9 days on one node.
More nodes would raise the global batch above the paper's 8.

### Training from scratch vs the reference checkpoint

We trained `03_shear_flow_ddp.py` from scratch with the notebook's settings (batch 8 = 1 trajectory
per GPU × 8 GPUs, SOAP lr 1e-4, 40 windows × 5 cells) using `run_shear_flow_repro.sh`, and **stopped
at step 2,600** (job 5558706, 3 h 41 min, about 5.1 s per step). The full 144,000 steps would take
about 8.4 days of compute on one node (about nine 24-hour jobs), against the paper's 8 days on one
H200, so we compared the loss curve instead of training to the end.

**The training loss matches the reference checkpoint's history over those 2,600 steps**
(200-step means of the training loss, VRMSE²):

| Step | 500 | 1,000 | 1,500 | 2,000 | 2,500 |
|---|---|---|---|---|---|
| Ours | 0.458 | 0.131 | 0.087 | 0.069 | 0.057 |
| Reference (`shear_flow_redo_part_2.pt`) | 0.439 | 0.142 | 0.096 | 0.073 | 0.061 |

- The two curves overlap from the first steps; ours is 4–10% lower from step 1,000 on, within the
  noise of different random draws of trajectories and frames.
- The test loss is too noisy to compare this way: each value is one test trajectory with 12
  frames, and ours is computed every 250 steps (`TEST_EVERY`) against the reference's every 25.
- This shows our setup (ROCm, 8-GPU data parallelism, The Well HDF5 input, the SOAP fix) trains the
  model the same way as the original; it does not show the final accuracy, which needs the full run.

Outputs in `hnode_runs/repro/`: `checkpoint.pt` (step 2,600), `repro_vs_reference.png` (training
loss curves), `repro_vs_reference.npz`, and the first lines of the two failed runs' logs.

### Larger batches: faster per hour, less efficient per sample

The reference trains at batch 8 on one H200. On Frontier, batch 8 is 1 trajectory on each of a
node's 8 GCDs, and a step is latency-bound (40 Newton-solved windows, one after another), so we
tried more trajectories per GPU.

**Memory limits it to 2 per GPU** (job 5560440, from the step-2,600 checkpoint):

| Trajectories per GPU | Typical step | Samples/s (8 GPUs) | Peak memory (of 64 GiB) |
|---|---|---|---|
| 1 | 4.1 s | 1.6 | 19.1 GiB |
| 2 | 5.0 s | 2.8 | 33.3 GiB |
| 4 | failed on the first backward ("HIP error: no kernel image is available"), at about 61 GiB estimated | | |
| 8 | out of memory (about 117 GiB needed) | | |

- Each trajectory costs about 14 GiB (12 decoded 256×512 frames and the Newton Jacobians, float64),
  plus about 5 GiB fixed.
- **Why the paper fits batch 8 on one GPU:** an H200 has 141 GB, and 5 + 8 × 14 ≈ 117 GiB fits. A
  Frontier GCD has 64 GB. An MI250X card has 128 GB, but as two separate 64 GB devices.
- Batching is nearly free for the solver: doubling the trajectories per GPU adds about 0.9 s per step.

**Across nodes**, RCCL needs the aws-ofi-rccl plugin. A 552 MB float64 all-reduce (the gradient
size) over 2 nodes took 435 ms over sockets and 33 ms with the plugin (job 5563103). The build under
`csc143/.../aws-ofi-rccl/rocm6.3.1` has only dangling symlinks; we used
`/sw/frontier/mde/lrn068_mde/phil836/aws-ofi-rccl/lib`, which links against ROCm 6.3.1 and libfabric
2.3.1. In training, the cross-node all-reduce took a median 0.05 s of a 6.9 s step on 2 nodes and
0.04 s on 4 nodes, with occasional slow steps (90th percentile 0.4 s and 0.7 s).

**Convergence** (from scratch, learning rate scaled as 1e-4 × √(batch/8), training loss VRMSE²):

| Trajectories seen | Batch 64, lr 2.83e-4 (4 nodes) | Batch 32, lr 2e-4 (2 nodes) | Batch 16, lr 1.4e-4 (1 node) | Batch 8 (1 node) | Reference |
|---|---|---|---|---|---|
| 9,600 | 0.887 | 0.850 | 0.236 | 0.107 | 0.117 |
| 16,000 | 0.877 | 0.247 | 0.101 | 0.069 | 0.073 |
| 20,800 | 0.559 | 0.128 | 0.077 | 0.056 | 0.058 |
| 29,000 | 0.220 | 0.076 | 0.058 | — | 0.043 |
| 59,000 | 0.056 | — | — | — | 0.021 |

| Hours of training | Batch 64 (4 nodes) | Batch 32 (2 nodes) | Batch 16 (1 node) | Batch 8 (1 node) |
|---|---|---|---|---|
| 1.0 | **0.156** | 0.282 | 0.265 | 0.209 |
| 1.5 | **0.070** | 0.100 | 0.121 | 0.123 |
| 1.8 | **0.057** | 0.076 | 0.093 | 0.099 |

To reach loss 0.08 (smoothed training loss):

| | Batch 64 (4 nodes) | Batch 32 (2 nodes) | Batch 16 (1 node) | Batch 8 (1 node) |
|---|---|---|---|---|
| Hours | **1.38** | 1.75 | 2.08 | 2.37 |
| Speed-up over batch 8 | **1.72×** | 1.35× | 1.14× | 1× |
| Node-hours | 5.5 | 3.5 | **2.1** | 2.4 |

- **Per trajectory seen, bigger batches need more data:** at 29,000 trajectories batch 16 is at
  1.3× the reference's loss and batch 32 at 1.8×; batch 64 is at 2.6× by 59,000. The bigger the
  batch, the longer the early plateau near 0.9: to about 8,000 trajectories at batch 32 and 18,000
  at batch 64.
- **Per hour, bigger batches win, and the gap grows with time:** batch 64's speed-up over batch 8
  is 1.4× to reach 0.10, 1.7× to reach 0.08 and about 2× to reach 0.057 (1.8 h against about 3.7 h).
- **Per node-hour, batch 16 on one node is the cheapest;** batch 32 and 64 cost about 1.5× and 2.3×
  the node-hours of batch 8 for their speed.
- **Step times:** 5.1 s (batch 8), 6.0 s (16), 6.9 s (32 and 64). With all 8 GCDs of a node in use,
  steps are about 10% slower than with 2 per node (an earlier batch-32 test on 8 nodes × 2 GCDs ran
  6.3 s steps, at 4× the node-hours). At batch 64 an epoch is only 14 steps, and each epoch's first
  step waits about 4 s for the data loaders.
- **Only the first 2–4 hours are measured.** The gains could change later in training, and a larger
  batch is a different training run: whether it ends as accurate as the reference is unknown.

Rough cost of a full 144,000-step reproduction, scaling batch 8's 8.2 days by the speed-ups above
(the lower speed-up measured to 0.08, the higher at the end of each run where known):

| Setup | Wall-clock | Node-hours | Same training as the reference |
|---|---|---|---|
| Batch 8, 1 node | about 8.2 days | about 196 | Yes |
| Batch 16, 1 node | about 6.3–7.2 days | about 150–175 | No |
| Batch 32, 2 nodes | about 6 days | about 290 | No |
| Batch 64, 4 nodes | about 4.1–4.8 days | about 390–460 | No |

Outputs: `hnode_runs/batch_timing/` (memory and timing logs), `hnode_runs/multinode/` (all-reduce
benchmark), `hnode_runs/repro_b16/`, `hnode_runs/repro_b32_2n/` and `hnode_runs/repro_b64/`
(checkpoints, logs, `b16_vs_b8.png`; `repro_b64/batch_comparison.png` has all four batch sizes).

## Accuracy vs the paper (held-out evaluation)

`MODE=eval` with the pretrained checkpoint (`shear_flow_redo_part_2.pt`): full 200-frame rollouts
from frames 0–1 on all 224 valid + test trajectories. VRMSE as The Well defines it: per field,
√(mean squared error / (spatial variance + 1e-7)) in physical units, averaged over the 4 fields,
then over trajectories.

| Horizon | 1 | 10 | 25 | 50 | 75 | 100 | 150 | 199 | Mean |
|---|---|---|---|---|---|---|---|---|---|
| **All 224 held-out** | 0.032 | 0.031 | 0.035 | 0.047 | 0.048 | 0.045 | 0.043 | 0.044 | **0.042** |
| 47 unique (not in train) | 0.031 | 0.031 | 0.034 | 0.041 | 0.042 | 0.041 | 0.042 | 0.045 | 0.040 |
| 177 duplicates of train | 0.032 | 0.031 | 0.036 | 0.049 | 0.049 | 0.046 | 0.043 | 0.044 | 0.043 |
| Re = 1e4 (56) | 0.034 | 0.034 | 0.035 | 0.042 | 0.042 | 0.038 | 0.034 | 0.037 | 0.037 |
| Re = 5e4 (56) | 0.031 | 0.029 | 0.034 | 0.042 | 0.041 | 0.040 | 0.036 | 0.039 | 0.037 |
| Re = 1e5 (56) | 0.031 | 0.031 | 0.034 | 0.046 | 0.045 | 0.044 | 0.041 | 0.040 | 0.041 |
| Re = 5e5 (56) | 0.032 | 0.031 | 0.039 | 0.058 | 0.061 | 0.059 | 0.059 | 0.059 | 0.055 |
| **Paper Fig. 4c HNODE** (read off the plot) | ~0.03 | ~0.03 | ~0.045 | ~0.08 | ~0.10 | ~0.105 | ~0.125 | ~0.13 | — |

- **Short horizons match the paper** (~0.03). **At long horizons ours is 2–3× lower** than the
  paper's plotted curve, and levels off around 0.045 instead of rising to 0.13.
- **The overlap does not flatter the model:** the 47 unseen trajectories score slightly better
  (0.040) than the 177 duplicates (0.043).
- **Stable:** no blow-up over 200 frames for any group; higher Re is harder, as expected.
- **Why lower than the paper is unknown**, since the paper does not give its exact procedure:
  VRMSE computed or averaged differently (per field vs pooled, before or after the square root, a
  different ε, which matters late when the tracer fades), a different checkpoint, or a different
  horizon definition. With the training loss's definition (normalized units, ε = 1e-3) the mean is
  even lower, 0.031.
- The paper's baselines on the same plot: PhysiX ~0.25–0.5, C-U-Net up to about 40.

Outputs: `hnode_runs/eval_full/pretrained/` (`vrmse.npz` holds per-trajectory, per-frame, per-field
errors with split, Re, Sc and duplicate flags). The notebook's rollout figure for test trajectory 0
is reproduced exactly by the script's evaluation section (`hnode_runs/eval_traj0/`).

## Large latent dimensions (F = 512 and 1024)

The paper's models use F = 64. This section is about running HNODE with latent dimension 512 or 1024,
tested on shear-flow POD coefficients (`latent_baselines.py hnode` with `R=512` or `R=1024`).

### Where the cost goes
- **The network Jacobian is per cell, not per window.** `CrossAttnTransformer.forward` computes
  each cell's Jacobian with `vmap(jacrev)`: F backward passes giving an F × F block (u) and an
  F × 2F block (J tokens). The width stays 256, so the inputs enter through Linear(F → 256) and
  the output leaves through Linear(256 → F); each per-cell Jacobian therefore has rank ≤ 256.
- **The training memory was the final Newton step's graph.** The implicit-function gradient comes
  from one final Newton step with autograd on, `params − J⁻¹H`. Its gradient is
  `−J⁻¹ ∂H/∂θ − d(J⁻¹)·H`. The second term is multiplied by H, which is below 1e-12 at
  convergence, yet keeping it stored a backward pass through `jacrev` for every window. Detaching
  J drops that graph and leaves the gradient unchanged (relative difference 1.3e-16 at F = 64,
  5.4e-15 at F = 256).
- **At large F the linear solves dominate the time**: 18.6 s of a 23 s rollout at F = 1024 with
  the block solver. The network and its Jacobian take 4.5 s.

### Options added
The solver and network options are opt-in, so their defaults (and the gx project) are unchanged.
`latent_baselines.py` turns detach and the relative tolerance on by default; for converged windows
this changes results only at rounding level (about 1e-15).

| Where | Option | What it does |
|---|---|---|
| `hybrid_node_solver.py` | `solver.detach_final_jacobian = True` | Detaches J (dense tensor or `_BlockJacobian.detached()`) in the final Newton step. Same gradient, far less memory |
| `hybrid_node_solver.py` | `solver.relative_tol = True` | Converged when `max|H| < tol · max(1, max|params|)`. The residual's rounding floor grows with the state, so states of ~1e4 never reach an absolute 1e-12, and each such window ran all `max_iters` iterations |
| `neural_networks.py` | `net.jacobian_chunk_size = N` | Passes `chunk_size=N` to both `jacrev` calls. Not useful after the detach: no memory saved, up to 2× slower |
| `latent_baselines.py` | `HNODE_SOLVER` (`dense`/`block`) | Linear solver; use `block` at large F |
| `latent_baselines.py` | `HNODE_DETACH` (default `1`) | Sets `detach_final_jacobian` |
| `latent_baselines.py` | `HNODE_RELTOL` (default `1`) | Sets `relative_tol` |
| `latent_baselines.py` | `HNODE_MAX_ITERS` (solver default 100) | Newton iteration cap; 30 for the large-F runs |
| `latent_baselines.py` | `MICRO` (default 0 = whole local batch) | Trajectories per forward pass on a GPU; gradients are accumulated, so the global batch and the gradient are unchanged |

The step log of `latent_baselines.py` now also prints peak GPU memory.

### Profile: one MI250X GCD, batch 1, 40 windows × 5 cells
`run_profile_latent_dim.sh` (job 5629560) runs `profile_latent_dim.py` on POD coefficients of real
shear-flow frames with a randomly initialized N_θ (width 256, 3 blocks, 4 heads) and dt = 0.01.
Newton needed 3–4 iterations per window at every F. Training step, time / peak memory:

| F | Unknowns per window | Dense (default) | Block | Block + detach |
|---|---|---|---|---|
| 64 | 832 | 3.4 s / 6.7 GiB | 3.9 s / 6.4 GiB | **3.3 s / 0.4 GiB** |
| 128 | 1,664 | 4.9 s / 14.2 GiB | 4.7 s / 12.6 GiB | **4.5 s / 0.8 GiB** |
| 256 | 3,328 | 8.7 s / 31.3 GiB | 6.7 s / 25.7 GiB | **5.9 s / 2.2 GiB** |
| 512 | 6,656 | out of memory | 11.9 s / 55 GiB | **10.6 s / 6.6 GiB** |
| 1024 | 13,312 | out of memory | out of memory | **23.8 s / 23.3 GiB** |

Rollout without gradients at F = 1024: 96 s dense, 23 s block. Adding `chunk64` to block + detach
made steps up to 2× slower with the same memory. After the detach, memory grows roughly as F², so
F = 2048 (about 90 GiB per trajectory) needs more: a low-rank (≤ 256) Jacobian or splitting a
trajectory across GPUs. Results: `hnode_runs/latent_dim_profile/sweep/` (`summary.txt`,
`latent_dim_profile.png`).

### Two things that went wrong in the first tests
1. **Newton stalls on blown-up rollouts.** A 10-step test (`STEPS=10`) reaches the 199-frame
   horizon at step 6 with an almost untrained model. Its rollouts grow from |state| ≈ 16 to
   ≈ 1e5 over 40 windows. Newton then fails in two ways: genuinely, with |H| from 0.1 to 25; and at
   the rounding floor, at about 1e-11 against the absolute 1e-12. Every stalled window ran 100
   iterations; 715 of them in 28 minutes. On the first 8 windows from real frames, Newton converges
   in 3–4 iterations to 1e-14 at R = 512, dense and block alike
   (`hnode_runs/latent_dim_profile/newton_diag/`). Fixes: `HNODE_RELTOL=1`, `HNODE_MAX_ITERS=30`.
   In a real run, the 199-frame horizon starts at step 1,501, after training on horizons 10 and 50.
2. **R = 1024 with 2 trajectories per GCD runs out of memory** at the first 199-frame step:
   45.7 GiB allocated plus 17.3 GiB reserved but fragmented. `expandable_segments` is "not
   supported on this platform" (PyTorch ROCm). `MICRO=1` (one trajectory per forward pass,
   accumulated) brings the peak to 25.3 GiB.

Test runs, global batch 16 on 8 GCDs, 2 trajectories per GCD:

| R | Job | Settings | Peak memory (h = 10 / 50 / 199) | s/step at h = 199 (untrained, blown up) |
|---|---|---|---|---|
| 512 | 5630651 | block, detach, reltol, 30 iterations | 4.2 / 6.3 / 14.0 GiB | 6–15 |
| 1024 | 5630675 | same | 9.3 / 17.4 GiB / out of memory | — |
| 1024 | 5630684 | same + `MICRO=1` | 6.0 / 10.0 / 25.3 GiB | 14–31 |

### How to run
From the repo directory, one row = HNODE (8 GCDs, chained 2-hour jobs) + the other time steppers
+ the evaluation with the POD decoder:

```bash
# F = 512: 2 trajectories per GCD fit (14 GiB)
HNODE_SOLVER=block HNODE_MAX_ITERS=30 HNODE_CHAIN=6 \
    ./submit_latent_row.sh ../hnode_runs/pod/full 512 ../hnode_runs/rows/pod512 pod:512

# F = 1024: one trajectory per forward pass (25 GiB); same global batch 16
HNODE_SOLVER=block HNODE_MAX_ITERS=30 MICRO=1 HNODE_CHAIN=8 \
    ./submit_latent_row.sh ../hnode_runs/pod/full 1024 ../hnode_runs/rows/pod1024 pod:1024
```

`HNODE_DETACH=1` and `HNODE_RELTOL=1` are already the defaults. A short test first:
`sbatch -t 01:00:00 --export=ALL,JOB=hnode,LATENT_DIR=…/pod/full,R=1024,OUT=…,HNODE_SOLVER=block,HNODE_MAX_ITERS=30,MICRO=1,STEPS=10,CKPT_EVERY=5 run_latent_job.sh`.
Delete the test's `OUT/hnode/` before rerunning, or it resumes from its checkpoint.

The stored POD basis has 1,024 modes, so R ≤ 1024 without recomputing it. Profiling:
`sbatch run_profile_latent_dim.sh` (all cases), or
`python profile_latent_dim.py F block+detach OUT.json` for one case.

In other code (e.g. a large-F model elsewhere):

```python
solver = HybridNODESolver(net, ..., linear_solver="block")
solver.detach_final_jacobian = True   # same gradient, far less memory
solver.relative_tol = True            # tolerance scaled by the state size
solver.max_iters = 30
```

### What to expect
- **The POD floor limits accuracy, not HNODE**: projection VRMSE 0.178 at R = 512 and 0.122 at
  R = 1024, vs 0.278 at R = 128. The Swin-autoencoder row is already at 0.12 (LSTM 0.119,
  HNODE 9k 0.121) with 64 dimensions. These rows test whether HNODE trains and scales at large F.
- **Cost of the finished runs** (3,000 steps, 8 GCDs): the R = 1024 `floor:64` run took 11.1 hours
  of training (40,029 s). The results are in the next section.
- **Possible use for the full Swin-HNODE:** `detach_final_jacobian` should also cut the end-to-end
  training memory (about 14 GiB per trajectory now, which limits the batch per GCD). Untested.

## Does a larger latent dimension help HNODE?

**Short answer: no, not as trained here.** Going from 64 to 1,024 POD or Fourier POD coefficients
does not make HNODE more accurate, and there is no consistent trend with r. Above r ≈ 256 the
small scales get worse. A better 64-dimensional latent space (Swin) helps much more than more
POD modes.

### Setup
- **Latent spaces:** POD and Fourier POD of shear flow (`pod_basis.py`, 1,024 modes each), at
  r = 64, 128, 256, 512 and 1024. One "row" per basis and r (`hnode_runs/rows/{pod,fpod}R/`).
- **Time steppers on the frozen latents** (`latent_baselines.py`): HNODE with SOAP at lr 1e-4 and
  HNODE_DT = 1, plus LSTM, NODE, Koopman and DMD at their best learning rate (`best.json`). All get
  3,000 steps at global batch 16, horizons 10 → 50 → 199, and a latent-space MSE loss.
- **HNODE coefficient scalings (`SCALE`):** `mode` (every coordinate to unit std), `shared` (one
  factor, the largest std) and `floor:64` (unit std for the 64 largest coordinates, the 64th std
  for the rest). `floor:64` is the main series; it equals `mode` at r = 64.
- **For comparison, 64-dimensional Swin latents:** the two-stage Swin autoencoder
  (`ae_train.py`, `hnode_runs/ae/swin64/row/`) with the same time steppers, and the pretrained
  end-to-end Swin-HNODE checkpoint (144,001 steps at batch 8 with a pixel loss).
- **Evaluation:** `eval_latent_rollout.py`, VRMSE as in "Accuracy vs the paper", over frames
  1–199 and the 224 held-out trajectories. Tables: `summarize_latent_dim.py`.

### VRMSE versus r

Mean VRMSE (lower is better; "projection" is the basis's own coefficients, the best a time
stepper can do in that space):

| | r = 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|
| POD projection | 0.371 | 0.278 | 0.233 | 0.178 | 0.122 |
| POD HNODE `floor:64` | 4.22 | 4.17 | 4.39 | 7.16 | 3.51 |
| POD HNODE `shared` | 7.83 | 7.34 | 6.61 | 3.45 | 6.32 |
| POD LSTM | 1.15 | 1.29 | 1.13 | 1.20 | 1.50 |
| Fourier POD projection | 0.330 | 0.284 | 0.212 | 0.172 | 0.118 |
| Fourier POD HNODE `floor:64` | 4.08 | 4.51 | 4.73 | 7.80 | 3.42 |
| Fourier POD HNODE `shared` | 8.44 | 8.19 | 6.94 | 6.31 | 9.29 |
| Fourier POD LSTM | 1.19 | 1.01 | 1.40 | 1.20 | 1.63 |

Median over trajectories (less affected by the pressure and v_y frames whose variance is near
zero):

| | r = 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|
| POD projection | 0.156 | 0.121 | 0.095 | 0.070 | 0.049 |
| POD HNODE `floor:64` | 0.482 | 0.499 | 0.521 | 0.637 | 0.677 |
| POD LSTM | 0.237 | 0.238 | 0.240 | 0.235 | 0.248 |
| Fourier POD HNODE `floor:64` | 0.491 | 0.508 | 0.537 | 0.623 | 0.633 |

64-dimensional Swin latents:

| Model | Mean | Median |
|---|---|---|
| Swin autoencoder projection | 0.090 | 0.072 |
| Swin autoencoder + HNODE, 3,000 steps | 0.219 | 0.213 |
| Swin autoencoder + HNODE, 9,000 steps | 0.121 | 0.105 |
| Swin autoencoder + LSTM | 0.119 | 0.106 |
| Pretrained Swin-HNODE (end to end) | 0.042 | 0.035 |

- **The projection improves steadily with r** (0.37 → 0.12), so the extra modes carry
  information. **HNODE does not use it:** the `floor:64` mean stays at about 3.5–4.7 apart from r = 512, and
  the median gets worse with r (0.48 → 0.68). Its gap to the projection grows from about 11× to
  about 29× (mean), and from 3× to 14× (median).
- **No consistent trend.** The best scaling changes with r (`shared` is best at POD r = 512 and
  worst elsewhere), and the r = 512 spike of `floor:64` appears in both bases. Each point is one
  seed, so differences of this size may be training noise.
- **The small gain of `floor:64` at r = 1024 comes from pressure and v_y.** Per field, POD r = 64
  → 1024: tracer 0.40 → 0.48, pressure 8.86 → 6.59, v_x 0.155 → 0.154, v_y 7.46 → 6.79. Fourier
  POD is the same.
- **The LSTM beats HNODE 2–6× at every r** and stays close to flat. In these coordinates the
  bottleneck is learning the dynamics with HNODE, not the latent space.
- **The same 3,000 steps on 64 Swin dimensions** give 0.219, about 14× better than the best POD-family
  HNODE (3.00, POD r = 128 `mode (Sept)`). At 9,000 steps HNODE matches the LSTM, near its own projection (0.121 vs 0.090).

### Kinetic-energy spectra
`spectral_energy.py` decodes each rollout and computes the shell-averaged kinetic-energy spectrum
E(k, t) and the error spectrum |û_pred − û_true|²/2, in physical units, with k in units of 2π
(domain 1 × 2). Error ÷ true energy per band, frames 100–199, energy-weighted over the 224
trajectories (1 = no skill):

| Model | k < 3.5 | 3.5–15.5 | 15.5–63.5 | ≥ 63.5 |
|---|---|---|---|---|
| POD HNODE `floor:64`, r = 64 | 0.029 | 0.53 | 1.4 | 1.2 |
| POD HNODE `floor:64`, r = 256 | 0.028 | 0.73 | 4.1 | 4.1 |
| POD HNODE `floor:64`, r = 512 | 0.041 | 1.8 | 36 | 34 |
| POD HNODE `floor:64`, r = 1024 | 0.028 | 5.0 | 189 | 45 |
| Fourier POD HNODE `floor:64`, r = 1024 | 0.027 | 5.3 | 115 | 3.9 |
| POD LSTM, r = 1024 | 0.0023 | 0.20 | 4.0 | 3.0 |
| POD projection, r = 1024 | 1.7e-5 | 0.012 | 1.2 | 1.7 |
| Swin autoencoder projection | 2.7e-4 | 0.079 | 16 | 3,197 |
| Swin autoencoder + HNODE, 3,000 steps | 0.0065 | 0.18 | 18 | 3,294 |
| Swin autoencoder + HNODE, 9,000 steps | 0.0010 | 0.093 | 15 | 3,143 |
| Pretrained Swin-HNODE | 7.4e-5 | 0.014 | 4.8 | 324 |

- **Large scales do not improve with r.** HNODE's k < 3.5 error stays at about 0.03 from r = 64
  to 1024, about 1,600× above the r = 1024 projection.
- **Larger r adds spurious small-scale energy.** Above r ≈ 256 the model's energy at
  k > 3.5 grows during the rollout, to 5.5× the truth at 3.5–15.5 and 188× at 15.5–63.5 for
  POD r = 1024. The LSTM in the same space stays within about 4×.
- **The Swin models are 3–400× better than POD r = 64 HNODE at k < 15.5,** but their decoders add a broadband
  small-scale floor above k ≈ 10–15. It comes from 4-pixel patch blockiness (spikes at k = 64,
  90.5, 128, 181), is already in the autoencoder projection and in the pretrained model from
  frame 1, and stays flat after about frame 50. It is about 20% of the velocity error over frames
  100–199. At 15.5–63.5 every Swin model is worse than Fourier POD HNODE at r = 64 (1.3).
- **The kinetic energy is mostly v_x** (the mean shear), so these bands barely see the v_y and
  pressure changes that move the mean VRMSE.
- The ratios are dominated by the Re = 5e5 trajectories. Per-trajectory medians are 5–10× larger
  but rank the models the same way. The 47 unique trajectories give the same conclusions.

### Caveats
- **One seed per point** and a fixed 3,000-step budget. Large r may need longer training or a
  different learning rate (the r = 1024 run took 11 hours for 3,000 steps). This is the one
  setting that could still change the answer.
- **Unequal budgets on the Swin side:** the pretrained model trained end to end for 144,001 steps
  with a pixel loss; the POD rows and the Swin autoencoder rows used 3,000 (or 9,000) steps with a
  latent loss.
- **The solver settings are not a confound.** The large-r rows used `block`, `relative_tol` and 30
  Newton iterations; the September 64-dim runs used `dense`, an absolute tolerance and 100. On the
  trained Swin-autoencoder HNODE models (3k and 9k steps; `hnode_runs/solver_check/`, job 5658085)
  every combination gives the same rollouts to 5e-15 (relative), the same VRMSE to 6 digits, the same
  Newton iterations (4.1–4.5 per window, at most 6; `relative_tol` saves about 1%), and the same
  training gradients at h = 10 and 50 to 3e-14. Block rollouts are 3× faster (11 vs 33–39 s per rank).
- **The pretrained encoder is not a projection floor.** It is trained only on frames 0–1, so its
  latents of later frames decode badly (VRMSE 0.9 at frame 10, 8 at frame 199).

### Outputs
- `hnode_runs/rows/latent_dim_vrmse.png`, and per row `eval/compare.txt` and `eval/<model>/vrmse.npz`.
- `hnode_runs/spectra/`:
  - `spectra_{pod,fpod,swin}_{spectrogram,bands,spectra}.png` (E(k, t), band energy and error
    versus time, E(k) at 4 frames).
  - `spectra_vs_r.png` (band error versus r, with the Swin models at the left).
  - `spectra_unique_*` (the same for the 47 unique trajectories).
  - `spectra_summary.txt`.
- `hnode_runs/latent_export/heldout/`: pretrained Swin-HNODE latents and rollouts for valid + test
  (`run_encode.sh`); they reproduce the `MODE=eval` VRMSE (0.0424) to 3e-13 per trajectory.

Rerun the spectra: `sbatch -q debug -o spectra-%j.out hnode_runs/spectra/run_spectra.sh`
(about 7 minutes on one node).
