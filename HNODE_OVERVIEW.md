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

The block system built in `_build_A_and_M_p0` (four blocks of rows):

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
   neighbours. `_build_indexers` scatters those blocks into the full matrix.
3. Run damped Newton, halving the step if the residual grows, until the residual is below `1e-12`.

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
| The weak form: finite-element matrices for (J, v), (u, v′) and (J′, q) | `_build_A_and_M_p0`, which builds `M_rt`, `G`, `M_p0` with scikit-fem |
| λ multipliers and continuity between windows | The last two unknowns per field; `u_next = lambda_2`, `J_init = J_hat[:, -1]` in `evolve_batch` |
| Local transformer N | `CrossAttnTransformer.forward` in `neural_networks.py`: 3 tokens (J_left, u, J_right), Jacobian from `vmap(jacrev)` |
| Newton solve of H = 0 | `_timestep` (damped Newton) and `_compute_J_and_residual` |
| Implicit-differentiation gradient | The final Newton step taken with autograd on, after a `no_grad` loop |
| All windows solved together via Schur complement | `backprop_type="schur"` |

## Files

| File | Role |
|---|---|
| `hybrid_node_solver.py` | Core: the finite-element matrices, Newton solve, rollout, and Schur-complement backprop |
| `neural_networks.py` | `CrossAttnTransformer`, the per-cell N_θ and its Jacobian |
| `ode_datasets.py`, `pca_datasets.py` | Data sampled on the staggered node/midpoint grid |
| `01_lorenz.py` | Simplest case: the 3-D state is used directly (start here) |
| `02_flow_past_cylinder.py` | State is 64 PCA modes, conditioned on `z = log(Re)` |
| `03_shear_flow.py`, `04_plasma_gun.ipynb` | A Swin-V2 encoder/decoder maps fields to a latent space; HNODE runs there, trained end to end |

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
   10, 20, 25, 40, …), the dense Jacobian (unknowns² × 8 bytes × batch) must fit in memory, and
   Newton must still converge in a few iterations.
3. Re-measure with the trained checkpoint: a stiffer trained network may need more Newton
   iterations in long windows.
4. A checkpoint trained with one window size can be rolled out with another: rebuild the solver
   with the new `N`.

For shear flow, `N = 20` (10 cells per window) looks best here: about 1.7× faster per training
step, with the same results.

## Per-window system: H(p) = A·p + M·N_θ(p) − b = 0

Per field, `p = [J_0…J_5, u_0…u_4, λ₁, λ₂]` (13 numbers). The 13 equations are:

| Rows | Equation | Meaning |
|---|---|---|
| 6 (J rows) | `(J, v) + (u, v') − λ-terms = 0` | J = du/dt |
| 5 (u rows) | `J_{c+1} − J_c + h·N_θ(u_c, J_c, J_{c+1}) = 0` | J changes according to the network |
| 1 | `J_0 − J_init = 0` | Initial velocity |
| 1 | `λ₁ − u_init = 0` | Initial state |

### Shapes (one window, batch size B, shear flow)

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
        Δp = −solve(Jac, H)                   # 832×832 dense LU (_dense_solve)
        old_p = p
        p = p + eta·Δp
Jac, H = _compute_J_and_residual(p)           # one more step WITH autograd:
p = p + eta · (−solve(Jac, H))                # attaches the implicit gradient −Jac⁻¹ ∂H/∂θ
```

- **Cost** is dominated by the network Jacobian: 64 backward passes through the transformer per cell.
  The 832×832 solve is cheap on a GPU.
- **Measured iteration counts:** Lorenz (trained) 5–6, cylinder flow (trained) 6–7,
  shear flow (untrained, random input) 3. The residual reaches about 1e-16 each time.
- **Quirks** (from reading the code, harmless in practice):
  1. A rejected step reuses Δp computed at the rejected point, applied at `old_p` with half the step size.
     This is not a textbook backtracking line search.
  2. The final step is scaled by eta, so the gradient is too if eta < 1 at convergence. Usually eta = 1.

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
   probably workable with smaller windows; F = 1024 (about 13,000 unknowns per window at M = 5)
   would need the sparse solver (cuDSS) and a cheaper Jacobian strategy. This is estimated from
   the scaling, not tested.
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
| Safety checks | Every step, ranks confirm they all-reduce the same set of gradients; every 50 steps, parameter checksums are compared and rank 0's weights re-broadcast if they drifted (e.g. SOAP eigendecomposition round-off) |
| Rank 0 only | Logging, test loss, checkpoint saves and the evaluation figures (saved as `shear_flow_ddp_loss.png` and `shear_flow_ddp_rollout.png`) |
| MIOpen cache | Per rank in `/tmp`, to avoid lock contention on a shared database |
| Data source | `DATA_FORMAT=well` (default): The Well HDF5 files under `WELL_DATA_PATH`; `DATA_FORMAT=npy`: the local per-frame folders |
| Test knobs | `CHECKPOINT_PATH` (`none` = from scratch), `SAVE_PATH`, `NUM_EPOCHS`, `MAX_STEPS`, `RUN_EVAL=0` |

### Running
```bash
sbatch run_shear_flow_ddp.sh                          # 1 node × 8 GPUs, account fus184
MAX_STEPS=5 RUN_EVAL=0 sbatch run_shear_flow_ddp.sh   # short smoke test
```
Or, inside an allocation:
`srun -N1 -n8 -c7 --gpus-per-task=1 --gpu-bind=closest python -u 03_shear_flow_ddp.py`

### Data: The Well HDF5 files, read directly

**Location:** `/lustre/orion/world-shared/fus184/jyc/the_well/datasets/shear_flow/`, downloaded with
`the-well-download --base-path <dir> --dataset shear_flow --split train` (and `--split test`).

| | |
|---|---|
| Files | One HDF5 file per (Re, Sc) pair and split: 28 in `data/train` (32 trajectories each, 896 total) and 28 in `data/test` (4 each, 112 total); about 16.8 GB per train file. No `valid` split was downloaded (not used) |
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

**Verification:** with the same seed, `WellShearFlowDataset` and `SubsamplingDataset` return
identical samples (all four fields, dtype and values) for train and test, and identical full
200-frame trajectories. The loader delivers about 0.22 s per sample with 4 workers, after a 25 s
startup.

### Test status

| Date | Where | Result |
|---|---|---|
| 2026-09-23 | Login node, 1 MI210, 1 rank NCCL, local `.npy` data | Init, broadcast, a full training step (40 windows, 3 Newton iterations each, residual about 1e-16), all-reduce and `optimizer.step()` ran; stopped in the test evaluation because only `test/traj_0` exists locally |
| 2026-09-24 | Login node, 1 MI210, 1 rank NCCL, The Well data, from scratch, 2 steps | **Complete:** 896/112 trajectories; step 0 took 88 s (compilation plus the first test evaluation), step 1 took 4.7 s; parameter sync check, checkpoint save (5.1 GB) and both evaluation figures all worked |
| 2026-09-24 | Compute node, 8 GPUs (job 5539007, 5 steps, from the pretrained checkpoint) | Submitted; pending |

RCCL refuses two ranks on one GPU ("Duplicate GPU detected"), so multi-rank NCCL can only be
tested on a compute node. With 8 ranks an epoch is 112 steps per rank, so 10 epochs is about
1,120 steps.
