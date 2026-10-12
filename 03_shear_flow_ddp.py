#!/usr/bin/env python
# coding: utf-8

# # HNODE: Shear Flow (data-parallel training)
#
# Data-parallel version of 03_shear_flow.py: one process per GPU, launched with srun.
#
#     srun -N1 -n8 -c7 --gpus-per-task=1 --gpu-bind=closest python -u 03_shear_flow_ddp.py
#
# (or submit run_shear_flow_ddp.sh). Run without srun, it trains as a single process.
#
# Gradients are averaged across ranks by hand after loss.backward() instead of wrapping the models
# in DistributedDataParallel. nn_rhs_model runs inside HybridNODESolver: it is called ~40 times with
# autograd per backward (one final Newton step per window), through vmap(jacrev(...)), and the solver
# reads attributes (dropout_rate, forward_original) that a DDP wrapper would hide. One flattened
# all-reduce over encoder, decoder and nn_rhs_model is equivalent to DDP and avoids all of that.
#
# Data: by default the full dataset is read directly from The Well's HDF5 files (WELL_DATA_PATH,
# 896 train / 112 test trajectories) through the_well's WellDataset. DATA_FORMAT=npy uses the
# per-frame .npy folders (shear_flow_frames_train / _test) instead.
#
# Environment overrides (for testing): DATA_FORMAT ("well" or "npy"), WELL_DATA_PATH,
# CHECKPOINT_PATH ("none" = train from scratch), SAVE_PATH, NUM_EPOCHS, MAX_STEPS (stop after this many
# optimizer steps; 0 = no limit), TOTAL_STEPS (stop once the loss history, which a resumed checkpoint
# carries, has this many steps; 0 = no limit), TIME_LIMIT (seconds; save and stop cleanly once a step
# ends after it, for chained jobs; 0 = none), TEST_EVERY (rank-0 test loss every N steps, default 25),
# BATCH_PER_RANK (trajectories per GPU, default 1: 8 GPUs give the notebook's batch 8), LR (SOAP
# learning rate, default 1e-4),
# RUN_EVAL (0 skips the final evaluation), DDP_BACKEND (default "nccl", which is RCCL on ROCm; "gloo"
# lets ranks share one GPU).

import os
import time


def _env_int(names, default):
    for name in names:
        if name in os.environ:
            return int(os.environ[name])
    return default


# Rank info from torchrun-style variables if present, else from the srun step.
if "SLURM_STEP_NUM_TASKS" in os.environ and "RANK" not in os.environ:
    rank = _env_int(["SLURM_PROCID"], 0)
    local_rank = _env_int(["SLURM_LOCALID"], 0)
    world_size = _env_int(["SLURM_STEP_NUM_TASKS"], 1)
else:
    rank = _env_int(["RANK"], 0)
    local_rank = _env_int(["LOCAL_RANK"], 0)
    world_size = _env_int(["WORLD_SIZE"], 1)
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")  # fine for a single node
os.environ.setdefault("MASTER_PORT", "29500")
# Per-rank MIOpen cache: ranks sharing the default database on a network filesystem contend for its lock.
_user = os.environ.get("USER", "user")
os.environ.setdefault("MIOPEN_USER_DB_PATH", f"/tmp/miopen-{_user}-{rank}")
os.environ.setdefault("MIOPEN_CUSTOM_CACHE_DIR", f"/tmp/miopen-{_user}-{rank}")

import torch
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import random
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from einops import rearrange

from hybrid_node_solver import HybridNODESolver
from neural_networks import CrossAttnTransformer
from swin_transformer_v2 import SwinEncoder, SwinDecoder
from soap import SOAP

torch.set_default_dtype(torch.float64)

if torch.cuda.is_available():
    # With --gpus-per-task=1 each rank sees a single device; otherwise pick by local rank.
    gpu_index = local_rank if torch.cuda.device_count() > 1 else 0
    torch.cuda.set_device(gpu_index)
    device = torch.device("cuda", gpu_index)
else:
    device = torch.device("cpu")

dist.init_process_group(
    backend=os.environ.get("DDP_BACKEND", "nccl" if device.type == "cuda" else "gloo"),
    rank=rank,
    world_size=world_size,
)


def log(*args):
    if rank == 0:
        print(*args, flush=True)


# Different seeds per rank so each rank draws different frames; model weights are broadcast from rank 0.
SEED = 0
torch.manual_seed(SEED + rank)
np.random.seed(SEED + rank)
random.seed(SEED + rank)

log(f"World size: {world_size}, backend: {dist.get_backend()}, device: {device}")
print(f"[rank {rank}] local_rank={local_rank} device={device}", flush=True)


# ## Configuration

# "well": The Well's HDF5 files (WELL_DATA_PATH/train, WELL_DATA_PATH/test).
# "npy": per-frame folders, parent_dir/traj_N/frame_M.npy + parent_dir/stats.npy
DATA_FORMAT = os.environ.get("DATA_FORMAT", "well")
WELL_DATA_PATH = os.environ.get(
    "WELL_DATA_PATH", "/lustre/orion/world-shared/fus184/jyc/the_well/datasets/shear_flow/data"
)
TRAIN_DATA_PATH = "./shear_flow_frames_train"
TEST_DATA_PATH = "./shear_flow_frames_test"
# Normalization stats. The Well's stats.yaml has the same values rounded to 5 digits; this file has
# the full-precision values the checkpoint was trained with.
STATS_PATH = os.path.join(TRAIN_DATA_PATH, "stats.npy")

# Path to a saved checkpoint to resume training, or None to train from scratch.
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "./shear_flow_redo_part_2.pt")
if CHECKPOINT_PATH.lower() == "none":
    CHECKPOINT_PATH = None
SAVE_PATH = os.environ.get("SAVE_PATH", "shear_flow_checkpoint.pt")
SAVE_EVERY = 200
SYNC_CHECK_EVERY = 50  # compare parameter checksums across ranks every N steps


# ## Dataset


class SubsamplingDataset(Dataset):
    """
    Expected directory structure::

        parent_directory/
            traj_0/
                scalars.npy          # scalar parameters for this trajectory (Re, Sc)
                frame_0.npy
                frame_1.npy
                ...
            traj_1/
                ...
            stats.npy                # dataset mean and std per channel
    """

    def __init__(
        self,
        parent_directory,
        n_trajectories,
        n_frames_per_trajectory=200,
        bdf_degree=2,
        N=4,
        seed=None,
    ):
        self.parent_directory = parent_directory
        self.n_trajectories = n_trajectories
        self.n_frames_per_trajectory = n_frames_per_trajectory
        self.bdf_degree = bdf_degree
        self.seed = seed
        self.N = N

        if not os.path.exists(parent_directory):
            raise FileNotFoundError(f"Data directory not found: {parent_directory}")

    def __len__(self):
        return self.n_trajectories

    def __getitem__(self, idx):
        if self.seed is not None:
            random.seed(self.seed + idx)

        traj_dir = os.path.join(self.parent_directory, f"traj_{idx}")
        scalars = torch.from_numpy(np.load(os.path.join(traj_dir, "scalars.npy")))

        start_index = 1
        start_indices = list(range(start_index - self.bdf_degree + 1, start_index + 1))
        indices_to_decode = list(range(start_index, self.n_frames_per_trajectory))
        random.shuffle(indices_to_decode)
        indices_to_decode = indices_to_decode[: self.N]

        start_frames = torch.stack(
            [
                torch.from_numpy(np.load(os.path.join(traj_dir, f"frame_{i}.npy")))
                for i in start_indices
            ]
        )
        target_frames = torch.stack(
            [
                torch.from_numpy(np.load(os.path.join(traj_dir, f"frame_{i}.npy")))
                for i in indices_to_decode
            ]
        )
        indices_to_decode = torch.tensor(indices_to_decode) - start_index

        return {
            "start_frames": start_frames.double(),
            "target_frames": target_frames.double(),
            "scalars": scalars.double(),
            "indices_to_decode": indices_to_decode,
        }

    def full_trajectory(self, idx):
        """All frames (T, H, W, C) and scalars [Re, Sc] of one trajectory, as float64."""
        traj_dir = os.path.join(self.parent_directory, f"traj_{idx}")
        frames = torch.stack(
            [
                torch.from_numpy(np.load(os.path.join(traj_dir, f"frame_{t}.npy")))
                for t in range(self.n_frames_per_trajectory)
            ]
        )
        scalars = torch.from_numpy(np.load(os.path.join(traj_dir, "scalars.npy")))
        return frames.double(), scalars.double()


class WellShearFlowDataset(Dataset):
    """
    The same samples as SubsamplingDataset, read from The Well's HDF5 files via WellDataset.

    WellDataset only reads contiguous time windows, so each sample loads the whole trajectory
    (full_trajectory_mode, ~400 MiB) and keeps the 2 start frames and N target frames in the
    worker; only those (~28 MiB) reach the training process. Trajectories are numbered in sorted
    file order, so index 0 is Re = 1e4, Sc = 0.1, trajectory 0 (the same as the local traj_0).
    Channels come out as [tracer, pressure, velocity_x, velocity_y].
    """

    def __init__(self, split_directory, bdf_degree=2, N=4, seed=None):
        from the_well.data import WellDataset

        if not os.path.exists(split_directory):
            raise FileNotFoundError(f"Data directory not found: {split_directory}")
        self.well = WellDataset(
            path=split_directory,
            n_steps_input=bdf_degree,
            full_trajectory_mode=True,
            max_rollout_steps=10_000,  # the default (100) would truncate trajectories
            return_grid=False,
            boundary_return_type=None,
            use_normalization=False,
        )
        self.bdf_degree = bdf_degree
        self.seed = seed
        self.N = N

    def __len__(self):
        return len(self.well)

    def full_trajectory(self, idx):
        """All frames (T, H, W, C) and scalars [Re, Sc] of one trajectory, as float64."""
        sample = self.well[idx]
        frames = torch.cat([sample["input_fields"], sample["output_fields"]])
        # float32, like the per-frame scalars.npy files (so Sc = 0.1 is 0.10000000149).
        scalars = sample["constant_scalars"]
        return frames.double(), scalars.double()

    def __getitem__(self, idx):
        if self.seed is not None:
            random.seed(self.seed + idx)

        frames, scalars = self.full_trajectory(idx)
        n_frames_per_trajectory = frames.shape[0]

        start_index = 1
        start_indices = list(range(start_index - self.bdf_degree + 1, start_index + 1))
        indices_to_decode = list(range(start_index, n_frames_per_trajectory))
        random.shuffle(indices_to_decode)
        indices_to_decode = indices_to_decode[: self.N]

        # .clone() so the full trajectory can be freed in the worker.
        start_frames = frames[start_indices].clone()
        target_frames = frames[indices_to_decode].clone()
        indices_to_decode = torch.tensor(indices_to_decode) - start_index

        return {
            "start_frames": start_frames,
            "target_frames": target_frames,
            "scalars": scalars,
            "indices_to_decode": indices_to_decode,
        }


latent_dim = 64
dt = 1e-2
num_time_steps_per_data_sample = 200
N = 40  # number of training rollouts
# batch_size = 8
batch_size = int(os.environ.get("BATCH_PER_RANK", 1))  # per rank; the global batch is batch_size * world_size

delta_t = dt * num_time_steps_per_data_sample / N
N_t = (
    num_time_steps_per_data_sample // N
) + 1  # collocation points per rollout (M = N_t - 1)

if DATA_FORMAT == "well":
    train_dataset = WellShearFlowDataset(
        os.path.join(WELL_DATA_PATH, "train"), bdf_degree=2, N=12
    )
    test_dataset = WellShearFlowDataset(
        os.path.join(WELL_DATA_PATH, "test"), bdf_degree=2, N=12
    )
elif DATA_FORMAT == "npy":
    train_dataset = SubsamplingDataset(
        # TRAIN_DATA_PATH, n_trajectories=896, n_frames_per_trajectory=200, bdf_degree=2, N=12
        TRAIN_DATA_PATH, n_trajectories=1, n_frames_per_trajectory=200, bdf_degree=2, N=12
    )
    test_dataset = SubsamplingDataset(
        TEST_DATA_PATH, n_trajectories=112, n_frames_per_trajectory=200, bdf_degree=2, N=12
    )
else:
    raise ValueError(f"DATA_FORMAT must be 'well' or 'npy', got {DATA_FORMAT!r}")

# Each rank gets an equal share of trajectories (padded by repetition), so every rank takes the
# same number of steps per epoch and the gradient all-reduces line up.
train_sampler = DistributedSampler(
    train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=SEED
)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    sampler=train_sampler,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    drop_last=True,
)
# The test loss is computed on rank 0 only.
test_dataloader = None
if rank == 0:
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )

_stats = np.load(STATS_PATH, allow_pickle=True).item()
frame_mean = torch.tensor(_stats["mean"], dtype=torch.float64).to(device)
frame_std = torch.tensor(_stats["std"], dtype=torch.float64).to(device)

# Conditioning: log10(Re) and log10(Sc), normalised by training-set statistics
reynolds = torch.tensor([1e4, 5e4, 1e5, 5e5]).log10()
schmidt = torch.tensor([0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]).log10()
z_mean = torch.tensor([reynolds.mean(), schmidt.mean()]).unsqueeze(0).to(device)
z_std = torch.tensor([reynolds.std(), schmidt.std()]).unsqueeze(0).to(device)

log(
    f"Data: {DATA_FORMAT}. Train trajectories: {len(train_dataset)} ({len(train_sampler)} per rank), "
    f"Test trajectories: {len(test_dataset)}"
)
log(f"delta_t={delta_t:.4f}, N_t={N_t} (M={N_t-1} intervals per rollout)")
log(f"Global batch: {batch_size * world_size} ({batch_size} per rank)")


# ## Model

encoder = SwinEncoder(
    img_size=(256, 512),
    patch_size=4,
    in_chans=4,
    latent_dim=latent_dim,
    embed_dim=96,
    depths=(2, 2, 6, 2),
    num_heads=(3, 6, 12, 24),
    window_size=(8, 8),
    mlp_ratio=4,
    drop=0.0,
    attn_drop=0.0,
    drop_path_rate=0.0,
    final_channels=64,
).to(device)

decoder = SwinDecoder(
    img_size=(256, 512),
    patch_size=4,
    out_chans=4,
    latent_dim=latent_dim,
    embed_dim=96,
    depths=(2, 2, 6, 2),
    num_heads=(3, 6, 12, 24),
    window_size=(8, 8),
    mlp_ratio=4,
    drop=0.0,
    attn_drop=0.0,
    drop_path_rate=0.0,
).to(device)

encoder = torch.compile(encoder)
decoder = torch.compile(decoder)

nn_rhs_model = CrossAttnTransformer(
    problem_dim=latent_dim,
    latent_dim=64 * 4,
    num_blocks=3,
    num_heads=4,
    z_dim=2,
    dropout_rate=0,
    activation=torch.nn.Tanh(),
    use_self_attention_z_block=True,
).to(device)

solver = HybridNODESolver(
    nn_rhs_model,
    delta_t,
    N_t,
    latent_dim,
    N,
    backprop_type="matrix_free",
    use_manual_jacobian=True,
    use_sparse_solver=False,
    use_cpu_solver=False,
    verbose=False,
    device=device,
)

models = [encoder, decoder, nn_rhs_model]
all_params = [p for m in models for p in m.parameters() if p.requires_grad]
log(f"Total parameters: {sum(p.numel() for p in all_params):,}")


# ## Data-parallel helpers


def broadcast_parameters():
    """Copy rank 0's parameters and buffers to every rank."""
    with torch.no_grad():
        for m in models:
            for t in list(m.parameters()) + list(m.buffers()):
                dist.broadcast(t, src=0)


def allreduce_gradients():
    """Average gradients over ranks with one flattened all-reduce (equivalent to DDP).
    Returns whether the averaged gradient is finite (the same answer on every rank)."""
    params = [p for p in all_params if p.grad is not None]
    # Every rank must all-reduce the same tensors; a mismatch would hang or corrupt the collective.
    counts = torch.tensor([len(params), -len(params)], device=device)
    dist.all_reduce(counts, op=dist.ReduceOp.MAX)
    if counts[0] != len(params) or -counts[1] != len(params):
        raise RuntimeError(f"[rank {rank}] ranks disagree on which parameters have gradients")
    grads = [p.grad for p in params]
    flat = _flatten_dense_tensors(grads)
    dist.all_reduce(flat)
    flat.div_(world_size)
    for g, synced in zip(grads, _unflatten_dense_tensors(flat, grads)):
        g.copy_(synced)
    return bool(torch.isfinite(flat).all())


def check_parameter_sync(step):
    """Re-broadcast from rank 0 if ranks' parameters have drifted apart (e.g. SOAP eigh round-off)."""
    with torch.no_grad():
        checksum = torch.stack(
            [
                sum(p.double().sum() for p in all_params),
                sum(p.double().abs().sum() for p in all_params),
            ]
        )
        cmax, cmin = checksum.clone(), checksum.clone()
        dist.all_reduce(cmax, op=dist.ReduceOp.MAX)
        dist.all_reduce(cmin, op=dist.ReduceOp.MIN)
        drift = ((cmax - cmin).abs() / cmax.abs().clamp_min(1e-30)).max().item()
    if drift > 1e-12:
        log(f"Step {step}: parameters drifted across ranks (rel. {drift:.2e}); re-broadcasting from rank 0")
        broadcast_parameters()
    else:
        log(f"Step {step}: parameters in sync across ranks (rel. diff {drift:.1e})")


# ## Optimizer and Checkpoint


def get_param_groups(models, weight_decay=1e-4):
    """Separate transformer params into decay / no-decay groups (standard practice)."""
    if not isinstance(models, (list, tuple)):
        models = [models]
    decay, no_decay = [], []
    for model in models:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if (
                param.ndim <= 1
                or "bias" in name
                or "norm" in name.lower()
                or "ln" in name
            ):
                no_decay.append(param)
            else:
                decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


optimizer = SOAP(
    get_param_groups([nn_rhs_model, encoder, decoder]),
    precondition_frequency=10,
    lr=float(os.environ.get("LR", 1e-4)),  # the notebook's 1e-4 for batch 8
)
log(f"SOAP learning rate: {optimizer.param_groups[0]['lr']:g}")

train_loss_history, test_loss_history = [], []

# Every rank loads the same checkpoint (so optimizer state matches too); the broadcast below
# then guarantees identical weights, including when training from scratch.
if CHECKPOINT_PATH is not None:
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        encoder.load_state_dict(checkpoint["encoder"])
        decoder.load_state_dict(checkpoint["decoder"])
        nn_rhs_model.load_state_dict(checkpoint["nn_rhs_model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        train_loss_history = checkpoint["train_loss_history"]
        test_loss_history = checkpoint["test_loss_history"]
        del checkpoint
        log(f"Loaded checkpoint from {CHECKPOINT_PATH} ({len(train_loss_history)} steps)")
    else:
        log(f"Checkpoint not found at {CHECKPOINT_PATH}, training from scratch.")

broadcast_parameters()


# ## Training


def normalize(frames):
    return (frames - frame_mean) / frame_std


def vrmse_loss(pred, target, eps=1e-3):
    """Variance-normalised RMSE^2 over batch, time, and channel. pred/target: (B, T, H, W, C)."""
    spatial_dims = (-3, -2)
    mse = (pred - target).pow(2).mean(dim=spatial_dims)
    var = target.var(dim=spatial_dims)
    return (mse / (var + eps)).mean()


def compute_loss(batch):
    B = batch["start_frames"].shape[0]
    start_frames = normalize(batch["start_frames"].to(device))
    start_latents = rearrange(
        encoder(rearrange(start_frames, "B T H W C -> (B T) C H W")),
        "(B T) L -> B T L",
        B=B,
    )
    latent_batch = {
        "J_init": (start_latents[:, 1, :] - start_latents[:, 0, :]) / dt,
        "u_init": (start_latents[:, 0, :] + start_latents[:, 1, :]) / 2.0,
        "z": (batch["scalars"].to(device).log10() - z_mean) / z_std,
    }
    _, u_hat_rollout = solver.evolve_batch(latent_batch)
    latents_to_decode = rearrange(
        torch.vmap(lambda array, indices: array[indices])(
            u_hat_rollout, batch["indices_to_decode"].to(device)
        ),
        "B T L -> (B T) L",
    )
    reconstructed = rearrange(
        decoder(latents_to_decode), "(B T) C H W -> B T H W C", B=B
    )
    return vrmse_loss(reconstructed, normalize(batch["target_frames"].to(device)))


def save_checkpoint():
    """Save to SAVE_PATH, keeping the previous checkpoint as SAVE_PATH.prev; never save non-finite weights."""
    with torch.no_grad():
        finite = all(bool(torch.isfinite(p).all()) for p in all_params)
    if not finite:
        log(f"Non-finite parameters at step {len(train_loss_history)}: checkpoint NOT saved")
        return
    if os.path.exists(SAVE_PATH):
        os.replace(SAVE_PATH, SAVE_PATH + ".prev")
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "nn_rhs_model": nn_rhs_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "train_loss_history": train_loss_history,
            "test_loss_history": test_loss_history,
        },
        SAVE_PATH,
    )


def synchronize():
    if device.type == "cuda":
        torch.cuda.synchronize(device)


# ## Check modes (MODE=grad_check or MODE=eval); the default MODE=train continues below.

MODE = os.environ.get("MODE", "train")


def flat_grads():
    return torch.cat([p.grad.reshape(-1) for p in all_params if p.grad is not None])


if MODE == "grad_check":
    # The averaged gradient over world_size ranks (one sample each) must equal the gradient of the
    # mean loss over the same samples computed on one GPU (by accumulation on rank 0).
    if DATA_FORMAT != "well":
        raise ValueError("MODE=grad_check expects DATA_FORMAT=well")
    check_ds = WellShearFlowDataset(os.path.join(WELL_DATA_PATH, "train"), bdf_degree=2, N=12, seed=1234)
    collate = torch.utils.data.default_collate
    solver.verbose = False

    optimizer.zero_grad()
    my_loss = compute_loss(collate([check_ds[rank]]))
    my_loss.backward()
    allreduce_gradients()
    g_ddp = flat_grads().clone()
    rank_losses = [None] * world_size
    dist.all_gather_object(rank_losses, my_loss.item())

    if rank == 0:
        optimizer.zero_grad()
        ref_losses = []
        for i in range(world_size):
            loss_i = compute_loss(collate([check_ds[i]]))
            (loss_i / world_size).backward()
            ref_losses.append(loss_i.item())
        g_ref = flat_grads()
        rel = ((g_ddp - g_ref).norm() / g_ref.norm()).item()
        loss_diff = max(abs(a - b) for a, b in zip(rank_losses, ref_losses))
        log(f"grad_check: {world_size} ranks vs 1 GPU on samples 0..{world_size - 1} (train, seed 1234)")
        log(f"  per-sample losses (ranks): {[f'{x:.6e}' for x in rank_losses]}")
        log(f"  max |loss_rank - loss_single| = {loss_diff:.2e}")
        log(f"  gradient norm {g_ref.norm().item():.6e}; relative difference ||g_ddp - g_1gpu|| / ||g_1gpu|| = {rel:.2e}")
        log(f"  RESULT: {'PASS' if rel < 1e-8 else 'FAIL'} (threshold 1e-8)")
    dist.barrier()
    dist.destroy_process_group()
    raise SystemExit(0)

if MODE == "eval":
    # Paper-style test evaluation: full 200-frame rollouts from frames 0-1 on the held-out splits
    # (valid + test = 224 trajectories), per-frame VRMSE, averaged over trajectories.
    # VRMSE per field = sqrt(mean over space of (pred - target)^2 / (var over space of target + eps)),
    # then averaged over the 4 fields. Two variants are stored: The Well's definition (physical
    # units, eps = 1e-7) and the training loss's (normalized units, eps = 1e-3).
    import json

    if DATA_FORMAT != "well":
        raise ValueError("MODE=eval expects DATA_FORMAT=well")
    EVAL_OUT = os.environ.get("EVAL_OUT", "eval_out")
    EVAL_SPLITS = os.environ.get("EVAL_SPLITS", "valid,test").split(",")
    DUP_MAP = os.environ.get("DUP_MAP", "")
    dup_map = json.load(open(DUP_MAP)) if DUP_MAP and os.path.exists(DUP_MAP) else {}
    solver.verbose = False
    encoder.eval(); decoder.eval(); nn_rhs_model.eval()

    jobs = []
    datasets = {}
    for split in EVAL_SPLITS:
        datasets[split] = WellShearFlowDataset(os.path.join(WELL_DATA_PATH, split), bdf_degree=2, N=12)
        jobs += [(split, i) for i in range(len(datasets[split]))]
    my_jobs = jobs[rank::world_size]
    log(f"eval: {len(jobs)} trajectories from {EVAL_SPLITS}, {len(my_jobs)} per rank")

    results = []
    t0 = time.time()
    with torch.no_grad():
        for n_done, (split, idx) in enumerate(my_jobs):
            frames, scalars = datasets[split].full_trajectory(idx)
            frames = frames.to(device).unsqueeze(0)  # (1, T, H, W, C)
            frames_norm = normalize(frames)
            start_latents = rearrange(
                encoder(rearrange(frames_norm[:, :2], "B T H W C -> (B T) C H W")),
                "(B T) L -> B T L",
                B=1,
            )
            latent_batch = {
                "J_init": (start_latents[:, 1, :] - start_latents[:, 0, :]) / dt,
                "u_init": (start_latents[:, 0, :] + start_latents[:, 1, :]) / 2.0,
                "z": (scalars.unsqueeze(0).to(device).log10() - z_mean) / z_std,
            }
            _, u_hat_rollout = solver.evolve_batch(latent_batch)
            T = frames.shape[1]
            # Decode all cells in one call (one compiled shape); cell k is frame k + 1.
            recon_norm = rearrange(decoder(u_hat_rollout[0]), "T C H W -> T H W C")[: T - 1]
            target_norm = frames_norm[0, 1:]
            mse_n = (recon_norm - target_norm).pow(2).mean(dim=(1, 2))  # (T-1, C)
            var_n = target_norm.var(dim=(1, 2))
            vrmse_train = (mse_n / (var_n + 1e-3)).sqrt()
            vrmse_well = (mse_n * frame_std**2 / (var_n * frame_std**2 + 1e-7)).sqrt()
            meta = dup_map.get(f"{split}:{idx}", {})
            results.append(
                {
                    "split": split,
                    "idx": idx,
                    "Re": float(scalars[0]),
                    "Sc": float(scalars[1]),
                    "duplicate_of_train": meta.get("duplicate_of_train"),
                    "vrmse_well": vrmse_well.cpu().numpy(),
                    "vrmse_train": vrmse_train.cpu().numpy(),
                }
            )
            if rank == 0:
                log(f"  rank 0: {n_done + 1}/{len(my_jobs)} done ({time.time() - t0:.0f}s)")

    gathered = [None] * world_size
    dist.all_gather_object(gathered, results)
    if rank == 0:
        allres = sorted([r for part in gathered for r in part], key=lambda r: (r["split"], r["idx"]))
        os.makedirs(EVAL_OUT, exist_ok=True)
        vw = np.stack([r["vrmse_well"] for r in allres])  # (N, T-1, C)
        vt = np.stack([r["vrmse_train"] for r in allres])
        split_arr = np.array([r["split"] for r in allres])
        dup = np.array([r["duplicate_of_train"] is not None for r in allres]) if dup_map else None
        np.savez(
            os.path.join(EVAL_OUT, "vrmse.npz"),
            vrmse_well=vw, vrmse_train=vt, split=split_arr,
            idx=np.array([r["idx"] for r in allres]),
            Re=np.array([r["Re"] for r in allres]), Sc=np.array([r["Sc"] for r in allres]),
            duplicate=dup if dup is not None else np.array([]),
        )
        horizons = [1, 10, 25, 50, 75, 100, 150, 199]
        groups = {"all held-out": np.ones(len(allres), bool)}
        if dup is not None:
            groups["unique (not in train)"] = ~dup
            groups["duplicates of train"] = dup
        for re_val in sorted(set(r["Re"] for r in allres)):
            groups[f"Re = {re_val:g}"] = np.array([r["Re"] == re_val for r in allres])
        lines = [f"{'group':<24}{'n':>5}  " + "".join(f"{'t=' + str(t):>8}" for t in horizons) + f"{'mean':>8}"]
        curves = {}
        for name, mask in groups.items():
            if mask.sum() == 0:
                continue
            curve = vw[mask].mean(axis=(0, 2))  # average over trajectories and fields
            curves[name] = curve
            lines.append(
                f"{name:<24}{int(mask.sum()):>5}  "
                + "".join(f"{curve[t - 1]:>8.4f}" for t in horizons)
                + f"{curve.mean():>8.4f}"
            )
        curve_train_def = vt.mean(axis=(0, 2))
        lines.append(
            f"{'all (training-loss def.)':<24}{len(allres):>5}  "
            + "".join(f"{curve_train_def[t - 1]:>8.4f}" for t in horizons)
            + f"{curve_train_def.mean():>8.4f}"
        )
        summary = "\n".join(lines)
        open(os.path.join(EVAL_OUT, "summary.txt"), "w").write(summary + "\n")
        log("VRMSE (The Well definition) vs rollout horizon, averaged over trajectories and fields:\n" + summary)

        # Paper Fig. 4c HNODE curve, read off the published plot (approximate).
        paper_t = [1, 10, 25, 50, 75, 100, 150, 199]
        paper_v = [0.03, 0.03, 0.045, 0.08, 0.10, 0.105, 0.125, 0.13]
        fig, ax = plt.subplots(figsize=(7, 4))
        tt = np.arange(1, vw.shape[1] + 1)
        for name, color in [("all held-out", "tab:red"), ("unique (not in train)", "tab:purple"),
                            ("duplicates of train", "tab:orange")]:
            if name in curves:
                ax.semilogy(tt, curves[name], color=color, label=f"ours: {name} (n={int(groups[name].sum())})")
        ax.semilogy(paper_t, paper_v, "k--o", ms=4, label="paper Fig. 4c HNODE (approx.)")
        ax.set_xlabel("Rollout prediction horizon"); ax.set_ylabel("VRMSE")
        ax.set_title("Shear flow, held-out trajectories"); ax.legend(fontsize=8)
        plt.tight_layout(); plt.savefig(os.path.join(EVAL_OUT, "vrmse_vs_horizon.png"), dpi=130); plt.close(fig)
        log(f"Saved {EVAL_OUT}/vrmse.npz, summary.txt, vrmse_vs_horizon.png")
    dist.barrier()
    dist.destroy_process_group()
    raise SystemExit(0)


# SOLVER_VERBOSE=0 silences the per-solve "Converged after ..." lines (40 per step on rank 0).
solver.verbose = rank == 0 and os.environ.get("SOLVER_VERBOSE", "1") != "0"
skipped_steps = 0
test_iter = iter(test_dataloader) if rank == 0 else None
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", 10))
MAX_STEPS = int(os.environ.get("MAX_STEPS", 0))
TOTAL_STEPS = int(os.environ.get("TOTAL_STEPS", 0))
TIME_LIMIT = float(os.environ.get("TIME_LIMIT", 0))
TEST_EVERY = int(os.environ.get("TEST_EVERY", 25))
steps_this_run = 0
stop_training = False
# A resumed run continues the epoch count, so the sampler does not replay earlier epochs' order.
start_epoch = len(train_loss_history) // len(train_dataloader)
log(f"Starting at step {len(train_loss_history)} (epoch {start_epoch}); TOTAL_STEPS={TOTAL_STEPS}, "
    f"TIME_LIMIT={TIME_LIMIT:.0f}s, TEST_EVERY={TEST_EVERY}")
t_run_start = time.time()

t_prev = time.time()
for epoch in range(start_epoch, start_epoch + NUM_EPOCHS):
    train_sampler.set_epoch(epoch)
    for batch_ndx, batch in enumerate(train_dataloader):
        # Per-step timing: data wait, forward, backward, all-reduce, optimizer, other (test/save/sync).
        t_start = time.time()
        t_data = t_start - t_prev
        optimizer.zero_grad()
        loss = compute_loss(batch)
        synchronize()
        t_fwd = time.time()
        loss.backward()
        synchronize()
        t_bwd = time.time()
        grads_ok = allreduce_gradients()
        synchronize()
        t_comm = time.time()
        if grads_ok:
            optimizer.step()
        else:
            # One rank's loss or gradient went non-finite (e.g. a Newton solve diverged): skip the update
            # on every rank instead of letting the all-reduce spread NaN into all weights.
            skipped_steps += 1
            bad = torch.tensor([float(not torch.isfinite(loss.detach()).all())], device=device)
            bad_ranks = [torch.zeros_like(bad) for _ in range(world_size)]
            dist.all_gather(bad_ranks, bad)
            log(f"Step {len(train_loss_history)}: non-finite gradient (non-finite loss on ranks "
                f"{[r for r, b in enumerate(bad_ranks) if b.item() > 0]}); update skipped ({skipped_steps} so far)")
            optimizer.zero_grad()
        synchronize()
        t_opt = time.time()
        timings = torch.tensor(
            [t_data, t_fwd - t_start, t_bwd - t_fwd, t_comm - t_bwd, t_opt - t_comm],
            device=device,
        )
        timings_max = timings.clone()
        dist.all_reduce(timings_max, op=dist.ReduceOp.MAX)

        # Mean loss over ranks, for logging.
        global_loss = loss.detach().clone()
        dist.all_reduce(global_loss)
        global_loss = (global_loss / world_size).item()

        iteration = len(train_loss_history)
        train_loss_history.append(global_loss)
        steps_this_run += 1

        if rank == 0:
            if iteration % TEST_EVERY == 0:
                try:
                    test_batch = next(test_iter)
                except StopIteration:
                    test_iter = iter(test_dataloader)
                    test_batch = next(test_iter)
                with torch.no_grad():
                    test_loss = compute_loss(test_batch)
                test_loss_history.append(test_loss.item())
            else:
                test_loss_history.append(
                    test_loss_history[-1] if test_loss_history else float("nan")
                )

            if iteration % SAVE_EVERY == 0:
                save_checkpoint()

        if steps_this_run % SYNC_CHECK_EVERY == 0:
            check_parameter_sync(iteration)

        synchronize()
        t_prev = time.time()
        if rank == 0:
            d, f, b, c, o = timings.tolist()
            dm, fm, bm, cm, om = timings_max.tolist()
            log(
                f"Epoch {epoch} | Batch {batch_ndx}/{len(train_dataloader)} "
                f"| Train {global_loss:.6e} | Test {test_loss_history[-1]:.6e} "
                f"| step {t_prev - t_start + d:.1f}s: data {d:.2f} fwd {f:.2f} bwd {b:.2f} "
                f"allreduce {c:.2f} opt {o:.2f} other {t_prev - t_opt:.2f} "
                f"| slowest rank: data {dm:.2f} fwd {fm:.2f} bwd {bm:.2f} allreduce {cm:.2f} opt {om:.2f}"
            )

        # Stop conditions, decided identically on every rank (the time limit via an all-reduce).
        over_time = torch.tensor([float(TIME_LIMIT > 0 and time.time() - t_run_start > TIME_LIMIT)], device=device)
        dist.all_reduce(over_time, op=dist.ReduceOp.MAX)
        if over_time.item() > 0:
            log(f"Time limit reached after {steps_this_run} steps this run; saving and stopping")
        stop_training = (
            (MAX_STEPS and steps_this_run >= MAX_STEPS)
            or (TOTAL_STEPS and len(train_loss_history) >= TOTAL_STEPS)
            or over_time.item() > 0
        )
        if stop_training:
            break
    if stop_training:
        break

if device.type == "cuda":
    peak = torch.tensor([torch.cuda.max_memory_allocated(device) / 2**30], device=device)
    dist.all_reduce(peak, op=dist.ReduceOp.MAX)
    log(f"Peak GPU memory (max over ranks): {peak.item():.1f} GiB of "
        f"{torch.cuda.get_device_properties(device).total_memory / 2**30:.0f}")
check_parameter_sync(len(train_loss_history) - 1)
if rank == 0 and steps_this_run > 0:
    save_checkpoint()
    log(f"Saved checkpoint to {SAVE_PATH}")
dist.barrier()


# ## Evaluation (rank 0 only; figures are saved to files)

if rank == 0 and os.environ.get("RUN_EVAL", "1") != "0":
    # Loss curves
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.semilogy(train_loss_history, alpha=0.7, label="train")
    ax.semilogy(test_loss_history, alpha=0.7, label="test")
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("VRMSE")
    ax.set_title("Shear flow training loss")
    ax.legend()
    plt.tight_layout()
    plt.savefig("shear_flow_ddp_loss.png", dpi=150)
    plt.close(fig)

    # Simple rollout on traj_0 of the test data: tracer channel target vs model
    TRAJ_INDEX = 0
    time_indices = [0, 25, 50, 100, 199]
    tracer_channel = 0

    encoder.eval()
    decoder.eval()
    nn_rhs_model.eval()

    with torch.no_grad():
        frames, scalars = test_dataset.full_trajectory(TRAJ_INDEX)
        frames = frames[:num_time_steps_per_data_sample]

        frames = frames.unsqueeze(0).to(device)  # (1, T, H, W, C)
        frames_norm = normalize(frames)

        start_frames = frames_norm[:, :2]
        start_latents = rearrange(
            encoder(rearrange(start_frames, "B T H W C -> (B T) C H W")),
            "(B T) L -> B T L",
            B=1,
        )
        latent_batch = {
            "J_init": (start_latents[:, 1, :] - start_latents[:, 0, :]) / dt,
            "u_init": (start_latents[:, 0, :] + start_latents[:, 1, :]) / 2.0,
            "z": (scalars.unsqueeze(0).to(device).log10() - z_mean) / z_std,
        }
        _, u_hat_rollout = solver.evolve_batch(latent_batch)

        latents_to_decode = rearrange(u_hat_rollout, "B T L -> (B T) L")
        recon_norm = rearrange(decoder(latents_to_decode), "(B T) C H W -> B T H W C", B=1)
        recon_norm = torch.cat([frames_norm[:, :1], recon_norm[:, :-1]], dim=1)

        target = frames[0].cpu()
        recon = (recon_norm * frame_std + frame_mean)[0].cpu()

        # Per-frame VRMSE (the square root of the training loss's per-frame term, channel-averaged)
        # for comparison with the paper's VRMSE-vs-horizon curve. Frame 0 is the given input.
        mse = (recon_norm - frames_norm).pow(2).mean(dim=(-3, -2))  # (1, T, C)
        var = frames_norm.var(dim=(-3, -2))
        vrmse = (mse / (var + 1e-3)).sqrt().mean(dim=-1)[0].cpu().numpy()  # (T,)
    np.save("shear_flow_ddp_vrmse_traj0.npy", vrmse)
    log(
        "Test traj_0 VRMSE at horizon "
        + ", ".join(f"{t}: {vrmse[t]:.4f}" for t in [1, 10, 25, 50, 100, 150, 199])
    )
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.semilogy(np.arange(1, len(vrmse)), vrmse[1:], color="tab:red", label="HNODE (test traj_0)")
    ax.set_xlabel("Rollout prediction horizon")
    ax.set_ylabel("VRMSE")
    ax.set_title(f"Test traj_{TRAJ_INDEX}: VRMSE vs horizon")
    ax.legend()
    plt.tight_layout()
    plt.savefig("shear_flow_ddp_vrmse.png", dpi=150)
    plt.close(fig)

    tracer_all = torch.cat(
        [target[:, :, :, tracer_channel], recon[:, :, :, tracer_channel]], dim=0
    )
    vmin, vmax = tracer_all.min().item(), tracer_all.max().item()

    fig, axes = plt.subplots(
        2,
        len(time_indices),
        figsize=(3.0 * len(time_indices), 5.0),
        gridspec_kw={"hspace": 0.06, "wspace": 0.04},
    )
    row_labels = ["Target", "Model"]
    label_kw = dict(
        fontsize=9,
        color="white",
        bbox=dict(
            facecolor="black", alpha=0.55, edgecolor="none", boxstyle="round,pad=0.2"
        ),
    )

    for col, t_idx in enumerate(time_indices):
        for row, data in enumerate([target, recon]):
            ax = axes[row, col]
            ax.imshow(
                data[t_idx, :, :, tracer_channel].numpy(),
                cmap="RdBu_r",
                vmin=vmin,
                vmax=vmax,
                aspect="auto",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.text(
                    0.03,
                    0.97,
                    f"t={t_idx}",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    **label_kw,
                )
            if col == 0:
                ax.set_ylabel(row_labels[row], fontsize=11)

    fig.suptitle(f"Test traj_{TRAJ_INDEX}: tracer channel {tracer_channel}", y=0.98)
    plt.tight_layout()
    plt.savefig("shear_flow_ddp_rollout.png", dpi=150)
    plt.close(fig)
    log("Saved shear_flow_ddp_loss.png, shear_flow_ddp_rollout.png and shear_flow_ddp_vrmse.png")

dist.destroy_process_group()
