"""Shared pieces of the shear-flow setup (03_shear_flow_ddp.py) for scripts that reuse the pretrained
HNODE model: the Swin encoder/decoder, the HNODE time stepper, normalization, conditioning and
data access. Configuration matches 03_shear_flow_ddp.py and the checkpoint shear_flow_redo_part_2.pt.

Latent-space baselines (NODE, LSTM, Koopman) are compared with HNODE through:
    encode_latents.py      encode every trajectory once with the frozen encoder (plus the HNODE rollout)
    eval_latent_rollout.py decode any model's latent rollout and compute VRMSE like MODE=eval
"""
import os

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
WELL_DATA_PATH = os.environ.get(
    "WELL_DATA_PATH", "/lustre/orion/world-shared/fus184/jyc/the_well/datasets/shear_flow/data"
)
CHECKPOINT_PATH = os.path.join(HERE, "shear_flow_redo_part_2.pt")
STATS_PATH = os.path.join(HERE, "shear_flow_frames_train", "stats.npy")

LATENT_DIM = 64
DT = 1e-2
NUM_FRAMES = 200
N_WINDOWS = 40  # HNODE windows per rollout (5 cells each)


def distributed_env():
    """(rank, local_rank, world_size) from srun, torchrun, or a single process."""
    env = os.environ
    if "SLURM_STEP_NUM_TASKS" in env and "RANK" not in env:
        return int(env.get("SLURM_PROCID", 0)), int(env.get("SLURM_LOCALID", 0)), int(env["SLURM_STEP_NUM_TASKS"])
    return int(env.get("RANK", 0)), int(env.get("LOCAL_RANK", 0)), int(env.get("WORLD_SIZE", 1))


def pick_device(local_rank):
    if not torch.cuda.is_available():
        return torch.device("cpu")
    idx = local_rank if torch.cuda.device_count() > 1 else 0
    torch.cuda.set_device(idx)
    return torch.device("cuda", idx)


def frame_stats(device):
    """Per-channel mean and std the checkpoint was trained with (full precision)."""
    s = np.load(STATS_PATH, allow_pickle=True).item()
    return (torch.tensor(s["mean"], dtype=torch.float64, device=device),
            torch.tensor(s["std"], dtype=torch.float64, device=device))


def normalize_z(scalars):
    """Conditioning [Re, Sc] -> normalized log10, as in training. scalars: (..., 2)."""
    reynolds = torch.tensor([1e4, 5e4, 1e5, 5e5], dtype=torch.float64).log10()
    schmidt = torch.tensor([0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0], dtype=torch.float64).log10()
    mean = torch.stack([reynolds.mean(), schmidt.mean()]).to(scalars.device)
    std = torch.stack([reynolds.std(), schmidt.std()]).to(scalars.device)
    return (scalars.double().log10() - mean) / std


def _load(module, state):
    # The checkpoint was saved from torch.compile'd modules ("_orig_mod." prefix).
    module.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in state.items()})
    return module


def load_models(device, compile=True):
    """Pretrained (encoder, decoder, nn_rhs_model) in eval mode."""
    from neural_networks import CrossAttnTransformer
    from swin_transformer_v2 import SwinDecoder, SwinEncoder

    swin = dict(img_size=(256, 512), patch_size=4, latent_dim=LATENT_DIM, embed_dim=96, depths=(2, 2, 6, 2),
                num_heads=(3, 6, 12, 24), window_size=(8, 8), mlp_ratio=4, drop=0.0, attn_drop=0.0,
                drop_path_rate=0.0)
    encoder = SwinEncoder(in_chans=4, final_channels=64, **swin)
    decoder = SwinDecoder(out_chans=4, **swin)
    nn_rhs = CrossAttnTransformer(problem_dim=LATENT_DIM, latent_dim=64 * 4, num_blocks=3, num_heads=4, z_dim=2,
                                  dropout_rate=0, activation=torch.nn.Tanh(), use_self_attention_z_block=True)
    ck = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    encoder, decoder, nn_rhs = (_load(m, ck[k]).to(device).eval()
                                for m, k in [(encoder, "encoder"), (decoder, "decoder"), (nn_rhs, "nn_rhs_model")])
    if compile:
        encoder, decoder = torch.compile(encoder), torch.compile(decoder)
    return encoder, decoder, nn_rhs


def hnode_solver(nn_rhs, device):
    from hybrid_node_solver import HybridNODESolver

    delta_t = DT * NUM_FRAMES / N_WINDOWS
    return HybridNODESolver(nn_rhs, delta_t, NUM_FRAMES // N_WINDOWS + 1, LATENT_DIM, N_WINDOWS,
                            backprop_type="matrix_free", use_manual_jacobian=True, use_sparse_solver=False,
                            use_cpu_solver=False, verbose=False, device=device)


def hnode_initial_state(e0, e1, z):
    """HNODE's starting point from the latents of frames 0 and 1 (as in 03_shear_flow_ddp.py)."""
    return {"J_init": (e1 - e0) / DT, "u_init": (e0 + e1) / 2.0, "z": z}


def well_split(split):
    """The Well's WellDataset over one split, one full trajectory (200 frames) per item.
    Items are ordered by sorted file name, then trajectory index in the file."""
    from the_well.data import WellDataset

    return WellDataset(path=os.path.join(WELL_DATA_PATH, split), n_steps_input=2, full_trajectory_mode=True,
                       max_rollout_steps=10_000, return_grid=False, boundary_return_type=None,
                       use_normalization=False)


def full_trajectory(well, idx):
    """(frames (T, H, W, C) float64, scalars [Re, Sc] float64) of one trajectory."""
    sample = well[idx]
    frames = torch.cat([sample["input_fields"], sample["output_fields"]])
    return frames.double(), sample["constant_scalars"].double()


def split_index(split):
    """[(file name, trajectory index in file)] in WellDataset's item order."""
    import glob

    import h5py

    out = []
    for f in sorted(glob.glob(os.path.join(WELL_DATA_PATH, split, "*.hdf5"))):
        with h5py.File(f, "r") as fh:
            out += [(os.path.basename(f), i) for i in range(fh["t0_fields/tracer"].shape[0])]
    return out
