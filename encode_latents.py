#!/usr/bin/env python
"""Encode every shear-flow trajectory with the frozen pretrained encoder, for latent-space baselines.

    srun -N1 -n8 -c7 --gpus-per-task=1 --gpu-bind=closest python -u encode_latents.py OUT_DIR
    (or sbatch run_encode_latents.sh)

Writes OUT_DIR/{train,valid,test}.pt, each a dict:
    latents   (N, 200, 64)  encoder output of every frame (normalized frames, as in training)
    scalars   (N, 2)        [Re, Sc]
    z         (N, 2)        normalized log10 conditioning, as HNODE gets it
    file, traj              source HDF5 file and trajectory index in it (WellDataset item order)
    duplicate_of_train      train index this trajectory duplicates, or None (valid/test; needs DUP_MAP)
    hnode     (N, 199, 64)  HNODE latent rollout from frames 0-1, predicting frames 1..199 (valid/test)

Environment: SPLITS (default "train,valid,test"), HNODE_SPLITS (default "valid,test"),
DUP_MAP (default hnode_runs/eval_full/dup_map.json), MAX_TRAJ (per split, for testing; 0 = all).
"""
import json
import os
import sys
import time

import torch
import torch.distributed as dist

import shear_flow_common as C

torch.set_default_dtype(torch.float64)
CHUNK = 50  # frames per encoder call (one compiled shape; 200 = 4 chunks)


def main():
    out_dir = sys.argv[1]
    splits = os.environ.get("SPLITS", "train,valid,test").split(",")
    hnode_splits = set(os.environ.get("HNODE_SPLITS", "valid,test").split(","))
    dup_path = os.environ.get("DUP_MAP", os.path.join(C.HERE, "..", "hnode_runs", "eval_full", "dup_map.json"))
    max_traj = int(os.environ.get("MAX_TRAJ", 0))

    rank, local_rank, world_size = C.distributed_env()
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29501")
    user = os.environ.get("USER", "user")
    os.environ.setdefault("MIOPEN_USER_DB_PATH", f"/tmp/miopen-{user}-{rank}")
    os.environ.setdefault("MIOPEN_CUSTOM_CACHE_DIR", f"/tmp/miopen-{user}-{rank}")
    dist.init_process_group("gloo", rank=rank, world_size=world_size)  # only gathers Python objects
    device = C.pick_device(local_rank)
    log = (lambda *a: print(*a, flush=True)) if rank == 0 else (lambda *a: None)

    encoder, _, nn_rhs = C.load_models(device)
    solver = C.hnode_solver(nn_rhs, device)
    mean, std = C.frame_stats(device)
    dup_map = json.load(open(dup_path)) if os.path.exists(dup_path) else {}
    log(f"world size {world_size}, splits {splits}, HNODE rollout on {sorted(hnode_splits)}, "
        f"duplicate map: {dup_path if dup_map else 'none'}")

    for split in splits:
        well = C.well_split(split)
        index = C.split_index(split)
        assert len(index) == len(well), (len(index), len(well))
        n = min(len(well), max_traj) if max_traj else len(well)
        mine = list(range(rank, n, world_size))
        do_hnode = split in hnode_splits
        # Prefetch trajectories (~400 MiB each) in worker processes while the GPU encodes.
        loader = torch.utils.data.DataLoader(mine, batch_size=None, num_workers=3, prefetch_factor=2,
                                             collate_fn=lambda i: (i, *C.full_trajectory(well, i)))
        results, t0 = [], time.time()
        with torch.no_grad():
            for k, (i, frames, scalars) in enumerate(loader):
                frames = (frames.to(device) - mean) / std  # (T, H, W, C)
                x = frames.permute(0, 3, 1, 2)
                lat = torch.cat([encoder(x[s:s + CHUNK]) for s in range(0, x.shape[0], CHUNK)])
                z = C.normalize_z(scalars.to(device)).unsqueeze(0)
                r = {"i": i, "latents": lat.cpu(), "scalars": scalars, "z": z[0].cpu()}
                if do_hnode:
                    # Encode frames 0-1 as their own batch, exactly as MODE=eval does.
                    e01 = encoder(x[:2])
                    _, u_hat = solver.evolve_batch(C.hnode_initial_state(e01[:1], e01[1:], z))
                    r["hnode"] = u_hat[0, : x.shape[0] - 1].cpu()  # cell k = frame k + 1
                results.append(r)
                if rank == 0 and (k % 10 == 0 or k == len(mine) - 1):
                    log(f"  {split} rank 0: {k + 1}/{len(mine)} ({time.time() - t0:.0f}s)")

        gathered = [None] * world_size
        dist.all_gather_object(gathered, results)
        if rank == 0:
            allr = sorted((r for part in gathered for r in part), key=lambda r: r["i"])
            meta = [dup_map.get(f"{split}:{r['i']}", {}) for r in allr]
            out = {
                "latents": torch.stack([r["latents"] for r in allr]),
                "scalars": torch.stack([r["scalars"] for r in allr]),
                "z": torch.stack([r["z"] for r in allr]),
                "file": [index[r["i"]][0] for r in allr],
                "traj": [index[r["i"]][1] for r in allr],
                "duplicate_of_train": [m.get("duplicate_of_train") for m in meta] if split != "train" else None,
                "hnode": torch.stack([r["hnode"] for r in allr]) if do_hnode else None,
            }
            os.makedirs(out_dir, exist_ok=True)
            torch.save(out, os.path.join(out_dir, f"{split}.pt"))
            log(f"{split}: saved {len(allr)} trajectories, latents {tuple(out['latents'].shape)}"
                + (f", hnode {tuple(out['hnode'].shape)}" if do_hnode else "") + f" ({time.time() - t0:.0f}s)")
        dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
