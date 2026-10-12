"""VRMSE versus latent dimension r for each basis (POD, Fourier POD): one HNODE series per coefficient
scaling (SCALE: "mode", "shared", "floor:64"), the projection floor (oracle) and LSTM, from the rows'
eval/*/vrmse.npz (eval_latent_rollout.py).

    python summarize_latent_dim.py [ROWS_DIR] [OUT_PNG]

Rows are ROWS_DIR/{pod,fpod}R. HNODE runs are taken from each row's hnode*/history.json with SOAP at
lr 1e-4 and HNODE_DT = 1, grouped by history["scale"] (absent in the September runs, which used "mode" with the dense solver and
100 Newton iterations: series "mode (Sept)"; floor:K at R <= K is "mode", but is kept as its own series). Each run's directory name is printed. The other models
are the row's best learning rate (best.json). Statistics over frames 1-199 and the 4 channels (The Well VRMSE):
  mean    mean over the 224 held-out trajectories (valid + test)
  test    mean over the 112 test trajectories (HNODE variants are compared on valid during training)
  median  median over trajectories of the per-trajectory mean (the mean is dominated by pressure and v_y
          frames whose target variance is close to zero)
  train   mean of the training-loss VRMSE definition (variance + 1e-3 in the denominator)
"""
import json, os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
rows_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "hnode_runs", "rows")
out_png = sys.argv[2] if len(sys.argv) > 2 else os.path.join(rows_dir, "latent_dim_vrmse.png")
BASES = {"pod": "POD", "fpod": "Fourier POD"}
STATS = ["mean", "test", "median", "train"]


def score(path):
    f = os.path.join(path, "vrmse.npz")
    if not os.path.exists(f):
        return None
    d = np.load(f)
    per_traj = d["vrmse_well"].mean(axis=(1, 2))
    return {"mean": per_traj.mean(), "test": per_traj[d["split"] == "test"].mean(),
            "median": np.median(per_traj), "train": d["vrmse_train"].mean()}


table = {}  # (basis, R) -> {series: (run, stats)}
for name in sorted(os.listdir(rows_dir)):
    m = re.fullmatch(r"(f?pod)(\d+)", name)
    if not m:
        continue
    basis, R, row = m.group(1), int(m.group(2)), os.path.join(rows_dir, name)
    res = {}
    for run in sorted(os.listdir(row)):
        h = os.path.join(row, run, "history.json")
        if not (run.startswith("hnode") and os.path.exists(h)):
            continue
        d = json.load(open(h))
        if d.get("optimizer") != "soap" or d.get("lr") != 1e-4 or d.get("hnode_dt") != 1.0 or d["history"][-1]["step"] != 3000:
            continue
        s = score(os.path.join(row, "eval", run))
        if s is not None:
            res["hnode " + d.get("scale", "mode (Sept)")] = (run, s)
    best = os.path.join(row, "best.json")
    best = json.load(open(best)) if os.path.exists(best) else {}
    for model in ["oracle", "lstm"]:
        run = "oracle" if model == "oracle" else os.path.basename(best.get(model, {}).get("dir", ""))
        s = score(os.path.join(row, "eval", run)) if run else None
        if s is not None:
            res[model] = (run, s)
    table[basis, R] = res

SERIES = ["oracle", "hnode mode (Sept)", "hnode mode", "hnode floor:64", "hnode shared", "lstm"]
for basis, label in BASES.items():
    dims = sorted(R for b, R in table if b == basis)
    if not dims:
        continue
    for stat in STATS:
        print(f"\n{label}, VRMSE {stat} (frames 1-199)")
        print(f"{'':19s}" + "".join(f"{R:>9d}" for R in dims))
        for ser in SERIES:
            cells = [table[basis, R][ser][1][stat] if ser in table[basis, R] else None for R in dims]
            if any(c is not None for c in cells):
                print(f"{ser:19s}" + "".join(f"{c:9.3f}" if c is not None else f"{'-':>9s}" for c in cells))
    print("runs: " + "; ".join(f"{R}: " + ", ".join(f"{ser.split()[-1]}={table[basis, R][ser][0]}"
                                                    for ser in SERIES[1:5] if ser in table[basis, R]) for R in dims))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit(0)
fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
styles = {"oracle": "k--", "hnode mode (Sept)": "o--", "hnode mode": "o-", "hnode floor:64": "D-", "hnode shared": "s-", "lstm": "^:"}
for col, (basis, label) in enumerate(BASES.items()):
    dims = sorted(R for b, R in table if b == basis)
    for row_i, stat in enumerate(["mean", "median"]):
        ax = axes[row_i, col]
        for ser in SERIES:
            pts = [(R, table[basis, R][ser][1][stat]) for R in dims if ser in table[basis, R]]
            if pts:
                ax.plot(*zip(*pts), styles[ser], label="projection floor" if ser == "oracle" else ser)
        ax.set(xscale="log", yscale="log", title=f"{label}: {stat} VRMSE")
        if dims:
            ax.set_xticks(dims)
            ax.set_xticklabels([str(R) for R in dims])
        ax.grid(alpha=0.3)
        if row_i == 1:
            ax.set_xlabel("latent dimension r")
axes[0, 0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(out_png, dpi=120)
print(f"\nsaved {out_png}")
