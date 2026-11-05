"""Live training figures: loss curves and a fixed validation example (input track with the gap | model | truth)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ki_tracks import to_km  # noqa: E402


def loss_png(path, hist, name):
    """Training loss (running mean over 200 steps) and validation loss / 24-h infill error against step."""
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2), layout="constrained")
    tr = np.asarray(hist["train"])
    k = min(200, len(tr))
    ax[0].semilogy(hist["step"], tr, color="0.8", lw=0.5, label="train (each step)")
    ax[0].semilogy(hist["step"][k - 1:], np.convolve(tr, np.ones(k) / k, "valid"), color="k", label=f"train ({k}-step mean)")
    ax[0].semilogy(hist["val_step"], hist["val_loss"], "o-", color="tab:red", ms=3, label="validation parcels")
    ax[0].set(xlabel="step", ylabel="loss (acceleration + score MSE)")
    ax[0].legend(fontsize=7, frameon=False)
    ax[1].plot(hist["val_step"], hist["val_infill_km"], "o-", color="tab:red", ms=3)
    ax[1].set(xlabel="step", ylabel="median error in 24-h gaps (km)", ylim=(0, None))
    fig.suptitle(name, fontsize=9)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def example_png(path, d, j, k0, k1, path_model, title, parcels=None):
    """Row 1: a validation parcel's 24-h gap (input | filled | truth); row 2: a zoom on the gap."""
    pj = (d.val if parcels is None else parcels)[j]
    track = to_km(d, d.x[:, pj]) / 1e3
    gap = slice(k0, k1 + 1)
    filled = to_km(d, path_model) / 1e3
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.4), layout="constrained", sharex="row", sharey="row")
    for row, ax_row in enumerate(axes):
        for col, ax in enumerate(ax_row):
            before, after = track[: k0 + 1], track[k1:]
            ax.plot(*before.T, color="0.4", lw=1)
            ax.plot(*after.T, color="0.4", lw=1)
            if col == 1:
                ax.plot(*filled.T, color="tab:red", lw=2, label="model fills the gap")
            if col == 2:
                ax.plot(*track[gap].T, color="tab:blue", lw=2, label="true track in the gap")
            ax.plot(*track[k0], "ko", ms=4)
            ax.plot(*track[k1], "ks", ms=4)
            ax.set_aspect("equal", adjustable="box")
            if row == 1:
                both = np.concatenate([track[gap], filled])
                pad = 0.3 * np.ptp(both, axis=0).max() + 0.05
                ax.set_xlim(both[:, 0].min() - pad, both[:, 0].max() + pad)
                ax.set_ylim(both[:, 1].min() - pad, both[:, 1].max() + pad)
        ax_row[0].set_ylabel("y (1000 km, polar stereographic)")
    for ax, t in zip(axes[0], ["input: track with the gap removed", "model", "truth"]):
        ax.set_title(t, fontsize=9)
    for ax in axes[1]:
        ax.set_xlabel("x (1000 km)")
    axes[0, 1].legend(fontsize=7, frameon=False)
    axes[0, 2].legend(fontsize=7, frameon=False)
    fig.suptitle(title, fontsize=9)
    fig.savefig(path, dpi=110)
    plt.close(fig)
