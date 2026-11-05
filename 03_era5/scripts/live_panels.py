# Copy of roger/02_dinosaur/scripts/live_panels.py.
"""Live training figures: loss curves and fixed validation examples (context tracks | model | truth)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402



def loss_png(path, hist, name, val_label="validation error (km)"):
    """Training loss (running mean over 200 steps) and validation errors against step."""
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2), layout="constrained")
    tr = np.asarray(hist["train"])
    k = min(200, len(tr))
    ax[0].semilogy(hist["step"], tr, color="0.8", lw=0.5, label="train (each step)")
    ax[0].semilogy(hist["step"][k - 1:], np.convolve(tr, np.ones(k) / k, "valid"), color="k", label=f"train ({k}-step mean)")
    ax[0].set(xlabel="step", ylabel="loss (acceleration + score MSE)")
    ax[0].legend(fontsize=7, frameon=False)
    ax[1].plot(hist["val_step"], hist.get("val_km", hist.get("val_infill_km")), "o-", color="tab:red", ms=3, label="validation")
    for key, color in (("forecast_km", "tab:purple"), ("backcast_km", "tab:green")):
        if hist.get(key):
            ax[1].plot(hist["val_step"], hist[key], ".--", color=color, ms=3, label=key.split("_")[0])
    ax[1].set(xlabel="step", ylabel=val_label, ylim=(0, None))
    ax[1].legend(fontsize=7, frameon=False)
    fig.suptitle(name, fontsize=9)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def context_png(path, tok, centroid, roll, truth, n_forecast, title, n_show=8, span_tokens=17):
    """Rows: a forecast and a backcast validation sample; columns: context tracks | model tracks | true tracks.

    tok (B, Tc*K, 8) normalized context tokens; roll (H+1, B, Q, 3) and truth (B, H+1, Q, 3) in km (x, y shown).
    """
    from ctx_model import POS
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.6), layout="constrained", sharex="row", sharey="row")
    for row, b in enumerate((0, n_forecast)):
        ctx = tok[b].reshape(span_tokens, -1, 8)[..., 1:3].cpu().numpy() * POS[0] + centroid[b].cpu().numpy()
        for col, ax in enumerate(axes[row]):
            ax.plot(ctx[:, :, 0] / 1e3, ctx[:, :, 1] / 1e3, color="0.8", lw=0.5)
            if col == 1:
                ax.plot(roll[:, b, :n_show, 0] / 1e3, roll[:, b, :n_show, 1] / 1e3, color="tab:red", lw=1.5)
            if col == 2:
                ax.plot(truth[b, :, :n_show, 0] / 1e3, truth[b, :, :n_show, 1] / 1e3, color="tab:blue", lw=1.5)
            ax.plot(truth[b, 0, :n_show, 0] / 1e3, truth[b, 0, :n_show, 1] / 1e3, "ko", ms=3)
            ax.set_aspect("equal", adjustable="box")
        axes[row, 0].set_ylabel(("forecast" if row == 0 else "backcast") + "\ny (1000 km)")
    for ax, t in zip(axes[0], ["context: other parcels' observed 48 h (grey)", "model: 24 h from the black dots",
                               "truth"]):
        ax.set_title(t, fontsize=9)
    for ax in axes[1]:
        ax.set_xlabel("x (1000 km, polar stereographic)")
    fig.suptitle(title, fontsize=9)
    fig.savefig(path, dpi=110)
    plt.close(fig)
