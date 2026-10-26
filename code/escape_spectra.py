"""Input-power spectra P(omega) of the escape-sequence designs (results/escape_seq.npy): rows chain / truss, columns
optimised / explicit family, one curve per level s. Dots at the bottom: excited modes (share > 1 %), one row per level."""
import os, sys, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from models import chain, truss, spectrum, W0
S_LEVELS = [1, 3, 10, 30, 100, 300, 1e3, 3e3, 1e4, 3e4, 1e5]   # as in escape_seq.py

HERE = os.path.dirname(os.path.abspath(__file__))
MODE = sys.argv[1] if len(sys.argv) > 1 else "k"
R = np.load(os.path.join(HERE, "results", "escape_seq_%s.npy" % MODE), allow_pickle=True).item()
BOX = {"k": "$k\\in[\\bar k/s, \\bar k s]$, masses fixed", "m": "$m\\in[\\bar m/s, \\bar m s]$, stiffness fixed", "km": "$k\\in[\\bar k/s, \\bar k s]$, $m\\in[\\bar m/s, \\bar m s]$"}[MODE]
SHOW = [1, 10, 100, 1e3, 1e4, 1e5] if MODE != "km" else [1, 3, 10, 100, 1e3, 1e4]
WS = np.geomspace(0.1, 1e6, 4000)
models = {"chain": chain(), "truss": truss()}

fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharex=True, sharey=True)
for row, name in enumerate(["chain", "truss"]):
    mdl = models[name]
    for col, (which, lab) in enumerate([(0, "optimised"), (1, "explicit family")]):
        ax = axes[row, col]; seq = R[name][which]
        cols = plt.cm.viridis(np.linspace(0.95, 0.0, len(SHOW)))
        for j, (s, c) in enumerate(zip(SHOW, cols)):
            a = seq[S_LEVELS.index(s)]
            ax.loglog(WS, spectrum(mdl, a["k"], a["m"], WS), color=c, lw=1.2, label="$s=%g$  ($P$=%.1e W, $\\mu$=%.0e)" % (s, a["P"], a["mu"]))
            yb = 10 ** ((-20.6 if MODE != "k" else -12.6) + 0.3 * j); act = a["sh"] > 1e-2
            ax.scatter(a["wr"][act], np.full(act.sum(), yb), s=4 + 40 * a["sh"][act], color=c, clip_on=False, zorder=5)
            lo, hi = a["lo"], a["hi"]; ax.plot([lo, hi if np.isfinite(hi) else 1e6], [yb, yb], color=c, lw=0.8, alpha=0.6)
        ax.axvline(W0, color="crimson", ls="--", lw=1)
        ax.set_xlim(0.1, 1e6); ax.set_ylim(10 ** -21.7 if MODE != "k" else 10 ** -13.7, 1e2); ax.grid(True, alpha=0.3)
        ax.text(0.015, 0.975, "%s, %s: %s, no budgets" % (name, lab, BOX), transform=ax.transAxes, va="top", fontsize=9)
        ax.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.0, 0.93), framealpha=0.85)
        if col == 0: ax.set_ylabel("$P(\\omega)$ [W]")
        if row == 1: ax.set_xlabel("$\\omega$ [rad/s]")
fig.suptitle("Input-power spectra along the escape sequences, variables: %s (dashed line: $\\omega=300$ rad/s; dots: excited modes, share > 1 %%; bars: excited gap)" % MODE, fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "escape_spectra_%s.png" % MODE), dpi=150); fig.savefig(os.path.join(HERE, "escape_spectra_%s.pdf" % MODE))
print("written escape_spectra_%s.png/pdf" % MODE)
