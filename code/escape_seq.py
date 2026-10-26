"""Unbounded escape of the excited modes, chain and truss. Usage: python escape_seq.py [k|m|km]  (default k).

Design variables: k (stiffness only, masses fixed, mu_f = 1 constant), m (masses only, stiffness fixed) or km (both).
Boxes k in [kbar/s, kbar*s] and/or m in [mbar/s, mbar*s], no budgets, s = 1 ... 1e5. Two sequences per system:
  (a) optimised: min P over log10 of the variables with SQP (SLSQP, bounds only), continuation from the previous level;
  (b) explicit family (no optimisation): springs/bars attached to the loaded mass = kbar/s, all others = kbar*s;
      loaded mass = mbar*s, all other masses = mbar/s.
For each design: P, mu_k = f^T M^-1 f, excited gap (nearest excited mode below/above omega at share > 1 %), share of the
participation in the fixed band B = [omega/2, 2 omega], and the bound 2P/(g_B mu_k) of Theorem 1 on that share.
Writes results/escape_seq_<mode>.npy and escape_seq_<mode>.png/pdf."""
import os, sys, numpy as np
from scipy.optimize import minimize
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from models import chain, truss, power_and_grads, modes, excited_gap, W0, ETA

HERE = os.path.dirname(os.path.abspath(__file__))
MODE = sys.argv[1] if len(sys.argv) > 1 else "k"; VK, VM = "k" in MODE, "m" in MODE
S_LEVELS = [1, 3, 10, 30, 100, 300, 1e3, 3e3, 1e4, 3e4, 1e5]
if MODE == "km": S_LEVELS = S_LEVELS[:-2]   # k spans 1e-1..1e10 at s = 1e5: the lowest eigenvalue (~1e-5) is below eps*||K||
B = (W0 / 2, 2 * W0)


def g_of(wr, w=W0, eta=ETA):   # hysteretic g(omega_r; omega), stiffness-proportional, as in the manuscript
    d = eta * wr**2
    return w * d / ((wr**2 - w**2)**2 + d**2)


def g_B():   # min of g over the band: g is unimodal around omega, so the min is at an end point
    return min(g_of(np.array([B[0]]))[0], g_of(np.array([B[1]]))[0])


def analyse(model, k, m):
    P = power_and_grads(model, k, m)[0]; wr, p, mu = modes(model, k, m); sh = p / mu
    lo, hi = excited_gap(wr, p, mu)
    inB = (wr >= B[0]) & (wr <= B[1])
    return dict(P=P, mu=mu, lo=lo, hi=hi, shareB=sh[inB].sum(), bound=2 * P / g_B() / mu, wr=wr, sh=sh, k=k.copy(), m=m.copy())


def optimise(model, s, k_start, m_start):
    kbar, mbar = np.mean(model["k0"]), np.mean(model["m0"]); nk, nm = model["n_k"], model["n_m"]
    bk = [(np.log10(kbar / s), np.log10(kbar * s))] * nk if VK else []; bm = [(np.log10(mbar / s), np.log10(mbar * s))] * nm if VM else []
    x0 = np.concatenate([np.log10(k_start) if VK else [], np.log10(m_start) if VM else []]); bounds = bk + bm
    x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])
    def split(x):
        k = 10.0**x[:nk] if VK else k_start; m = 10.0**x[nk:] if VK and VM else (10.0**x if VM else m_start); return k, m
    def fun(x):
        k, m = split(x); P, gk, gm = power_and_grads(model, k, m)
        g = np.concatenate([gk * k if VK else [], gm * m if VM else []]) * np.log(10)   # d log10 P / d log10 x = x dP/dx / P
        return np.log10(P), g / (P * np.log(10))
    r = minimize(fun, x0, jac=True, method="SLSQP", bounds=bounds, options=dict(maxiter=3000, ftol=1e-14))
    k, m = split(r.x); return k, m, r.get("nit", r.get("nfev", -1))


def explicit(model, s):
    kbar, mbar = np.mean(model["k0"]), np.mean(model["m0"]); k = model["k0"].copy(); m = model["m0"].copy()
    if VK: k = np.full(model["n_k"], kbar * s); k[model["load_springs"]] = kbar / s
    if VM: m = np.full(model["n_m"], mbar / s); m[model["load_mass"]] = mbar * s
    return k, m


def load_springs(model):
    if model["name"] == "chain":
        ln = model["load_mass"]; return [ln, ln + 1]          # springs ln (left of mass ln) and ln+1 (right)
    ln = model["load_node"]; return [i for i, (a, b) in enumerate(model["bars"]) if ln in (a, b)]


results = {}
for mdl in (chain(), truss()):
    mdl["load_springs"] = load_springs(mdl)
    # check the spring indexing on the chain: dK of those springs must touch the loaded dof
    if mdl["name"] == "chain":
        for i in mdl["load_springs"]:
            assert abs(np.asarray(mdl["dK"][i])[mdl["load_mass"], mdl["load_mass"]]) > 0, "spring %d not attached to the loaded mass" % i
    k = mdl["k0"].copy(); m = mdl["m0"].copy(); seq_opt, seq_exp = [], []
    for s in S_LEVELS:
        # two starts per level: continuation from the previous level, and the explicit family of this level; keep the lower P
        k1, m1, nit1 = optimise(mdl, s, k, m); a1 = analyse(mdl, k1, m1)
        k2, m2, nit2 = optimise(mdl, s, *explicit(mdl, s)); a2 = analyse(mdl, k2, m2)
        if a2["P"] < a1["P"]: k, m, a, nit = k2, m2, a2, nit2; a["start"] = "explicit"
        else: k, m, a, nit = k1, m1, a1, nit1; a["start"] = "continuation"
        a["nit"] = nit; a["other_P"] = max(a1["P"], a2["P"]); seq_opt.append(a)
        e = analyse(mdl, *explicit(mdl, s)); seq_exp.append(e)
        print("%-5s s=%7g | opt: P=%.3e mu=%.2e P/mu=%.2e gap [%7.2f, %8.1f] shareB=%.1e bound=%.1e (%d it, from %s, other start %.1e) | explicit: P=%.3e mu=%.2e gap [%7.2f, %8.1f] shareB=%.1e bound=%.1e"
              % (mdl["name"], s, a["P"], a["mu"], a["P"] / a["mu"], a["lo"], a["hi"], a["shareB"], a["bound"], nit, a["start"][:4], a["other_P"], e["P"], e["mu"], e["lo"], e["hi"], e["shareB"], e["bound"]), flush=True)
    results[mdl["name"]] = (seq_opt, seq_exp)
np.save(os.path.join(HERE, "results", "escape_seq_%s.npy" % MODE), results, allow_pickle=True)

# ---------- figure: excited modes vs s (share > 1 %, marker area ~ share) and P vs s
fig, axes = plt.subplots(3, 2, figsize=(11, 11))
BOX = {"k": "$k\\in[\\bar k/s, \\bar k s]$, masses fixed", "m": "$m\\in[\\bar m/s, \\bar m s]$, stiffness fixed", "km": "$k\\in[\\bar k/s, \\bar k s]$, $m\\in[\\bar m/s, \\bar m s]$"}[MODE]
for col, name in enumerate(["chain", "truss"]):
    seq_opt, seq_exp = results[name]
    ax = axes[0, col]
    for seq, c, lab in ((seq_opt, "tab:blue", "optimised"), (seq_exp, "tab:orange", "explicit")):
        for s, a in zip(S_LEVELS, seq):
            act = a["sh"] > 1e-2
            ax.scatter(np.full(act.sum(), s), a["wr"][act], s=6 + 80 * a["sh"][act], color=c, alpha=0.7, edgecolor="none", label=lab if s == 1 else None)
        ax.plot(S_LEVELS, [a["lo"] for a in seq], color=c, lw=0.8, ls="--"); ax.plot(S_LEVELS, [a["hi"] for a in seq], color=c, lw=0.8, ls="--")
    ax.axhline(W0, color="crimson", ls="--", lw=1); ax.axhspan(B[0], B[1], color="crimson", alpha=0.07)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylabel("excited natural frequencies [rad/s]"); ax.set_title("%s: %s, no budgets" % (name, BOX), fontsize=10)
    ax.legend(fontsize=8, loc="upper left"); ax.grid(True, alpha=0.3)
    ax = axes[1, col]
    for seq, c, lab in ((seq_opt, "tab:blue", "optimised"), (seq_exp, "tab:orange", "explicit")):
        ax.loglog(S_LEVELS, [a["P"] for a in seq], "o-", color=c, label="$P$, " + lab)
        ax.loglog(S_LEVELS, [a["mu"] for a in seq], "^--", color=c, mfc="none", label="$\\mu_k$, " + lab)
        ax.loglog(S_LEVELS, [a["P"] / a["mu"] for a in seq], "d-.", color=c, mfc="none", label="$P/\\mu_k$, " + lab)
    ax.set_ylabel("$P$ [W],  $\\mu_k$ [1/kg],  $P/\\mu_k$"); ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)
    ax = axes[2, col]
    for seq, c, lab in ((seq_opt, "tab:blue", "optimised"), (seq_exp, "tab:orange", "explicit")):
        ax.loglog(S_LEVELS, [max(a["shareB"], 1e-16) for a in seq], "s-", color=c, label="share in $B$, " + lab)
        ax.loglog(S_LEVELS, [a["bound"] for a in seq], ":", color=c, label="bound $2P/(g_B\\mu_k)$, " + lab)
    ax.set_xlabel("$s$"); ax.set_ylabel("participation share in $B=[\\omega/2, 2\\omega]$"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
fig.suptitle("Unbounded escape, variables: %s. Dashed = excited gap; shaded = band $B$; share in $B$ is bounded by $2P/(g_B\\,\\mu_k)$ (Theorem 1)" % MODE, fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "escape_seq_%s.png" % MODE), dpi=150); fig.savefig(os.path.join(HERE, "escape_seq_%s.pdf" % MODE))
print("written escape_seq_%s.png/pdf" % MODE)
