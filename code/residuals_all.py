"""First-order residuals of every design saved in results/ (Table 1, relaxation levels, pairs, displacement objectives):
reduced gradient of log P (or log |u_j|) with the budget multipliers fitted on the variables farther than 0.1% from a bound,
relative to the gradient scale."""
import glob, numpy as np
from models import chain, truss, power_and_grads
KB, MB = (1e4, 5e5), (0.1, 5.0)
K_LEVELS = [1e5, 3e4, 1e4, 3e3, 1e3, 3e2]
M_LEVELS = {"chain": [(0.9, 1.1), (0.5, 2.0), (0.25, 4.0), (0.125, 8.0), (0.1, 10.0), (0.1, 10.9)], "truss": [(0.9, 1.1), (0.5, 2.0), (0.25, 4.0), (0.125, 8.0), (0.1, 10.0), (0.1, 19.9)]}
def resid(g_parts):
    gs = max(np.max(abs(g)) for g, _, _ in g_parts); worst = 0.0
    for g, x, (lo, hi) in g_parts:
        free = (x > lo * 1.001) & (x < hi * 0.999)
        if free.sum() > 1:
            a = x[free]; lam = (g[free] @ a) / (a @ a); worst = max(worst, np.max(abs(g[free] - lam * a)) / gs)
    return worst
worst_all = 0.0
for fn in sorted(glob.glob("results/*.npz")):
    base = fn.split("/")[-1][:-4]
    if base.startswith("disp_") or base.startswith("sym_best") or "theta" in base: continue
    name = "chain" if "chain" in base else "truss"; mdl = chain() if name == "chain" else truss(); d = np.load(fn)
    if "k" not in d.files or "m" not in d.files or "truncated" in base: continue
    if "pairs" in base: mdl = dict(mdl); mdl["f"] = d["f"]; vk, vm, kb, mb = True, False, KB, MB
    elif "relax" in base:
        vs = base.split("_")[2]; L = int(base[-1]); vk, vm = "k" in vs, "m" in vs
        kb = (K_LEVELS[L], 5e5) if vk else KB; mb = M_LEVELS[name][L] if vm else MB
    else:
        vs = base.split("_")[1]; vk, vm = "k" in vs, "m" in vs; kb, mb = KB, MB
    P, gk, gm = power_and_grads(mdl, d["k"], d["m"]); parts = []
    if vk: parts.append((gk * d["k"] / P, d["k"], kb))
    if vm: parts.append((gm * d["m"] / P, d["m"], mb))
    r = resid(parts); worst_all = max(worst_all, r); print("%-24s residual %.1e" % (base, r))
print("largest first-order residual over all designs: %.1e" % worst_all)
