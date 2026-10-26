"""Numbers quoted in Sections 4.2, 4.4 and 4.5 (SQP results)."""
import math, numpy as np
from scipy.linalg import eigh
from models import chain, truss, modes, W0, ETA
def g(wr): s = wr**2; d = ETA * s; return W0 * d / ((s - W0**2)**2 + d**2)
def iv(P, mu, tau=1e-2):
    c = 2 * P / (tau * mu); A = 2 * c * W0**2 + ETA * W0; D = A * A - 4 * c * c * (1 + ETA**2) * W0**4
    return (math.sqrt((A - math.sqrt(D)) / (2 * c * (1 + ETA**2))), math.sqrt((A + math.sqrt(D)) / (2 * c * (1 + ETA**2)))) if D > 0 else None
def gap(wr, sh, thr):
    ex = wr[sh >= thr]; lo = ex[ex < W0].max() if (ex < W0).any() else float("nan"); hi = ex[ex > W0].min() if (ex > W0).any() else float("inf"); return lo, hi
print("=== Table 1")
for name, mdl in (("chain", chain()), ("truss", truss())):
    P0 = 7.464e-5 if name == "chain" else 8.851e-5
    for c in ("k", "m", "km"):
        d = np.load("results/%s_%s.npz" % (name, c)); wr, p, mu = modes(mdl, d["k"], d["m"]); sh = p / mu; P = float(d["P"])
        lo, hi = gap(wr, sh, 1e-2); ivl = iv(P, mu)
        inside = (wr > lo) & (wr < hi); near = abs(wr - W0) < 0.2 * W0
        print("%s %-2s P=%.3e P0/P=%.0f mu=%.2f gap [%.0f,%.0f] Cor9 [%.0f,%.0f] P/mu=%.3e s; distinct %d/%d; max share inside gap %.1e at %.0f; within 20%%: %.1e; gaps thr 1e-3/1e-2/1e-1: %s %s %s; lowest excited %.1f; zero-share modes %d (inside gap %d); m_load %.2f"
              % (name, c, P, P0 / P, mu, lo, hi, ivl[0], ivl[1], P / mu, len(set(np.round(np.log10(d["all_P"]), 3))), len(d["all_P"]), sh[inside].max() if inside.any() else 0, wr[inside][np.argmax(sh[inside])] if inside.any() else 0, sh[near].max() if near.any() else 0, np.round(gap(wr, sh, 1e-3)), np.round(gap(wr, sh, 1e-2)), np.round(gap(wr, sh, 1e-1)), wr[sh >= 1e-2].min(), int(np.sum(sh < 1e-20)), int(np.sum((sh < 1e-20) & inside)), d["m"][mdl["load_mass"]]))
gB = min(g(0.8 * W0), g(1.2 * W0)); print("requirement B=[0.8w,1.2w], tau=1e-2: P/mu < %.3e s" % (1e-2 * gB / 2))
print("=== relaxation last levels")
for name, mdl in (("chain", chain()), ("truss", truss())):
    for c in ("k", "m", "km"):
        for L in (0, 5):
            d = np.load("results/%s_relax_%s_L%d.npz" % (name, c, L)); wr, p, mu = modes(mdl, d["k"], d["m"]); sh = p / mu; lo, hi = gap(wr, sh, 1e-2)
            top = np.argmax(sh); print("%s relax %-2s L%d P=%.3e mu=%.3f gap [%.1f, %.0f] top mode %.1f rad/s share %.2f; m_load %.2f" % (name, c, L, float(d["P"]), mu, lo, hi, wr[top], sh[top], d["m"][mdl["load_mass"]]))
print("=== escape k: two factors of the slack")
S = [1, 3, 10, 30, 100, 300, 1e3, 3e3, 1e4, 3e4, 1e5]; gBB = min(g(W0 / 2), g(2 * W0))
for mode in ("k", "m", "km"):
    a = np.load("results/escape_seq_%s.npy" % mode, allow_pickle=True).item()
    for name, mdl in (("chain", chain()), ("truss", truss())):
        for j in (0, len(a[name][0]) - 1):
            d = a[name][0][j]; K = mdl["assemble_K"](d["k"]); M = mdl["assemble_M"](d["m"]); lam, Phi = eigh(K, M); p = (Phi.T @ mdl["f"])**2; wr = np.sqrt(lam)
            inB = (wr >= W0 / 2) & (wr <= 2 * W0); share = p[inB].sum() / p.sum(); bound = 2 * d["P"] / (gBB * p.sum()); PB = 0.5 * (p[inB] * g(wr[inB])).sum()
            print("%s %s s=%g: share %.1e bound %.1e bound/share %.1e P_B/P %.2e weight factor %.1f; explicit P %.3e opt P %.3e ratio %.2f" % (mode, name, S[j], share, bound, bound / share if share > 0 else float("nan"), PB / d["P"], (2 * PB / (gBB * p.sum())) / share if share > 0 else float("nan"), a[name][1][j]["P"], d["P"], a[name][1][j]["P"] / d["P"]))
print("=== near-orthogonal modes inside the gap (truss), shares < 1e-5, 250-700 rad/s")
for mode in ("k", "km"):
    a = np.load("results/escape_seq_%s.npy" % mode, allow_pickle=True).item()
    for j in range(len(a["truss"][0])):
        d = a["truss"][0][j]; mdl = truss(); K = mdl["assemble_K"](d["k"]); M = mdl["assemble_M"](d["m"]); lam, Phi = eigh(K, M); p = (Phi.T @ mdl["f"])**2; wr = np.sqrt(lam); sh = p / p.sum()
        sel = (wr > 250) & (wr < 700) & (sh < 1e-5)
        if sel.any(): print("  %s s=%g: %d modes, %.0f-%.0f rad/s, shares %.0e-%.0e" % (mode, S[j], sel.sum(), wr[sel].min(), wr[sel].max(), sh[sel].min(), sh[sel].max()))
for name, mdl in (("chain", chain()), ("truss", truss())):
    for c in ("k", "km"):
        d = np.load("results/%s_%s.npz" % (name, c)); wr, p, mu = modes(mdl, d["k"], d["m"]); sh = p / mu; sel = (wr > 250) & (wr < 700) & (sh < 1e-5)
        if sel.any(): print("  Table1 %s %s: %d modes %.0f-%.0f rad/s shares %.0e-%.0e" % (name, c, sel.sum(), wr[sel].min(), wr[sel].max(), sh[sel].min(), sh[sel].max()))
