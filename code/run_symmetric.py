"""Truss with the bar stiffnesses constrained to mirror symmetry about the loaded node (bars only, sum k fixed,
box [1e4, 5e5]). Reduced variables: y_i = total stiffness of mirror pair i (25 pairs) and the central vertical bar
(1 bar, last variable, eliminated by the budget). dK_red,i = (dK_j1 + dK_j2)/w_i so that K = sum_i y_i dK_red,i.
Starts: uniform (mean of k0) and the symmetrised random initial design. SQP multi-start in the reduced space; checks the
exact orthogonality of the antisymmetric modes. Saves results/truss_k_sym.npz (best run)."""
import os, numpy as np
from models import truss, modes, excited_gap, power_and_grads, spectrum, W0
from sqp_driver import run_sqp, multistart

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "results"); NX = 11
t = truss(); bars = [tuple(b) for b in t["bars"]]; nid = lambda ix, iy: iy * NX + ix


def mirror_node(nn): ix, iy = nn % NX, nn // NX; return nid(NX - 1 - ix, iy)


key = {tuple(sorted(b)): i for i, b in enumerate(bars)}
mirror = np.array([key[tuple(sorted((mirror_node(i), mirror_node(j))))] for i, j in bars])
groups = []; seen = set()
for j in range(len(bars)):
    if j in seen: continue
    g = sorted({j, int(mirror[j])}); groups.append(g); seen.update(g)
groups.sort(key=len, reverse=True)          # pairs first, the single self-mirrored bar last (eliminated variable)
w = np.array([len(g) for g in groups], float); nr = len(groups)
assert w[-1] == 1 and np.all(w[:-1] == 2), w


def to_full(y): k = np.zeros(len(bars)); [k.__setitem__(g, y[i] / w[i]) for i, g in enumerate(groups)]; return k
def to_red(k): return np.array([sum(k[j] for j in g) for g in groups])


red = dict(t); red["n_k"] = nr; red["dK"] = [sum(np.asarray(t["dK"][j]) for j in g) / w[i] for i, g in enumerate(groups)]
red["assemble_K"] = lambda y: t["assemble_K"](to_full(y)); red["k0"] = to_red(t["k0"])
kb = (1e4 * w, 5e5 * w)

if __name__ == "__main__":
    starts = [("uniform", np.full(nr, np.mean(t["k0"])) * w, None), ("symmetrised random", to_red(0.5 * (t["k0"] + t["k0"][mirror])), None)]
    y, mm, P, hist, sname, allP = multistart(red, True, False, kb, (1, 1), starts, n_random=50)
    k = to_full(y); wr, p, mu = modes(t, k, t["m0"]); sh = p / mu; meth = "SQP"
    asym = np.max(abs(k - k[mirror])) / k.max(); nanti = int(np.sum(sh < 1e-20)); inside = int(np.sum((sh < 1e-20) & (wr > excited_gap(wr, p, mu)[0]) & (wr < excited_gap(wr, p, mu)[1])))
    print("symmetric, %d starts, %d distinct minima, best from %s: P=%.4e gap %s asym %.1e; modes with share < 1e-20: %d (%d inside the excited gap); k at kmin %d kmax %d"
          % (len(allP), len(set(np.round(np.log10(allP), 3))), sname, P, np.round(excited_gap(wr, p, mu), 1), asym, nanti, inside, int(np.sum(k < 1.001e4)), int(np.sum(k > 0.999 * 5e5))), flush=True)
    lo, hi = excited_gap(wr, p, mu); ws = np.geomspace(5, 3000, 2500)
    np.savez(os.path.join(OUT, "truss_k_sym.npz"), k=k, m=t["m0"], P=P, hist=hist, wr=wr, p=p, mu=mu, lo=lo, hi=hi, spec=spectrum(t, k, t["m0"], ws), ws=ws, start=sname, method=meth, all_P=allP)
    print("best symmetric: P=%.4e (%s, %s) saved to results/truss_k_sym.npz" % (P, sname, meth))
