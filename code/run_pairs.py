"""Loads at several points, chain and truss: checks (modal sum, total participation mu_f, sum rule) for five loads per
system, and constrained minimisation with SQP, multi-start (stiffness only, sum k fixed, box [1e4, 5e5]) for the adjacent and non-adjacent
antiphase pairs. Also Lemma 4 (path compliance) check for the non-adjacent pair. Results in results/pairs_<...>.npz."""
import os, numpy as np
from scipy.linalg import eigh
from models import chain, truss, modes, power_and_grads, spectrum, excited_gap, W0, ETA
from sqp_driver import multistart

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "results"); KB = (1e4, 5e5)


def g_of(wr, w=W0, eta=ETA):
    d = eta * wr**2; return w * d / ((wr**2 - w**2)**2 + d**2)


def with_load(model, fvec):
    m = dict(model); m["f"] = fvec; return m


def loads(model):
    n = model["n_dof"]; rng = np.random.default_rng(7)
    if model["name"] == "chain":
        e = lambda i: np.eye(n)[i]
        L = [("point force at $m_6$", e(5)), ("in-phase pair $m_4$, $m_8$", e(3) + e(7)), ("antiphase pair $m_4$, $m_8$", e(3) - e(7)),
             ("adjacent antiphase pair $m_6$, $m_7$", e(5) - e(6)), ("random load at all masses", rng.standard_normal(n))]
    else:
        free = model["free"]; idx = {d: q for q, d in enumerate(free)}
        ev = lambda node: np.eye(n)[idx[2 * node + 1]]   # vertical dof of a node
        ln = model["load_node"]
        L = [("vertical force at node %d" % ln, ev(ln)), ("in-phase pair, nodes %d, %d" % (ln - 2, ln + 2), ev(ln - 2) + ev(ln + 2)),
             ("antiphase pair, nodes %d, %d" % (ln - 2, ln + 2), ev(ln - 2) - ev(ln + 2)), ("adjacent antiphase pair, nodes %d, %d" % (ln, ln + 1), ev(ln) - ev(ln + 1)),
             ("random load at all free dofs", rng.standard_normal(n))]
    return L


def checks(model):
    rows = []
    for name, fv in loads(model):
        md = with_load(model, fv); k, m = md["k0"], md["m0"]
        wr, p, mu = modes(md, k, m); ws = np.geomspace(20, 700, 200); Pd = spectrum(md, k, m, ws)
        Pm = np.array([0.5 * np.sum(p * g_of(wr, w)) for w in ws]); err = np.max(abs(Pm - Pd) / Pd)
        wi = np.geomspace(1e-2, 1e5, 400000); Y = spectrum(md, k, m, wi) * 2   # Re(f^H Y f) = 2P/|.|... P = 1/2 Re(f^H Y f) -> Re = 2P
        integral = np.trapezoid(Y, wi); target = (np.pi - np.arctan(ETA)) / 2 * mu
        rows.append((name, err, mu, integral, target))
        print("%-5s %-42s modal-sum err %.1e  mu_f=%.3f  sum rule %.5f vs %.5f" % (model["name"], name, err, mu, integral, target), flush=True)
    return rows


def minimise_pair(model, tag, fv):
    md = with_load(model, fv); k, m, P, hist, sname, allP = multistart(md, True, False, KB, (1, 1), [("initial", None, None)], n_random=50)
    print("   %d starts, %d distinct minima, best from %s" % (len(allP), len(set(np.round(np.log10(allP), 3))), sname), flush=True)
    wr, p, mu = modes(md, k, m); lo, hi = excited_gap(wr, p, mu); P0 = power_and_grads(md, md["k0"], md["m0"])[0]
    np.savez(os.path.join(OUT, "pairs_%s_%s.npz" % (model["name"], tag)), k=k, m=m, P=P, P0=P0, hist=hist, all_P=allP, start=sname, wr=wr, p=p, mu=mu, lo=lo, hi=hi, f=fv,
             spec=spectrum(md, k, m, np.geomspace(5, 3000, 2500)), ws=np.geomspace(5, 3000, 2500))
    top = np.argsort(-p)[:3]
    print("%-5s %-8s P0=%.3e P=%.3e ratio %.1f mu=%.3f gap [%.1f, %.1f] iters %d; top modes: %s; k at kmin %d, at kmax %d"
          % (model["name"], tag, P0, P, P0 / P, mu, lo, hi, len(hist) - 1, "; ".join("%.0f (%.2f)" % (wr[i], p[i] / mu) for i in top),
             int(np.sum(k < KB[0] * 1.001)), int(np.sum(k > KB[1] * 0.999))), flush=True)
    return md, k


def path_lemma(model, md, k, tag):
    """Lemma 4 for the non-adjacent pair: p_r <= |F|^2 (sum over a path 1/k_l) wr^2, with the shortest path between the two loaded points."""
    wr, p, mu = modes(md, k, md["m0"])
    if model["name"] == "chain":
        path = [4, 5, 6, 7]          # springs between m_4 (index 3) and m_8 (index 7): k_4 .. k_7
    else:
        ln = model["load_node"]; bars = [tuple(b) for b in model["bars"]]
        path = [bars.index(tuple(sorted((ln - 2, ln - 1)))), bars.index(tuple(sorted((ln - 1, ln)))), bars.index(tuple(sorted((ln, ln + 1)))), bars.index(tuple(sorted((ln + 1, ln + 2))))]
    comp = np.sum(1.0 / k[path]); j = np.argmin(wr); jl = np.argsort(wr)[:3]
    print("%-5s %s Lemma 4: path springs %s, sum 1/k = %.3e; lowest excited modes: %s" % (model["name"], tag, path, comp,
          "; ".join("wr=%.1f p=%.3f bound=%.3f" % (wr[i], p[i], comp * wr[i]**2) for i in jl)), flush=True)


if __name__ == "__main__":
    allrows = {}
    for model in (chain(), truss()):
        allrows[model["name"]] = checks(model)
        L = dict(loads(model)); names = list(L)
        md, k = minimise_pair(model, "adj", L[names[3]])
        md2, k2 = minimise_pair(model, "nonadj", L[names[2]]); path_lemma(model, md2, k2, "nonadj")
    np.save(os.path.join(OUT, "pairs_checks.npy"), allrows, allow_pickle=True)
    print("done")
