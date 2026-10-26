"""Manuscript cases with SQP (SciPy SLSQP on the logarithms of the variables, explicit budget equalities). Each case is a dict; results go to results/<case>.npz (k, m, P, hist, wr, p, mu, spectrum).
Run:  python run_cases.py [case_name ...]   (no args = all)
Relaxation sweeps use continuation: level i starts from the optimum of level i-1; mass-varying families also try the
'mass-at-load' start and keep the better of the two."""
import os, sys
import numpy as np
from models import chain, truss, power_and_grads, modes, spectrum, excited_gap, check_gradients, W0
from sqp_driver import run_sqp, multistart, random_start

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"); os.makedirs(OUT, exist_ok=True)
WS = np.geomspace(5, 3000, 2500)
KB = (1e4, 5e5)                                   # reference box for stiffness
K_LEVELS = [1e5, 3e4, 1e4, 3e3, 1e3, 3e2]
N_RANDOM = 50
M_LEVELS = {"chain": [(0.9, 1.1), (0.5, 2.0), (0.25, 4.0), (0.125, 8.0), (0.1, 10.0), (0.1, 10.9)],
            "truss": [(0.9, 1.1), (0.5, 2.0), (0.25, 4.0), (0.125, 8.0), (0.1, 10.0), (0.1, 19.9)]}


def mass_at_load_start(model, mb):
    m = np.full(model["n_m"], mb[0]); ln = model["load_mass"]; m[ln] = model["m0"].sum() - mb[0] * (model["n_m"] - 1)
    if m[ln] > mb[1]:
        ex = m[ln] - mb[1]; m[ln] = mb[1]; oth = [i for i in range(model["n_m"]) if i != ln]; m[oth] += ex / len(oth)
    return m


def save(name, model, k, m, P, hist, extra=None):
    wr, p, mu = modes(model, k, m); lo, hi = excited_gap(wr, p, mu)
    np.savez(os.path.join(OUT, name + ".npz"), k=k, m=m, P=P, hist=hist, wr=wr, p=p, mu=mu, lo=lo, hi=hi, spec=spectrum(model, k, m, WS), ws=WS, **(extra or {}))
    print("%-34s P=%.3e  mu=%.3f  gap [%7.1f, %7.1f]  iters=%d" % (name, P, mu, lo, hi, len(hist)), flush=True)


def solve(model, vary_k, vary_m, kb, mb, k_start=None, m_start=None, **kw):
    k, m, P, hist, info = run_sqp(model, vary_k, vary_m, kb, mb, k_start, m_start, verbose=False, **kw)
    if info["k_box_ok"] is False or info["m_box_ok"] is False: print("   WARNING: box violated", info)
    return k, m, P, hist


def case_single(model, tag):
    """Table 1: k only, m only, both, with the reference box; masses in [0.1, 5]. Starts: the initial design, the
    mass-at-load design (mass families) and N_RANDOM log-uniform random designs; the lowest P is kept and all final
    values are saved (all_P)."""
    for vs, vk, vm in [("k", True, False), ("m", False, True), ("km", True, True)]:
        starts = [("initial", None, None)]
        if vm: starts.append(("mass-at-load", None, mass_at_load_start(model, (0.1, 5.0))))
        if vk and not vm:   # best design of the multi-start search over mirror-symmetric designs (run_sym_multistart.py)
            fs = os.path.join(OUT, "sym_best_%s.npz" % model["name"])
            if os.path.exists(fs): starts.append(("symmetric multi-start", np.load(fs)["k"], None))
        k, m, P, hist, sname, allP = multistart(model, vk, vm, KB, (0.1, 5.0), starts, n_random=N_RANDOM)
        nd = len(set(np.round(np.log10(allP), 3)))
        print("   %s %s: %d starts, %d distinct minima, best from %s" % (tag, vs, len(allP), nd, sname), flush=True)
        save("%s_%s" % (tag, vs), model, k, m, P, hist, extra={"start": sname, "all_P": allP})


def case_relax(model, tag):
    """Relaxation sweeps with continuation, three variable sets."""
    for vs, vk, vm in [("k", True, False), ("m", False, True), ("km", True, True)]:
        levels = K_LEVELS if vs == "k" else (M_LEVELS[model["name"]] if vs == "m" else list(zip(K_LEVELS, M_LEVELS[model["name"]])))
        kprev = mprev = None
        for i, lv in enumerate(levels):
            kb = (lv, KB[1]) if vs == "k" else (KB if vs == "m" else (lv[0], KB[1]))
            mb = (1.0, 1.0) if vs == "k" else (lv if vs == "m" else lv[1])
            cands = []
            if i == 0: cands.append(("uniform", None, None))
            else: cands.append(("continuation", kprev, mprev))
            if vm: cands.append(("mass-at-load", None if not vk else (kprev if i else None), mass_at_load_start(model, mb)))
            best = None
            for sname, ks, ms in cands:
                k, m, P, hist = solve(model, vk, vm, kb, mb, ks, ms)
                if best is None or P < best[2]: best = (k, m, P, hist, sname)
            kprev, mprev = best[0], best[1]
            save("%s_relax_%s_L%d" % (tag, vs, i), model, *best[:4], extra={"start": best[4], "level": str(lv)})


def case_truss_directions():
    for th in [90, 0, 45]:
        model = truss(theta_deg=th)
        k, m, P, hist = solve(model, True, False, KB, (1.0, 1.0))
        save("truss_theta%d_k" % th, model, k, m, P, hist)


CASES = {
    "chain_single": lambda: case_single(chain(), "chain"),
    "truss_single": lambda: case_single(truss(), "truss"),
    "chain_relax": lambda: case_relax(chain(), "chain"),
    "truss_relax": lambda: case_relax(truss(), "truss"),
    "truss_directions": case_truss_directions,
}

if __name__ == "__main__":
    names = sys.argv[1:] or list(CASES)
    for mdl in (chain(), truss()):
        ek, em = check_gradients(mdl, mdl["k0"], mdl["m0"]); print("%s gradient check: k %.1e, m %.1e" % (mdl["name"], ek, em))
    for nm in names:
        print("=== case", nm, flush=True); CASES[nm]()
    print("done")
