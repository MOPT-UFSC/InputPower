"""Sequential quadratic programming (SciPy SLSQP) for min P(w0) over stiffnesses and/or masses.
Same interface and return values as mma_driver.run_mma, so the case scripts can switch between the two.

Variables: the logarithms of the design variables (all of them; nothing is eliminated).
Objective: log P, with the analytic adjoint gradient (d log P / d log x = x dP/dx / P).
Budgets (budget="equality"): sum k = K0 and/or sum m = M0 as explicit equality constraints, normalised by K0, M0.
budget=None: box constraints only (escape sequences).
Bounds: per-family scalars or per-variable arrays, in the original variables.
Returns k, m, P, hist (iteration, P, 0, equality residual), info (budget use, box check, SLSQP status)."""
import numpy as np
from scipy.optimize import minimize
from models import power_and_grads


def run_sqp(model, vary_k, vary_m, k_bounds, m_bounds, k_start=None, m_start=None, budget="equality",
            max_iter=3000, ftol=1e-14, verbose=False, log_every=50):
    k0 = np.asarray(model["k0"], float) if k_start is None else np.asarray(k_start, float)
    m0 = np.asarray(model["m0"], float) if m_start is None else np.asarray(m_start, float)
    nk_full, nm_full = len(k0), len(m0); nk = nk_full if vary_k else 0; nm = nm_full if vary_m else 0
    K0, M0 = float(np.sum(model["k0"])), float(np.sum(model["m0"]))
    klo = np.broadcast_to(np.asarray(k_bounds[0], float), (nk_full,)); khi = np.broadcast_to(np.asarray(k_bounds[1], float), (nk_full,))
    mlo = np.broadcast_to(np.asarray(m_bounds[0], float), (nm_full,)); mhi = np.broadcast_to(np.asarray(m_bounds[1], float), (nm_full,))
    kc = np.clip(k0, klo, khi) if vary_k else k0; mc = np.clip(m0, mlo, mhi) if vary_m else m0
    x0 = np.concatenate([np.log(kc[:nk]) if vary_k else [], np.log(mc[:nm]) if vary_m else []])
    bounds = [(np.log(klo[j]), np.log(khi[j])) for j in range(nk)] + [(np.log(mlo[i]), np.log(mhi[i])) for i in range(nm)]

    def split(x):
        kk = np.exp(x[:nk]) if vary_k else k0; mm = np.exp(x[nk:]) if vary_m else m0; return kk, mm

    def fun(x):
        kk, mm = split(x); P, gk, gm = power_and_grads(model, kk, mm)
        g = np.concatenate([gk * kk / P if vary_k else [], gm * mm / P if vary_m else []])
        return np.log(P), g

    cons = []
    if budget == "equality":
        if vary_k:
            cons.append({"type": "eq", "fun": lambda x: np.exp(x[:nk]).sum() / K0 - 1.0,
                         "jac": lambda x: np.concatenate([np.exp(x[:nk]) / K0, np.zeros(nm)])})
        if vary_m:
            cons.append({"type": "eq", "fun": lambda x: np.exp(x[nk:]).sum() / M0 - 1.0,
                         "jac": lambda x: np.concatenate([np.zeros(nk), np.exp(x[nk:]) / M0])})
    hist = []; it = [0]

    def cb(x):
        it[0] += 1; kk, mm = split(x); P = power_and_grads(model, kk, mm)[0]
        res = max([abs(c["fun"](x)) for c in cons] + [0.0]); hist.append((it[0], P, 0.0, res))
        if verbose and it[0] % log_every == 0: print("  it %4d  P=%.4e  eq res %.1e" % (it[0], P, res), flush=True)

    r = minimize(fun, x0, jac=True, method="SLSQP", bounds=bounds, constraints=cons, callback=cb,
                 options=dict(maxiter=max_iter, ftol=ftol))
    kk, mm = split(r.x)
    # SLSQP satisfies the equalities to its own tolerance; a final rescaling of each family removes the residual exactly
    if budget == "equality":
        if vary_k: kk = np.clip(kk * K0 / kk.sum(), klo, khi)
        if vary_m: mm = np.clip(mm * M0 / mm.sum(), mlo, mhi)
    P = power_and_grads(model, kk, mm)[0]; hist.append((it[0] + 1, P, 0.0, 0.0))
    info = {"k_budget_use": kk.sum() / K0 if vary_k else None, "m_budget_use": mm.sum() / M0 if vary_m else None,
            "k_box_ok": bool(np.all(kk >= klo - 1e-9 * khi) and np.all(kk <= khi + 1e-9 * khi)) if vary_k else None,
            "m_box_ok": bool(np.all(mm >= mlo - 1e-9 * mhi) and np.all(mm <= mhi + 1e-9 * mhi)) if vary_m else None,
            "success": bool(r.success), "message": r.message, "nit": int(r.nit)}
    return kk, mm, P, hist, info


def random_start(model, vary_k, vary_m, k_bounds, m_bounds, rng):
    """Log-uniform random design inside the box, scaled family by family to the budget (clip and rescale until both hold)."""
    def draw(x0, lo, hi, total):
        n = len(x0); lo = np.broadcast_to(np.asarray(lo, float), (n,)); hi = np.broadcast_to(np.asarray(hi, float), (n,))
        x = np.exp(rng.uniform(np.log(lo), np.log(hi)))
        for _ in range(200):
            x = np.clip(x, lo, hi); x *= total / x.sum()
            if np.all(x >= lo * 0.999999) and np.all(x <= hi * 1.000001): break
        return np.clip(x, lo, hi)
    k = draw(model["k0"], k_bounds[0], k_bounds[1], model["k0"].sum()) if vary_k else model["k0"].copy()
    m = draw(model["m0"], m_bounds[0], m_bounds[1], model["m0"].sum()) if vary_m else model["m0"].copy()
    return k, m


def multistart(model, vary_k, vary_m, k_bounds, m_bounds, starts, n_random=50, seed=11, **kw):
    """Runs run_sqp from the given named starts [(name, k, m), ...] and from n_random random starts; returns the best
    (k, m, P, hist, name) and the sorted list of all final powers."""
    rng = np.random.default_rng(seed); runs = []
    for name, ks, ms in starts:
        k, m, P, hist, info = run_sqp(model, vary_k, vary_m, k_bounds, m_bounds, ks, ms, **kw); runs.append((P, k, m, hist, name))
    for i in range(n_random):
        ks, ms = random_start(model, vary_k, vary_m, k_bounds, m_bounds, rng)
        k, m, P, hist, info = run_sqp(model, vary_k, vary_m, k_bounds, m_bounds, ks, ms, **kw); runs.append((P, k, m, hist, "random %d" % i))
    runs.sort(key=lambda z: z[0]); P, k, m, hist, name = runs[0]
    return k, m, P, hist, name, np.array([z[0] for z in runs])
