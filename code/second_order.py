"""Second-order check of the reported constrained designs: for each design, the Hessian of the Lagrangian of log P in the
logarithms of the design variables (central differences of the adjoint gradient; the budget equalities contribute
-lambda diag(exp x)), projected on the tangent space of the active
constraints (budget equalities and bounds at their limits), and the signs of the reduced gradient on the active bounds.
A design is a strict local minimum if the projected Hessian is positive definite and the reduced gradient points into
the box on every active bound. Usage: python second_order.py"""
import numpy as np
from scipy.linalg import null_space
from models import chain, truss, power_and_grads, results_file
KB, MB = (1e4, 5e5), (0.1, 5.0)
def analyse(mdl, k, m, vk, vm, label):
    x = np.r_[np.log(k) if vk else [], np.log(m) if vm else []]; nk = len(k) if vk else 0
    def split(x):
        kk = np.exp(x[:nk]) if vk else k; mm = np.exp(x[nk:]) if vm else m; return kk, mm
    def grad(x):
        kk, mm = split(x); P, gk, gm = power_and_grads(mdl, kk, mm)
        return np.r_[gk * kk / P if vk else [], gm * mm / P if vm else []], P
    g0, P = grad(x); n = len(x); H = np.zeros((n, n)); h = 1e-4
    for i in range(n):
        e = np.zeros(n); e[i] = h; H[:, i] = (grad(x + e)[0] - grad(x - e)[0]) / (2 * h)
    H = 0.5 * (H + H.T)
    # equality constraints: sum k = const, sum m = const (gradients in x: k_j, m_i); active bounds
    A = []; lo = np.zeros(n, bool); hi = np.zeros(n, bool)
    if vk: a = np.zeros(n); a[:nk] = np.exp(x[:nk]); A.append(a); lo[:nk] = np.exp(x[:nk]) < KB[0] * 1.001; hi[:nk] = np.exp(x[:nk]) > KB[1] * 0.999
    if vm: a = np.zeros(n); a[nk:] = np.exp(x[nk:]); A.append(a); lo[nk:] = np.exp(x[nk:]) < MB[0] * 1.001; hi[nk:] = np.exp(x[nk:]) > MB[1] * 0.999
    act = lo | hi; free = ~act
    # multipliers of the equalities from the free variables (least squares), reduced gradient on active bounds
    Af = np.array([a[free] for a in A]).T; lam = np.linalg.lstsq(Af, g0[free], rcond=None)[0] if free.sum() else np.zeros(len(A))
    r = g0 - sum(l * a for l, a in zip(lam, A)); res_free = np.max(abs(r[free])) / np.max(abs(g0)) if free.any() else 0
    sign_ok = np.all(r[lo] >= -1e-6 * np.max(abs(g0))) and np.all(r[hi] <= 1e-6 * np.max(abs(g0)))
    # strict complementarity: smallest bound multiplier (|reduced gradient| on the active bounds) relative to the gradient scale
    strict = min(np.min(abs(r[act])) / np.max(abs(g0)) if act.any() else np.inf, np.inf)
    # Hessian of the Lagrangian: the budget c(x) = sum exp(x_j) - const has Hessian diag(exp(x_j)) in the log variables
    HL = H - sum(l * np.diag(a) for l, a in zip(lam, A))
    # tangent space: free variables, orthogonal to equality gradients
    Z = null_space(np.array([a[free] for a in A])) if free.sum() > len(A) else np.zeros((free.sum(), 0))
    Hf = HL[np.ix_(free, free)]; ev = np.linalg.eigvalsh(Z.T @ Hf @ Z) if Z.shape[1] else np.array([np.inf])
    print("%-32s P=%.4e  free %2d/%2d  residual %.1e  bound signs %s  smallest bound multiplier %.1e  tangent dim %2d  projected Hessian eig min %.2e max %.2e  -> %s" % (
        label, P, free.sum(), n, res_free, "ok" if sign_ok else "VIOLATED", strict, Z.shape[1], ev.min(), ev.max(), "second-order sufficient conditions hold numerically" if ev.min() > 0 and sign_ok and strict > 10 * res_free else "not verified"))
for name, mdl in (("chain", chain()), ("truss", truss())):
    for c, vk, vm in (("k", True, False), ("m", False, True), ("km", True, True)):
        d = np.load(results_file("%s_%s.npz" % (name, c))); analyse(mdl, d["k"], d["m"], vk, vm, "Table 1 %s %s" % (name, c))
