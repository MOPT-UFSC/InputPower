"""Numbers quoted in Section 4.3 for the pair designs (results/pairs_*.npz): powers, bars at the bounds, main modes,
static compliance, compliance bound of the lowest excited mode, largest ratio participation/bound, low-frequency bound."""
import numpy as np
from models import chain, truss, modes, excited_gap, results_file
for name, mdl in (("chain", chain()), ("truss", truss())):
    for tag in ("adj", "nonadj"):
        d = np.load(results_file("pairs_%s_%s.npz" % (name, tag))); k, m, f = d["k"], d["m"], d["f"]; K = mdl["assemble_K"](k)
        md = dict(mdl); md["f"] = f; wr, p, mu = modes(md, k, m); sh = p / mu; C = f @ np.linalg.solve(K, f)
        lo, hi = excited_gap(wr, p, mu); top = np.argsort(-p)[:4]
        print("%s %s: P0=%.3e P=%.3e (x%.1f) mu=%.2f gap [%.0f, %.0f]; k at kmin %d, kmax %d; top modes: %s" % (name, tag, float(d["P0"]), float(d["P"]), float(d["P0"]) / float(d["P"]), mu, lo, hi, int(np.sum(k < 1.001e4)), int(np.sum(k > 0.999 * 5e5)), "; ".join("%.0f rad/s (%.2f)" % (wr[i], sh[i]) for i in top)))
        print("   above 800 rad/s: %d modes carry %.2f of mu; compliance f^H K^-1 f/|F|^2 = %.3e m/N; max p_r/(C w_r^2) = %.2f" % (int(np.sum(wr > 800)), sh[wr > 800].sum(), C, np.max(p / (C * wr**2))))
        r0 = np.argmin(wr[sh > 1e-2]); w0 = wr[sh > 1e-2][r0]; p0 = p[sh > 1e-2][r0]
        print("   lowest excited mode %.1f rad/s: p = %.3f, compliance bound C w^2 = %.3f; participation below 60 rad/s %.3f, bound 60^2 C = %.3f; below 100: %.3f vs %.3f" % (w0, p0, C * w0**2, p[wr <= 60].sum(), 3600 * C, p[wr <= 100].sum(), 1e4 * C))
        if name == "chain":
            path = [4, 5, 6, 7] if tag == "nonadj" else [6]; S = np.sum(1 / k[path]); print("   chain path springs %s: sum 1/k = %.3e, path bound of lowest excited mode %.3f, max p/(path bound) over modes %.2f" % (path, S, S * w0**2, np.max(p / (S * wr**2))))
