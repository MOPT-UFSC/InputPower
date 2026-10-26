"""Export the data files (paper3/data/n_*.dat) and the TikZ truss drawings for the rewritten Section 5.
Reads results/*.npz and results/escape_seq_*.npy produced by run_cases.py, run_pairs.py and escape_seq.py."""
import os, numpy as np
from scipy.linalg import eigh
from models import chain, truss, modes, power_and_grads, spectrum, W0, ETA

HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "results")
PAPER = os.path.normpath(os.path.join(HERE, "..", "figures")); D = os.path.join(PAPER, "data"); os.makedirs(D, exist_ok=True)
models = {"chain": chain(), "truss": truss()}
S_LEVELS = [1, 3, 10, 30, 100, 300, 1e3, 3e3, 1e4, 3e4, 1e5]


def decimate(x, cols, tol=0.1, max_gap=40):
    """Indices to keep so that every curve (log10) changes by at most tol between kept points, or at most max_gap steps."""
    L = [np.log10(np.maximum(c, 1e-300)) for c in cols]; keep = [0]; last = 0
    for i in range(1, len(x)):
        if any(abs(l[i] - l[last]) > tol for l in L) or i - last >= max_gap or i == len(x) - 1: keep.append(i); last = i
    return np.array(keep)


def write(name, header, cols):
    np.savetxt(os.path.join(D, name), np.column_stack(cols), header=header, comments="", fmt="%.8e")


def g_of(wr, w=W0, eta=ETA):
    d = eta * wr**2; return w * d / ((wr**2 - w**2)**2 + d**2)


# ------------------------------------------------------------ 1. decomposition with a localised dashpot (chain and truss)
ww = np.linspace(20, 700, 1400)
for name, mdl, dof in (("chain", models["chain"], 8), ("truss", models["truss"], None)):
    K = mdl["assemble_K"](mdl["k0"]); M = mdl["assemble_M"](mdl["m0"]); f = mdl["f"]; n = len(f)
    if dof is None:   # vertical dof of the node three to the left of the loaded node, top chord
        free = mdl["free"]; dof = free.index(2 * (mdl["load_node"] - 3) + 1)
    C = np.zeros((n, n)); C[dof, dof] = 80.0
    lam, Phi = eigh(K, M); Pn = np.zeros_like(ww); tmin = np.zeros_like(ww); tmax = np.zeros_like(ww); Pp = np.zeros_like(ww); errp = 0.0
    for i, w in enumerate(ww):
        u = np.linalg.solve(K - w**2 * M + 1j * w * C, f); Pn[i] = 0.5 * np.real(f @ (1j * w * u))
        t = 0.5 * np.real(1j * w * (Phi.T @ f) * (Phi.T @ M @ u)); tmin[i] = t.min(); tmax[i] = t.max()
        up = np.linalg.solve(K * (1 + 1j * ETA) - w**2 * M, f); Pp[i] = 0.5 * np.real(f @ (1j * w * up))
        tp = 0.5 * (Phi.T @ f)**2 * g_of(np.sqrt(lam), w); errp = max(errp, abs(tp.sum() - Pp[i]) / Pp[i])
    write("n_nonprop_%s.dat" % name, "w P Prmin Prmax", [ww, Pn, tmin, tmax])
    print("%s: proportional modal sum max rel err %.1e; localised dashpot: min P_r/P = %.3g, max = %.3g" % (name, errp, (tmin / Pn).min(), (tmax / Pn).max()))

from run_symmetric import mirror as mirror_truss
# ------------------------------------------------------------ 2. constrained minimization: table rows, relaxation spectra, designs
print("\nconstrained (sum fixed, box [1e4,5e5], m in [0.1,5]):")
for name in ("chain", "truss"):
    mdl = models[name]; P0 = power_and_grads(mdl, mdl["k0"], mdl["m0"])[0]
    for vs in ("k", "m", "km"):
        fn = os.path.join(R, "%s_%s.npz" % (name, vs))
        if not os.path.exists(fn): print("  missing", fn); continue
        d = np.load(fn); sh = d["p"] / float(d["mu"]); near = abs(d["wr"] - W0) < 0.2 * W0
        print("  %-5s %-2s P0=%.3e P=%.3e P0/P=%6.1f mu=%.3f gap [%6.1f, %7.1f] max share within 20%% of w: %.1e  iters %d  start %s  m_load %.2f"
              % (name, vs, P0, float(d["P"]), P0 / float(d["P"]), float(d["mu"]), float(d["lo"]), float(d["hi"]), sh[near].max() if near.any() else 0, len(d["hist"]) - 1, d["start"], d["m"][mdl["load_mass"]]))
        if vs == "k": print("     modes with zero participation (share < 1e-20): %d, of which inside the excited gap %d; mirror asymmetry max|k-k_mirror|/kmax = %.1e; distinct minima %d of %d starts" % (int(np.sum(sh < 1e-20)), int(np.sum((sh < 1e-20) & (d["wr"] > float(d["lo"])) & (d["wr"] < float(d["hi"])))), float(np.max(abs(d["k"] - (d["k"][::-1] if name == "chain" else d["k"][mirror_truss]))) / d["k"].max()), len(set(np.round(np.log10(d["all_P"]), 3))) if "all_P" in d.files else -1, len(d["all_P"]) if "all_P" in d.files else -1))
        elif "all_P" in d.files: print("     distinct minima %d of %d starts" % (len(set(np.round(np.log10(d["all_P"]), 3))), len(d["all_P"])))
        write("n_modes_%s_%s.dat" % (name, vs), "wr share", [d["wr"], sh])
        if name == "chain":
            write("n_design_chain_%s.dat" % vs, "j k_init k_opt", [np.arange(mdl["n_k"]), mdl["k0"] / 1e5, d["k"] / 1e5])
            write("n_mass_chain_%s.dat" % vs, "i m_init m_opt", [np.arange(1, mdl["n_m"] + 1), mdl["m0"], d["m"]])
    # relaxation spectra: w, initial, L0..L5 ; excited modes per level
    for vs in ("k", "m", "km"):
        cols = []; marks = []
        for L in range(6):
            fn = os.path.join(R, "%s_relax_%s_L%d.npz" % (name, vs, L))
            if not os.path.exists(fn): break
            d = np.load(fn); ws = d["ws"]; cols.append(d["spec"]); sh = d["p"] / float(d["mu"]); act = sh > 1e-2
            marks += [(L, w_, s_) for w_, s_ in zip(d["wr"][act], sh[act])]
            marks.append((L, float(d["lo"]), -1)); marks.append((L, float(d["hi"]) if np.isfinite(d["hi"]) else 3000.0, -2))
            print("  %-5s relax %-2s L%d P=%.3e mu=%.3f gap [%6.1f, %7.1f] iters %d" % (name, vs, L, float(d["P"]), float(d["mu"]), float(d["lo"]), float(d["hi"]), len(d["hist"]) - 1))
        if not cols: print("  missing relax", name, vs); continue
        sp0 = spectrum(mdl, mdl["k0"], mdl["m0"], ws); sub = decimate(ws, [sp0], 0.1, 40)
        write("n_relax_%s_%s_P0.dat" % (name, vs), "w P", [ws[sub], sp0[sub]])
        for L, c in enumerate(cols):
            sub = decimate(ws, [c], 0.1, 40); write("n_relax_%s_%s_L%d.dat" % (name, vs, L), "w P", [ws[sub], c[sub]])
        for L in range(len(cols)):
            rows = [(w_, s_) for (L_, w_, s_) in marks if L_ == L and s_ > 0]
            write("n_relaxmodes_%s_%s_L%d.dat" % (name, vs, L), "wr share", [np.array([r[0] for r in rows]), np.array([r[1] for r in rows])])

# ------------------------------------------------------------ 3. pairs
print("\npairs:")
for name in ("chain", "truss"):
    for tag in ("adj", "nonadj"):
        fn = os.path.join(R, "pairs_%s_%s.npz" % (name, tag))
        if not os.path.exists(fn): print("  missing", fn); continue
        d = np.load(fn); sh = d["p"] / float(d["mu"]); write("n_pairs_%s_%s_modes.dat" % (name, tag), "wr share", [d["wr"], sh])
        if name == "chain": write("n_pairs_chain_%s_springs.dat" % tag, "j k_init k_opt", [np.arange(12), models["chain"]["k0"] / 1e5, d["k"] / 1e5])
        print("  %-5s %-6s P0=%.3e P=%.3e ratio %.1f gap [%6.1f, %7.1f]; k at kmin %d, kmax %d" % (name, tag, float(d["P0"]), float(d["P"]), float(d["P0"]) / float(d["P"]), float(d["lo"]), float(d["hi"]), int(np.sum(d["k"] < 1e4 * 1.001)), int(np.sum(d["k"] > 5e5 * 0.999))))

# ------------------------------------------------------------ 4. escape sequences
print("\nescape:")
WSE = np.geomspace(0.1, 1e6, 4000)
for mode in ("k", "m", "km"):
    fn = os.path.join(R, "escape_seq_%s.npy" % mode)
    if not os.path.exists(fn): print("  missing", fn); continue
    Rm = np.load(fn, allow_pickle=True).item(); levels = S_LEVELS[:len(Rm["chain"][0])]
    if mode == "km": levels = levels[:7]   # s <= 1e3: beyond, the modal sum departs from the direct power by > 1e-4 (rounding level of K)
    for name in ("chain", "truss"):
        mdl = models[name]
        for which, tag in ((0, "opt"), (1, "exp")):
            seq = Rm[name][which][:len(levels)]
            P0, mu0 = seq[0]["P"], seq[0]["mu"]
            write("n_escape_%s_%s_%s.dat" % (mode, name, tag), "s P mu lo hi shareB bound Pn mun Pmun",
                  [np.array(levels), np.array([a["P"] for a in seq]), np.array([a["mu"] for a in seq]), np.array([a["lo"] for a in seq]),
                   np.array([a["hi"] if np.isfinite(a["hi"]) else np.nan for a in seq]), np.array([a["shareB"] if a["shareB"] > 0 else np.nan for a in seq]), np.array([a["bound"] for a in seq]),
                   np.array([a["P"] / P0 for a in seq]), np.array([a["mu"] / mu0 for a in seq]), np.array([(a["P"] / a["mu"]) / (P0 / mu0) for a in seq])])
            rows = [(s, w_, s_) for s, a in zip(levels, seq) for w_, s_ in zip(a["wr"], a["sh"]) if s_ > 1e-2]
            write("n_escapemodes_%s_%s_%s.dat" % (mode, name, tag), "s wr share", [np.array([r[0] for r in rows]), np.array([r[1] for r in rows]), np.array([r[2] for r in rows])])
            write("n_escape_%s_%s_%s_Pmu.dat" % (mode, name, tag), "s Pmu", [np.array(levels), np.array([a["P"] / a["mu"] for a in seq])])
        show = [v for v in (1, 10, 100, 1e3, 1e4, 1e5) if v in levels]
        npts = []
        for s in show:
            c = spectrum(mdl, Rm[name][0][levels.index(s)]["k"], Rm[name][0][levels.index(s)]["m"], WSE); sub = decimate(WSE, [c], 0.1, 40); npts.append(len(sub))
            write("n_escapespec_%s_%s_s%d.dat" % (mode, name, int(np.log10(s))), "w P", [WSE[sub], c[sub]])
        print("  spectra %s %s: points kept per curve %s of %d" % (mode, name, npts, len(WSE)))
        a = Rm[name][0][:len(levels)]; print("  %-2s %-5s opt P: %s ; mu: %s" % (mode, name, " ".join("%.2e" % x["P"] for x in a), " ".join("%.1e" % x["mu"] for x in a)))
        print("          gap lo: %s ; hi: %s" % (" ".join("%.2f" % x["lo"] for x in a), " ".join("%.0f" % x["hi"] if np.isfinite(x["hi"]) else "none" for x in a)))
        e = Rm[name][1][:len(levels)]; print("          exp P: %s" % " ".join("%.2e" % x["P"] for x in e))

# ------------------------------------------------------------ 5. TikZ truss drawings: initial (model figure) and constrained designs k / m / km
t = models["truss"]; nodes = t["nodes"]; bars = t["bars"]


def truss_tikz(k, m, label, scale=1.0, arrow=True, kmax=5e5, mmax=5.0):
    s = ["\\begin{scope}[yshift=%.2fcm]" % 0.0]
    for (a, b), kk in zip(bars, k):
        lw = 0.25 + 2.2 * kk / kmax
        s.append("\\draw[line width=%.2fpt, black!80] (%.1f,%.1f)--(%.1f,%.1f);" % (lw, nodes[a, 0], nodes[a, 1], nodes[b, 0], nodes[b, 1]))
    for i, (x, y) in enumerate(nodes):
        r = 0.05 + 0.13 * np.sqrt(m[i] / mmax)
        s.append("\\fill[blue!60!black] (%.1f,%.1f) circle (%.3fcm);" % (x, y, r))
    for i in t["fix_nodes"]:
        s.append("\\draw[fill=black] (%.1f,%.1f) ++(-0.12,-0.2) -- ++(0.24,0) -- ++(-0.12,0.2) -- cycle;" % (nodes[i, 0], nodes[i, 1] - 0.0))
    ln = t["load_node"]
    if arrow: s.append("\\draw[-{Stealth[length=5pt]}, very thick, red!70!black] (%.1f,%.1f)--(%.1f,%.1f);" % (nodes[ln, 0], nodes[ln, 1] + 0.95, nodes[ln, 0], nodes[ln, 1] + 0.22))
    s.append("\\node[font=\\scriptsize, anchor=west] at (-0.3,%.2f) {%s};" % (nodes[:, 1].max() + 0.6, label))
    s.append("\\end{scope}")
    return "\n".join(s)


with open(os.path.join(PAPER, "fig_truss_model.tex"), "w") as fh:
    fh.write("\\begin{tikzpicture}[x=1cm,y=1cm]\n" + truss_tikz(np.full(51, 1.1e5), np.ones(22), "", kmax=5e5) +
             "\n\\node[font=\\small, red!70!black, anchor=south west] at (%.1f,%.2f) {$F\\mathrm{e}^{\\iu\\omega t}$};\n\\end{tikzpicture}\n" % (nodes[t["load_node"], 0] + 0.15, nodes[t["load_node"], 1] + 0.55))
blocks = []
for j, (vs, lab) in enumerate([("k", "(c) truss, bars only"), ("m", "(d) truss, masses only"), ("km", "(e) truss, bars and masses")]):
    fn = os.path.join(R, "truss_%s.npz" % vs)
    if not os.path.exists(fn): continue
    d = np.load(fn); blocks.append("\\begin{scope}[yshift=%.2fcm]\n%s\n\\end{scope}" % (-2.25 * j, truss_tikz(d["k"], d["m"], lab)))
with open(os.path.join(PAPER, "fig_truss_designs.tex"), "w") as fh:
    fh.write("\\begin{tikzpicture}[x=1cm,y=1cm]\n" + "\n".join(blocks) + "\n\\end{tikzpicture}\n")
print("\nwritten data/n_*.dat, fig_truss_model.tex, fig_truss_designs.tex")
