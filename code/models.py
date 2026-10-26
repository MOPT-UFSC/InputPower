"""Models for the input-power minimisation examples (procedural).
Two structures with the same conventions as gen_data.py / gen_truss.py of the manuscript:
  chain : 11 lumped masses, 12 springs, fixed-fixed, point force at mass 6 (index 5)
  truss : 11 x 2 lumped masses, 51 bars with axial stiffness only, four end nodes fixed, force at top mid node at angle theta
Design variables: spring/bar stiffnesses k (any subset), nodal masses m (any subset).
Damping: hysteretic, Z = K(1 + i eta) - w^2 M.
Each model is a dict with: assemble_K(k), assemble_M(m), dK (list of dK/dk_j), f (load vector), k0, m0, n_k, n_m,
                           mass_dofs (list: for each mass i, the dof indices it multiplies)."""
import numpy as np
from scipy.linalg import eigh

ETA = 0.02
W0 = 300.0


# ------------------------------------------------------------------ chain
def chain(seed=1, n=11, load_index=5):
    nk = n + 1
    dK = []
    for j in range(nk):
        D = np.zeros((n, n))
        if j < n: D[j, j] += 1
        if j > 0: D[j - 1, j - 1] += 1
        if 0 < j < n: D[j, j - 1] -= 1; D[j - 1, j] -= 1
        dK.append(D)
    rng = np.random.default_rng(seed)
    k0 = 1e5 * (1 + 0.2 * rng.random(nk)); m0 = np.ones(n)
    f = np.zeros(n); f[load_index] = 1.0
    return dict(name="chain", n_dof=n, n_k=nk, n_m=n, dK=dK, k0=k0, m0=m0, f=f,
                assemble_K=lambda k: sum(k[j] * dK[j] for j in range(nk)),
                assemble_M=lambda m: np.diag(m),
                mass_dofs=[[i] for i in range(n)], load_mass=load_index)


# ------------------------------------------------------------------ truss
def truss(seed=1, NX=11, NY=2, theta_deg=90.0):
    nodes = np.array([(ix, iy) for iy in range(NY) for ix in range(NX)], float); NN = len(nodes); NDOF = 2 * NN
    nid = lambda ix, iy: iy * NX + ix
    bars = []
    for iy in range(NY):
        for ix in range(NX - 1): bars.append((nid(ix, iy), nid(ix + 1, iy)))
    for ix in range(NX): bars.append((nid(ix, 0), nid(ix, 1)))
    for ix in range(NX - 1): bars.append((nid(ix, 0), nid(ix + 1, 1))); bars.append((nid(ix + 1, 0), nid(ix, 1)))
    bars = np.array(bars); NB = len(bars)
    fix_nodes = [nid(0, 0), nid(0, 1), nid(NX - 1, 0), nid(NX - 1, 1)]
    fix = sorted([2 * q for q in fix_nodes] + [2 * q + 1 for q in fix_nodes]); free = [d for d in range(NDOF) if d not in fix]
    load_node = nid(NX // 2, 1)
    dK = []
    for b, (i, j) in enumerate(bars):
        nvec = (nodes[j] - nodes[i]) / np.linalg.norm(nodes[j] - nodes[i]); kb = np.outer(nvec, nvec)
        D = np.zeros((NDOF, NDOF)); d = [2 * i, 2 * i + 1, 2 * j, 2 * j + 1]
        D[np.ix_(d, d)] += np.block([[kb, -kb], [-kb, kb]]); dK.append(D[np.ix_(free, free)])
    rng = np.random.default_rng(seed)
    k0 = 1e5 * (1 + 0.2 * rng.random(NB)); m0 = np.ones(NN)
    fv = np.zeros(NDOF); t = np.radians(theta_deg); fv[2 * load_node] = np.cos(t); fv[2 * load_node + 1] = np.sin(t); f = fv[free]
    free_idx = {d: q for q, d in enumerate(free)}
    mass_dofs = [[free_idx[d] for d in (2 * i, 2 * i + 1) if d in free_idx] for i in range(NN)]
    def assemble_M(m):
        diag = np.zeros(len(free))
        for i in range(NN):
            for q in mass_dofs[i]: diag[q] = m[i]
        return np.diag(diag)
    return dict(name="truss", n_dof=len(free), n_k=NB, n_m=NN, dK=dK, k0=k0, m0=m0, f=f,
                assemble_K=lambda k: sum(k[b] * dK[b] for b in range(NB)),
                assemble_M=assemble_M, mass_dofs=mass_dofs, load_mass=load_node,
                nodes=nodes, bars=bars, fix_nodes=fix_nodes, load_node=load_node, free=free)


# ------------------------------------------------------------------ response, gradients, modes
def power_and_grads(model, k, m, w=W0, eta=ETA):
    """P(w) and adjoint gradients dP/dk (all springs) and dP/dm (all masses). Real load."""
    K = model["assemble_K"](k); M = model["assemble_M"](m); f = model["f"]
    Z = K * (1 + 1j * eta) - w**2 * M
    u = np.linalg.solve(Z, f)
    P = 0.5 * np.real(f @ (1j * w * u))
    gk = np.array([0.5 * np.real(-1j * w * (1 + 1j * eta) * (u @ D @ u)) for D in model["dK"]])
    gm = np.array([0.5 * np.real(-1j * w * (-w**2) * sum(u[q] * u[q] for q in dofs)) for dofs in model["mass_dofs"]])
    return P, gk, gm


def modes(model, k, m):
    """Natural frequencies, load participations p_r = (phi_r^T f)^2 and total mu_f."""
    lam, Phi = eigh(model["assemble_K"](k), model["assemble_M"](m))
    wr = np.sqrt(np.maximum(lam, 0)); p = (Phi.T @ model["f"])**2
    return wr, p, p.sum()


def spectrum(model, k, m, ws, eta=ETA):
    K = model["assemble_K"](k); M = model["assemble_M"](m); f = model["f"]
    return np.array([0.5 * np.real(f @ (1j * w * np.linalg.solve(K * (1 + 1j * eta) - w**2 * M, f))) for w in ws])


def cluster_participation(wr, p, rtol=1e-6):
    """Groups numerically repeated eigenvalues: consecutive sorted eigenvalues lambda = wr^2 whose relative difference
    is below rtol form one cluster. Returns the frequency of each cluster (its first member) and the participation
    summed over the cluster, p_E, which does not depend on the basis chosen in the eigenspace."""
    order = np.argsort(wr); wr = np.asarray(wr)[order]; p = np.asarray(p)[order]; lam = wr**2
    new = np.r_[True, np.diff(lam) > rtol * np.maximum(lam[1:], 1e-300)]; grp = np.cumsum(new) - 1
    return wr[new], np.bincount(grp, weights=p)


def excited_gap(wr, p, mu, w=W0, thr=1e-2, rtol=1e-6):
    """Edges of the excited gap around w: the nearest natural frequencies below and above w whose normalized
    participation p_E / mu exceeds thr, with the participation summed over clusters of numerically repeated
    eigenvalues (relative tolerance rtol on lambda = wr^2), as defined in the paper (Section 4.2)."""
    wc, pc = cluster_participation(wr, p, rtol); act = pc / mu > thr
    lo = wc[act][wc[act] < w]; hi = wc[act][wc[act] > w]
    return (lo.max() if len(lo) else np.nan), (hi.min() if len(hi) else np.nan)


def count_groups(all_P, rtol=2e-3):
    """Number of groups of distinct final values of the power among the starts (Table 1, column 'groups'): the values
    are sorted, the first value of a group is its reference, the following values join the group while
    P / P_ref - 1 <= rtol, and the first value beyond that limit opens the next group."""
    n = 0; ref = None
    for P in sorted(float(x) for x in all_P):
        if ref is None or P / ref - 1 > rtol: n += 1; ref = P
    return n


def results_dir():
    """Directory of the saved results (results/ next to this file, or code/results/ of the repository when this file is
    the copy in supplement/). The results of the runs reported in the paper come in code/results/compactado.zip."""
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    for d in (os.path.join(here, "results"), os.path.join(here, "..", "code", "results")):
        if os.path.isdir(d): return os.path.normpath(d)
    return os.path.join(here, "results")


def results_file(name):
    """Path of a saved result; stops with instructions when the file is missing."""
    import os, sys
    fn = os.path.join(results_dir(), name)
    if not os.path.exists(fn):
        sys.exit("missing %s\nThe results of the runs reported in the paper are in code/results/compactado.zip: run, from the repository root,\n"
                 "    python -m zipfile -e code/results/compactado.zip code/results\nor produce the file with the script listed in README.md." % fn)
    return fn


def check_gradients(model, k, m, h_k=1.0, h_m=1e-4):
    P, gk, gm = power_and_grads(model, k, m)
    fk = np.array([(power_and_grads(model, k + h_k * np.eye(len(k))[j], m)[0] - power_and_grads(model, k - h_k * np.eye(len(k))[j], m)[0]) / (2 * h_k) for j in range(len(k))])
    fm = np.array([(power_and_grads(model, k, m + h_m * np.eye(len(m))[i])[0] - power_and_grads(model, k, m - h_m * np.eye(len(m))[i])[0]) / (2 * h_m) for i in range(len(m))])
    return np.max(abs(gk - fk)) / np.max(abs(fk)), np.max(abs(gm - fm)) / np.max(abs(fm))
