# Modal participation bounds for active input power minimization

Code, results and supplementary data for the manuscript

> O. M. Silva, E. L. Cardoso, M. M. Neves, A. Lenzi, *Modal participation bounds for active input power minimization in vibrating structures*, submitted to the Journal of Sound and Vibration (2026).

The manuscript proves a band participation inequality for the active input power of a harmonic load on a discrete structure with proportional damping, and studies the spectral paths along which the power can vanish. The numerical part illustrates the results on two systems, a fixed-fixed mass-spring chain (11 masses, 12 springs) and a planar truss (11 × 2 lumped masses, 51 bars, 36 degrees of freedom), with the analysis frequency ω = 300 rad/s, hysteretic damping η = 0.02 and a unit point force at the central mass or at the top mid-span node. All minimizations use a single method, SciPy's SLSQP on the logarithms of the design variables, with the stiffness and mass budgets as explicit equality constraints.

Everything here is procedural Python (NumPy, SciPy, mpmath for the 40-digit checks, Matplotlib for the diagnostic plots). The figures of the paper are pgfplots/TikZ, generated from exported data files.

## Layout

```
code/            scripts and the models (run from inside this directory)
code/results/    results of the runs reported in the paper (NumPy .npz/.npy)
logs/            console output of those runs, as cited in the paper
supplement/      supplementary material: final designs of every reported case as CSV, and scripts to recompute them
figures/         data files and generators of the pgfplots figures; figures/figs/ has the rendered PDFs
```

## Installation

```
pip install -r requirements.txt
```

Python ≥ 3.9. The figure pipeline additionally needs a LaTeX distribution with pgfplots and TikZ.

## The two models (`code/models.py`)

`chain()` and `truss()` return a dictionary with the initial stiffnesses `k0` (k_j = 10^5 (1 + 0.2 ξ_j) N/m, ξ_j uniform, NumPy default generator with seed 1), the masses `m0` (1 kg each), the load vector `f`, the assembly functions `assemble_K`, `assemble_M` and the element matrices `dK` (one per spring or bar). `power_and_grads(model, k, m)` returns the active input power and its adjoint gradients with respect to every stiffness and mass (Eq. (gradient) of the paper); `modes` returns the natural frequencies, the participations p_r = (φ_rᵀ f)² and their total μ_f = fᴴ M⁻¹ f; `excited_gap` returns the gap of natural frequencies around ω whose normalized participation exceeds 10⁻² (summed over numerically repeated eigenvalues); `spectrum` returns P(ω) on a grid.

`code/sqp_driver.py` has the optimizer: `run_sqp` (SLSQP on log k, log m, objective log P with the analytic gradient, bounds as box, budgets Σk/K₀ − 1 = 0 and Σm/M₀ − 1 = 0 as equalities, ftol 10⁻¹⁴, 3000 iterations, final exact rescaling of each family), `random_start` (log-uniform design in the box, scaled to the budgets) and `multistart`.

## The examples, script by script

Each entry gives the section, table or figure of the paper, the script, its arguments, the results it writes and the log of the run used in the paper.

| Paper | Script | Results | Log |
|---|---|---|---|
| Section 4.1, gradient check and Fig. 3 (modal decomposition, proportional and non-proportional damping) | `export_paper.py` (gradient check is printed at the start of `run_cases.py`) | `figures/data/n_terms_truss.dat`, `n_nonprop_*.dat` (chain terms of Fig. 3(a): `figures/gen_weight_decomp.py`) | `export_paper.log` |
| Section 4.1, modal sum, participation total and sum rule for five loads | `section4_numbers.py` | printed | `section4_numbers.log` |
| Remark 1 (Caughey series, prescribed modal ratios) and the dashpot counterexample of Section 5 | `classical_damping.py` | printed | `classical_damping.log` |
| **Table 1** and Fig. 4: constrained minimization with stiffness, mass or both; 52 starts per problem | `run_cases.py chain_single truss_single` | `results/chain_k.npz`, `chain_m.npz`, `chain_km.npz`, `truss_*.npz` (keys below) | `sqp_single.log` |
| Symmetric start used in Table 1 for the stiffness-only problems (200 random starts over mirror-symmetric designs) | `run_sym_multistart.py 200` (uses `run_symmetric.py`) | `results/sym_best_chain.npz`, `sym_best_truss.npz` | in `sqp_single.log` |
| Section 4.2, 300 random starts in the full space (stiffness only) | `run_fullspace_starts.py` | `results/fullspace_chain.npy`, `fullspace_truss.npy` | `fullspace.log` |
| Section 4.2, displacement objectives \|u₆\| and \|u₉\| on the chain, 31 starts each | `run_disp.py` | `results/disp_u6.npz`, `disp_u9.npz` | `sqp_disp.log` |
| **Fig. 5**: relaxation sweeps by continuation (six levels of the bounds, three variable sets) | `run_cases.py chain_relax truss_relax` | `results/*_relax_{k,m,km}_L0..L5.npz` | `sqp_relax.log` |
| Section 4.3 and **Fig. 6**: adjacent and non-adjacent antiphase pairs, 51 starts | `run_pairs.py`, numbers quoted in the text by `pairs_numbers.py` | `results/pairs_{chain,truss}_{adj,nonadj}.npz`, `pairs_checks.npy` | `sqp_pairs.log` |
| Section 4.4 and **Figs. 7, 8, 9**: unbounded escape sequences, s = 1 … 10⁵, optimized and explicit designs | `escape_seq.py k`, `escape_seq.py m`, `escape_seq.py km` | `results/escape_seq_{k,m,km}.npy` | `sqp_escape_{k,m,km}.log` |
| **Table 2**: verification of ten representative designs (eigenproblem residuals, mass orthogonality, modal sum, dissipative form, 40-digit recomputation) | `verify_extreme.py` | `code/verify_extreme_table.tex` (rows of Table 2), printed | `verify_extreme.log` |
| Section 4.4, error of the modal sum along the sequences | `modal_errors.py` | printed | `modal_errors.log` |
| Section 4.1, first-order residuals of every constrained design | `residuals_all.py` | printed | `residuals_all.log` |
| Section 4.1, second-order check of the Table 1 designs (projected Hessian of the Lagrangian, strict complementarity) | `second_order.py` | printed | `second_order.log` |
| Section 4.5: minimization with a truncated modal basis (participation threshold 10⁻² or 10⁻³) | `escape_trunc.py k 1e-2`, `escape_trunc.py k 1e-3`, `escape_trunc.py km 1e-2` | `results/escape_trunc_{k,km}_thr*.npy` | `trunc_{k,km}_1e-*.log` |
| Supplementary material | `export_supplement.py`, checked by `supplement_check.py` | `../supplement/*.csv` | |

Diagnostic plots (not in the paper): `escape_spectra.py k|m|km` draws the input-power spectra of the escape sequences with Matplotlib; `summarise.py [prefix]` tabulates every `results/*.npz`.

`run_cases.py` also has a case `truss_directions` (point force at 0°, 45° and 90°) that is not reported in the paper.

### Result file formats

`results/<system>_<vars>.npz` and `<system>_relax_<vars>_L<i>.npz` (Table 1, Fig. 5): `k`, `m` final design; `P` final power; `hist` iteration history (iteration, P, 0, equality residual); `wr`, `p`, `mu` natural frequencies, participations and their total; `lo`, `hi` edges of the excited gap; `spec`, `ws` input-power spectrum P(ω) on the grid `ws`; `start` name of the start that gave the best design; `all_P` final powers of all starts (sorted); `level` index of the relaxation level.

`results/pairs_*.npz`: as above plus `f` (load vector) and `P0` (initial power). `results/disp_*.npz`: `disp`, `disp0` final and initial |u_j|, `u` the complex u_j, `all_disp`, `all_P`, `all_lo`, `all_hi` over the 31 starts. `results/sym_best_*.npz`: best mirror-symmetric design and `all_P` over the 200 reduced-space starts.

`results/escape_seq_<vars>.npy`: a dictionary `{"chain": (opt, exp), "truss": (opt, exp)}`; `opt` and `exp` are lists over the levels s = 1, 3, 10, …, 10⁵ of dictionaries with `P`, `mu`, `lo`, `hi`, `shareB` (participation share in B = [ω/2, 2ω]), `bound` (2P/(g_B μ)), `wr`, `sh` (normalized participations), `k`, `m`, `start`, `nit`, `other_P` (power of the other start). Load with `np.load(path, allow_pickle=True).item()`.

`results/escape_trunc_<vars>_thr<t>.npy`: list over the levels of dictionaries with `s`, `k`, `m`, `P_model` (power seen by the truncated model), `P_full` (true power), `w_peak`, `share_peak`, `wr`, `sh`, `hist`, `nS` (number of retained modes). `results/fullspace_*.npy`: the 300 final powers.

## Reproducing the runs

From `code/`, about one hour on two cores for the whole set:

```
OMP_NUM_THREADS=2 python run_sym_multistart.py 200
OMP_NUM_THREADS=2 python run_cases.py chain_single truss_single chain_relax truss_relax
python run_pairs.py
python escape_seq.py k ; python escape_seq.py m ; python escape_seq.py km
python run_disp.py ; python run_fullspace_starts.py
python escape_trunc.py k 1e-2 ; python escape_trunc.py k 1e-3 ; python escape_trunc.py km 1e-2
python verify_extreme.py ; python second_order.py ; python residuals_all.py ; python modal_errors.py
python section4_numbers.py ; python pairs_numbers.py ; python classical_damping.py
python export_paper.py ; python export_supplement.py ; python supplement_check.py
```

`run_cases.py` reads `results/sym_best_*.npz` to include the symmetric start, so `run_sym_multistart.py` goes first. The random starts use fixed seeds (`multistart` seed 11, `run_disp.py` seed 3, `run_fullspace_starts.py` seed 5, `run_sym_multistart.py` and the random load of `run_pairs.py` seed 7), so the runs are reproducible up to the floating-point differences between BLAS implementations, which can change the iterates of SLSQP near the tolerance and hence the number of distinct final values among the starts.

## Figures

`figures/` holds the pgfplots sources `fig_*.tex` as included in the manuscript, the data files they read (`data/n_*.dat`, written by `code/export_paper.py`; `data/g_curves.dat` and `data/ex1_power.dat`, written by `figures/gen_weight_decomp.py`), the generator `gen_figs.py` that writes `fig_relax.tex`, `fig_escape_freq.tex`, `fig_escape_P.tex`, `fig_escape_spec.tex`, `fig_decomp.tex` and `fig_designs_chain.tex` from the data (the two truss drawings are written by `export_paper.py`), and standalone wrappers in `figs/` to render each figure to PDF:

```
cd figures && python gen_weight_decomp.py && python gen_figs.py
cd figs && for n in fig_relax fig_escape_freq fig_escape_P fig_escape_spec fig_pairs fig_decomp; do TEXINPUTS=..: pdflatex -jobname=$n ${n}_standalone.tex; done
```

## Supplementary material

`supplement/` contains the final stiffness and mass vectors of every design reported in the paper as CSV files (one row per spring, bar or mass, with its index), the load vectors of the pair examples, and the scripts that rebuild the two systems and recompute the power, the participations, the gaps and the checks of Table 2 from those vectors. See `supplement/README.md`.

## Acknowledgement of AI assistance

Claude (Anthropic) was used to assist in the development of the code and in its organization into this repository. The formulation, the numerical experiments and the verification of the results are the responsibility of the authors.

## License

MIT, see `LICENSE`.
