# Modal participation bounds for active input power minimization

Code, results and supplementary data for the manuscript (repository: https://github.com/MOPT-UFSC/InputPower)

> O. M. Silva, E. L. Cardoso, M. M. Neves, A. Lenzi, *Modal participation bounds for active input power minimization in vibrating structures*, submitted to the Journal of Sound and Vibration (2026).

The manuscript proves a band participation inequality for the active input power of a harmonic load on a discrete structure with proportional damping, and studies the spectral paths along which the power can vanish. The numerical part illustrates the results on two systems, a fixed-fixed mass-spring chain (11 masses, 12 springs) and a planar truss (11 × 2 lumped masses, 51 bars, 36 degrees of freedom), with the analysis frequency ω = 300 rad/s, hysteretic damping η = 0.02 and a unit point force at the central mass or at the top mid-span node. All minimizations use a single method, SciPy's SLSQP on the logarithms of the design variables, with the stiffness and mass budgets as explicit equality constraints.

Everything here is procedural Python (NumPy, SciPy, mpmath for the 40-digit checks, Matplotlib for the diagnostic plots). The figures of the paper are pgfplots/TikZ, generated from exported data files.

## Layout

```
code/                     scripts and the models; every script is run from inside this directory
code/results/             compactado.zip: results of the runs reported in the paper (NumPy .npz/.npy); extract before use
logs/                     console output of the runs reported in the paper, and recount_groups.log (see below)
supplement/               data_.zip: final designs of every reported case as CSV; copies of the check scripts; README.md
figures/                  pgfplots sources fig_*.tex of the paper, their generators, data.zip with the data files they read
figures/figs/             standalone wrappers that render each figure to PDF (output written here)
figures/rendered/         the PDFs of the figures as they appear in the submitted manuscript (provided, for comparison)
```

Provided files are the archives, the scripts, the logs, `figures/*.tex`, `figures/rendered/*.pdf` and `code/verify_extreme_table.tex` (the rows of Table 2). Everything extracted from an archive or written by a script (`code/results/*.npz`, `*.npy`, `supplement/*.csv`, `figures/data/*.dat`, `figures/figs/*.pdf`) is listed in `.gitignore`: the archives are the versioned copies, see "Updating the archives" below.

## Installation

```
pip install -r requirements.txt
```

Python ≥ 3.9. The figure pipeline additionally needs a LaTeX distribution with pgfplots and TikZ (TeX Live 2022 or later; `pdflatex` is enough).

## Data preparation

The results, the supplement designs and the figure data are stored as ZIP archives. From the root of the repository:

```
python -m zipfile -e code/results/compactado.zip code/results
python -m zipfile -e supplement/data_.zip supplement
python -m zipfile -e figures/data.zip figures/data
```

The first archive holds 60 files (`code/results/*.npz`, `*.npy`), the second 134 CSV files (`supplement/*.csv`), the third 169 data files (`figures/data/*.dat`). A script that needs a result that is not there stops with the extraction command above, or names the script that produces the file.

## Two ways of using the repository

**Checking the published results** uses only the saved designs and takes minutes. From `code/`:

| Check | Command | What it does |
|---|---|---|
| Table 1 designs | `python supplement_check.py 'table1*'` | recomputes, from the CSV files, the direct power, the modal sum, the total participation μ_f and the excited gap of the six designs (and of any other design with another pattern, or all of them with no argument) |
| Groups of final power values (column "groups" of Table 1) | `python recount_groups.py` | recounts the groups from the saved `all_P` arrays with the 0.2 % rule; prints the earlier rounding rule alongside |
| Numbers quoted in Section 4 | `python section4_numbers.py`, `python pairs_numbers.py` | gaps at three thresholds, Corollary 9 intervals, near-orthogonal modes, pair designs |
| First- and second-order conditions | `python residuals_all.py`, `python second_order.py` | residuals of every constrained design; projected Hessian of the Table 1 designs |
| Table 2 | `python verify_extreme.py` | eigenproblem residuals, modal sum, dissipative form and 40-digit recomputation of ten designs of the escape sequences; writes `verify_extreme_table.tex` |
| Modal-sum error along the escape sequences | `python modal_errors.py` | |
| Remark 1 and the dashpot counterexample of Section 5 | `python classical_damping.py` | needs no saved result |
| Figure data and figures | `python export_paper.py`, then see "Figures" | rewrites `figures/data/n_*.dat` and the two truss drawings from the saved results |

**Rerunning the optimizations** regenerates `code/results/` and takes about one hour on two cores. From `code/`:

```
OMP_NUM_THREADS=2 python run_sym_multistart.py 200          # symmetric start of Table 1 -> results/sym_best_*.npz
OMP_NUM_THREADS=2 python run_cases.py chain_single truss_single chain_relax truss_relax
python run_pairs.py
python escape_seq.py k ; python escape_seq.py m ; python escape_seq.py km
python run_disp.py ; python run_fullspace_starts.py
python escape_trunc.py k 1e-2 ; python escape_trunc.py k 1e-3 ; python escape_trunc.py km 1e-2
python export_paper.py ; python export_supplement.py
```

`run_cases.py` reads `results/sym_best_*.npz` to include the symmetric start, so `run_sym_multistart.py` goes first (it prints a note when the file is absent). The random starts use fixed seeds (`multistart` seed 11, `run_disp.py` seed 3, `run_fullspace_starts.py` seed 5, `run_sym_multistart.py` and the random load of `run_pairs.py` seed 7), so the runs are reproducible up to the floating-point differences between BLAS implementations, which can change the iterates of SLSQP near the tolerance and hence the number of groups of final values among the starts. The checks above can then be repeated on the new results.

## The two models (`code/models.py`)

`chain()` and `truss()` return a dictionary with the initial stiffnesses `k0` (k_j = 10^5 (1 + 0.2 ξ_j) N/m, ξ_j uniform, NumPy default generator with seed 1), the masses `m0` (1 kg each), the load vector `f`, the assembly functions `assemble_K`, `assemble_M` and the element matrices `dK` (one per spring or bar). `power_and_grads(model, k, m)` returns the active input power and its adjoint gradients with respect to every stiffness and mass (Eq. (gradient) of the paper); `modes` returns the natural frequencies, the participations p_r = (φ_rᵀ f)² and their total μ_f = fᴴ M⁻¹ f; `spectrum` returns P(ω) on a grid.

Two definitions used throughout the code and the paper live in this file:

- `excited_gap(wr, p, mu)`: the nearest natural frequencies below and above ω whose normalized participation exceeds 10⁻², with the participation summed over clusters of numerically repeated eigenvalues (`cluster_participation`, relative tolerance 10⁻⁶ on λ = ω_r²) before the threshold is applied, so that the result does not depend on the basis chosen in a repeated eigenspace. No repeated eigenvalues occur in the designs of Table 1 at that tolerance.
- `count_groups(all_P)`: the number of groups of distinct final values of the power among the starts (column "groups" of Table 1): the values are sorted, the first value of a group is its reference, the following values join the group while P/P_ref − 1 ≤ 0.2 %, and the first value beyond that limit opens the next group. The campaign logs in `logs/` were written with an earlier rule (rounding log₁₀ P to three decimals), which counts slightly more groups; `logs/recount_groups.log` has both counts for every saved `all_P`, and the paper reports the 0.2 % rule.

`code/sqp_driver.py` has the optimizer: `run_sqp` (SLSQP on log k, log m, objective log P with the analytic gradient, bounds as box, budgets Σk/K₀ − 1 = 0 and Σm/M₀ − 1 = 0 as equalities, ftol 10⁻¹⁴, 3000 iterations, final exact rescaling of each family), `random_start` (log-uniform design in the box, scaled to the budgets) and `multistart`.

## The examples, script by script

Each entry gives the section, table or figure of the paper, the script, its arguments, the results it writes and the log of the run used in the paper.

| Paper | Script | Results | Log |
|---|---|---|---|
| Section 4.1, gradient check and Fig. 3 (modal decomposition, proportional and non-proportional damping) | `export_paper.py` (gradient check is printed at the start of `run_cases.py`) | `figures/data/n_terms_*.dat`, `n_nonprop_*.dat` | `export_paper.log` |
| Section 4.1, modal sum, participation total and sum rule for five loads | `section4_numbers.py` | printed | `section4_numbers.log` |
| Remark 1 (Caughey series, prescribed modal ratios) and the dashpot counterexample of Section 5 | `classical_damping.py` | printed | `classical_damping.log` |
| **Table 1** and Fig. 4: constrained minimization with stiffness, mass or both; 52 starts per problem | `run_cases.py chain_single truss_single` | `results/chain_k.npz`, `chain_m.npz`, `chain_km.npz`, `truss_*.npz` (keys below) | `sqp_single.log`; groups recounted in `recount_groups.log` |
| Symmetric start used in Table 1 for the stiffness-only problems (200 random starts over mirror-symmetric designs) | `run_sym_multistart.py 200` (uses `run_symmetric.py`) | `results/sym_best_chain.npz`, `sym_best_truss.npz` | in `sqp_single.log` |
| Section 4.2, 300 random starts in the full space (stiffness only) | `run_fullspace_starts.py` | `results/fullspace_chain.npy`, `fullspace_truss.npy` | `fullspace.log` |
| Section 4.2, displacement objectives \|u₆\| and \|u₉\| on the chain, 31 starts each | `run_disp.py` | `results/disp_u6.npz`, `disp_u9.npz` | `sqp_disp.log` |
| **Fig. 5**: relaxation sweeps by continuation (six levels of the bounds, three variable sets) | `run_cases.py chain_relax truss_relax` | `results/*_relax_{k,m,km}_L0..L5.npz` | `sqp_relax.log` |
| Section 4.3 and **Fig. 6**: adjacent and non-adjacent antiphase pairs, 51 starts | `run_pairs.py`, numbers quoted in the text by `pairs_numbers.py` | `results/pairs_{chain,truss}_{adj,nonadj}.npz`, `pairs_checks.npy` | `sqp_pairs.log` |
| Section 4.4 and **Figs. 7, 8, 9**: unbounded escape sequences, s = 1 … 10⁵, optimized and explicit designs | `escape_seq.py k`, `escape_seq.py m`, `escape_seq.py km` | `results/escape_seq_{k,m,km}.npy` | `sqp_escape_{k,m,km}.log` |
| **Table 2**: verification of ten representative designs | `verify_extreme.py` | `code/verify_extreme_table.tex`, printed | `verify_extreme.log` |
| Section 4.4, error of the modal sum along the sequences | `modal_errors.py` | printed | `modal_errors.log` |
| Section 4.1, first-order residuals of every constrained design | `residuals_all.py` | printed | `residuals_all.log` |
| Section 4.1, second-order check of the Table 1 designs (projected Hessian of the Lagrangian, strict complementarity) | `second_order.py` | printed | `second_order.log` |
| Section 4.5: minimization with a truncated modal basis (participation threshold 10⁻² or 10⁻³) | `escape_trunc.py k 1e-2`, `escape_trunc.py k 1e-3`, `escape_trunc.py km 1e-2` | `results/escape_trunc_{k,km}_thr*.npy` | `trunc_{k,km}_1e-*.log` |
| Supplementary material | `export_supplement.py` (writes to `../supplement/`), checked by `supplement_check.py` | `supplement/*.csv` | |

Diagnostic plots (not in the paper): `escape_spectra.py k|m|km` draws the input-power spectra of the escape sequences with Matplotlib; `summarise.py [prefix]` tabulates every `results/*.npz`. `run_cases.py` also has a case `truss_directions` (point force at 0°, 45° and 90°) that is not reported in the paper.

### Result file formats

`results/<system>_<vars>.npz` and `<system>_relax_<vars>_L<i>.npz` (Table 1, Fig. 5): `k`, `m` final design; `P` final power; `hist` iteration history (iteration, P, 0, equality residual); `wr`, `p`, `mu` natural frequencies, participations and their total; `lo`, `hi` edges of the excited gap; `spec`, `ws` input-power spectrum P(ω) on the grid `ws`; `start` name of the start that gave the best design; `all_P` final powers of all starts (sorted); `level` index of the relaxation level.

`results/pairs_*.npz`: as above plus `f` (load vector) and `P0` (initial power). `results/disp_*.npz`: `disp`, `disp0` final and initial |u_j|, `u` the complex u_j, `all_disp`, `all_P`, `all_lo`, `all_hi` over the 31 starts. `results/sym_best_*.npz`: best mirror-symmetric design and `all_P` over the 200 reduced-space starts.

`results/escape_seq_<vars>.npy`: a dictionary `{"chain": (opt, exp), "truss": (opt, exp)}`; `opt` and `exp` are lists over the levels s = 1, 3, 10, …, 10⁵ of dictionaries with `P`, `mu`, `lo`, `hi`, `shareB` (participation share in B = [ω/2, 2ω]), `bound` (2P/(g_B μ)), `wr`, `sh` (normalized participations), `k`, `m`, `start`, `nit`, `other_P` (power of the other start). Load with `np.load(path, allow_pickle=True).item()`.

`results/escape_trunc_<vars>_thr<t>.npy`: list over the levels of dictionaries with `s`, `k`, `m`, `P_model` (power seen by the truncated model), `P_full` (true power), `w_peak`, `share_peak`, `wr`, `sh`, `hist`, `nS` (number of retained modes). `results/fullspace_*.npy`: the 300 final powers.

## Figures

`figures/` holds the pgfplots sources `fig_*.tex` as included in the manuscript and the scripts that write them. The data files they read are all produced by two scripts: `code/export_paper.py` writes `figures/data/n_*.dat` from the saved results (among them `n_terms_chain.dat` and `n_terms_truss.dat`, the modal terms of Fig. 3(a, b), and `n_nonprop_*.dat`, its panels (c, d)) and the two truss drawings `fig_truss_model.tex` and `fig_truss_designs.tex`; `figures/gen_weight_curves.py` writes `figures/data/g_curves.dat` (the weight curves of Fig. 1). `figures/gen_figs.py` then writes `fig_relax.tex`, `fig_escape_freq.tex`, `fig_escape_P.tex`, `fig_escape_spec.tex`, `fig_decomp.tex` and `fig_designs_chain.tex` from the data (`fig_pairs.tex` is hand-written and reads `n_pairs_*.dat`). The standalone wrappers in `figures/figs/` render each figure to a PDF in that directory:

```
cd code && python export_paper.py                      # or extract figures/data.zip instead
cd ../figures && python gen_weight_curves.py && python gen_figs.py
cd figs && for n in fig_relax fig_escape_freq fig_escape_P fig_escape_spec fig_pairs fig_decomp; do pdflatex -interaction=nonstopmode -jobname=$n ${n}_standalone.tex; done
```

The wrappers read `../fig_*.tex` and `../data/`, so no `TEXINPUTS` setting is needed. The PDFs generated this way land in `figures/figs/`; the ones in the submitted manuscript are provided in `figures/rendered/` for comparison. (Fig. 1, the weight curves, Fig. 2, the two systems, and Fig. 4, the designs, are drawn directly in the manuscript from `g_curves.dat`, `fig_truss_model.tex`, `fig_designs_chain.tex` and `fig_truss_designs.tex`.)

## Supplementary material

`supplement/` contains, in `data_.zip`, the final stiffness and mass vectors of every design reported in the paper as CSV files (one row per spring, bar or mass, with its index) and the load vectors of the pair examples, together with copies of the scripts that rebuild the two systems and recompute the power, the participations, the gaps and the checks of Table 2 from those vectors. The copies are identical to the files in `code/` and find the results in `code/results/` when run from `supplement/`. See `supplement/README.md`.

## Updating the archives

After a rerun, the archives are rebuilt from the root of the repository with

```
python -m zipfile -c code/results/compactado.zip code/results/*.npz code/results/*.npy
python -m zipfile -c supplement/data_.zip supplement/*.csv
python -m zipfile -c figures/data.zip figures/data/*.dat
```

(`python -m zipfile -c` stores each file under its base name, without directories, which is the layout the extraction commands of "Data preparation" expect.)

## Acknowledgement of AI assistance

Claude (Anthropic) was used to assist in the development of the code and in its organization into this repository. The formulation, the numerical experiments and the verification of the results are the responsibility of the authors.

## License

MIT, see `LICENSE`.
