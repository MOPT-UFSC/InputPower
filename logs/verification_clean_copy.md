# Verification in a clean copy of the repository (2026-10-26)

Clean copy = the versioned files only (archives, scripts, logs, figure sources). Commands as in README.md; outputs abridged.

## 1. Data preparation (repository root)
```
$ python -m zipfile -e code/results/compactado.zip code/results
$ python -m zipfile -e supplement/data_.zip supplement
$ python -m zipfile -e figures/data.zip figures/data
$ ls code/results | wc -l ; ls supplement/*.csv | wc -l ; ls figures/data | wc -l
61
134
175
```
(61 = 60 result files plus the archive; 175 = the 169 figure data files plus six stale files of the first archive, removed when the archive was rebuilt in step 4.)

## 2. Table 1 designs recomputed from the CSV files (code/)
```
$ python supplement_check.py 'table1*'
table1_chain_k.csv                            P_direct 2.5991e-06  |modal-direct|/P 3.9e-15  mu 1.0000  gap [126, 1226]
table1_chain_km.csv                           P_direct 2.2149e-07  |modal-direct|/P 1.0e-13  mu 0.2000  gap [75, 1916]
table1_chain_m.csv                            P_direct 1.5745e-06  |modal-direct|/P 7.9e-14  mu 0.2000  gap [215, 672]
table1_truss_k.csv                            P_direct 3.0358e-06  |modal-direct|/P 2.1e-15  mu 1.0000  gap [137, 603]
table1_truss_km.csv                           P_direct 1.0593e-07  |modal-direct|/P 1.4e-13  mu 0.2000  gap [76, 884]
table1_truss_m.csv                            P_direct 9.1348e-07  |modal-direct|/P 6.1e-14  mu 0.2000  gap [177, 558]
```

## 3. Recount of the groups of final power values (code/)
```
$ python recount_groups.py
design                       starts log10-round  0.2% rule
Table 1 chain stiffness          52         39         37
Table 1 chain masses             52         28         26
Table 1 chain stiffness, masses     52         32         28
Table 1 truss stiffness          52         46         44
Table 1 truss masses             52         38         34
Table 1 truss stiffness, masses     52         51         47
pairs chain adj                  51         33         31
pairs chain nonadj               51         43         42
pairs truss adj                  51         49         48
pairs truss nonadj               51         47         47
chain min |u6| (on |u|)          31         19         18
chain min |u9| (on |u|)          31         27         26
symmetric search chain          200         24         20
300 full-space starts chain     300         95         88
symmetric search truss          200        138        129
300 full-space starts truss     300        226        194
```

## 4. Export of the figure data from the saved results (code/), compared with the extracted figures/data
```
$ mv figures/data figures/data_from_zip ; python export_paper.py
          exp P: 9.04e-05 1.30e-05 2.89e-07 2.91e-09 8.15e-11 3.03e-12 8.18e-14

written data/n_*.dat, fig_truss_model.tex, fig_truss_designs.tex
$ python gen_weight_curves.py   (figures/)
$ diff -rq figures/data figures/data_from_zip
Only in figures/data_from_zip: n_modes_truss_k_sym.dat
Only in figures/data_from_zip: n_relaxmodes_chain_k.dat
Only in figures/data_from_zip: n_relaxmodes_chain_m.dat
Only in figures/data_from_zip: n_relaxmodes_truss_k.dat
Only in figures/data_from_zip: n_relaxmodes_truss_km.dat
Only in figures/data_from_zip: n_relaxmodes_truss_m.dat
```

## 5. Figure sources regenerated and compared with the versioned ones (figures/)
```
$ cp fig_*.tex /tmp/fig_versioned/ ; python gen_figs.py ; diff
figures written
same fig_decomp.tex
same fig_designs_chain.tex
same fig_escape_P.tex
same fig_escape_freq.tex
same fig_escape_spec.tex
same fig_pairs.tex
same fig_relax.tex
same fig_truss_designs.tex
same fig_truss_model.tex
```

## 6. Figures compiled (figures/figs/)
```
$ for n in fig_relax fig_escape_freq fig_escape_P fig_escape_spec fig_pairs fig_decomp; do pdflatex -interaction=nonstopmode -jobname=$n ${n}_standalone.tex; done
fig_relax: exit 0, 1 page, 202369 bytes (rendered copy: 202369 bytes)
fig_escape_freq: exit 0, 1 page, 83842 bytes (rendered copy: 83842 bytes)
fig_escape_P: exit 0, 1 page, 101947 bytes (rendered copy: 101947 bytes)
fig_escape_spec: exit 0, 1 page, 179539 bytes (rendered copy: 179539 bytes)
fig_pairs: exit 0, 1 page, 100754 bytes (rendered copy: 100754 bytes)
fig_decomp: exit 0, 1 page, 412858 bytes (rendered copy: 412828 bytes)
```

The six files only in the archive were stale leftovers of earlier runs (not read by any figure); the archive figures/data.zip was rebuilt from this export (169 files). All other files are byte-identical.

## 7. Remaining checks (code/), exit codes and last line of each
```
$ python section4_numbers.py   -> exit 0, 0 s; last line:   Table1 truss km: 7 modes 284-622 rad/s shares 2e-09-3e-06
$ python pairs_numbers.py   -> exit 0, 1 s; last line:    lowest excited mode 31.1 rad/s: p = 0.393, compliance bound C w^2 = 0.426; participation below 60 rad/s 0.411, bound 60^2 C = 1.587; below 100: 0.4
$ python residuals_all.py   -> exit 0, 0 s; last line: largest first-order residual over all designs: 3.4e-07
$ python second_order.py   -> exit 0, 1 s; last line: Table 1 truss km                 P=1.0593e-07  free 14/73  residual 1.9e-07  bound signs ok  smallest bound multiplier 5.3e-07  tangent dim 12  projec
$ python modal_errors.py   -> exit 0, 1 s; last line: km truss explicit  s=1:2e-15 s=3:1e-14 s=10:3e-13 s=30:8e-11 s=100:3e-08 s=300:1e-06 s=1000:3e-04 s=3000:2e-02 s=10000:3e+00
$ python classical_damping.py   -> exit 0, 1 s; last line:   maximum of P(c) at c = 6.8e+02 N s/m, P = 3.278e-04 W
$ python verify_extreme.py   -> exit 0, 15 s; last line: table rows written
$ python export_supplement.py   -> exit 0, 0 s; last line: 134 CSV files written to /tmp/clean/supplement
$ diff -rq supplement (CSV written by export_supplement.py) against the archive
134 CSV files compared
$ cmp code/verify_extreme_table.tex (written now) with the versioned copy
same
```

## 8. Supplement copies run from supplement/
```
$ python supplement_check.py 'escape_k_truss_optimized_s100000'
escape_k_truss_optimized_s100000.csv          P_direct 1.2236e-10  |modal-direct|/P 3.2e-06  mu 1.0000  gap [1, 97358]
$ python modal_errors.py
km truss optimized s=1:6e-15 s=3:1e-15 s=10:2e-13 s=30:6e-13 s=100:8e-09 s=300:3e-07 s=1000:5e-06 s=3000:6e-04 s=10000:2e-01
km truss explicit  s=1:2e-15 s=3:1e-14 s=10:3e-13 s=30:8e-11 s=100:3e-08 s=300:1e-06 s=1000:3e-04 s=3000:2e-02 s=10000:3e+00
```
