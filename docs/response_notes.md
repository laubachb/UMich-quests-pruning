# Reviewer response: evidence collected so far (working notes, 2026-10-08)

Status key: DONE = result on disk, committed; GPU = needs model training; PENDING = job running.

## Reviewer 1

**#1 What "retained fraction" means.** DONE. `results/stats_train/retention_accounting_*.csv`,
`figures/out/retention_accounting.png`. Retention is a fraction of atomic *environments*; frames are
kept whole and the force loss is masked per atom. At 1 % stratified retention 62 % of frames still
enter training; at 5 % global retention diamond 300 K and graphite 300 K keep **1 environment each**
while HD liquid 3.0 g/cc 8000 K takes 41 % of the budget. Same pattern on MoNbTaVW (liquid 64 % of the
budget; divacancy 1 env). Revision: define retention explicitly, add the accounting figure.

**#2 Test-set correlation with training trajectories.** DONE. `results/test_distance/summary.csv`.
Nearest-neighbour descriptor distance test->train vs train->train (leave-one-out): ratio 0.98-1.37
for 11 of 12 groups (graphite 300 K: 2.6, i.e. test is *farther*). Test environments are no closer to
the training set than training environments are to each other: a fresh sample of the same trajectory,
not near-duplicates. New datasets use a contiguous last-20 % held-out block per group (trajectory-
disjoint where the source is ordered). PENDING: same analysis for MoNbTaVW, SiO2.

**#3 Elbow specification.** DONE (re-derived). `pruning/funiq_sweep.py`, `pruning/elbow_logh.py`,
`figures/out/funiq_curves.png`. Grid: 40 log-spaced h in [0.005, 0.5]; f_uniq(h) = H(h)/H_max on the
full group. f_uniq vs log10 h is a sigmoid; elbow = midpoint in log h between the inflection and the
lower maximum-curvature bend of a cubic spline. Reproduces the paper's table (crystal h* 0.016/0.020/
0.032/0.039 vs 0.017/0.018/0.035/0.045; f_uniq RMS 0.09 over 14 groups). Chord/kneedle rules do NOT
reproduce it (range-dependent). Subsampling 25-100 % moves h* by at most one grid step for all groups
(`results/funiq/groups_s0/elbows_all.csv`). ACTION: get the original elbow script / Supp. Tab. 2 from
coauthors to state the published procedure exactly.

**#4 Runs, seeds, uncertainty.** DONE. `results/stats_test/`, `figures/out/improvement_ci.png`.
7-10 independent runs per retention level, 20 at 100 %; bootstrap 95 % CIs on the median relative
improvement. Crystals +20 to +40 % with CIs well clear of zero; LD liquids ~0; HD liquids **-1 to -3 %,
CIs exclude zero** (state this plainly; the text currently says "near parity"). Outlier runs flagged
in `per_point_summary.csv`. Seeds: paper runs used a fixed training seed with varying data order; new
runs use seeds 0,1,2 (E1-E5).

## Reviewer 2

**#1 Elbow exists for uniform distributions too.** PENDING null test (`pruning/funiq_null.py`, job
running): f_uniq(h) on a covariance-matched Gaussian and a column-shuffled surrogate of each group's
descriptors, same n/d/h-grid; compare elbow h* and f_uniq with the real group. Agree in part. The elbow is an operational scale,
not a redundancy proof. Evidence that it is informative: elbow h* differs by 7x between diamond 300 K
and HD liquids; f_uniq(0.015) says 99 % of liquid environments are unique, the elbow says 20-35 %.
Robustness: see R1 #3. PENDING: same curves on SiO2 (14 polymorphs + liquid/amorphous) and MoNbTaVW.

**#2 Bandwidth does not enter FPS; Stratified-Fixed == Stratified-Adaptive at equal retention.**
Correct, and the revision must say so. The adaptive step only sets the per-group retention. GPU
experiments isolate it: E4 (paper budget, 31.2 % overall, per-group 0.17-0.55) vs E5 (global and
stratified at a flat 31.2 %). E4_midlo uses the re-derived rule (33.2 %).

**#3 Why entropy at all if FPS does the selection.** DONE. `results/stats_train/radius_vs_elbow.csv`.
The FPS covering radius at the elbow retention fraction equals the elbow bandwidth: r/h* median 1.08
(IQR 1.04-1.17) over the 12 non-cold-curve groups. Retaining f_uniq(h*) by FPS therefore leaves every
discarded environment within ~h* of a kept one: the entropy elbow and the geometric covering radius are
the same length scale, which is why h never needs to enter FPS. A kernel-free alternative (knee of
log r(k) vs log fraction) gives similar budgets for crystals (0.35-0.42 vs 0.30-0.34) but larger ones for
HD liquids (0.5-0.68 vs 0.34); correlation with the elbow budget 0.61. GPU: random baselines (E1) answer
whether FPS matters at all.

**Minor: prior redundancy-minimising generation.** Add citations (active learning by ChIMES; FPS /
CUR / DFT-MD subsampling literature).

## Reviewer 3

**#1 Originality (sample all fundamental states).** The accounting figure shows global FPS does *not*
do this (1 environment for two crystal groups at 5 %); stratification is the mechanism. E5 vs E4
tests whether per-group budgets add anything beyond equal-fraction stratification.

**#2 Multi-element systems.** DONE (selection side). SiO2 (Erhard 2022, Zenodo 6353684; 16 groups:
13 polymorphs + liquid/quench/amorphous) and MoNbTaVW (Byggmastar 2021, ColabFit; 19 configuration
sets: alloys, liquids, point defects, surfaces) with contiguous held-out test blocks per group;
QUESTS multicomponent descriptor. Findings mirror carbon and are sharper:
- Global FPS at 5 %: SiO2 liquid+quench+amorphous take 98.5 % of the budget, low-cristobalite keeps
  0 environments, most polymorphs 1-16. MoNbTaVW liquid takes 64 %; divacancy 1 env.
  (`figures/out/retention_accounting_{sio2,monbtavw}.png`)
- f_uniq(h) sigmoids per group; log-h elbow rule gives h* 0.02-0.06 (SiO2) and 0.015-0.19 (MoNbTaVW,
  defect groups show a two-step decline: bulk-like environments merge first). Stable to 25-75 %
  subsampling (<= 1 grid step). (`figures/out/funiq_curves_{sio2,monbtavw}.png`)
- Test/train NN-distance ratio ~1 for MoNbTaVW (held-out block is as independent as a random split);
  SiO2: ratios 1.0-2.2, the contiguous held-out block is farther than a random split would be.
GPU: 66 jobs each (E1-E5), `training/jobs_{sio2,monbtavw}.txt`.

**#3 Length.** Editorial: cut abstract to <200 words, compress Sec. 1.1 and 2.1 background.

## Reviewer 4

**#1 Density rescaling / fixed f_uniq target as alternatives.** Budgets exist for both a fixed
f_uniq target (`budget_ftarget_0.3.json`) and the fixed h=0.015 (`budget_fixed_h015.json`, 84 %
retention, i.e. no pruning for liquids). Density rescaling conflates T and rho groups; argue in text.

**#2 How "Full" is built in Fig. 2.** DONE: Full = union of per-group budgets (paper: 31.2 %
overall); dashed line = f_uniq elbow of the pooled curve. Pooled f_uniq(0.015) re-derived = 0.873
(paper 0.884); MoNbTaVW 0.913, SiO2 0.880: the default bandwidth calls ~90 % of environments unique in
all three chemistries. Pooled log-h elbows in `results/{funiq,ext/*/funiq}/global_s0/elbows_logh.csv`.

**#3 Other subsampling strategies / random.** GPU: random_global and random_stratified at 5/10/20 %.

**#4 Single component, single dataset, single MLIP.** See R3 #2. MLIP: NequIP only (state as limitation).

**#5 Train MAE > test MAE.** DONE. Composition, not a bug: per group train ~ test; test MAEs
reweighted to the training mix give 0.42 eV/A vs train Full 0.39. The test set has no cold-curve
groups and a different liquid share. Also: the Nov-2025 CSVs are training-set MAEs (relabelled).

**#5b Fig. 3 is circular.** Agree it is descriptive; keep as supplementary or merge with the curves.

**#6 RDFs insensitive.** DONE: force-tail analysis (`results/stats_test/force_tail.csv`). In the
top-10 % |F| bin at 5 % retention stratified beats global by 21-36 % on crystals; HD liquids -1 to
-3 %, i.e. the same ordering as the mean MAE but with larger absolute errors (0.54 vs 0.56 eV/A Full).
Proposes error-vs-|F| as the discriminating metric; RDF retained as a sanity check.

## Data corrections to state in the revision
MD: 1 fs x 6000 steps (not 0.5 fs x 10 000). "diamond 3.68 g/cc" frames are 3.56 g/cc. Test tag
"LD_liquid_0.5gcc_2000" is 1.0 g/cc. LR-scheduler threshold 0.2. 10 duplicate frame pairs in the
training set (675 unique of 685).
