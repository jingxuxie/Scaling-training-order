# When Does Training Order Buy Compute?
## An Exact Curriculum–Mixture Boundary

**Nine-page revision (v3): September 21, 2026.** This is a complete, anonymous submission-format research draft, with full written proofs and executed CPU experiments. It has not been submitted or independently reviewed. Start with [`paper/main.pdf`](paper/main.pdf) or the [plain-language summary](docs/PLAIN_LANGUAGE_SUMMARY.md).

**Complete reference audit:** all 31 works were located and checked against primary records; 17 entries received bibliographic updates. The current PDF has **nine main-text pages and 33 pages total**, including the expanded bibliography. See [the complete reference audit](docs/REFERENCE_AUDIT.md).

## Manuscript polish in v3

**Local ICLR 2027 formatting and figure update:** the supplied template and its
support files are installed in `paper/`, with the standard anonymous review
header. All 11 figures now use a shared academic style with bottom/left axes,
serif typography, distinct colors and line styles, and unboxed legends. The
styling pass retained nine main-text pages and 32 pages total; the subsequent
reference audit adds one bibliography page. See
[`docs/FIGURE_STYLE.md`](docs/FIGURE_STYLE.md) for figure-only regeneration and
[`docs/iclr2027_formatting_audit.json`](docs/iclr2027_formatting_audit.json) for
the formatting checks at that stage. Historical v3 audit files below describe the
original archive before this presentation update.

The current main text is **exactly nine pages**, with **33 pages in the complete PDF** after the reference audit. The introduction, Section 3 model setup, and Section 6 experiments have been expanded with plain-language motivation, precise information/compute assumptions, and a fuller explanation of the baselines and negative results. The main theorem and numerical values are unchanged. See [the detailed revision note](docs/REVISION_V3.md).

This pass reran all 212 tests and checked that the recorded result files and scientific computation code are unchanged. It did not rerun the full experiment suites. The preceding numerical reproduction record is retained with that distinction explicit.

## Main scientific result

The earlier draft bounded two-phase schedules. The central theorem now covers **every measurable switching schedule** in the stated symmetric two-source family. If `epsilon` is the initial error mass orthogonal to the main direction, `phi` the source angle, and `B` the training work, balanced static mixing is globally optimal **if and only if**

```text
epsilon >= exp(-2 * B * cos(phi)).
```

Below the boundary, two mixed phases suffice for strict improvement. Above it, adding more phases cannot help. For a fixed positive orthogonal mass, there is a finite budget after which static mixing is exactly optimal. The best gain just below the boundary is quadratic, with sharp leading coefficient 1/8. These results assume a schedule chosen before drawing the initial error, not a feedback policy observing each realized error.

The revision also proves an isotropic additive-noise lower bound and implements exact expected Gaussian SGD for two distinct domain-sampling conventions. A curvature bound supplies a lower/upper bracket over **all static mixture weights**, rather than relying on a dense grid alone.

## What the experiments show

| Experiment | Result |
|---|---|
| Arbitrary-switch audit | 1,200 schedules; no observed lower-bound violations |
| Boundary signs | All 150 checks agree at 80-digit precision |
| Near-boundary gain | 45 checks approach coefficient 1/8 |
| Additive-noise audit | 480 cases; no observed violations |
| Exact SGD, domains sampled per example | 186 / 432 settings improve by over 1% against the lower static bracket; median candidate gain 1.0033 |
| Exact SGD, domains sampled per batch | 220 / 432; median candidate gain 1.0112 |
| Monte Carlo vs exact SGD moments | 32 / 32 means within three standard errors; 8,192 trajectories per condition |
| Retained finite-data / nonlinear stress tests | Median unchanged preparation gains 0.9944 / 0.8980 |
| Retained paid pilot | 23 / 2,160 accepted; accepted median gain 1.400, all-outcome median 0.403 |

Gain means baseline loss divided by candidate loss; below one is worse. The exact-SGD medians use the upper static-risk bracket (the best evaluated constant weight), while the >1% counts use the lower bracket covering every static weight. The exact SGD candidate library includes static mixing and knows population moments. Its nonnegative candidate gain is therefore not itself evidence of usefulness, and planning cost is not charged. Per-example mixing is the primary convention. Float64 bracket evaluations are not directed-rounding machine certificates.

## Reproduce

The tested environment is Python 3.13, using NumPy/SciPy and no GPU, model download, paid API, or external dataset. From this directory:

```bash
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
python -m pytest -q
python experiments/run.py --suite all --out reproduced/results
python experiments/extend.py --suite all --out reproduced/results
python scripts/verify_results.py --generated reproduced/results
```

Set `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1` before Python for predictable CPU usage. Every suite can also run separately; use `--help` for names. In the prior revision's recorded same-environment audit, **212 tests passed and all 21 raw CSVs, four derived CSVs, and two summary JSONs reproduced byte-for-byte**. The 212-test suite was rerun for v3, and all recorded result files remain byte-identical to that revision. Cross-platform comparison uses declared numerical tolerances. The supplied GitHub Actions workflow runs these checks on pushes and pull requests.

To regenerate the canonical results and figures:

```bash
python experiments/run.py --suite all --out results
python experiments/extend.py --suite all --out results
python experiments/summarize.py
python experiments/report.py
python experiments/report_v2.py
python scripts/write_references.py
python scripts/build_paper.py
```

The LaTeX build requires `pdflatex` and the packages listed in `paper/main.tex`; BibTeX is not needed. The manuscript cites 31 references. Its main text fits within the ICLR 2027 submission limit. The local template is byte-identical to the pinned official ICLR 2027 source. The build checks that hash and the nine-page main-text marker; provenance and final author checks are in [`docs/SUBMISSION_CHECKLIST.md`](docs/SUBMISSION_CHECKLIST.md).

## Files and scope

`src/training_order/` contains dynamics, bounds, SGD moments, and certificates. `experiments/` contains all controlled experiments and summaries. `paper/` includes full proofs and eight manuscript figures (11 figure files are generated, including ancillary plots). `results/` includes every scientific table and separate timing metadata. `docs/` records the claim boundaries, proof audit, reproduction audit, and author-facing submission checklist.

The key assumptions are mirror-source geometry, a fixed optimizer, identity target risk in the main theorem, and a precommitted schedule for a known initial second moment. We do not claim the same boundary for arbitrary Hessians, feedback policies, or neural networks. The paid-pilot experiment is separate from the oracle SGD planner. Negative transfers are retained.

The branch descends from the original repository and the earlier research draft. See [`docs/PUBLISHING.md`](docs/PUBLISHING.md) for importing the Git bundle without rewriting remote history. Substantial AI assistance is disclosed in the paper. Human author sign-off, independent proof/novelty review, and actual conference submission are not claimed.
