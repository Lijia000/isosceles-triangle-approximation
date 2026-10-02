# Multi-Norm Isosceles Triangle Approximation

[简体中文说明](README_zh.md)

This repository contains the numerical code supporting an isosceles-triangle
approximation problem. For an input triangle with three positive side lengths,
the code evaluates admissible isosceles targets under angle/coefficient bounds
and compares the optimal residuals in the $\ell_1$, $\ell_2$, and
$\ell_\infty$ norms. It generates decision-region and cost figures and
exhaustively validates the formulas on a perimeter-one triangle grid.

The programs implement the numerical formulation used by the associated paper;
they are not a general-purpose geometry library.

## Repository contents

### Source files

| File | Purpose |
| --- | --- |
| `generate_figures.py` | Generates the publication-style decision-region and cost figures. |
| `verify_multinorm_grid.py` | The sole supported exhaustive numerical verification program. |
| `requirements.txt` | Runtime dependencies. |
| `README.md`, `README_zh.md` | English and Simplified Chinese documentation. |

`verify_multinorm.py` is a legacy, redundant verifier. Do **not** use it and
do **not** upload it to the public code repository.

### Generated outputs

`generate_figures.py` writes paired PDF and PNG files. With the recommended
command below, the $\ell_1$ pairs are:

```text
1-90_decision_regions.{pdf,png}       1-90_costs.{pdf,png}
45-90_decision_regions.{pdf,png}      45-90_costs.{pdf,png}
30-50_decision_regions.{pdf,png}      30-50_costs.{pdf,png}
90-120_decision_regions.{pdf,png}     90-120_costs.{pdf,png}
balance_decision_regions.{pdf,png}    balance_costs.{pdf,png}
```

When `--paper --english` is used, it also writes:

```text
1-90_l2_decision_regions.{pdf,png}    1-90_l2_costs.{pdf,png}
1-90_linf_decision_regions.{pdf,png}  1-90_linf_costs.{pdf,png}
```

The figure generator also writes `figure_generation_results.json` unless
`--no-results-json` is supplied. The verifier writes
`verification_results.json` and `verification_summary.tex`.

Generated images, JSON records, and TeX summaries are outputs rather than
source. They are not needed for a standalone code repository.
`verification_results.json` may be included optionally as a reproducibility
record.

## Prerequisites and installation

Use Python 3.8 through 3.11. The upper SciPy bound is intentional:
the independent linear-program controls use SciPy's `revised simplex` method,
which was removed in newer SciPy releases.

```bash
conda activate base
python -m pip install -r requirements.txt
```

The required packages are NumPy, SciPy, and Matplotlib. A Conda environment is
not otherwise required; any compatible Python environment can be used.

## Reproduce the figures

Run this command from the directory containing the scripts:

```bash
python generate_figures.py --paper --balance --english --output-dir .
```

It produces the ten $\ell_1$ files and four $\ell_2$/$\ell_\infty$ files
listed above, plus `figure_generation_results.json`, in the current working
directory. The default output directory for the $\ell_1$ pairs is `pic/`.

### Figure-generator options

| Option | Meaning |
| --- | --- |
| `--paper` | Select the four standard angle intervals: `[1°,90°]`, `[45°,90°]`, `[30°,50°]`, and `[90°,120°]`. This is the default interval set. |
| `--balance` | Also include the balanced coefficient case `(k1,k2)=(0.75,1.25)`. |
| `--alpha A --beta B` | Generate one explicit angle interval `[A,B]` in degrees; provide both options and do not combine with `--paper`. |
| `--interactive` | Prompt for `alpha` and `beta` in degrees. |
| `--step S` | Set the input-triangle lattice spacing; it is not a target-side discretization step. Default: `0.01`. |
| `--surf-n N` | Set subdivisions per side of the interpolated surface mesh. Default: `220`. |
| `--output-dir DIR` | Destination for $\ell_1$ figure pairs and `figure_generation_results.json`. Default: `pic/`. |
| `--english` / `--russian` | Use English or Russian figure text. Without either option, figure labels are Chinese. The two options are mutually exclusive. |
| `--hide-tie-lines` | Omit tie segments from decision figures and their legends. |
| `--output-stem STEM` | Replace the filename prefix for every $\ell_1$ output pair. |
| `--no-results-json` | Suppress `figure_generation_results.json`. |

The $\ell_2$ and $\ell_\infty$ figures are produced only for
`--paper --english` and are written beside `generate_figures.py`; this is
intentional current script behavior.

## Exhaustive verification

Run only the supported verifier:

```bash
python verify_multinorm_grid.py
```

The verification uses perimeter 1 and input-grid step `0.01`. It retains
strict triangle inequalities and all ordered, position-preserving side triples:
**1,176 triples for each of five settings**:

1. `[1°,90°]`
2. `[45°,90°]`
3. `[30°,50°]`
4. `[90°,120°]`
5. `(k1,k2)=(3/4,5/4)`

For every triple, it checks all three base-edge branches and the two-branch
global rule. Independent validation controls are:

| Norm | Formula validation control |
| --- | --- |
| $\ell_1$ | Finite breakpoint enumeration and an independent linear program. |
| $\ell_2$ | Direct continuous minimization of the reduced squared objective. |
| $\ell_\infty$ | An independent epigraph linear program. |

The verifier records formula/control discrepancies, feasibility violations,
reconstruction errors, and $\ell_1$ tie counts in
`verification_results.json`; it also emits the generated TeX macro file
`verification_summary.tex`.

## Recommended public GitHub upload set

Upload only:

```text
generate_figures.py
verify_multinorm_grid.py
requirements.txt
README.md
README_zh.md
```

Optionally add `verification_results.json` as a record of a completed run.
Do not upload `verify_multinorm.py`, generated figures, paper LaTeX sources,
publisher templates/classes, bibliography files, PDFs, or other paper assets.
Those materials are unnecessary for a standalone numerical-code repository.
