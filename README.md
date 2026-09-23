# Two-Regime Operating Laws for Optimal Speed in Delayed and Quantized Pursuit Control

Reproduction package — version 1.0.0

- **Author:** BongKeun Song
- **Affiliation:** Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU), Faculty of Engineering, Erlangen, Germany
- **Corresponding author:** bongkeun.song@fau.de
- **Journal:** Nonlinear Dynamics (under consideration, 2026)
- **Package DOI (concept, all versions):** <CONCEPT_DOI>

This package contains the code, the stored simulation results, the third-party-controller harness and
the reference engine behind the paper. It does **not** contain the manuscript: the article text is
distributed by the journal, and nothing here depends on having it.

## Layout

```
code/                    analysis and experiment scripts
  derivation/            the derivation series: 152 scripts and 211 stored result files
  nav2/                  the harness that drives the third-party navigation stack
data/                    stored results of the reference-system experiment series (phase1-phase15)
figures/                 the figures as the paper prints them, drawn by the scripts in code/
docs/                    pre-registrations, results notes, and two model notes
engine/crowd_control_engine/   the crowd simulation engine used as the study's reference system
ENVIRONMENT.md           versions and every departure from the third-party stack's defaults
```

## Prerequisites

Python 3.10 or newer with `numpy`, `scipy`, `matplotlib` and `sympy` (`pip install -r
requirements.txt`). The derivation scripts run on CPU; the longest interval-certification runs take
hours, and their stored summaries are shipped so that nothing has to be re-run to read the results.

The Nav2 experiments need ROS 2 Jazzy and the navigation stack itself, which is **not** vendored;
`ENVIRONMENT.md` pins the versions and lists every parameter that departs from the stack's defaults.

Some derivation scripts open the engine through an absolute path from the machine they were run on.
`code/derivation/README.md` gives the one command that rewrites those lines to read the environment
variable `ENGINE_DIR` instead:

```bash
export ENGINE_DIR=$PWD/engine/crowd_control_engine
```

## Reproducing the results

Run these from `code/derivation/` unless stated otherwise. Each writes its JSON beside itself and
prints what it computed.

| command | what it reproduces |
|---|---|
| `python d131_corner_energy_converged_2026-09-23.py` | the two plant-class constants, integrated to convergence: u* = 0.4984 (heading command) and 0.3078 (curvature command), with the sign study and the step-size refinement that supersede the earlier Euler values |
| `python d56_cross_block_consistency_2026-09-07.py` | the pooled third-party margin, u* = 0.302908 |
| `python d102_relocate_nav2_2026-09-13.py` | the three locators (three-point parabola, whole-bowl fit, symmetric window) and the spread between them |
| `python d103_corner_law_refit_2026-09-13.py` | the corner exponent and its interval |
| `python d107i_orbit_structure_summary_2026-09-13.py` | the existence band of the two periodic orbits and the closed forms for its edges |
| `python d119d_birth_multiplier_closed_form_2026-09-14.py` | the border-fold points, including the exact 4/7 |
| `python d127_closed_form_chain_margin_2026-09-14.py` | the quantized operating margin u* = 0.7475 and the 4.1% mean error against the measured optima |
| `python d132_nav2_tuning_regret_converged_2026-09-23.py` | the tuning-regret comparison of the operating law against rules that each drop one of its ingredients |
| `python d120d_krawczyk_stability_proof_2026-09-14.py 1 0.02 0.02 1 6 0.30 0.34 demo` | a short interval-certification run; the full runs' summaries are shipped as `d120d_summary_*.json` |

Figures, run from `code/`:

```bash
python make_fig12_bifurcation_atlas.py       # the bifurcation atlas
python make_fig13_chain_and_regret_table.py  # the analytic chain, and the regret table as markdown
python make_graphical_abstract.py            # the paper's graphical abstract
python make_figures_2026-07-19.py            # the reference-system figures (needs ENGINE_DIR)
```

## What is measured where

- `data/` holds the reference-system series: latency, look-ahead, window, smoothing, jitter, dropout,
  geometry, and the transfer ladder. The `phase*` scripts in `code/` produced them.
- `code/derivation/` holds the analysis that turns those measurements into the laws, and the
  third-party-controller blocks (`nav2_block_*.json`).
- `docs/PREREG_ACTUATOR.md` and `docs/PREREG_GRACEFUL.md` are the pre-registrations, written before the
  runs they predict; `docs/RESULTS_GRACEFUL.md` reports what those runs returned.

## Licence

Code: MIT (`LICENSE-CODE.txt`). Stored results, figures and notes: CC BY 4.0 (`LICENSE-CONTENT.md`).

## Citing

See `CITATION.md` and `CITATION.cff`. Please cite the package through its concept DOI, and the paper
separately.
