# The derivation scripts

These are the scripts that produced the study's analytical and numerical results, shipped as they were
run. They are the record of what produced each number, so they are not rewritten for tidiness; the one
edit made for distribution is that the paths which pointed at the engine on the machine that ran them
now point at `engine/crowd_control_engine` inside this package, so they work here without setup.

Each script writes its JSON beside itself and prints what it computed. Nothing has to be re-run to read
the results: every output file that the paper draws on ships with the script that made it.

## Where the main results come from

| script | result |
|---|---|
| `d131_corner_energy_converged_2026-09-23.py` | the two plant-class constants, integrated to convergence: u* = 0.4984 (heading command), 0.3078 (curvature command) |
| `d133_relocate_nav2_converged_2026-09-23.py` | the three locators on the third-party curves, and their spread against the derived constant |
| `d132_nav2_tuning_regret_converged_2026-09-23.py` | the tuning-regret comparison against rules that each drop one ingredient of the law |
| `d134_graceful_predictions_recomputed_2026-09-23.py` | every prediction in the second controller's table, recomputed from the exact-law engine |
| `d127_closed_form_chain_margin_2026-09-14.py` | the quantized operating margin u* = 0.7475 and its 4.1% mean error |
| `d116*`, `d118`, `d119*` | the word family W_n, its existence intervals, the grazing death (Proposition 2) and the border-fold points including the exact 4/7 (Proposition 3) |
| `d120d_krawczyk_stability_proof_2026-09-14.py` | the interval certification; the full runs' summaries ship as `d120d_summary_*.json` |
| `d107*`, `d113` | the section map, its validated integrator, and the existence band of the two orbits |
| `d56`, `d62`, `d100`, `d102` | the third-party blocks: pooled margin, isolated delay terms, the second controller's verdict |

`d38` and `d102` are kept as what was run before the constants were integrated to convergence; `d131`
and `d133` supersede them, and the paper quotes the latter.

## Running them

Python 3.10+ with `numpy`, `scipy`, `matplotlib`, `sympy` and `PyYAML` (see `requirements.txt`). The
certification runs take hours; everything else is seconds to minutes.

The scripts behind the third-party-controller section drive a ROS 2 navigation stack, which is not
vendored here; `ENVIRONMENT.md` pins the versions and lists every parameter that departs from the
stack's defaults.
