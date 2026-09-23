# The derivation scripts, and the one path in them you must change

These scripts are shipped **exactly as they were run**, apart from the engine path and the name of the environment variable that overrides it. They are the record of
what produced each number, so they are not rewritten for tidiness.

One consequence is visible immediately: 24 of them open the crowd engine
(`engine/crowd_control_engine`, shipped) through an absolute path,

```python
REPO = pathlib.Path(r'C:\Users\...\_letg_main_read\crowd_control_engine')
```

and three more open a zip of it the same way. On any other machine that path
does not exist, and those scripts fail on import.

**The fix is one command.** It rewrites exactly those lines to read the
environment variable `ENGINE_DIR`, leaving every other line untouched:

```bash
export ENGINE_DIR=/path/to/crowd_control_engine
python3 code/derivation/retarget_engine_path.py
```

Run it with `--dry-run` first to see the exact lines it would change.

Scripts that do not touch the crowd engine are unaffected and run as
shipped — including **every script behind Sec. VII**, which runs against the
Nav2 stack rather than that engine.

## Why the JSONs are here and not under `data/`

Each d-script reads and writes its results with `HERE / '<name>.json'`, i.e. in
its own directory. The package therefore keeps a script and its result side by
side, which is also what v1.1 did for `d1`–`d35`.

This is worth stating because the first build of this package did it the other
way, putting the new results under `data/derivation/`, and so produced a package
in which 51 of the shipped scripts could not find their own inputs. The defect
was introduced by the packaging rather than by the research, which is the class
of failure this project's process notes call repair-induced; it is recorded here
rather than quietly fixed.

## Added in v1.4

| script | what it establishes |
|---|---|
| `d102_relocate_nav2` | the Nav2 constant under all three curve-minimum estimators (0.3029 / 0.2840 / 0.2949) |
| `d103_corner_law_refit` | the corner exponent per leg, q = 0.843 +- 0.026 |
| `d104_freq_iso_rerun_compare` | the freq_iso block run twice: session reproducibility and grid refinement, per estimator |
| `d105_probe_track_unknown` | whether `track_unknown_space` in the local costmap changes what the controller does (needs the ROS 2 workspace; see ENVIRONMENT.md) |

## Added in v1.5 (manuscript v23b: Secs. 5-6, 8.4, App. B; Supplement S9)

| script | what it establishes |
|---|---|
| `d107i_orbit_structure_summary` | the (+) and (+,-,+) orbits, q_bc and q_lo against the section map (Sec. 5.3, App. B.1) |
| `d113` (integrator v3) | the validated event-by-event section map; use it for every section-map number (d27, d28 are defective) |
| `d116`, `d116b`, `d116i` | the word family W_n, its existence intervals and the grazing death of every word (Prop. 3) |
| `d118_word_multiplier_closed_form` | the multiplier recursion (Thm. 3) against finite differences and v3 |
| `d119b`, `d119c`, `d119d` | deaths for n = 1-8, births as 2-D root problems, P_B in closed form, u = 4/7 and the border-fold points (Prop. 4) |
| `d120d_krawczyk_stability_proof` | the interval certification of Table 2; run per u-row with a checkpoint (`d120d_rows_<tag>_n<n>_*.jsonl`, not shipped) and a summary JSON (shipped). Hours of CPU per word |
| `d121_sympy_border_polynomials` | the W_1 end-window border, the W_2 border curve and its degree-22 border-fold point (needs sympy, mpmath) |
| `d122`, `d123` | the stable-member scan, the short-excursion-last check (58/58), proof aggregates |
| `d124`-`d126` | why the earlier margin evaluation ran 9-16% low (seeds, global cubic, grid edge) |
| `d127_closed_form_chain_margin` | the analytic chain: closed-form sigma_theta -> one u* = 0.7475 for every latency, mean speed error 4.1% (Sec. 6.2, Fig. 4) |
| `d128`, `d128b` | Nav2 tuning regret against rules that drop one ingredient of the law, and the operating window (Table 5) |
| `d129_border_fold_tangency` | the fold-to-border distance near each border-fold point, against the quadratic contact of Colombo-Dercole |
