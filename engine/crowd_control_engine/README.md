# Crowd-control engine — reproduction materials

The simulation engine used as the reference system of

**"Two-Regime Operating Laws for Optimal Speed in Delayed and Quantized Pursuit Control"**

and, earlier, by the author's study published as *Scientific Reports* **16**, 29469 (2026),
https://doi.org/10.1038/s41598-026-72284-6, whose citation metadata is in `CITATION.cff`.

> Author: BongKeun Song, Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU), Faculty of Engineering, Erlangen, Germany (bongkeun.song@fau.de).

This repository contains the simulation code and raw experimental outputs (JSON) supporting the manuscript. Approximately 33,200 Monte Carlo simulation runs across seven experiments are reproduced by the scripts below.

---

## Summary of Key Findings

- **Empirical scaling**: v*(α, τ) = C(α) × τ⁻ᵝ with β ≈ 0.5 across four trajectories and a 10-fold latency range (τ = 100–1000 ms).
- **Collapse**: RMSE(v) curves from six latency conditions align onto a single master function of x = v√τ, with R²_collapse = 0.956.
- **Invariant**: x* = v*√τ ≈ 1.347 m·s⁻¹ᐟ², CV = 6.4%.
- **Practical formula**: v*(α, τ) ≈ 1.347 × cos¹·⁶⁸⁸(α/2) × τ⁻⁰·⁵¹¹.

---

## Repository Contents

### Core simulator

| File | Role |
|---|---|
| `simulation_main.py` | Discrete-time simulation framework (Δt = 1/60 s, delayed-state retrieval, vote aggregation, exponential smoothing). All experiment scripts import from this module. |

### Experiment scripts and JSON outputs

| Script | Output JSON | Manuscript reference |
|---|---|---|
| `unified_vstar_experiment.py` | `unified_vstar_results.json` | Table I; Table III; Fig. 1; Fig. 2 (G1 + G2 main sweep) |
| `exp1_fixed.py` | `exp1_fixed_results.json` | Table IV (E1 — vote-noise σ sweep) |
| `exp2_fixed_exp3_supplement.py` | `exp2_fixed_results.json` | Table V (E2 — VOTE_INT proportional) |
| `exp2_fixed_exp3_supplement.py` | `exp3_quadratic_fixed.json` | Table VI quadratic τ=100 ms (E3 supplement, ceiling resolved) |
| `expa_quantization.py` | `expa_results.json` | Fig. 4 (EXP-A — directional resolution n_dirs ∈ {4, 8, 16, 36, 360}) |
| `theta_autocorr_fixed.py` | `theta_autocorr_results.json` | Fig. 5 (E4 — direction error analysis) |
| `corner_freq_experiment.py` | `corner_freq_results.json` | §VII.C.2 (CF — corner-frequency, n = 11,700, r = +0.94) |

### Partially superseded script (retained for provenance)

| Script | Output JSON | Notes |
|---|---|---|
| `exp123_final.py` | `exp3_results.json` | **Only the EXP3 Fixed/Majority outputs are used in the paper** (β = 0.498 for FV, β = 0.493 for MV; Table VI). Its EXP1 outputs were superseded by `exp1_fixed.py` (incorrect σ patching); its EXP2 outputs were superseded by `exp2_fixed_exp3_supplement.py` (WIN patch did not propagate to VOTE_INT); and its EXP3 quadratic τ=100 ms row was superseded by `exp3_quadratic_fixed.json` (ceiling artifact). The script is retained so the provenance of `exp3_results.json` is fully reproducible. |

### Not included

- Figure-generation scripts are not provided. Figures in the manuscript were produced from the JSON outputs above.
- The earlier file `exp2_results.json` (buggy WIN patch output) is omitted; its corrected counterpart is `exp2_fixed_results.json`.

---

## Code → Manuscript Run Counts

| Group | Approx. runs |
|---|---|
| Main sweep (G1 + G2) | 9,360 |
| Noise (E1) | 3,510 |
| VOTE_INT (E2) | 1,170 |
| Aggregation (E3) | 1,425 |
| Directional resolution (EXP-A) | 5,850 |
| Direction-error analysis (E4) | 195 |
| Corner-frequency (CF) | 11,700 |
| **Total** | **≈ 33,200** |

---

## Reproduction

### Environment

- Python ≥ 3.9
- Dependencies:
  - `numpy`
  - `scipy`
  - `matplotlib` (used only by `theta_autocorr_fixed.py` for diagnostic plot output; not required for JSON regeneration)
  - `json`, `math`, `random` (Python standard library)

See `requirements.txt`.

### Execution order

The scripts are mostly independent, but two of them (`exp2_fixed_exp3_supplement.py` and `exp123_final.py`) read `unified_vstar_results.json` as a baseline. Run in the following order:

```bash
# 1. Main sweep (required baseline for E2/E3)
python unified_vstar_experiment.py

# 2. Robustness experiments
python exp1_fixed.py
python exp2_fixed_exp3_supplement.py
python exp123_final.py     # only EXP3 FV/MV outputs used

# 3. Generality / mechanism experiments
python expa_quantization.py
python theta_autocorr_fixed.py
python corner_freq_experiment.py
```

Each script writes its output JSON to the working directory. The JSON files in this repository are the exact outputs used to generate the manuscript tables and figures.

### Notes on simulation parameters

- All experiments use Monte Carlo MC = 15 per condition.
- Default crowd size N = 150, troll ratio 5% (unless varied).
- Aggregation window WIN = 0.3 s; vote interval VOTE_INT = 18 frames at 60 Hz.
- 8-direction vote quantization is the default; EXP-A sweeps this from 4 to 360.

---

## Citation

Author: BongKeun Song; the machine-readable citation is in `CITATION.cff`.

---

## License

This repository is released under the **Creative Commons Attribution 4.0 International License (CC BY 4.0)**. See `LICENSE`.
