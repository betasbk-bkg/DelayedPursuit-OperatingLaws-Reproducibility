# Pre-registration — E12, the actuator-lag term (2026-09-08, written **before** the runs)

*English edition of the pre-registration. It follows the record as written: nothing has been
added, removed or reordered. The Korean original is retained by the author.*

**Purpose**: to answer the standing objection that *"the plant has no dynamics"*. Instead of switching
on a physics engine and adding **uncontrolled** realism, **one missing term** is added and swept.

---

## 1. Prediction — no free parameter

The impulse response of a first-order lag `v̇ = (u − v)/T_act` is `(1/T)e^(−t/T)`, whose **first moment
is exactly T**. Theorem 2 composes mean delays, so

> **τ_tot = τ + T_ctrl/2 + T_sim/2 + T_act**  ← **coefficient s_act = 1**

- **Primary prediction**: `s_act = 1.000`
- **Null hypothesis**: `s_act = 0` (actuator dynamics do not enter τ_tot)
- **Secondary prediction**: once T_act is in the budget, **u\* stays at 0.306 across all six rows.**
  Because the sweep is in u-space, a term missing from the budget shows up as **drift in u\*** — the
  same way E10 caught the excess of the Euler integration.

**What s_act = 0 would look like**: the true τ_tot would be `tt − T_act`, so the observed u\* would be
inflated by `tt/(tt−T_act)`, which in the last row is a factor **3.3** → it would leave the sweep
window [0.188, 0.442] and report as an **upper EDGE**. The null is therefore distinguishable not only
by the coefficient but by **how the row fails**.

## 2. Design

| item | value | reason |
|---|---|---|
| T_act grid | 0, 0.025, 0.05, 0.10, 0.20, 0.30 s | leverage: at the top, **2.3×** the rest of the budget (0.13 s) |
| injected τ | fixed at 0.10 s | as in E1′ |
| geometry | square, side 10 m (**seg/L = 16.67**, L = 0.6 m) | corner isolation |
| sweep | u-space **[0.1884, 0.4424]**, 9 points, `u_cap = 0.85·u_c` | as in E8–E11 |
| wheel | 2.75 | unchanged |
| plant | `loopback_actuator` (**diagnostic build**) | the headline runs stay on upstream |

**54 points in total, estimated machine time ≈ 3.1 hours**
(136 min of driving + 50 min of overhead; the overhead is 55 s/point, back-computed from E10.)

**Additional prediction (entered immediately after launch, with zero rows of data)**: d48 fits
`τ_tot = τ + s·T_act + c`, so **c absorbs T_ctrl/2 + T_sim/2 = 30.0 ms** here. E1′ returned **29.87 ms**
at the same place. This block therefore predicts **two numbers at once (s_act = 1.000, c = 30.0 ms)**,
not one.

## 3. What was checked before the runs (before trusting any coefficient)

If the failure mode is that the parameter is **silently ignored**, the coefficient comes out 0 and I
would read that **as a result** — "Theorem 2 fails". So it was measured first (`verify_actuator.py`):

| requested T_act | measured T̂ (linear.x) | r² | measured T̂ (angular.z) | r² |
|---|---|---|---|---|
| 0.30 s | **0.3001 s** (+0.0%) | 1.000 | 0.2916 s (−2.8%) | 0.990 |
| 0 | α = 1 exactly, max\|applied − commanded\| = **0.0** | — | **2.2e−19** | — |

**At T_act = 0 this build is identical to upstream to machine precision.** The first row of the sweep
is therefore its own control.

## 4. A signal that is already visible

Two smoke points (v = 1.5 fixed, one lap of the 10 m square):

| T_act | τ_tot | u | RMSE |
|---|---|---|---|
| 0 | 0.130 | 0.0975 | 0.1427 m |
| 0.30 | 0.430 | **0.3225** | **0.1101 m** |

**Adding delay reduced the error by 23%**, because u moved from 0.098 to 0.323 — that is, toward
u\* = 0.306. It is a two-point preview of Theorem 1's cancellation working *for actuator lag as well*;
two points are a preview only, and the block below is the measurement.

## 5. What this experiment opens

**Heredia & Ollero (2007)** non-dimensionalise their pure-pursuit stability analysis by the **steering
time constant T**. Until now this study's τ_tot had no such term at all, so it met that literature only
in a limit. Putting T_act in **brings the experiment inside that family.**

---

**Scripts**: `nav2/loopback_actuator.py`, `verify_actuator.py`, `experiment.py --block actuator`.
**Output**: `nav2_block_actuator_2026-09-08.json`.
