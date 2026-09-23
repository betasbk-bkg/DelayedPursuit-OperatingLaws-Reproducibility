# The vote layer moves u\* — the derived and the earlier published value belong to different plants (2026-09-08)

*English edition of the working note. The measurements, tables and conclusions follow the record as
written; the passage that drafted wording for a manuscript is given as a plain statement of the
result. The Korean original is retained by the author.*

**Scripts**: `d85_crowd_ellipse_2026-09-08.py` (crowd vs deterministic, circle vs ellipse) ·
`d83_ellipse_scaling_2026-09-08.py` · `d84_stadium_sweep_2026-09-08.py`

---

## 1. What the problem was

Two numbers had been compared side by side throughout the project:

| | value | source |
|---|---|---|
| derived | **0.4995** | `d38`, heading-servo linearisation |
| earlier published measurement (circle) | **0.7109** | x\*(433) = 1.404 → v\* = 2.134 → u\* |

A 42% difference. But **the two values do not belong to the same plant.**

`d38` solves a **deterministic** loop with delay, hold and a first-order lag only. The earlier
measurement is that loop with a **150-agent, 8-direction vote layer** on top of it. Treating them as
the same yardstick was the mistake.

---

## 2. Measurement — four cells, one protocol

DUR 240 s, τ = 433 ms, bowl estimator. Deterministic and crowd were run **on both the circle and the
ellipse**. Running the circle as a control is the point of the design: with the ellipse alone one
cannot separate the vote layer from geometry × vote layer.

| | circle | ellipse | difference |
|---|---|---|---|
| **deterministic** | **0.5164** | **0.5103** | **−1.2%** |
| **crowd** (MC 8, grid 1.00–3.50, 51-point bowl) | **0.7206** | **0.5252** | **−27%** |
| earlier published | 0.7109 | 0.562 | −21% |
| **how far the vote layer pushes** | **+39.5%** | **+2.9%** | |

**The crowd circle reproduces the published 0.7109 to within 1.4%.**

**The direction matters.** The geometry dependence does not come from the ellipse departing; it comes
from **the circle being pushed**. The vote layer moves the circle by 40% and the ellipse by only 3%.

> **The crowd ellipse cannot be pinned to better than 5%.** Two runs give 0.5538 (MC 10, narrow grid,
> 26 points) and 0.5252 (MC 8, wide grid, 51 points) — **5.4% apart**, and −6.5% from the published
> 0.562. The circle (51-point bowl) is firm and the ellipse is not; the distinction has to be kept
> when either is quoted.

---

## 3. Three things settle at once

### 3.1 On the deterministic plant the smooth-geometry margin is **shape-invariant**

| geometry | u\* | condition |
|---|---|---|
| circle R = 10 | 0.5164 | Λ = ∞ |
| ellipse a = 12, b = 6 | 0.5103 | ℓ/Λ = 0.57 |
| stadium (`d84`) | **0.5083 ± 0.0034** | R/L 1–6 × S/L 2.5–20 |

**A fourfold change of curvature radius, an eightfold change of straight length, and three different
ways for curvature to vary (circle → ellipse → stadium) leave u\* where it is**, and that value agrees
with d38's 0.4995 to within **2–3%**.

### 3.2 An open question of the earlier study closes differently

What that manuscript left open:

> *"a margin formula for arbitrary smooth geometries (ellipse u\* = 0.562)"*

**What is to be found is not a function of geometry.** Deterministically there is one smooth-geometry
margin (0.51), and the geometry dependence is **made by the vote layer**. The formula has to be a
function of quantization.

### 3.3 My quasi-static explanation is discarded

Earlier the same day I explained the ellipse's departure by `ℓ/Λ = 0.57`, i.e. quasi-static failure.
**That was wrong.** `d83` scales the ellipse by 1–3×, changing ℓ/Λ threefold from 0.57 to 0.19, and
u\* does not move; and the deterministic ellipse equals the circle to begin with (−1.2%). Quasi-static
departure acts on the **size** of the RMSE, not on the **location of the optimum**.

---

## 4. u\* is insensitive to almost everything, and sensitive to the vote layer

Collecting what was measured that day gives a consistent hierarchy.

| perturbation | on RMSE | **on u\*** |
|---|---|---|
| horizon truncation (low speed, −30% bias) | large | **+0.3%** (local estimate) |
| non-isolated corners (zigzag, seg/L 3.54) | wings on the curve | **+0.9%** |
| quasi-static departure (ellipse ℓ/Λ 0.57) | present | **0%** |
| arclength-monotone vs global projection | up to 20% | **−0.1 to −0.3%** |
| shape (circle → ellipse → stadium) | — | **±0.4%** |
| size harmonics (zigzag) | — | **−5.7%** |
| **the vote layer (quantization)** | — | **+39.5% (circle), +2.9% (ellipse)** |

**The first-order effect is what sits on top of the plant.** Geometry, projection and horizon are all
second order. And that is **the same level of statement** as the actuator lag entering τ_tot with
coefficient 1.06 on the Nav2 side: add a term to the plant and the optimum moves; how the path is
measured matters less.

---

## 5. What this means for a merged account

**When the two sets of numbers are quoted side by side, the plant has to be named.**

The corner margin of heading-servo pure pursuit is **u\* = 0.51 on the deterministic plant**, and it is
invariant to curvature radius (R/L 1–6), straight length (S/L 2.5–20) and the way curvature varies
(circle 0.5164, ellipse 0.5103, stadium 0.5083 ± 0.0034). Adding the 8-direction vote aggregation
moves the margin substantially **only on constant-curvature paths**: the circle to **0.72** (+39.5%),
the ellipse to **0.53** (+2.9%). The margin reported by the earlier study is therefore a property of
the quantized aggregation layer **engaging with a constant turn rate**, not a property of the tracking
geometry.

That turns the relation between the two sets of numbers from a contradiction into a separation of
layers.

---

## 5b. A candidate mechanism is in the earlier study's own result (iii)

Why is the circle the one that is pushed? From that study's E4, result (iii):

> *"The theta_err autocorrelation shows ... a secondary peak at lag k = 13. The period k × WIN =
> 13 × 0.3 s = 3.9 s coincides with the time for the Circle trajectory to rotate 45 deg at
> v = 2.0 m/s"*

**On a circle the heading turns at a constant rate, so the direction snap engages periodically with the
45° vote grid.** On an ellipse the turn rate varies with curvature and the engagement is broken up.
That study **observed the periodic structure without interpreting it**; here it becomes the candidate
mechanism for a 40% push in u\*.

**It is testable**: a stadium has constant curvature on its arcs and zero on its straights, so the
engagement can only happen on the arcs. If the crowd stadium's shift lands between the circle's and the
ellipse's, the explanation is supported. **It has not been run.**

---

## 6. What stays honest

- ~~There is no settled value for the crowd circle~~ — **done**: 0.7206 (51-point bowl, widened grid).
  What is unsettled instead is the **crowd ellipse**, whose two runs differ by 5.4%.
- **Only one τ (433 ms) was measured.** Whether the vote layer's shift varies with τ was not examined.
  If it does not, the shift in u\* is a pure layer effect; if it does, it interacts with delay.
- **There is no mechanism for the +39.5% (§5b is a candidate only).** That quantization produces
  direction error was quantified by `d67`/`d68`, but **by how much and in which direction** it moves
  the optimum is not derived.
