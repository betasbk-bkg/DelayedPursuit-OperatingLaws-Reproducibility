# `nav2_graceful_controller` — results

Pre-registration: `PREREG_GRACEFUL_2026-09-08.md` (written before any sweep data).
Derivations: `d97` (linearised family), `d99` (exact law), `d98` (isolation).
Verdict: `d100`. Block: `experiment.py --block graceful`, 113.6 min, 41 runs.

---

## 1. The blocker, and why the error message named the wrong cause

The run log listed this controller as failing with
`Collision detected in trajectory`, with the costmap suspected. It was neither.

`graceful_controller.cpp:200-222` picks a motion target by walking the plan
backwards:

```cpp
if (dist_to_target > params_->max_lookahead) {continue;}
...
} else if (dist_to_target < params_->min_lookahead) { break; }
```

`make_params.py` pinned `min_lookahead = max_lookahead = L`, copying what RPP
requires. Under that pin the first candidate to survive the `continue` trips the
`break`; the loop ends having simulated nothing, and control falls through to
line 250, which throws `NoValidControl("Collision detected in trajectory")` —
**the same message whether a trajectory collided or none was ever built.** That
is why every costmap, footprint, `track_unknown_space`, frame and path-geometry
variant failed identically: none touched the selection window.

With `min_lookahead = 0.05` the controller drives, 0 collision messages.

An earlier "fix" was actively harmful and is withdrawn: `track_unknown_space:
False` makes an off-map footprint point **fatal**, since `footprintCostAtPose`
returns `-1.0` off the map, `static_cast<unsigned char>(-1.0) == 255 ==
NO_INFORMATION`, and that branch is `is_tracking_unknown ? false : true`.

## 2. It is not a third class. It is RPP's equation at a different point.

Linearising the Park–Kuipers law (`smooth_control_law.cpp:114-122`,
`ego_polar_coords.hpp:60-70`) in `d38`'s variables gives `delta = e + e' - g`,
`phi = e - g + P`, and with `A = k_delta + 1 + k_phi`, `B = k_delta k_phi`:

```
e'' = -(A+B) e(x-u) - A e'(x-u) + (A+B) g(x-u) - B P(x-u)
characteristic:  lam^2 + exp(-lam u) (A lam + (A+B)) = 0
```

**`A = 2, B = 0` reproduces `d38`'s curvature-command plant to 4.4e-16** — that
is Nav2 RPP, and the anchor returns `u* = 0.3100` / `u_c = 0.5202` against the
published `0.3060` / `0.520494`. RPP and the graceful controller are two members
of one family, and `k_phi`, `k_delta` are dynamically settable ROS parameters, so
the family index is a knob. That is what turned a single-number comparison into
a predicted transition.

## 3. The transition was predicted and it happened — 4 of 4

| `k_phi` | (A, B) | predicted | measured |
|---|---|---|---|
| 0.5 | (3.5, 1) | optimum, `u*` 0.1118 | **0.1244** [0.1180, 0.1462] ✓ |
| 1.0 | (4, 2) | optimum, `u*` 0.0778 | **0.0715** [0.0715, 0.0847] ✓ |
| 2.0 | (5, 4) | **no interior optimum** | none, min at low edge ✓ |
| **3.0 (shipped)** | (6, 6) | **no interior optimum** | none, min at low edge ✓ |

**The sentence the paper gets:** the graceful controller *as shipped* has no
optimal speed; lowering `k_phi` below ~2 brings one back. The boundary was
placed between 1 and 2 in advance.

**The mechanism was named in advance and is not the gain.** The decomposition
row `(A, B) = (6, 0)` — graceful's gain with RPP's absent feedforward — keeps an
optimum at `u* = 0.0980`. It is the `B P(x-u)` term, feedback on the path tangent
at the look-ahead point, that removes it: a controller that already anticipates
the corner has nothing left to gain from the delay anticipating for it. RPP has
no such term, which is why RPP has an optimum.

This is a second, independent boundary on the claim, and a different one from
Stanley's (`d94`): Stanley has no optimum because it has **no look-ahead**;
graceful has one and still loses the optimum because it **also feeds the tangent
forward**. "Look-ahead pursuit" was necessary but is not sufficient.

## 4. The linearisation would have been wrong; the exact law was not

| `k_phi` | measured | band | linearised (`d97`) | exact law (`d99`) |
|---|---|---|---|---|
| 0.5 | 0.1244 | [0.118, 0.146] | 0.2270 (+82%) | **0.1118 (−10%)** |
| 1.0 | 0.0715 | [0.072, 0.085] | 0.0995 (+39%) | **0.0778 (+9%, in band)** |

The square's corner is 90°, and the plugin's `atan(-k_phi phi)` and `sin(delta)`
saturate there — at `k_phi = 3`, `k_phi phi ~ 4.7`, where `atan` returns a third
of the linear value. For RPP this never mattered (`kappa = 2y/d^2` is exactly
linear in `y`, so `alpha` scales out of the argmin), which is why `d38`'s linear
treatment sufficed there and does not suffice here. The exact-law engine was
validated by reproducing RPP's known `u*` through the same integrator before it
was allowed to issue a graceful prediction.

## 5. Three denominators that are measured, not configured

| quantity | configured | measured | note |
|---|---|---|---|
| speed | `v_linear_max` | ratio **1.000** at `beta` 0 | at shipped `beta` 0.4 the ratio is **0.810 → 0.599, varying with speed** |
| look-ahead | `max_lookahead` 0.600 | **0.5398 ± 0.3%** | target chosen by path distance, law driven by Euclidean distance |
| control period | 1/20 = 0.0500 | **0.0500**, p95 0.0505 | holds despite a trajectory simulation per candidate per cycle |

The `beta` row is the reason the headline sweep runs at `beta = 0`: with the
shipped speed schedule the swept parameter is not the speed, and the ratio is not
even constant, so no single `u` describes a row. The shipped configuration is
kept as a control row and reported at its measured speed.

## 6. Corrections made in the course of this block

* **The bowl estimator is biased on these curves.** It fits every point within
  `factor x min`; that is unbiased only when the selection is symmetric about the
  vertex. These curves rise far more steeply than they fall, so the window
  reaches further on the shallow side and drags the vertex down — on the
  `k_phi` 1.0 row it returned 0.0503 against a grid minimum of 0.0903. Replaced
  by a window symmetric in `u` by construction, with the spread over window size
  reported as a band. This does **not** overturn the bowl elsewhere: on the reference engine
  main's shallow, near-symmetric curves `d78`'s argument stands. The estimator is
  matched to the curve shape.
* **The first "no optimum" test measured convexity, not a minimum.** It asked
  whether a point lay below the *chord* through its neighbours; any increasing,
  upward-bending curve puts every interior point below its chord. It duly
  reported an 11.6% "dip" on the `k_phi` 2.0 row, produced entirely by the last
  point rising. Replaced by "below both neighbours", scored against the shallower
  one.
* **The noise threshold used the wrong quantity.** The 1.269% figure is the
  row-to-row spread of the `u*` *extraction*; the dip test compares RMSE
  *points*, whose measured noise is 0.7%. the criteria record notes
  swapping these two as an existing defect in this project; this nearly repeated
  it.
* **Reported `u* = 0.1118` as "−5%" once**, from the biased bowl value. With the
  corrected locator it is −10%.

## 7. The one wiggle: replicated away, and the noise figure was borrowed

The shipped row is monotone except at `u = 0.130`, where the RMSE sat 2.95%
below its shallower neighbour against a 1.98% threshold. That threshold was
2 sd of a **0.7% single-point noise figure borrowed from the reference engine's RPP runs**,
which is precisely the move this project rules out elsewhere. So it was measured
here instead — two further launches, the three speeds bracketing the feature,
repeated:

| `v` | n | replicated mean | repeat spread | the block's value |
|---|---|---|---|---|
| 0.4775 | 2 | 0.02702 | 0.40% | 0.02835 |
| 0.5398 | 2 | 0.02804 | 2.93% | **0.02754** |
| 0.6021 | 2 | 0.02877 | 6.14% | 0.03019 |

**The replicated sequence is monotone increasing (0.02702 → 0.02804 → 0.02877).
The wiggle does not reproduce.** The shipped row's "no interior optimum" stands
without qualification, and the global minimum was at the low edge either way.

Two things follow for the block's error bars, and both are worth stating rather
than burying:

* **The borrowed 0.7% was too small, and the verdict script now uses the
  measured value.** Pooling the three cells gives a single-point relative sd of
  **2.78%** — four times the borrowed figure — which puts the dip threshold at
  7.87% and the 2.95% feature far below it. `d100` was changed to take 2.78%,
  and with it the shipped row reads `none (min at low edge)` rather than
  `edge with a dip`, so the manuscript, the verdict JSON and this document now
  say the same thing. The measurement is weak on its own terms (n = 2 per cell,
  three cells, 3 degrees of freedom) and is used anyway, because a weak
  measurement of the right quantity beats a precise measurement of a different
  one — the reasoning the criteria record already notes.
* **The reproducibility unit is the launch, not the point.** Two repeats inside
  one launch agreed to 0.4% at `v = 0.4775`, while the same speed differed by
  4.7% between launches. Comparisons within a row (one launch) are therefore
  tighter than comparisons across rows. This does not affect any conclusion in
  §3 — those are within-row shape judgements and the `u*` bands in §4 are wider
  than either figure — but it is the right caveat on the band widths.

Replication limited to 2 launches; `g_replicate.json`.
