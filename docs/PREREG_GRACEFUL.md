# Pre-registration — `nav2_graceful_controller`

**Written 2026-09-08, before any sweep data exists.** Derivation: `d97_graceful_class_2026-09-08.py`.
Every number below is a prediction, not a fit.

---

## 1. Why this controller, and what it is for

The paper's claim so far rests on two Nav2 controllers that both aim at a
look-ahead point. A referee's obvious question is whether the optimum is a
property of *delayed tracking* or a property of *this way of tracking*. Two
things already bear on it:

* Stanley (`d94`) has **no** look-ahead and has **no** interior optimum. That
  draws one boundary of the claim.
* `nav2_graceful_controller` does have a look-ahead but a different law. It is
  the only *second implementation within a class* available, and until today it
  had never run.

## 2. It is not a new class. It is the same equation at a different point.

From the source (`smooth_control_law.cpp:114-122`, `ego_polar_coords.hpp:60-70`)
the commanded curvature is the Park–Kuipers law in egocentric polar coordinates
`(r, phi, delta)` of the motion target:

```
kappa = -(1/r) [ k_delta (delta - atan(-k_phi phi))
                 + (1 + k_phi/(1 + (k_phi phi)^2)) sin(delta) ]
```

Linearising in `d38`'s variables (`x` = arclength/L, `e` = lateral error/L,
`psi = e'`, `g` the preview integral, `P(x) = Psi_p(x+1) - Psi_p(x)` the path's
turn over one look-ahead) gives `delta = e + e' - g` and `phi = e - g + P`, so
with `A = k_delta + 1 + k_phi` and `B = k_delta k_phi`

```
e'' = -(A+B) e(x-u) - A e'(x-u) + (A+B) g(x-u) - B P(x-u)          (*)
```

**Setting `A = 2, B = 0` reproduces `d38`'s curvature-command plant verbatim** —
that is Nav2 RPP, and the integrator agrees to 4.4e-16. So RPP and the graceful
controller are two members of one family indexed by `(A, B)`; the characteristic
equation is

```
lam^2 + exp(-lam u) (A lam + (A+B)) = 0
```

and the RPP anchor comes back at `u* = 0.3100` / `u_c = 0.5202` against the
published `0.3060` / `0.520494`.

**`k_phi` and `k_delta` are exposed ROS parameters and both are dynamically
settable** (`parameter_handler.cpp:140-166`), so `(A, B)` is a knob. That turns
a single-number comparison into a predicted *curve*, and a curve cannot be luck.

## 3. The predictions

| case | `k_phi` | `k_delta` | A | B | `u_c` | predicted `u*` |
|---|---|---|---|---|---|---|
| RPP anchor | — | — | 2 | 0 | 0.5202 | 0.3100 |
| `k_phi` 0.5 | 0.5 | 2 | 3.5 | 1 | 0.3336 | **0.2270** |
| `k_phi` 1.0 | 1.0 | 2 | 4 | 2 | 0.2900 | **0.0995** |
| `k_phi` 2.0 | 2.0 | 2 | 5 | 4 | 0.2350 | **none** |
| **shipped** | **3.0** | **2** | **6** | **6** | **0.2005** | **none** |
| `k_phi` 8.0 | 8.0 | 2 | 11 | 16 | 0.1204 | **none** |
| decomposition | — | — | 6 | 0 | 0.2313 | 0.0980 |

The headline is a **transition**, not a value:

> **The shipped graceful controller is predicted to have NO interior optimum.
> Lowering `k_phi` brings one back, somewhere between `k_phi` 1.0 and 2.0.**

For the shipped defaults `J(u)/min J` rises monotonically from the first grid
point — 1.000, 1.027, 1.055, 1.121, 1.248, 1.432, 1.861, 2.887, 4.662 at
u = 0.004 … 0.184 — so the prediction is a monotone RMSE(v), the same
phenomenology Stanley showed.

**The mechanism is named in advance, and it is not the gain.** The decomposition
row `(A, B) = (6, 0)` — graceful's gain with RPP's absent feedforward — still has
an optimum at `u* = 0.0980`. It is the `B P(x-u)` term, the feedback on the
*path tangent at the look-ahead point*, that removes it: a controller that
already anticipates the corner has nothing left to gain from the delay doing the
anticipating for it. RPP has no such term, which is why RPP has an optimum.

## 4. What would refute this

* An interior optimum at the shipped defaults, located away from the lower edge.
* No optimum at `k_phi` 0.5 or 1.0.
* An optimum at `k_phi` 0.5 or 1.0 landing far from 0.2270 / 0.0995 once
  expressed in the *measured* `u` (see §5).
* The transition sitting outside `k_phi` in [1, 2].

## 5. Three denominators that are measured, not assumed

The `u` above is `v tau_tot / L`, and on this controller none of the three is
what the configuration file says.

1. **`v` is not `v_linear_max`.** `smooth_control_law.cpp:66-73` modulates speed
   by curvature, `v = v_max/(1 + beta |kappa|^lambda)`, then by proximity to the
   target, then clamps. A first drive achieved 0.340 m/s at `v_linear_max` 0.500
   — ratio 0.680, and varying along the path. The sweep therefore runs at
   `beta = 0`, and the achieved speed is measured and required to be within 2%
   of `v_linear_max`. (The shipped `beta = 0.4` is kept as a separate control
   row, with its achieved speed measured.)
2. **`L` is not `max_lookahead`.** The target is the farthest plan pose within
   `max_lookahead` by *integrated path* distance; the control law then uses the
   *Euclidean* `r` to it. Measured: 0.5142 m against `max_lookahead` 0.600 —
   14% apart, larger than most effects in this paper. `u` is formed with the
   measured value.
3. **`T_ctrl` is not `1/freq` by assumption.** The graceful controller simulates
   a full trajectory per candidate target per cycle; the first drive logged one
   `Control loop missed its desired rate of 20 Hz. Current loop rate is 3.30 Hz`
   (once in ~1600 cycles). The period is measured from the controller's own
   `/cmd_vel_nav` publication times.

A self-test measures all three, plus `k_phi` set/read-back and
`goal_status == 4`, and the sweep does not run unless it passes.

## 6. Correction to the record

`DAY_2026-09-08.md` §8 lists this controller as blocked by
`Collision detected in trajectory`, with the costmap suspected. **That diagnosis
was wrong, and the cause was mine.** `graceful_controller.cpp:200-222` selects a
motion target by walking the plan backwards with

```
if (dist_to_target > max_lookahead)  continue;
else if (dist_to_target < min_lookahead)  break;
```

and `make_params.py` pinned `min_lookahead = max_lookahead = L`, copying what RPP
requires. Under that pin the first candidate to survive the `continue` trips the
`break`, the loop ends having simulated nothing, and control falls to line 250,
which throws `NoValidControl("Collision detected in trajectory")` — **the same
message whether a trajectory collided or no trajectory was ever built.** That is
why every costmap, footprint, `track_unknown_space`, frame and path-geometry
variant failed identically: none of them touched the selection window. With
`min_lookahead = 0.05` the controller drives, with 0 collision messages.

One earlier "fix" was actively harmful and is reverted in the reasoning:
`track_unknown_space: False` makes an off-map footprint point *fatal*, because
`footprintCostAtPose` returns `-1.0` off the map and
`static_cast<unsigned char>(-1.0) == 255 == NO_INFORMATION`, whose branch is
`is_tracking_unknown ? false : true` (`graceful_controller.cpp:375-405`).
