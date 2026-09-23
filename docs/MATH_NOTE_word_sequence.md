# Word sequence of the quantized loop — multipliers, births, deaths (2026-09-14)

Proof-strengthening pass for the ND submission. Notation as in d116: q = kΔ/2,
u = delay margin, θ = u/(2q), section state ζ at a forward crossing, c = 1 − u − 2qζ,
M(τ) the exact phase integral, W_n = (−,+)ⁿ,+ with backward/forward crossings b_j, f_j,
S_j = Σ_{i≤j}(f_i − b_i).

## Proved results

**P1 (final segment and return map).** After the last delay window of W_n closes,
M(τ) = qτ² + cτ + u + 2qS_n exactly (the window terms sum to 2qS_n). The next forward
crossing T solves qT² + cT + u + 2qS_n = 1 and P(ζ) = ζ + 1 − T. At a fixed point T = 1
and c = a = 1 − u − q − 2qS_n.
*Proof.* Σ_e s_e(τ − τ_e − θ) over the n pairs equals Σ_j(b_j − f_j) = −S_n. □

**P2 (universal grazing death).** W_n dies when the final-segment parabola touches M = 0:
c² = 4q(u + 2qS_n). With u + 2qS_n = 1 − q − c this is c² + 4qc − 4q(1 − q) = 0, so
c = 2√q(1 − √q) and S_n = ((√q − 1)² − u)/(2q), for every n. n = 0 gives (1 + √u)². □

**T1 (multiplier of every word in closed form).** Let β_j = −2qb_j − c and
φ_j = 2qf_j + c + 2q (the slopes of M at b_j and f_j). Then S_n′ = dS_n/dc obeys
S′_j = A_j S′_{j−1} − B_j, S′_0 = 0, with A_j = (1 − 2q/φ_j)(1 − 2q/β_j) and
B_j = f_j/φ_j + (1 − 2q/φ_j) b_j/β_j, and

    P′ = (c − 4q² S_n′)/(c + 2q).

For n = 0, P′ = (1 − u − q)/(1 − u + q); at its death P′ = −√u.
*Proof.* Implicit differentiation of G(b_j) = u + 2qS_{j−1}, of the f_j quadratic, and of
the final crossing condition, with dc/dζ = −2q and ∂M/∂τ(1) = c + 2q. □
*Checks (d118).* Against a finite difference of P: 4.9 × 10⁻⁹; against the validated
integrator v3: 2.3 × 10⁻⁹ (55 fixed points, n = 0–3).

**P3 (slopes at any short-excursion border).** If f_j − b_j = θ, the parabola that
carries M on (b_j, f_j) vanishes at both ends, so it is symmetric about b_j + θ/2:
β_j = u/2, φ_j = 2q + u/2, A_j = (u − 4q)/(u + 4q), B_j = (f_j + b_j)/φ_j — for every n. □
*Check (d119d).* 83 borders, n = 1–3: 8.4 × 10⁻¹⁴.

**T2 (birth of W_1).** On the short-excursion border S = θ, c = 1 − 2u − q, the f-equation
gives b = (q − 1 + 3u/2)/(2q), f + b = (q − 1 + 2u)/q, the b-equation gives
q = q_lo(u) = 1 + √(1 − (2u − 1)² + u²/4), and by T1 and P3

    P_B = (q − 1 + 2u)(4q − u) / ((4q + u)(q + 1 − 2u)).

P_B = 1 reduces to q(14u − 8) = 0. Hence:
- u < 4/7: P_B < 1 and W_1 is born stable at the border q_lo (border collision);
- u > 4/7: P_B > 1; the stable W_1 is born at a smooth saddle-node q_F < q_lo inside the
  ordering domain (P′ = 1 there), and its unstable twin runs from the fold to the border.
The short-excursion and end-window borders cross where additionally b = 1 − 2θ, i.e.
q = 7u/2 − 1 and 8u² − 9u + 2 = 0: u = (9 + √17)/16 ≈ 0.8202. □
*Checks (d119c, d119d).* P_B at 25 borders: 3.7 × 10⁻¹⁴; q_B − q_lo: 1.3 × 10⁻¹⁵;
P′ = 1 at all 34 folds solved (|P_F − 1| < 10⁻⁶), each inside the ordering domain;
the type (border vs fold) agrees with P_B < 1 at all 92 borders; stable members confirmed
by v3 at 92 of 92.

**P4 (the short excursion at birth is the last one; W_2 in closed form).** At every
short-excursion birth of W_2 and W_3 in d119c (58 of 58) the excursion with f_j − b_j = θ is
j = n, so by the parabola symmetry of P3, b_n = −(c + u/2)/(2q) linearly, as for W_1.
Eliminating (sympy, d121; factor chosen by a 60-digit perturbation-ratio test):
- W_1 end-window border: 36q²u² − 96q²u + 48q² − 8qu³ + 64qu² − 32qu + 3u⁴ − 8u³ + 4u² = 0;
  the switch back to border type there is the root u = 0.834458974556 of the irreducible
  607u⁸ − 3104u⁷ + 7312u⁶ − 7808u⁵ + 416u⁴ + 6656u³ − 5888u² + 2048u − 256.
- W_2 short-excursion border: an algebraic curve of degree 8 in q and 8 in u
  (d121 JSON); at 8 border points of d119c (u = 0.2–0.9) its value is 10⁻¹²–10⁻¹⁴ of its
  value 10⁻³ away.
- W_2 codimension-two point: a root of a degree-22 integer polynomial,
  u = 0.5578470225516988 (numerical 0.5578470226).

## Computer-assisted stability proof (in progress, 2026-09-14)

Krawczyk operator on the square-root-free crossing system (unknowns b_j, f_j, c;
parameters appear once per equation), outward-rounded interval arithmetic, T1 multiplier
enclosed on the certified box, root sides certified by β_j > 0, φ_j > 0 (d120d). Earlier
attempts through the square-root recursion (d120, d120b, d120c) failed by near-cancellation
in c² − 4qr near births; kept as the record.
- Depth-3 run (4-way splits, before the sensitivity-based start): proved area 2.367 (n = 1),
  2.422 (n = 2), 2.434 (n = 3); unproved 9.4 × 10⁻⁴, 3.1 × 10⁻³, 2.0 × 10⁻² — for n = 1, 2 all
  stored failures lie within 2 % of the birth; for n = 3 also a strip at u ≈ 0.94, where the
  solution is ~13× more sensitive to u than to q.
- Rigorous per box; NOT yet rigorous: that the boxes cover the whole existence set and that
  the certified solution is W_n's physical branch (planned: exclusion boxes, interval death
  condition, interval ordering checks).

## Computed, not proved

| result | value | script |
|---|---|---|
| codimension-two u (short-excursion border), n = 1–4 | 4/7, 0.557847, 0.550126, 0.544958 (decreasing in n) | d119d |
| W_1 on the end-window border: switch back to border type | u = 0.834459 (closed form not derived) | d119d |
| fold width q_lo − q_F (W_1) | up to 4.4 × 10⁻³ (u = 0.80) | d119c |
| stable member along the existence intervals, n = 1–5, u = 0.10–0.95 | a fixed point with \|P′\| < 1 at all 6,188 points; closest approach to \|P′\| = 1 is −0.983 (W_1 at death, u = 0.95) | inline scan on d118 |
| multiplier at death, n = 1–8, u = 0.05–0.95 | 152 deaths, P′ ∈ [−0.9927, −0.2958]: no flip before grazing | d119b (B) |

**A residual explained.** App. B's "q_lo agrees with the integrated map to 0.0041" was
not error: the map's birth lies below q_lo by the fold width for 4/7 < u < (9 + √17)/16
(u = 0.70, 0.75, 0.80: −0.0008, −0.0017, −0.0041 against widths 0.0008, 0.0021, 0.0044),
and at u = 0.85 the birth is on the end-window border (−0.0026). What remains is the
continuation's grid resolution (≈ 4 × 10⁻⁴).

## Still open
- Stability of W_n for general n as a theorem (|P′| < 1 on the stable branch): the
  recursion of T1 gives the multiplier exactly, but no sign argument closes it yet;
  a computer-assisted (interval-arithmetic) proof over a bounded (u, q, n) range is the
  realistic route.
- Completeness of the attractor set (only W_n): computed (census), not proved.
- Closed forms for n ≥ 2 births and for the end-window switch (needs a CAS: sympy is not
  installed on this machine).

## Scripts
d118 (multiplier closed form, checks, scan) · d119 (superseded) · d119b (part B valid,
part A failed) · d119c (births as 2-D root problems) · d119d (closed-form P_B, u = 4/7,
codimension-two points).
