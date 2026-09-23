"""
EXP1 재실행 — σ_noise 직접 패치 (올바른 버전)

gen_votes 내부 ±3°, ±30°을 monkey-patch로 변경.
σ_scale 범위: 0.5, 1.0, 2.0, 5.0, 10.0, 20.0
(작은 변화부터 양자화 스텝을 초과하는 극단값까지)

8방향 양자화 스텝 = 45°, 허용 오차 = ±22.5°
예측:
  σ_accurate=±3°×scale, σ_other=±30°×scale
  scale ≤ 7: noise < quantization → v* 무변화 예상
  scale > 7: noise ≥ quantization → v* 변화 가능

실행: python exp1_fixed.py
소요: ~20~30분
결과: exp1_fixed_results.json
"""

import sys, numpy as np, json, time
from scipy import stats
sys.path.insert(0, '.')
import simulation_main as sm
from simulation_main import Circle

# ── gen_votes monkey-patch ────────────────────────────────
ORIGINAL_GEN_VOTES = sm.gen_votes

def patched_gen_votes(ideal_angle, prev_angle, troll_ratio,
                      N_agents, rng, sigma_scale=1.0):
    """gen_votes with σ_noise scaled by sigma_scale"""
    n_troll = round(N_agents * troll_ratio)
    n_troll = min(n_troll, N_agents)
    remaining = N_agents - n_troll
    n_accurate = round(remaining * 0.7368)
    n_slow     = round(remaining * 0.2105)
    n_other    = remaining - n_accurate - n_slow
    if n_other < 0:
        n_accurate += n_other
        n_other = 0

    assert n_accurate + n_slow + n_troll + n_other == N_agents
    assert n_accurate >= 0 and n_slow >= 0

    angles = np.empty(n_accurate + n_slow + n_other)
    idx = 0

    # ★ σ_noise scaled: ±3° → ±(3×scale)°
    sigma_acc = 3.0 * sigma_scale
    angles[idx:idx+n_accurate] = ideal_angle + rng.uniform(
        -sigma_acc, sigma_acc, n_accurate)
    idx += n_accurate

    # slow agents: lagged (σ_scale 영향 없음)
    diff = ideal_angle - prev_angle
    if diff > 180: diff -= 360
    if diff < -180: diff += 360
    if n_slow > 0:
        lag = rng.uniform(0.2, 0.5, n_slow)
        angles[idx:idx+n_slow] = prev_angle + diff * (1-lag)
        idx += n_slow

    # ★ σ_noise scaled: ±30° → ±(30×scale)°
    sigma_other = 30.0 * sigma_scale
    if n_other > 0:
        angles[idx:idx+n_other] = ideal_angle + rng.uniform(
            -sigma_other, sigma_other, n_other)
        idx += n_other

    non_troll_votes = sm.angle_to_dir(angles[:idx])
    troll_votes = rng.integers(0, 8, n_troll) if n_troll > 0 else np.array([], dtype=int)

    return np.concatenate([non_troll_votes, troll_votes]).astype(int)


# ── 무결성 체크 ───────────────────────────────────────────
print("=" * 65)
print("EXP1 재실행 — σ_noise 직접 패치")
print("=" * 65)

QUANT_STEP = 45.0
print(f"\n8방향 양자화 스텝 = {QUANT_STEP}°, 허용 오차 = ±{QUANT_STEP/2}°")
print(f"\n{'σ_scale':>9} {'σ_acc(°)':>10} {'σ_other(°)':>12} {'vs quant':>12}")
print("-" * 50)

SIGMA_SCALES = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
for s in SIGMA_SCALES:
    sa = 3.0*s; so = 30.0*s
    flag = ">> quantization!" if so > QUANT_STEP/2 else ("≈ quant" if so > 10 else "< quant")
    print(f"  ×{s:<6.1f}  {sa:>9.1f}°  {so:>11.1f}°  {flag}")

print(f"\n예측: scale ≤ 7 → v* 무변화 (noise < quantization)")
print(f"      scale > 7 → v* 변화 가능 (noise ≥ quantization)")

# ── v* 측정 ───────────────────────────────────────────────
SPEEDS = [0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.25,2.5,3.0,4.0,5.0]
TAU_VALS = {100:6, 433:26, 1000:60}
N, MC, TROLL = 150, 15, 0.05
traj = Circle()

def measure_vstar(sigma_scale, tau_frames):
    """σ_noise 패치 후 v* 측정"""
    # gen_votes를 patched 버전으로 교체
    original = sm.gen_votes
    def _patched(ia, pa, tr, na, rng):
        return patched_gen_votes(ia, pa, tr, na, rng, sigma_scale)
    sm.gen_votes = _patched

    orig_delay = sm.DELAY_F
    sm.DELAY_F = tau_frames

    try:
        rmses = []
        for v in SPEEDS:
            r = sm.run_condition(traj, N, TROLL, MC,
                                 method='fixed', speed_override=v,
                                 seed_base=31, seed_offset=N)
            rmses.append(r['rmse_mean'])
    finally:
        sm.gen_votes = original
        sm.DELAY_F = orig_delay

    idx = int(np.argmin(rmses))
    vstar = float(SPEEDS[idx])
    if 1 <= idx <= len(SPEEDS)-2:
        try:
            c = np.polyfit(SPEEDS[idx-1:idx+2], rmses[idx-1:idx+2], 2)
            if c[0] > 0:
                vf = -c[1]/(2*c[0])
                if SPEEDS[idx-1] <= vf <= SPEEDS[idx+1]:
                    vstar = float(vf)
        except: pass
    return vstar, [float(r) for r in rmses]


# ── 실험 실행 ─────────────────────────────────────────────
total = len(SIGMA_SCALES)*len(TAU_VALS)*len(SPEEDS)*MC
print(f"\n총 runs: {total:,} | 예상: ~{total*0.08/60:.0f}분")
print("\n" + "="*65)

results = {}
t0 = time.time()

print(f"\n{'σ_scale':>9} {'τ(ms)':>7} {'v*':>8} {'x*=v√τ':>8} {'RMSE_min':>10}")
print("-" * 48)

for scale in SIGMA_SCALES:
    results[scale] = {}
    for tau_ms, tau_f in TAU_VALS.items():
        v, rmses = measure_vstar(scale, tau_f)
        x = v * np.sqrt(tau_ms/1000)
        rmin = min(rmses)
        results[scale][tau_ms] = {'vstar': v, 'x_star': x,
                                   'rmse_min': rmin, 'rmses': rmses}
        print(f"  ×{scale:<6.1f}  {tau_ms:>5}  {v:>7.3f}  {x:>7.4f}  {rmin:>9.4f}")

print(f"\n총 소요: {time.time()-t0:.0f}s")

# ── 분석 ─────────────────────────────────────────────────
print("\n" + "="*65)
print("분석: σ_scale vs x* = v*×√τ")
print("="*65)

print(f"\n{'σ_scale':>9} {'x*(100ms)':>11} {'x*(433ms)':>11} {'x*(1000ms)':>12} {'mean':>8} {'CV':>7}")
print("-"*60)

scale_list, x_mean_list = [], []
for scale in SIGMA_SCALES:
    xs = [results[scale][t]['x_star'] for t in [100,433,1000]]
    xm = np.mean(xs); cv = np.std(xs)/xm
    scale_list.append(scale)
    x_mean_list.append(xm)
    print(f"  ×{scale:<6.1f}  {xs[0]:>10.4f}  {xs[1]:>10.4f}  {xs[2]:>11.4f}  {xm:>7.4f}  {cv:>6.4f}")

# σ_scale vs x* 관계
sl, ic, r, p, _ = stats.linregress(np.log(scale_list), np.log(x_mean_list))
print(f"\n  x* ∝ σ^{sl:.3f}  (R²={r**2:.4f}, p={p:.5f})")

# 변화 분석
x_baseline = x_mean_list[1]  # scale=1.0
print(f"\n  기준 (σ×1.0): x*={x_baseline:.4f}")
print(f"\n  {'σ_scale':>9} {'x*':>8} {'변화율':>9} {'판정'}")
print("-" * 40)
for scale, xm in zip(scale_list, x_mean_list):
    delta_pct = (xm/x_baseline - 1)*100
    if abs(delta_pct) < 5:
        judge = "무변화 ✅ (quantization 마스킹)"
    elif abs(delta_pct) < 15:
        judge = "소폭 변화 ⚠️"
    else:
        judge = "유의미한 변화 ❌"
    print(f"  ×{scale:<6.1f}  {xm:>7.4f}  {delta_pct:>+8.1f}%  {judge}")

# 이론 검증
print(f"\n[이론 검증] 8방향 양자화 마스킹 가설")
print(f"  가설: σ_noise < 양자화 허용오차(22.5°)이면 v* 변화 없음")
print(f"  σ_other (±30°×scale):")
for scale in SIGMA_SCALES:
    so = 30*scale
    masked = "마스킹 예상" if so < 22.5 else "초과 → 효과 가능"
    print(f"    ×{scale:.1f}: σ_other={so:.0f}° {masked}")

# 결론
print(f"\n{'='*65}")
print("결론")
print(f"{'='*65}")
if abs(x_mean_list[-1]/x_baseline - 1)*100 > 15:
    print(f"""
  scale=20에서 x*가 유의미하게 변함
  → σ_noise가 충분히 크면 v*에 영향
  → quantization floor를 초과하는 noise는 v*를 변경
  → 논문: "v* is σ-independent within the quantization-limited regime"
""")
else:
    print(f"""
  모든 σ_scale에서 x*가 일정
  → v*는 σ_noise에 완전히 무관
  → 8방향 양자화가 noise 효과를 완전히 마스킹
  → 논문: "v* is σ-independent due to quantization domination"
""")

# 저장
with open('exp1_fixed_results.json', 'w') as f:
    json.dump({
        'sigma_scales': SIGMA_SCALES,
        'tau_vals': list(TAU_VALS.keys()),
        'quantization_step': QUANT_STEP,
        'results': {str(s): {
            str(t): {'vstar': d['vstar'], 'x_star': d['x_star'],
                     'rmse_min': d['rmse_min']}
            for t, d in sd.items()}
            for s, sd in results.items()},
        'analysis': {
            'power_law_exponent': sl,
            'R2': r**2,
            'baseline_x_star': x_baseline,
        }
    }, f, indent=2)
print("저장: exp1_fixed_results.json")
