"""
θ_err + Autocorrelation 측정 (수정판)
simulation_main.py의 noise가 하드코딩(±3°, ±30°)되어 있어서
σ scaling 실험 제거, θ_err 측정 + autocorr만 진행

검증 내용:
  θ_err: aggregate된 방향 vs 이상적 방향의 오차
    → σ_θ ∝ N^? (1/N? 1/√N?)
    → θ_err의 분포 확인

  Autocorrelation:
    → ρ(k)가 빠르게 0으로 수렴 → 독립 스텝 가정 정당화

실행: python theta_autocorr_fixed.py
"""

import sys, numpy as np, json, time
import matplotlib.pyplot as plt
from scipy import stats
sys.path.insert(0, '.')

import simulation_main as sm
from simulation_main import (
    Circle, Square, DIRS, gen_votes,
    DT, VOTE_INT, DELAY_F, LOOK, FRAMES, SMOOTH
)

print("=" * 60)
print("θ_err + Autocorrelation 측정")
print("=" * 60)
print(f"  DT={DT:.4f}s, VOTE_INT={VOTE_INT}, DELAY_F={DELAY_F}")
print(f"  FRAMES={FRAMES}, LOOK={LOOK}")
print(f"  noise: accurate±3°, other±30° (하드코딩)")


def measure_theta_err(traj, N_agents, troll_ratio, speed, seed):
    """
    θ_err = 집계 방향 - 이상적 방향 (degrees)
    매 VOTE_INT마다 기록
    """
    rng = np.random.default_rng(seed)
    pos = traj.start()
    vel = np.zeros(2)
    pos_hist = [pos.copy()]
    prev_angle = 0.0
    cur_dir = np.array([1., 0.])

    theta_errs = []

    for f in range(FRAMES):
        if f % VOTE_INT == 0:
            delay_idx = max(0, len(pos_hist) - 1 - DELAY_F)
            delayed_pos = pos_hist[delay_idx]

            _, arc = traj.closest(delayed_pos)
            look_ahead_point = traj.at(arc + LOOK)
            ideal_dir = look_ahead_point - delayed_pos
            norm = np.linalg.norm(ideal_dir)
            if norm > 1e-10:
                ideal_dir /= norm
            ideal_angle = np.degrees(np.arctan2(ideal_dir[1], ideal_dir[0]))

            votes = gen_votes(ideal_angle, prev_angle, troll_ratio, N_agents, rng)
            prev_angle = ideal_angle

            vote_vectors = DIRS[votes]
            blend = vote_vectors.mean(axis=0)
            gamma = np.linalg.norm(blend)
            if gamma > 1e-10:
                cur_dir = blend / gamma

            actual_angle = np.degrees(np.arctan2(cur_dir[1], cur_dir[0]))

            # circular difference
            err = actual_angle - ideal_angle
            err = (err + 180) % 360 - 180
            theta_errs.append(err)

        vel += SMOOTH * (cur_dir * speed - vel)
        pos = pos + vel * DT
        pos_hist.append(pos.copy())

    return np.array(theta_errs)


# ── 실험 1: σ_θ vs N ──────────────────────────────────────
print("\n" + "=" * 60)
print("실험 1: σ_θ vs N (N=10~200)")
print("=" * 60)

N_VALUES = [10, 25, 50, 100, 150, 200]
SPEED_REF = 2.0
MC_THETA = 15
TROLL = 0.05
traj = Circle()

n_vals, sigma_theta_vals = [], []
t0 = time.time()

print(f"\n{'N':>6} {'σ_θ (°)':>10} {'mean_θ (°)':>12}")
print("-" * 32)

for N in N_VALUES:
    all_errs = []
    for mc in range(MC_THETA):
        seed = mc * 31 + N
        errs = measure_theta_err(traj, N, TROLL, SPEED_REF, seed)
        all_errs.extend(errs.tolist())

    all_errs = np.array(all_errs)
    sigma_theta = np.std(all_errs)
    mean_theta  = np.mean(all_errs)
    n_vals.append(N)
    sigma_theta_vals.append(sigma_theta)
    print(f"  {N:>4}  {sigma_theta:>9.4f}°  {mean_theta:>11.4f}°")

# power law fit
n_arr  = np.array(n_vals, dtype=float)
st_arr = np.array(sigma_theta_vals)
sl, ic, r, p, _ = stats.linregress(np.log(n_arr), np.log(st_arr))

print(f"\n  σ_θ ∝ N^{sl:.3f}  (R²={r**2:.4f}, p={p:.5f})")
if abs(sl + 0.5) < 0.15:
    print("  → σ_θ ∝ N^(-0.5) = 1/√N ✅")
    print("  → crowd averaging: 집계 에러가 √N으로 감소")
elif abs(sl + 1.0) < 0.15:
    print("  → σ_θ ∝ N^(-1) = 1/N")
    print("  → 강한 averaging 효과")
else:
    print(f"  → σ_θ ∝ N^{sl:.2f} — 혼합 효과")

print(f"\n  참고: 이 실험의 noise는 하드코딩 (±3°, ±30°)")
print(f"  → σ_θ vs N만 측정 가능, σ scaling 불가")


# ── 실험 2: troll ratio vs σ_θ ────────────────────────────
print("\n" + "=" * 60)
print("실험 2: troll ratio vs σ_θ")
print("=" * 60)
print("  troll이 늘어나면 집계 오차가 커지는가?")

TROLL_VALS = [0.05, 0.10, 0.20, 0.30, 0.40]
N_REF = 150
sigma_by_troll = {}
print(f"\n{'troll':>8} {'σ_θ (°)':>10}")
print("-" * 22)

for tr in TROLL_VALS:
    all_errs = []
    for mc in range(MC_THETA):
        seed = mc * 31 + 150
        errs = measure_theta_err(traj, N_REF, tr, SPEED_REF, seed)
        all_errs.extend(errs.tolist())
    sigma_by_troll[tr] = np.std(all_errs)
    print(f"  {tr:>6.0%}   {sigma_by_troll[tr]:>9.4f}°")

# troll vs σ_θ 상관
troll_arr = np.array(list(sigma_by_troll.keys()))
st_troll  = np.array(list(sigma_by_troll.values()))
sl_t, ic_t, r_t, p_t, _ = stats.linregress(troll_arr, st_troll)
print(f"\n  troll vs σ_θ: r={r_t:.3f}, p={p_t:.4f}")
print(f"  → troll ↑ → σ_θ ↑ {'✅' if r_t > 0.9 else '⚠️'}")


# ── 실험 3: Autocorrelation ───────────────────────────────
print("\n" + "=" * 60)
print("실험 3: Autocorrelation — 독립 스텝 가정 검증")
print("=" * 60)

all_errs_corr = []
for mc in range(30):  # 충분한 포인트
    seed = mc * 31 + 150
    errs = measure_theta_err(traj, 150, 0.05, SPEED_REF, seed)
    all_errs_corr.extend(errs.tolist())

all_errs_corr = np.array(all_errs_corr)
n_total = len(all_errs_corr)
sig_thr = 1.96 / np.sqrt(n_total)

print(f"\n  총 θ_err 포인트: {n_total}")
print(f"  평균: {np.mean(all_errs_corr):.4f}°")
print(f"  표준편차: {np.std(all_errs_corr):.4f}°")
print(f"  95% 유의 임계값: ±{sig_thr:.4f}")

max_lag = 25
acf = []
for k in range(max_lag+1):
    if k == 0:
        acf.append(1.0)
    else:
        c = np.corrcoef(all_errs_corr[:-k], all_errs_corr[k:])[0,1]
        acf.append(float(c))

print(f"\n  Autocorrelation ρ(k) (처음 10개):")
print(f"  {'k':>4} {'ρ(k)':>9} {'유의':>8}")
print("-" * 28)
k_c = max_lag
for k in range(min(11, max_lag+1)):
    sig = "유의" if (abs(acf[k]) > sig_thr and k > 0) else ("기준" if k==0 else "NS")
    if abs(acf[k]) <= sig_thr and k > 0 and k_c == max_lag:
        k_c = k
    print(f"  {k:>3}  {acf[k]:>8.4f}  {sig}")

tau_over_win = DELAY_F // VOTE_INT
print(f"\n  Correlation length k_c ≈ {k_c}")
print(f"  τ/WIN = DELAY_F/VOTE_INT = {DELAY_F}/{VOTE_INT} = {tau_over_win:.1f}")
print(f"  조건 k_c << τ/WIN: {k_c} << {tau_over_win:.1f} "
      f"{'✅' if k_c < tau_over_win else '❌'}")

if k_c < tau_over_win:
    print("  → 스텝 독립성 확인 ✅")
    print("  → v×√(τ/WIN) 누적 구조 정당화")


# ── 시각화 ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Panel 1: σ_θ vs N
ax = axes[0]
ax.scatter(n_vals, sigma_theta_vals, s=100, color='steelblue',
           zorder=5, label='data')
n_fit = np.linspace(8, 220, 100)
st_fit = np.exp(ic) * n_fit**sl
ax.plot(n_fit, st_fit, 'r--', linewidth=2,
        label=f'σ_θ ∝ N^{sl:.2f} (R²={r**2:.3f})')
ax.set_xlabel('N (crowd size)', fontsize=12)
ax.set_ylabel('σ_θ (degrees)', fontsize=12)
ax.set_title('(a) Direction Error σ_θ vs N', fontsize=12)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# Panel 2: σ_θ vs troll
ax = axes[1]
ax.plot([t*100 for t in TROLL_VALS], st_troll,
        'o-', color='coral', linewidth=2, markersize=8)
ax.set_xlabel('Troll ratio (%)', fontsize=12)
ax.set_ylabel('σ_θ (degrees)', fontsize=12)
ax.set_title('(b) Direction Error vs Troll Ratio', fontsize=12)
ax.grid(alpha=0.3)

# Panel 3: Autocorrelation
ax = axes[2]
lags = np.arange(max_lag+1)
colors_bar = ['navy' if k == 0 else
              ('steelblue' if abs(acf[k]) > sig_thr else 'lightblue')
              for k in lags]
ax.bar(lags, acf, color=colors_bar)
ax.axhline(y= sig_thr, color='red', linestyle='--',
           linewidth=1.5, label=f'95% CI (±{sig_thr:.3f})')
ax.axhline(y=-sig_thr, color='red', linestyle='--', linewidth=1.5)
ax.axvline(x=k_c, color='orange', linestyle='-',
           linewidth=2.5, label=f'k_c = {k_c}')
ax.set_xlabel('Lag k (aggregation steps)', fontsize=12)
ax.set_ylabel('Autocorrelation ρ(k)', fontsize=12)
ax.set_title('(c) Autocorrelation of θ_err\n'
             f'(τ/WIN={tau_over_win:.0f}, k_c={k_c})', fontsize=12)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('theta_autocorr_plot.png', dpi=180, bbox_inches='tight')

print(f"\n총 소요: {time.time()-t0:.0f}s")
print("저장: theta_autocorr_plot.png")

# 결과 저장
with open('theta_autocorr_results.json', 'w') as f:
    json.dump({
        'sigma_vs_N': {
            'N_values': N_VALUES, 'sigma_theta': sigma_theta_vals,
            'power_law': sl, 'R2': r**2
        },
        'sigma_vs_troll': {
            'troll_values': TROLL_VALS,
            'sigma_theta': list(st_troll),
            'r': r_t
        },
        'autocorr': {
            'acf': acf, 'k_c': k_c,
            'sig_threshold': sig_thr,
            'tau_over_win': tau_over_win,
            'independence': k_c < tau_over_win
        }
    }, f, indent=2)
print("저장: theta_autocorr_results.json")
