"""
EXP-A: 방향 양자화 수 변화 실험
8방향 → 16방향 → 36방향 → continuous (360방향 근사)

목적: β=0.5 scaling이 quantization에 의존하는가?

시나리오 A: β=0.5 유지 → quantization이 원인 아님 (더 근본적 구조)
시나리오 B: β가 변함  → quantization이 핵심 원인 확정

실행: python expa_quantization.py
소요: ~30~45분
결과: expa_results.json
"""

import sys, numpy as np, json, time
from scipy import stats
sys.path.insert(0, '.')
import simulation_main as sm
from simulation_main import Circle

DT = 1/60
TAU_ALL = {100:6, 200:12, 300:18, 433:26, 600:36, 1000:60}
SPEEDS  = [0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.25,2.5,3.0,4.0,5.0]
N, MC   = 150, 15
TROLL   = 0.05

# ── n_dirs별 방향 집합 생성 ───────────────────────────────
def make_dirs(n):
    """n방향 단위벡터 배열 생성 (8방향 DIRS와 동일 구조)"""
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.stack([np.cos(angles), np.sin(angles)], axis=1)

def angle_to_dir_n(angles_deg, dirs):
    """각도 배열을 n방향 중 가장 가까운 방향 인덱스로 변환"""
    n = len(dirs)
    step = 360.0 / n
    indices = (np.round(np.array(angles_deg) / step) % n).astype(int)
    return indices

def patched_gen_votes_n(ideal_angle, prev_angle, troll_ratio,
                         N_agents, rng, n_dirs=8):
    """gen_votes를 n방향 양자화로 패치"""
    dirs = make_dirs(n_dirs)

    n_troll    = round(N_agents * troll_ratio)
    remaining  = N_agents - n_troll
    n_accurate = round(remaining * 0.7368)
    n_slow     = round(remaining * 0.2105)
    n_other    = remaining - n_accurate - n_slow

    angles = np.empty(n_accurate + n_slow + n_other)
    idx = 0

    # accurate agents
    angles[idx:idx+n_accurate] = ideal_angle + rng.uniform(-3, 3, n_accurate)
    idx += n_accurate

    # slow agents (lagged)
    if n_slow > 0:
        diff = ideal_angle - prev_angle
        diff = (diff + 180) % 360 - 180
        lag = rng.uniform(0.2, 0.5, n_slow)
        angles[idx:idx+n_slow] = prev_angle + diff * (1 - lag)
        idx += n_slow

    # other agents
    if n_other > 0:
        angles[idx:idx+n_other] = ideal_angle + rng.uniform(-30, 30, n_other)
        idx += n_other

    # 양자화: n방향으로 스냅
    non_troll_votes = angle_to_dir_n(angles[:idx], dirs)

    # troll: n방향 중 무작위
    troll_votes = (rng.integers(0, n_dirs, n_troll)
                   if n_troll > 0 else np.array([], dtype=int))

    return np.concatenate([non_troll_votes, troll_votes]).astype(int)


# ── v* 측정 (n_dirs 패치) ────────────────────────────────
def measure_vstar_ndirs(traj, tau_f, n_dirs):
    """n_dirs 방향 양자화로 v* 측정"""
    dirs = make_dirs(n_dirs)
    original_gv = sm.gen_votes
    original_dirs = sm.DIRS
    orig_delay = sm.DELAY_F

    # DIRS 패치 (aggregate에서 사용)
    sm.DIRS = dirs
    sm.DELAY_F = tau_f

    def _patched(ia, pa, tr, na, rng):
        return patched_gen_votes_n(ia, pa, tr, na, rng, n_dirs)
    sm.gen_votes = _patched

    try:
        rmses = []
        for v in SPEEDS:
            r = sm.run_condition(traj, N, TROLL, MC,
                                 method='fixed', speed_override=v,
                                 seed_base=31, seed_offset=N)
            rmses.append(r['rmse_mean'])
    finally:
        sm.gen_votes = original_gv
        sm.DIRS = original_dirs
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


# ── 무결성 확인 ───────────────────────────────────────────
print("=" * 60)
print("EXP-A: Quantization 방향 수 변화")
print("=" * 60)

N_DIRS_LIST = [4, 8, 16, 36, 360]  # 4=극단, 8=기준, 360≈continuous

print(f"\n방향 수별 각도 분해능:")
for n in N_DIRS_LIST:
    step = 360/n
    sigma_th = step/2 / (3**0.5)  # uniform [-step/2, step/2] RMS
    tag = "(기준)" if n==8 else ("(≈continuous)" if n==360 else "")
    print(f"  {n:>4}방향: step={step:>7.2f}°, σ_θ_quant≈{sigma_th:>6.2f}° {tag}")

total = len(N_DIRS_LIST)*len(TAU_ALL)*len(SPEEDS)*MC
print(f"\n총 runs: {total:,} | 예상: ~{total*0.08/60:.0f}분")

# ── 실험 실행 ─────────────────────────────────────────────
print("\n" + "="*60)
print("실험 실행")
print("="*60)

traj = Circle()
results = {}
t0 = time.time()

for n_dirs in N_DIRS_LIST:
    results[n_dirs] = {}
    for tau_ms, tau_f in TAU_ALL.items():
        v, rmses = measure_vstar_ndirs(traj, tau_f, n_dirs)
        x = v * np.sqrt(tau_ms/1000)
        results[n_dirs][tau_ms] = {'vstar': v, 'x_star': x, 'rmses': rmses}
        elapsed = time.time()-t0
        print(f"  {n_dirs:>4}방향  τ={tau_ms:>4}ms: v*={v:.3f}, x*={x:.4f}")

print(f"\n총 소요: {time.time()-t0:.0f}s")

# ── 분석 ─────────────────────────────────────────────────
print("\n" + "="*60)
print("분석: n_dirs vs β")
print("="*60)

print(f"\n{'n_dirs':>8} {'σ_θ(°)':>8} {'x*_mean':>9} {'x*_CV':>8} {'β':>8} {'R²':>8}")
print("-"*55)

beta_list, ndirs_list = [], []
x8_mean = None

for n_dirs in N_DIRS_LIST:
    tau_list = sorted(results[n_dirs].keys())
    vstars = [results[n_dirs][t]['vstar'] for t in tau_list]
    xstars = [results[n_dirs][t]['x_star'] for t in tau_list]

    sl, ic, r, p, se = stats.linregress(np.log(tau_list), np.log(vstars))
    beta = -sl
    xm  = np.mean(xstars)
    cv  = np.std(xstars)/xm
    sigma_th = (360/n_dirs/2) / (3**0.5)

    if n_dirs == 8:
        x8_mean = xm

    ok = "✅" if (cv < 0.1 and abs(beta-0.5) < 0.1) else "⚠️"
    print(f"  {n_dirs:>6}    {sigma_th:>7.2f}  {xm:>8.4f}  {cv:>7.4f}  {beta:>7.3f}  {r**2:>7.4f} {ok}")
    beta_list.append(beta)
    ndirs_list.append(n_dirs)

# β vs n_dirs 상관
corr, p_corr = stats.pearsonr(ndirs_list, beta_list)
print(f"\n  β vs n_dirs: r={corr:.3f}, p={p_corr:.4f}")

# 판정
print("\n" + "="*60)
print("결론")
print("="*60)

beta_range = max(beta_list) - min(beta_list)
b360 = beta_list[-1]  # continuous (360)
b8   = beta_list[1]   # 8방향 기준

print(f"\n  β(4방향)  = {beta_list[0]:.3f}")
print(f"  β(8방향)  = {b8:.3f} (기준)")
print(f"  β(16방향) = {beta_list[2]:.3f}")
print(f"  β(36방향) = {beta_list[3]:.3f}")
print(f"  β(360방향)= {b360:.3f} (≈continuous)")
print(f"\n  β 범위: {beta_range:.3f}")
print(f"  8→360 변화: {(b360-b8):+.3f}")

if abs(b360 - b8) < 0.05:
    print(f"""
  → ✅ β=0.5는 quantization에 무관하게 성립
  → 더 근본적인 구조 (delay + zero-order hold)가 원인
  → 논문 claim: "β=0.5 is a fundamental property of 
                 delayed discrete-update control systems,
                 independent of vote quantization"
  → 이론이 더 GENERAL해짐 ← 노벨티 강화!
""")
elif beta_range > 0.1:
    print(f"""
  → ⚠️  β가 n_dirs에 의존
  → quantization이 β=0.5의 원인 중 하나
  → 논문 claim: "β=0.5 emerges from the interaction of 
                 delay and 8-direction quantization"
  → 이론이 더 SPECIFIC해짐 (quantization 없으면 다름)
""")
else:
    print(f"  → β가 약하게 변함 — 추가 분석 필요")

# σ_θ vs n_dirs 예측
print("\n[σ_θ 예측 vs n_dirs]")
for n in N_DIRS_LIST:
    pred = (360/n/2) / (3**0.5)
    print(f"  {n:>4}방향: σ_θ_pred ≈ {pred:.2f}°")
print(f"  → σ_θ가 n_dirs에 의존하면 diffusion 이론 지지")
print(f"  → σ_θ가 n_dirs에 무관하면 다른 오차 원인")

# 저장
with open('expa_results.json', 'w') as f:
    json.dump({
        'n_dirs_list': N_DIRS_LIST,
        'tau_vals': list(TAU_ALL.keys()),
        'results': {str(n): {str(t): {'vstar': d['vstar'], 'x_star': d['x_star']}
                    for t, d in nd.items()}
                    for n, nd in results.items()},
        'beta': {str(n): beta_list[i] for i, n in enumerate(N_DIRS_LIST)},
    }, f, indent=2)
print("\n저장: expa_results.json")
