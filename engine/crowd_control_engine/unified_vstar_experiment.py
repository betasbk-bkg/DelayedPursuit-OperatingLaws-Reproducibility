"""
통합 실험 — v*(α, τ) 모델 검증
LE (latency) + TG (geometry) 통합

검증 목표:
  ① β ≈ 0.5: v*(τ) ∝ τ^(-0.5)
  ② separability: v*(α,τ) = f(α) × g(τ)
  ③ γ 결정: v*(α) ∝ cos^γ(α/2)
  ④ 통합 모델: v*(α,τ) = A × cos^γ(α/2) × τ^(-β)

실험 구조:
  E1: Circle + Square × 6τ × 3troll × 13speed × MC=15 (7,020 runs)
  E2: Hexagon + Triangle × 3τ × 2troll × 13speed × MC=15 (2,340 runs)
  합계: 9,360 runs (~5~8시간)

실행: python3 unified_vstar_experiment.py
simulation_main.py와 같은 폴더에서 실행
결과: unified_vstar_results.json
"""

import sys, numpy as np, json, time
from scipy import stats, optimize
sys.path.insert(0, '.')
import simulation_main as sm
from simulation_main import Circle, Square

DT = 1/60

# ── Polygon trajectory 클래스 ─────────────────────────────
class RegularPolygon:
    """
    정다각형 trajectory
    exterior angle = 360°/n
    n=3: 삼각형 α=120°
    n=4: 사각형 α=90° (Square와 동일하지만 독립 구현)
    n=6: 육각형 α=60°
    
    radius: 외접원 반지름 (scale 통제)
    """
    def __init__(self, n_sides, radius=10.0):
        self.name = f'polygon_{n_sides}'
        self.n = n_sides
        self.exterior_angle = 360.0 / n_sides  # 이게 corner angle α

        angles = np.linspace(0, 2*np.pi, n_sides, endpoint=False)
        verts = np.array([[radius*np.cos(a), radius*np.sin(a)]
                          for a in angles])
        verts = np.vstack([verts, verts[0]])  # 닫힌 경로

        self.segs = [(verts[i], verts[i+1]) for i in range(n_sides)]
        self.lens = [np.linalg.norm(b-a) for a,b in self.segs]
        self.circ = sum(self.lens)
        self.cum  = np.array([0]+list(np.cumsum(self.lens)))

        # corner angle 검증
        corners = []
        for i in range(len(self.segs)-1):
            v1 = self.segs[i][1]-self.segs[i][0]
            v2 = self.segs[i+1][1]-self.segs[i+1][0]
            d = np.clip(np.dot(v1/np.linalg.norm(v1),
                               v2/np.linalg.norm(v2)),-1,1)
            corners.append(np.degrees(np.arccos(d)))
        self.actual_corner = float(np.mean(corners))

    def closest(self, p):
        bd,bp,ba = 1e10, self.segs[0][0], 0.
        for i,(a,b) in enumerate(self.segs):
            v=b-a; l2=v@v
            if l2<1e-10: continue
            t=np.clip((p-a)@v/l2, 0, 1)
            pt=a+t*v; d=np.linalg.norm(p-pt)
            if d<bd: bd,bp,ba=d,pt,self.cum[i]+t*self.lens[i]
        return bp,ba

    def at(self, arc):
        arc=arc%self.circ
        for i,(a,b) in enumerate(self.segs):
            if arc<=self.cum[i+1]+1e-9:
                t=(arc-self.cum[i])/self.lens[i]
                return a+np.clip(t,0,1)*(b-a)
        return self.segs[-1][1]

    def start(self):
        return self.segs[0][0].copy()


# ── 실험 파라미터 ─────────────────────────────────────────
# τ 조건
TAU_ALL = {100:6, 200:12, 300:18, 433:26, 600:36, 1000:60}
TAU_SUB = {100:6, 433:26, 1000:60}  # E2용 (대표값 3개)

SPEEDS = [0.25,0.5,0.75,1.0,1.25,1.5,1.75,
          2.0,2.25,2.5,3.0,4.0,5.0]
N  = 150
MC = 15

# E1 trajectory (α=0°, 90°)
E1_TRAJS = {
    'circle': (Circle(), 0.0),
    'square': (Square(), 90.0),
}

# E2 trajectory (α=60°, 120°) — 육각형, 삼각형
# radius=10으로 통일 (Square side=4×2=8m 비슷한 scale)
hexagon  = RegularPolygon(6, radius=10.0)
triangle = RegularPolygon(3, radius=10.0)

E2_TRAJS = {
    'hexagon':  (hexagon,  hexagon.actual_corner),
    'triangle': (triangle, triangle.actual_corner),
}

E1_TROLLS = [0.05, 0.20, 0.40]
E2_TROLLS = [0.05, 0.40]

# 총 runs 계산
E1_runs = len(E1_TRAJS)*len(TAU_ALL)*len(E1_TROLLS)*len(SPEEDS)*MC
E2_runs = len(E2_TRAJS)*len(TAU_SUB)*len(E2_TROLLS)*len(SPEEDS)*MC
total = E1_runs + E2_runs

# ── 무결성 확인 ───────────────────────────────────────────
print("="*65)
print("실험 설계 무결성 확인")
print("="*65)

print("\n[E1 Trajectory]")
print(f"{'Name':<12} {'α':>8} {'circ(m)':>10}")
print("-"*34)
for name, (t, alpha) in E1_TRAJS.items():
    print(f"  {name:<10} {alpha:>7.1f}°  {t.circ:>9.2f}")

print("\n[E2 Trajectory] — 중간 α 검증용")
print(f"{'Name':<12} {'α설계':>8} {'α실제':>9} {'circ(m)':>10}")
print("-"*44)
for name, (t, alpha) in E2_TRAJS.items():
    print(f"  {name:<10} {t.exterior_angle:>7.1f}°  {t.actual_corner:>8.2f}°  {t.circ:>9.2f}")

print(f"\n[τ 조건 — E1: 6개, E2: 3개]")
print(f"{'τ(ms)':>8} {'frames':>8} {'실제ms':>10}")
print("-"*30)
for ms, f in TAU_ALL.items():
    tag = " ← E2 포함" if ms in TAU_SUB else ""
    print(f"  {ms:>6}  {f:>7}  {f*DT*1000:>9.1f}{tag}")

print(f"\n[통제 변수]")
print(f"  N=150, method=fixed, MC=15")
print(f"  speeds: {SPEEDS[0]}~{SPEEDS[-1]} m/s ({len(SPEEDS)}개)")
print(f"  monkey-patch: sm.DELAY_F 복원 확인됨 ✅")

print(f"\n[총 runs]")
print(f"  E1: {E1_runs:,} runs")
print(f"  E2: {E2_runs:,} runs")
print(f"  합계: {total:,} runs")
print(f"  예상: ~{total*0.08/3600:.1f}~{total*0.12/3600:.1f}시간")

# ── v* 측정 함수 ──────────────────────────────────────────
def measure_vstar(traj, tau_frames, troll, speeds=SPEEDS, mc=MC):
    """DELAY_F monkey-patch 후 v* 측정"""
    orig = sm.DELAY_F
    sm.DELAY_F = tau_frames
    try:
        rmses = []
        for v in speeds:
            r = sm.run_condition(traj, N, troll, mc,
                                 method='fixed', speed_override=v,
                                 seed_base=31, seed_offset=N)
            rmses.append(r['rmse_mean'])
    finally:
        sm.DELAY_F = orig  # 반드시 복원

    # quadratic fit으로 v* 정밀 추정
    idx = int(np.argmin(rmses))
    vstar = float(speeds[idx])
    if 1 <= idx <= len(speeds)-2:
        try:
            c = np.polyfit(speeds[idx-1:idx+2],
                           rmses[idx-1:idx+2], 2)
            if c[0] > 0:
                vf = -c[1]/(2*c[0])
                if speeds[idx-1] <= vf <= speeds[idx+1]:
                    vstar = float(vf)
        except: pass

    return float(vstar), [float(r) for r in rmses]


# ── 실험 실행 ─────────────────────────────────────────────
print("\n" + "="*65)
print(f"실험 시작 (MC={MC})")
print("="*65)

results = {}
t0 = time.time()
done = 0

# E1: Circle + Square, 전체 τ
print("\n--- E1: Circle + Square ---")
for tname, (traj, alpha) in E1_TRAJS.items():
    results[tname] = {'alpha': alpha, 'data': {}}
    for tau_ms, tau_f in TAU_ALL.items():
        results[tname]['data'][tau_ms] = {}
        for troll in E1_TROLLS:
            vstar, rmses = measure_vstar(traj, tau_f, troll)
            results[tname]['data'][tau_ms][troll] = {
                'vstar': vstar, 'rmses': rmses
            }
            done += len(SPEEDS)*MC
            elapsed = time.time()-t0
            eta = elapsed/done*(E1_runs+E2_runs-done)/60 if done>0 else 0
            print(f"  {tname:<8} τ={tau_ms:>4}ms tr={troll:.0%}: "
                  f"v*={vstar:.3f}  [ETA {eta:.0f}min]")

# E2: Hexagon + Triangle, 대표 τ
print("\n--- E2: Hexagon + Triangle ---")
for tname, (traj, alpha) in E2_TRAJS.items():
    results[tname] = {'alpha': alpha, 'data': {}}
    for tau_ms, tau_f in TAU_SUB.items():
        results[tname]['data'][tau_ms] = {}
        for troll in E2_TROLLS:
            vstar, rmses = measure_vstar(traj, tau_f, troll)
            results[tname]['data'][tau_ms][troll] = {
                'vstar': vstar, 'rmses': rmses
            }
            done += len(SPEEDS)*MC
            elapsed = time.time()-t0
            eta = elapsed/done*(E1_runs+E2_runs-done)/60 if done>0 else 0
            print(f"  {tname:<8} τ={tau_ms:>4}ms tr={troll:.0%}: "
                  f"v*={vstar:.3f}  [ETA {eta:.0f}min]")

total_time = time.time()-t0
print(f"\n총 소요: {total_time/3600:.2f}시간 ({total_time/60:.0f}분)")


# ── 분석 ─────────────────────────────────────────────────
print("\n" + "="*65)
print("분석 1: β 추정 (τ exponent)")
print("="*65)

tau_list = sorted(TAU_ALL.keys())
log_tau  = np.log(tau_list)

beta_est = {}
for tname in ['circle', 'square']:
    alpha = results[tname]['alpha']
    v5 = [results[tname]['data'][t][0.05]['vstar'] for t in tau_list]
    log_v = np.log(v5)
    sl, ic, r, p, _ = stats.linregress(log_tau, log_v)
    A = np.exp(ic); beta = -sl
    beta_est[tname] = {'A': A, 'beta': beta, 'r2': r**2, 'p': p}
    print(f"\n{tname} (α={alpha:.0f}°):")
    print(f"  v*(τ) = {A:.3f} × τ^(-{beta:.3f})")
    print(f"  R²={r**2:.4f}, p={p:.5f}")

# β가 동일한지
b_c = beta_est['circle']['beta']
b_s = beta_est['square']['beta']
print(f"\n  β(circle) = {b_c:.3f}")
print(f"  β(square) = {b_s:.3f}")
print(f"  β 차이    = {abs(b_c-b_s):.3f} {'✅ 동일로 볼 수 있음' if abs(b_c-b_s)<0.1 else '❌ 다름'}")
beta_mean = (b_c + b_s) / 2
print(f"  β 평균    = {beta_mean:.3f} {'≈ 0.5 (1/√τ)' if abs(beta_mean-0.5)<0.1 else ''}")

print("\n" + "="*65)
print("분석 2: Separability 검증")
print("="*65)
print("  v*(circle,τ) / v*(square,τ) = τ와 무관해야 함")
print(f"\n  {'τ(ms)':>8} {'v*(circle)':>12} {'v*(square)':>12} {'ratio':>8}")
print("  " + "-"*46)
ratios = []
for tau_ms in tau_list:
    vc = results['circle']['data'][tau_ms][0.05]['vstar']
    vs = results['square']['data'][tau_ms][0.05]['vstar']
    ratio = vc/vs
    ratios.append(ratio)
    print(f"  {tau_ms:>6}    {vc:>11.3f}    {vs:>11.3f}  {ratio:>7.3f}")

ratio_cv = np.std(ratios)/np.mean(ratios)
print(f"\n  ratio CV = {ratio_cv:.3f} {'✅ separable (CV<0.1)' if ratio_cv<0.1 else '❌ not separable'}")

print("\n" + "="*65)
print("분석 3: γ 추정 (corner angle exponent)")
print("="*65)

# τ=433ms (기준)에서 α vs v* 관계
tau_ref = 433
alpha_list, vstar_list = [], []
for tname in ['circle', 'hexagon', 'square', 'triangle']:
    if tname not in results: continue
    alpha = results[tname]['alpha']
    if tau_ref in results[tname]['data']:
        v = results[tname]['data'][tau_ref][0.05]['vstar']
        alpha_list.append(alpha)
        vstar_list.append(v)
        print(f"  {tname:<10} α={alpha:.1f}°: v*={v:.3f}")

if len(alpha_list) >= 3:
    # cos^γ(α/2) fitting
    def model_gamma(alpha_deg, A_ref, gamma):
        return A_ref * np.cos(np.radians(np.array(alpha_deg)/2))**gamma

    try:
        popt, pcov = optimize.curve_fit(
            model_gamma,
            alpha_list, vstar_list,
            p0=[2.0, 2.0], bounds=([0.5, 0.1], [5.0, 10.0])
        )
        A_ref, gamma = popt
        pred = model_gamma(alpha_list, A_ref, gamma)
        rmse = np.sqrt(np.mean((np.array(vstar_list)-pred)**2))
        print(f"\n  v*(α) = {A_ref:.3f} × cos^{gamma:.3f}(α/2)")
        print(f"  RMSE = {rmse:.4f} m/s")
        print(f"  γ {'≈ 2.0 (cos² 가설 지지)' if abs(gamma-2.0)<0.5 else f'= {gamma:.2f} (cos² 아님)'}")
    except Exception as e:
        print(f"  Fitting 실패: {e}")

print("\n" + "="*65)
print("분석 4: 통합 모델 v*(α,τ) = A × cos^γ(α/2) × τ^(-β)")
print("="*65)

# 3-parameter fitting
all_alpha, all_tau, all_vstar = [], [], []
for tname in results:
    alpha = results[tname]['alpha']
    for tau_ms, troll_data in results[tname]['data'].items():
        v = troll_data[0.05]['vstar']
        all_alpha.append(alpha)
        all_tau.append(tau_ms)
        all_vstar.append(v)

all_alpha = np.array(all_alpha)
all_tau   = np.array(all_tau, dtype=float)
all_vstar = np.array(all_vstar)

def unified_model(X, A, gamma, beta):
    alpha_deg, tau_ms = X
    return A * np.cos(np.radians(alpha_deg/2))**gamma * tau_ms**(-beta)

try:
    popt, pcov = optimize.curve_fit(
        unified_model,
        (all_alpha, all_tau), all_vstar,
        p0=[200.0, 2.0, 0.5],
        bounds=([1.0, 0.1, 0.1], [10000.0, 8.0, 2.0])
    )
    A_fit, gamma_fit, beta_fit = popt
    pred_all = unified_model((all_alpha, all_tau), *popt)
    rmse_all = np.sqrt(np.mean((all_vstar-pred_all)**2))
    r2_all = 1 - np.sum((all_vstar-pred_all)**2)/np.sum((all_vstar-np.mean(all_vstar))**2)

    print(f"\n  v*(α,τ) = {A_fit:.2f} × cos^{gamma_fit:.3f}(α/2) × τ^(-{beta_fit:.3f})")
    print(f"  R² = {r2_all:.4f}")
    print(f"  RMSE = {rmse_all:.4f} m/s")
    print(f"\n  γ = {gamma_fit:.3f} {'→ cos² 가설 지지 ✅' if abs(gamma_fit-2)<0.5 else '→ cos² 아님'}")
    print(f"  β = {beta_fit:.3f} {'→ 1/√τ 지지 ✅' if abs(beta_fit-0.5)<0.1 else '→ √ 아님'}")

except Exception as e:
    print(f"  Fitting 실패: {e}")

# ── 결과 저장 ─────────────────────────────────────────────
save = {
    'params': {'MC': MC, 'N': N, 'speeds': SPEEDS,
               'tau_all': {str(k):v for k,v in TAU_ALL.items()},
               'tau_sub': {str(k):v for k,v in TAU_SUB.items()}},
    'results': {
        tname: {
            'alpha': d['alpha'],
            'data': {
                str(tau): {
                    str(tr): {'vstar': tv['vstar'], 'rmses': tv['rmses']}
                    for tr, tv in trd.items()
                }
                for tau, trd in d['data'].items()
            }
        }
        for tname, d in results.items()
    }
}
with open('unified_vstar_results.json', 'w') as f:
    json.dump(save, f, indent=2)
print(f"\n결과 저장: unified_vstar_results.json")
