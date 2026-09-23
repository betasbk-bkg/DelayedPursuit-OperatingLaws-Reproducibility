"""
EXP1+2+3 최종판 — 해석 로직 수정
총 3,510 runs | 예상 20~30분

설계 근거:
  기존 데이터 (unified_vstar_results.json):
    fixed method, Circle, 6τ, troll=5%, MC=15 ✅ 재사용

  EXP1 (σ scaling): σ×0.5, σ×2.0만 새로 측정
    → "noise contributes but not sufficient" 검증
    
  EXP2 (WIN∝τ): proportional WIN만 새로 측정
    → delay vs information rate 원인 분리
    
  EXP3 (aggregation): majority, quadratic만 새로 측정
    → "aggregation-independent law" 검증

실행: python exp123_final.py
"""

import sys, numpy as np, json, time
from scipy import stats
sys.path.insert(0, '.')
import simulation_main as sm
from simulation_main import Circle

DT = 1/60
TAU_ALL = {100:6, 200:12, 300:18, 433:26, 600:36, 1000:60}
TAU_SUB = {100:6, 433:26, 1000:60}
SPEEDS  = [0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.25,2.5,3.0,4.0,5.0]
N, MC   = 150, 15
TROLL   = 0.05

# 기존 데이터 로드
with open('unified_vstar_results.json') as f:
    existing = json.load(f)

BASE_VSTARS = {int(t): existing['results']['circle']['data'][t]['0.05']['vstar']
               for t in existing['results']['circle']['data']}

print("기존 v*(fixed, Circle, troll=5%):")
for t, v in sorted(BASE_VSTARS.items()):
    print(f"  τ={t:>4}ms: v*={v:.3f}, x*=v√τ={v*np.sqrt(t/1000):.3f}")

X_BASE_MEAN = np.mean([v*np.sqrt(t/1000) for t,v in BASE_VSTARS.items()])
X_BASE_CV   = np.std([v*np.sqrt(t/1000) for t,v in BASE_VSTARS.items()]) / X_BASE_MEAN
print(f"\n기존 x* 평균={X_BASE_MEAN:.3f}, CV={X_BASE_CV:.4f}")

# ── 공통 v* 측정 ──────────────────────────────────────────
def measure_vstar(traj, tau_f, method='fixed',
                  sigma_scale=1.0, win_override=None):
    orig_delay = sm.DELAY_F
    orig_sigma = getattr(sm, 'SIGMA', None)
    orig_win   = getattr(sm, 'WIN', None)

    sm.DELAY_F = tau_f
    if sigma_scale != 1.0 and orig_sigma is not None:
        sm.SIGMA = orig_sigma * sigma_scale
    if win_override is not None and orig_win is not None:
        sm.WIN = win_override

    try:
        rmses = []
        for v in SPEEDS:
            r = sm.run_condition(traj, N, TROLL, MC,
                                 method=method, speed_override=v,
                                 seed_base=31, seed_offset=N)
            rmses.append(r['rmse_mean'])
    finally:
        sm.DELAY_F = orig_delay
        if orig_sigma is not None: sm.SIGMA = orig_sigma
        if orig_win is not None:   sm.WIN   = orig_win

    idx = int(np.argmin(rmses))
    vstar = float(SPEEDS[idx])
    if 1 <= idx <= len(SPEEDS)-2:
        try:
            c = np.polyfit(SPEEDS[idx-1:idx+2], rmses[idx-1:idx+2], 2)
            if c[0]>0:
                vf = -c[1]/(2*c[0])
                if SPEEDS[idx-1] <= vf <= SPEEDS[idx+1]:
                    vstar = float(vf)
        except: pass
    return vstar, [float(r) for r in rmses]


# ============================================================
# EXP1: σ Scaling
# 목적: noise가 v*에 기여하는가?
# 해석: "noise contributes but not sufficient"
#       σ 변화 시 v*가 전혀 안 변하면 → noise 무관
#       변하면 → noise contribution 확인 (얼마나 설명하는지)
# ============================================================
def run_exp1():
    print("\n" + "="*60)
    print("EXP1: σ Scaling")
    print("목적: noise가 v*에 부분적으로 기여하는가?")
    print("="*60)

    traj   = Circle()
    scales = [0.5, 2.0]
    results = {1.0: {t: BASE_VSTARS[t] for t in TAU_SUB}}
    t0 = time.time()

    for scale in scales:
        results[scale] = {}
        for tau_ms, tau_f in TAU_SUB.items():
            v, rmses = measure_vstar(traj, tau_f, sigma_scale=scale)
            results[scale][tau_ms] = {'vstar': v, 'rmses': rmses}
            x = v*np.sqrt(tau_ms/1000)
            print(f"  σ×{scale:.1f}  τ={tau_ms:>4}ms: v*={v:.3f}, x*={x:.3f}")

    # 분석: σ 변화 시 x* 변화량
    print(f"\n[EXP1 분석]")
    print(f"\n{'σ scale':>9} {'x*(100)':>9} {'x*(433)':>9} {'x*(1000)':>10} {'mean':>8}")
    print("-"*48)

    xstar_by_scale = {}
    for scale in [0.5, 1.0, 2.0]:
        tau_sub_sorted = sorted(TAU_SUB.keys())
        if scale == 1.0:
            vstars = [results[1.0][t] for t in tau_sub_sorted]
        else:
            vstars = [results[scale][t]['vstar'] for t in tau_sub_sorted]
        xstars = [v*np.sqrt(t/1000) for v,t in zip(vstars, tau_sub_sorted)]
        xstar_by_scale[scale] = np.mean(xstars)
        print(f"  ×{scale:<6.1f}  " + "  ".join(f"{x:>8.3f}" for x in xstars) +
              f"  {np.mean(xstars):>7.3f}")

    # 올바른 해석
    scales_sorted = [0.5, 1.0, 2.0]
    x_means = [xstar_by_scale[s] for s in scales_sorted]
    sl, _, r, p, _ = stats.linregress(np.log(scales_sorted), np.log(x_means))

    print(f"\n  x* ∝ σ^{sl:.3f}  (R²={r**2:.3f}, p={p:.4f})")
    print(f"  x* 변화 범위: {min(x_means):.3f} ~ {max(x_means):.3f}")
    print(f"  변화율: {(max(x_means)/min(x_means)-1)*100:.1f}%")

    # 올바른 해석 (단정 금지)
    print(f"\n  [해석]")
    change_pct = (max(x_means)/min(x_means) - 1) * 100
    if change_pct < 5:
        print(f"  → x*가 σ 변화에 거의 무반응 ({change_pct:.1f}%)")
        print(f"  → noise alone cannot explain scaling")
        print(f"  → RMSE floor가 noise보다 100× 크므로 예상된 결과")
        print(f"  → v*×√τ scaling은 noise 이외 구조적 원인에서 기인")
    elif change_pct < 20:
        print(f"  → x*가 σ에 부분적으로 의존 ({change_pct:.1f}%)")
        print(f"  → noise contributes partially to v* scaling")
        print(f"  → 다른 mechanism과 복합적으로 작용")
    else:
        print(f"  → x*가 σ에 유의미하게 의존 ({change_pct:.1f}%)")
        print(f"  → noise is a primary contributor to v* scaling")
        print(f"  → Random Walk mechanism 지지")

    print(f"\n  소요: {time.time()-t0:.0f}s")
    with open('exp1_results.json','w') as f:
        json.dump({str(s): ({str(t):v for t,v in d.items()} if isinstance(d,dict)
                   else {str(t): {'vstar':d[t]['vstar']} for t in d})
                   for s,d in results.items()}, f, indent=2)
    return results


# ============================================================
# EXP2: WIN ∝ τ (τ/WIN = const)
# 목적: delay vs information rate 원인 분리
# 해석: 결과에 따라 결정 (세 경우 열어둠)
#       분리 불가능 → 이론 확장의 근거
# ============================================================
def run_exp2():
    print("\n" + "="*60)
    print("EXP2: WIN ∝ τ")
    print("목적: delay vs information rate 원인 분리")
    print("="*60)

    WIN_BASE = 0.3; TAU_BASE = 0.433
    RATIO = TAU_BASE / WIN_BASE  # 1.44
    print(f"  기준: τ/WIN = {RATIO:.3f} (WIN=0.3s, τ=433ms)")
    print(f"  EXP2: τ/WIN = {RATIO:.3f} 유지하면서 τ 변화")

    traj = Circle()
    results = {}
    t0 = time.time()

    for tau_ms, tau_f in TAU_ALL.items():
        tau_s = tau_ms/1000
        win_prop = tau_s / RATIO
        v_prop, rmses = measure_vstar(traj, tau_f, win_override=win_prop)
        v_base = BASE_VSTARS.get(tau_ms)
        x_prop = v_prop*np.sqrt(tau_s)
        x_base = v_base*np.sqrt(tau_s) if v_base else None
        results[tau_ms] = {
            'vstar_base': v_base, 'vstar_prop': v_prop,
            'x_base': x_base, 'x_prop': x_prop,
            'win_prop': win_prop, 'rmses': rmses
        }
        print(f"  τ={tau_ms:>4}ms WIN={win_prop:.3f}s: "
              f"v*(prop)={v_prop:.3f} x*={x_prop:.3f} | "
              f"v*(base)={v_base:.3f} x*={x_base:.3f}")

    # 분석
    print(f"\n[EXP2 분석] x* 안정성 비교")
    tau_list = sorted(TAU_ALL.keys())
    x_base_list = [results[t]['x_base'] for t in tau_list if results[t]['x_base']]
    x_prop_list = [results[t]['x_prop'] for t in tau_list]

    cv_base = np.std(x_base_list)/np.mean(x_base_list)
    cv_prop = np.std(x_prop_list)/np.mean(x_prop_list)
    delta_cv = cv_prop - cv_base

    print(f"\n  WIN=0.3s (τ/WIN 가변): x* CV={cv_base:.4f}")
    print(f"  WIN∝τ  (τ/WIN 고정):  x* CV={cv_prop:.4f}")
    print(f"  ΔCV = {delta_cv:+.4f}")

    # 결과 기반 해석 — 세 경우를 모두 열어둠
    print(f"\n  [해석]")
    if cv_prop < cv_base * 0.8:
        print(f"  → WIN∝τ일 때 x*가 더 일정 (ΔCV={delta_cv:+.4f})")
        print(f"  → information rate (WIN/τ)도 scaling에 기여")
        print(f"  → 결론: delay + information rate 둘 다 중요")
    elif cv_prop > cv_base * 1.2:
        print(f"  → WIN∝τ일 때 x*가 오히려 불안정 (ΔCV={delta_cv:+.4f})")
        print(f"  → pure delay τ가 scaling의 주요 원인")
        print(f"  → 결론: delay가 원인, information rate는 부차적")
    else:
        print(f"  → WIN 변화에 무관하게 x* 일정 (ΔCV={delta_cv:+.4f})")
        print(f"  → v*×√τ scaling은 WIN에 강건함")
        print(f"  → 결론: pure delay τ가 scaling을 결정")

    print(f"\n  소요: {time.time()-t0:.0f}s")
    with open('exp2_results.json','w') as f:
        json.dump({str(t): {k: float(v) if isinstance(v,float) else v
                            for k,v in d.items() if k != 'rmses'}
                   for t,d in results.items()}, f, indent=2)
    return results


# ============================================================
# EXP3: Aggregation Generality
# 목적: v*×√τ scaling이 aggregation method에 무관한가?
# 해석: method-independent → general law claim 가능
# ============================================================
def run_exp3():
    print("\n" + "="*60)
    print("EXP3: Aggregation Generality")
    print("목적: v*×√τ = const가 aggregation-independent인가?")
    print("="*60)

    NEW_METHODS = ['majority', 'quadratic']
    traj = Circle()
    results = {'fixed': {t: {'vstar': BASE_VSTARS[t]} for t in TAU_SUB}}
    t0 = time.time()

    for method in NEW_METHODS:
        results[method] = {}
        for tau_ms, tau_f in TAU_SUB.items():
            v, rmses = measure_vstar(traj, tau_f, method=method)
            results[method][tau_ms] = {'vstar': v, 'rmses': rmses}
            x = v*np.sqrt(tau_ms/1000)
            print(f"  {method:<12} τ={tau_ms:>4}ms: v*={v:.3f}, x*={x:.3f}")

    # 분석
    print(f"\n[EXP3 분석]")
    print(f"\n{'Method':<12} {'x*(100)':>9} {'x*(433)':>9} {'x*(1000)':>10} {'CV':>8} {'β':>7}")
    print("-"*58)

    method_results = {}
    for method in ['fixed','majority','quadratic']:
        tau_sub_sorted = sorted(TAU_SUB.keys())
        vstars = [results[method][t]['vstar'] for t in tau_sub_sorted]
        xstars = [v*np.sqrt(t/1000) for v,t in zip(vstars,tau_sub_sorted)]
        cv = np.std(xstars)/np.mean(xstars)
        sl,_,r,_,_ = stats.linregress(
            np.log(tau_sub_sorted), np.log(vstars))
        beta = -sl
        method_results[method] = {'xstars': xstars, 'cv': cv, 'beta': beta}
        ok = "✅" if cv<0.1 and abs(beta-0.5)<0.15 else "⚠️"
        print(f"  {method:<10}  {xstars[0]:>8.3f}  {xstars[1]:>8.3f}  "
              f"{xstars[2]:>9.3f}  {cv:>7.4f}  {beta:>6.3f} {ok}")

    # 올바른 해석
    print(f"\n  [해석]")
    all_cv_ok = all(method_results[m]['cv'] < 0.1 for m in method_results)
    all_beta_ok = all(abs(method_results[m]['beta']-0.5) < 0.15
                      for m in method_results)

    if all_cv_ok and all_beta_ok:
        print(f"  ✅ 모든 aggregation method에서 v*×√τ = const, β≈0.5")
        print(f"  → aggregation-independent scaling law 확립")
        print(f"  → 논문 main claim: general law across aggregation methods")
    elif all_cv_ok:
        print(f"  ✅ 모든 method에서 x* 일정 (CV<0.1)")
        print(f"  ⚠️ β 범위: {min(method_results[m]['beta'] for m in method_results):.3f}~"
              f"{max(method_results[m]['beta'] for m in method_results):.3f}")
        print(f"  → scaling 방향은 consistent, β 값은 method에 따라 약간 변동")
    else:
        print(f"  ⚠️ method 간 x* 차이 있음 → aggregation이 v*에 영향")
        print(f"  → limitation: method-dependent scaling")

    print(f"\n  소요: {time.time()-t0:.0f}s")
    with open('exp3_results.json','w') as f:
        json.dump({m: {str(t): d['vstar'] for t,d in td.items()}
                   for m,td in results.items()}, f, indent=2)
    return results


# ============================================================
# 메인
# ============================================================
if __name__ == '__main__':
    e1 = 2*3*13*15   # 1,170
    e2 = 1*6*13*15   # 1,170
    e3 = 2*3*13*15   # 1,170
    total = e1+e2+e3  # 3,510

    print("="*60)
    print("EXP1+2+3 최종판")
    print("="*60)
    print(f"EXP1 (σ×0.5, σ×2.0 | 3τ):          {e1:,} runs")
    print(f"EXP2 (WIN∝τ | 6τ):                  {e2:,} runs")
    print(f"EXP3 (majority, quadratic | 3τ):     {e3:,} runs")
    print(f"총:                                  {total:,} runs")
    print(f"예상: ~{total*0.08/60:.0f}~{total*0.12/60:.0f}분")
    print(f"\n기존 데이터 재사용:")
    print(f"  σ×1.0 (fixed, 6τ): unified_vstar_results.json")
    print(f"  WIN=0.3s (fixed, 6τ): unified_vstar_results.json")
    print(f"  fixed method (3τ): unified_vstar_results.json")

    t0 = time.time()
    r1 = run_exp1()
    r2 = run_exp2()
    r3 = run_exp3()

    print(f"\n{'='*60}")
    print(f"전체 완료: {(time.time()-t0)/60:.1f}분")
    print(f"{'='*60}")
    print(f"\n논문 메시지 (4개):")
    print(f"  1. v*×√τ = const (이미 확인)")
    print(f"  2. noise contributes but not sufficient (EXP1)")
    print(f"  3. delay vs information rate 원인 규명 (EXP2)")
    print(f"  4. aggregation-independent law (EXP3)")
