"""
EXP2 수정 + EXP3 quadratic 보완

EXP2 수정: WIN이 아닌 VOTE_INT를 직접 패치
EXP3 보완: quadratic τ=100ms, speed range 10m/s로 확장
"""

import sys, numpy as np, json, time
from scipy import stats
sys.path.insert(0, '.')
import simulation_main as sm
from simulation_main import Circle

DT = 1/60
TAU_ALL = {100:6, 200:12, 300:18, 433:26, 600:36, 1000:60}
SPEEDS_STD = [0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.25,2.5,3.0,4.0,5.0]
SPEEDS_EXT = [0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.25,2.5,3.0,4.0,5.0,6.0,7.0,8.0,10.0]
N, MC = 150, 15
TROLL = 0.05

with open('unified_vstar_results.json') as f:
    existing = json.load(f)
BASE_VSTARS = {int(t): existing['results']['circle']['data'][t]['0.05']['vstar']
               for t in existing['results']['circle']['data']}

def measure_vstar(traj, tau_f, method='fixed',
                  vote_int_override=None, speeds=SPEEDS_STD):
    orig_delay    = sm.DELAY_F
    orig_vote_int = sm.VOTE_INT  # ← VOTE_INT 직접 패치

    sm.DELAY_F = tau_f
    if vote_int_override is not None:
        sm.VOTE_INT = vote_int_override

    try:
        rmses = []
        for v in speeds:
            r = sm.run_condition(traj, N, TROLL, MC,
                                 method=method, speed_override=v,
                                 seed_base=31, seed_offset=N)
            rmses.append(r['rmse_mean'])
    finally:
        sm.DELAY_F    = orig_delay
        sm.VOTE_INT   = orig_vote_int

    idx = int(np.argmin(rmses))
    vstar = float(speeds[idx])
    if 1 <= idx <= len(speeds)-2:
        try:
            c = np.polyfit(speeds[idx-1:idx+2], rmses[idx-1:idx+2], 2)
            if c[0]>0:
                vf = -c[1]/(2*c[0])
                if speeds[idx-1] <= vf <= speeds[idx+1]:
                    vstar = float(vf)
        except: pass
    return vstar, rmses


# ── EXP2 수정 ────────────────────────────────────────────
print("="*60)
print("EXP2 수정: VOTE_INT 직접 패치 (WIN 비례)")
print("="*60)

WIN_BASE = 0.3; TAU_BASE = 0.433
RATIO = TAU_BASE / WIN_BASE  # 1.44
print(f"기준 τ/WIN = {RATIO:.3f}")
print(f"VOTE_INT 기준 = {sm.VOTE_INT} frames")

traj = Circle()
results_exp2 = {}
t0 = time.time()

for tau_ms, tau_f in TAU_ALL.items():
    tau_s = tau_ms/1000
    win_prop = tau_s / RATIO
    vote_int_prop = max(1, int(win_prop / DT))  # VOTE_INT ∝ τ
    
    v_prop, _ = measure_vstar(traj, tau_f,
                               vote_int_override=vote_int_prop)
    v_base = BASE_VSTARS.get(tau_ms)
    x_prop = v_prop*np.sqrt(tau_s)
    x_base = v_base*np.sqrt(tau_s) if v_base else None

    results_exp2[tau_ms] = {
        'vstar_base': v_base, 'vstar_prop': v_prop,
        'x_base': x_base, 'x_prop': x_prop,
        'win_prop': win_prop, 'vote_int_prop': vote_int_prop
    }
    print(f"  τ={tau_ms:>4}ms VOTE_INT={vote_int_prop:>3}: "
          f"v*(prop)={v_prop:.3f} x*={x_prop:.3f} | "
          f"v*(base)={v_base:.3f} x*={x_base:.3f} "
          f"diff={v_prop-v_base:+.3f}")

# 분석
tau_list = sorted(TAU_ALL.keys())
x_base_list = [results_exp2[t]['x_base'] for t in tau_list]
x_prop_list = [results_exp2[t]['x_prop'] for t in tau_list]
cv_base = np.std(x_base_list)/np.mean(x_base_list)
cv_prop = np.std(x_prop_list)/np.mean(x_prop_list)
delta_cv = cv_prop - cv_base

print(f"\n  CV (VOTE_INT=18 고정): {cv_base:.4f}")
print(f"  CV (VOTE_INT∝τ):       {cv_prop:.4f}")
print(f"  ΔCV = {delta_cv:+.4f}")

if cv_prop < cv_base*0.8:
    print("  → VOTE_INT∝τ일 때 x*가 더 일정 → information rate 중요")
elif cv_prop > cv_base*1.2:
    print("  → VOTE_INT∝τ일 때 x*가 불안정 → pure delay τ가 원인")
else:
    print("  → 두 조건 유사 → delay τ가 주요 원인, WIN은 부차적")

with open('exp2_fixed_results.json','w') as f:
    json.dump({str(t): {k: v for k,v in d.items()}
               for t,d in results_exp2.items()}, f, indent=2)
print(f"\n  소요: {time.time()-t0:.0f}s | 저장: exp2_fixed_results.json")


# ── EXP3 보완: quadratic τ=100ms, speed 확장 ─────────────
print("\n" + "="*60)
print("EXP3 보완: quadratic τ=100ms speed 10m/s까지 확장")
print("="*60)

tau_f_100 = TAU_ALL[100]
t0 = time.time()

v_q100, rmses_q100 = measure_vstar(traj, tau_f_100,
                                    method='quadratic',
                                    speeds=SPEEDS_EXT)
x_q100 = v_q100 * np.sqrt(0.1)
print(f"\n  quadratic τ=100ms: v*={v_q100:.3f}, x*={x_q100:.3f}")
print(f"  (이전: v*=5.0 = speed ceiling 걸림)")

if v_q100 < 5.0:
    print(f"  ✅ 진짜 v* 찾음 — ceiling 문제 해결")
else:
    print(f"  ⚠️ 여전히 v*≥5.0 — 매우 빠른 속도가 optimal")

# quadratic 전체 결과 업데이트
print(f"\n  quadratic β 재추정:")
with open('exp3_results.json') as f:
    exp3 = json.load(f)

TAU_SUB = [100, 433, 1000]
vstars_q = [v_q100,
            exp3['quadratic']['433'],
            exp3['quadratic']['1000']]
xstars_q = [v*np.sqrt(t/1000) for v,t in zip(vstars_q, TAU_SUB)]
sl,_,r,_,_ = stats.linregress(np.log(TAU_SUB), np.log(vstars_q))
cv_q = np.std(xstars_q)/np.mean(xstars_q)
print(f"  x* = {xstars_q}")
print(f"  β = {-sl:.3f}, CV = {cv_q:.4f}")
ok = "✅" if cv_q<0.1 and abs(-sl-0.5)<0.15 else "⚠️"
print(f"  {ok}")

with open('exp3_quadratic_fixed.json','w') as f:
    json.dump({'vstar_100': v_q100, 'x_100': x_q100,
               'rmses_100': rmses_q100, 'speeds': SPEEDS_EXT}, f, indent=2)
print(f"\n  소요: {time.time()-t0:.0f}s | 저장: exp3_quadratic_fixed.json")

# 최종 요약
print("\n" + "="*60)
print("전체 결과 요약")
print("="*60)
print(f"""
EXP1: σ scaling
  → v*가 σ에 완전히 무관 (소수점 4자리 동일)
  → noise-independent scaling law 확립
  → 이론: "노이즈가 아닌 순수 latency 구조가 v* 결정"

EXP2 (수정):
  → VOTE_INT∝τ 조건에서 x* CV = {cv_prop:.4f} (기존 {cv_base:.4f})
  → {'pure delay가 scaling 원인' if abs(delta_cv)<0.02 else 'information rate도 기여'}

EXP3:
  → fixed/majority: β≈0.50, CV<0.07 ✅
  → quadratic τ=100ms: v*={v_q100:.3f} x*={x_q100:.3f}
""")
