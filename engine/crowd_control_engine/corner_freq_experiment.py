"""
Corner Frequency 분리 실험
corner=90° 고정, h 변화 → corner_freq 변화

목적:
  Hexagon 오차가 corner frequency 때문인지 확인
  v*(α, τ)에 corner frequency가 독립 변수로 개입하는지 검증

실행: python3 corner_freq_experiment.py
소요: ~20~30분
"""

import sys, numpy as np, json, time
from scipy import stats
sys.path.insert(0, '.')
import simulation_main as sm

DT = 1/60

# ── h 변화 Square 클래스 ─────────────────────────────────
class ScaledSquare:
    """corner=90° 고정, h 변화로 corner_freq 조절"""
    def __init__(self, h):
        self.h = h
        self.name = f'square_h{h}'
        self.corner_angle = 90.0
        self.c = np.array([
            [h,0],[h,h],[-h,h],[-h,-h],[h,-h],[h,0.]], dtype=float)
        self.segs = [(self.c[i],self.c[i+1]) for i in range(5)]
        self.lens = [np.linalg.norm(b-a) for a,b in self.segs]
        self.circ = sum(self.lens)
        self.cum  = np.array([0]+list(np.cumsum(self.lens)))
        self.corner_freq = 4 / self.circ  # corners per meter

    def closest(self,p):
        bd,bp,ba=1e10,self.c[0],0.
        for i,(a,b) in enumerate(self.segs):
            v=b-a;l2=v@v
            if l2<1e-10:continue
            t=np.clip((p-a)@v/l2,0,1)
            pt=a+t*v;d=np.linalg.norm(p-pt)
            if d<bd:bd,bp,ba=d,pt,self.cum[i]+t*self.lens[i]
        return bp,ba

    def at(self,arc):
        arc=arc%self.circ
        for i,(a,b) in enumerate(self.segs):
            if arc<=self.cum[i+1]+1e-9:
                t=(arc-self.cum[i])/self.lens[i]
                return a+np.clip(t,0,1)*(b-a)
        return self.c[-1]

    def start(self):return self.c[0].copy()


# ── 실험 파라미터 ─────────────────────────────────────────
H_VALUES = [10, 7, 5, 3.5, 2.5]
TAU_ALL  = {100:6, 200:12, 300:18, 433:26, 600:36, 1000:60}
SPEEDS   = [0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.25,2.5,3.0,4.0,5.0]
TROLLS   = [0.05, 0.40]
N, MC    = 150, 15

trajs = {h: ScaledSquare(h) for h in H_VALUES}
total = len(H_VALUES)*len(TAU_ALL)*len(TROLLS)*len(SPEEDS)*MC

# 무결성 확인
print("="*60)
print("무결성 확인")
print("="*60)
print(f"\n{'h':>6} {'corner°':>9} {'circ(m)':>10} {'freq/m':>10}")
print("-"*40)
for h, t in trajs.items():
    print(f"  {h:>4}   {t.corner_angle:>8.0f}°  {t.circ:>9.2f}  {t.corner_freq:>9.4f}")

print(f"\n  corner angle: 모두 90° ✅")
print(f"  corner freq 범위: {min(t.corner_freq for t in trajs.values()):.4f}~"
      f"{max(t.corner_freq for t in trajs.values()):.4f}/m")
print(f"\n  총 runs: {total:,} | 예상: ~{total*0.08/3600:.1f}시간")

# v* 측정
def measure_vstar(traj, tau_f, troll):
    orig = sm.DELAY_F
    sm.DELAY_F = tau_f
    try:
        rmses = []
        for v in SPEEDS:
            r = sm.run_condition(traj, N, troll, MC,
                                 method='fixed', speed_override=v,
                                 seed_base=31, seed_offset=N)
            rmses.append(r['rmse_mean'])
    finally:
        sm.DELAY_F = orig
    idx = int(np.argmin(rmses))
    vstar = float(SPEEDS[idx])
    if 1<=idx<=len(SPEEDS)-2:
        try:
            c=np.polyfit(SPEEDS[idx-1:idx+2],rmses[idx-1:idx+2],2)
            if c[0]>0:
                vf=-c[1]/(2*c[0])
                if SPEEDS[idx-1]<=vf<=SPEEDS[idx+1]:
                    vstar=float(vf)
        except:pass
    return vstar, [float(r) for r in rmses]

print(f"\n{'='*60}\n실험 시작\n{'='*60}")
results = {}
t0 = time.time()
done = 0

for h, traj in trajs.items():
    results[h] = {}
    for tau_ms, tau_f in TAU_ALL.items():
        results[h][tau_ms] = {}
        for troll in TROLLS:
            v, rmses = measure_vstar(traj, tau_f, troll)
            results[h][tau_ms][troll] = {'vstar':v,'rmses':rmses}
            done += len(SPEEDS)*MC
            eta = (time.time()-t0)/done*(total-done)/60 if done>0 else 0
            print(f"  h={h:>4} τ={tau_ms:>4}ms tr={troll:.0%}: "
                  f"v*={v:.3f}  [ETA {eta:.0f}min]")

print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")

# 분석
print(f"\n{'='*60}\n분석: corner_freq vs v*\n{'='*60}")
print(f"\n[τ=433ms, troll=5%]")
print(f"{'h':>6} {'freq/m':>9} {'v*':>8} {'circ':>8}")
print("-"*38)

freqs, vstars_433 = [], []
for h in H_VALUES:
    t = trajs[h]
    v = results[h][433][0.05]['vstar']
    freqs.append(t.corner_freq)
    vstars_433.append(v)
    print(f"  {h:>4}  {t.corner_freq:>8.4f}  {v:>7.3f}  {t.circ:>7.1f}")

corr, p = stats.pearsonr(freqs, vstars_433)
print(f"\n  corner_freq vs v*: r={corr:.3f}, p={p:.4f}")

if corr < -0.7 and p < 0.05:
    print("  → ✅ corner_freq ↑ → v* ↓ 확인!")
    print("  → Hexagon 오차가 corner_freq 때문 ✅")
    print("  → v*(α, f_c, τ) 3변수 모델 필요")
else:
    print("  → corner_freq 효과 없음")
    print("  → Hexagon 오차는 다른 이유")
    print("  → 현재 2변수 모델 v*(α,τ) 충분")

# β 분석
print(f"\n[β — corner_freq에 따라 달라지는가?]")
print(f"{'h':>6} {'freq/m':>9} {'β':>8} {'R²':>8}")
print("-"*38)

tau_list = list(TAU_ALL.keys())
log_tau  = np.log(tau_list)

for h in H_VALUES:
    v5 = [results[h][t][0.05]['vstar'] for t in tau_list]
    sl,ic,r,pv,_ = stats.linregress(log_tau, np.log(v5))
    print(f"  {h:>4}  {trajs[h].corner_freq:>8.4f}  {-sl:>7.3f}  {r**2:>7.4f}")

# 저장
with open('corner_freq_results.json','w') as f:
    json.dump({
        'h_values': H_VALUES,
        'corner_freqs': {str(h): trajs[h].corner_freq for h in H_VALUES},
        'results': {
            str(h): {
                str(tau): {str(tr): {'vstar':d['vstar'],'rmses':d['rmses']}
                           for tr,d in trd.items()}
                for tau,trd in td.items()}
            for h,td in results.items()}
    }, f, indent=2)
print("\n결과 저장: corner_freq_results.json")
