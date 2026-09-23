"""
this study self-audit -- 2026-07-19. Recomputes the four corrected claims of report §9.
Requires: expa_results.json + unified_vstar_results.json from crowd_control_engine,
and the closure-pack result JSONs, paths below.
"""
import os
import json, os, numpy as np
from scipy import stats

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
MAIN = os.environ.get("ENGINE_DIR", HERE)  # set to crowd_control_engine dir

ea = json.load(open(os.path.join(MAIN, "expa_results.json")))
uv = json.load(open(os.path.join(MAIN, "unified_vstar_results.json")))
Ls = json.load(open(os.path.join(HERE, "phase2_L_sweep_results_2026-07-18.json")))

taus = [100,200,300,433,600,1000]

# 1. beta(n=4) artifact
v4 = [ea['results']['4'][str(t)]['vstar'] for t in taus]
b_all = -stats.linregress(np.log(np.array(taus)/1000), np.log(v4)).slope
b_no1 = -stats.linregress(np.log(np.array(taus[1:])/1000), np.log(v4[1:])).slope
print(f"1. beta(n=4): all pts {b_all:.3f} | excl saturated tau=100: {b_no1:.3f} (artifact confirmed)")

# 2. x*(L=1) proper refinement
for r in Ls['runs']:
    if r['name']=='circle_n8' and r['L']==1.0:
        sp = Ls['speed_grids']['1.0']; rm = r['rmses']
        sub = list(zip(sp,rm))[2:]
        ss, rr = zip(*sub)
        i = int(np.argmin(rr))
        c = np.polyfit(ss[i-1:i+2], rr[i-1:i+2], 2)
        vf = -c[1]/(2*c[0])
        print(f"2. x*(L=1) refined = {vf*np.sqrt(0.433):.3f} vs pred 0.674 "
              f"({100*(vf*np.sqrt(0.433)-0.674)/0.674:+.1f}%, not '0%')")

# 3. raw plateau vs anchor
cr = uv['results']['circle']['data']
lows = [x for t in map(str,taus) for x in cr[t]['0.05']['rmses'][:3]]
print(f"3. raw plateau mean {np.mean(lows):.4f} vs anchor 0.4534 "
      f"({100*(np.mean(lows)-0.4534)/0.4534:+.1f}%; '0.6%' was fit-parameter only)")

# 4. u_tot* drift (8-dir circle)
obs = {0.1:3.931,0.2:3.231,0.3:2.577,0.433:2.134,0.6:1.758,1.0:1.218,1.5:0.869}
us = {t: v*(t+0.2333)/2 for t,v in obs.items()}
print("4. u_tot*(circle,8dir):", {t: round(u,3) for t,u in us.items()},
      "-> drifts 0.65->0.75, NOT constant; min()-law honest accuracy ~5%")
