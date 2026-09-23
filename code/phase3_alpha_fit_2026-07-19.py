"""
this study -- alpha-sweep discrimination analysis. 2026-07-19
Combines new (dodeca/octa/penta) + existing (hex/sq/tri, rotation-test protocol)
u* data and tests:
  (1) tau-invariance of u_tot per polygon (lag-regime membership)
  (2) functional form: u*(alpha) = A*cos^p(alpha/2) -- is p = 1?
  (3) amplitude: A = 1/2 (sagitta anchor) vs 1/e/cos45 = 0.5206 (critical-damping
      anchor) -- 4.2% apart; discrimination limited by c-systematics (+-3%)
  (4) small-alpha upturn (corner->dither crossover) diagnostics
"""
import os
import json, numpy as np
from scipy import optimize

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
C_LAG, L = 0.2333, 2.0

new = json.load(open(BASE + r"\phase3_alpha_sweep_results_2026-07-19.json", encoding='utf-8'))
rot = json.load(open(BASE + r"\phase1_rotation_results_2026-07-18.json", encoding='utf-8'))

pts = []   # (alpha, tau_s, u)
def add(alpha, taus_dict):
    for t, td in taus_dict.items():
        tau_s = int(t)/1000
        u = td['vstar']*(tau_s+C_LAG)/L
        pts.append((alpha, tau_s, u))

for cname, cd in new['cases'].items():
    add(cd['alpha'], cd['taus'])
for cname, alpha in [('hex_rot0',60.),('sq_rot0',90.),('tri_rot0',120.),('tri_rot30',120.)]:
    add(alpha, rot['cases'][cname]['taus'])

print(f"{'alpha':>6} {'tau':>6} {'u_tot':>7}")
by_alpha = {}
for a, t, u in sorted(pts):
    print(f"{a:>6.0f} {t:>6.3f} {u:>7.3f}")
    by_alpha.setdefault(a, []).append(u)

print("\nper-alpha mean u* and tau-CV:")
alphas, umeans, ustds = [], [], []
for a in sorted(by_alpha):
    us = np.array(by_alpha[a])
    alphas.append(a); umeans.append(us.mean()); ustds.append(us.std(ddof=0))
    print(f"  alpha={a:>5.0f}: u*={us.mean():.3f}  CV={100*us.std()/us.mean():.1f}%  (n={len(us)})")
alphas = np.array(alphas); umeans = np.array(umeans)

def model(a, A, p):
    return A*np.cos(np.radians(a/2))**p

popt, pcov = optimize.curve_fit(model, alphas, umeans, p0=[0.5,1.0])
perr = np.sqrt(np.diag(pcov))
pred = model(alphas, *popt)
print(f"\nfree fit: A = {popt[0]:.4f} +- {perr[0]:.4f}   p = {popt[1]:.3f} +- {perr[1]:.3f}")
print(f"  H1 anchors: A=0.5 (sagitta/continuum), p=1  | H1': A=1/e/cos45=0.5206")
for a, um, pr in zip(alphas, umeans, pred):
    print(f"  alpha={a:>5.0f}: obs={um:.3f}  fit={pr:.3f}  resid={100*(um-pr)/um:+.1f}%")

# fixed-p=1 fit for amplitude
poptA, pcovA = optimize.curve_fit(lambda a, A: model(a, A, 1.0), alphas, umeans, p0=[0.5])
print(f"\np=1 fixed: A = {poptA[0]:.4f} +- {np.sqrt(pcovA[0,0]):.4f}")
print(f"  distance to 1/2:      {abs(poptA[0]-0.5)/np.sqrt(pcovA[0,0]):.1f} sigma")
print(f"  distance to 0.5206:   {abs(poptA[0]-0.5206)/np.sqrt(pcovA[0,0]):.1f} sigma")
print(f"  (plus +-3% systematic from c -- state in report)")

# zero-param H1 residuals
print("\nzero-param H1 (0.5*cos(a/2)) residuals:")
for a, um in zip(alphas, umeans):
    h1 = 0.5*np.cos(np.radians(a/2))
    print(f"  alpha={a:>5.0f}: obs={um:.3f} pred={h1:.3f}  {100*(um-h1)/h1:+.1f}%")
