"""x_q first-principles bracket (Sec. III-D / VIII). Pre-registered target 0.989.
Open-loop: measured vote-mixture transfer function convolved with diffusion
dither sigma_d = sigma_theta*sqrt(T_c)*x/L (all inputs independent of v* data)
-> x_q = 2.25. Delayed-loop dither amplification sqrt(M(u*)) ~ 1.47 -> ~1.53.
Residual ~1.5x factor: closed-loop snap-timing dynamics (open problem).
Set the engine path below, then run."""
import os
import sys, numpy as np
sys.path.insert(0, os.environ.get("ENGINE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine", "crowd_control_engine")))
import simulation_main as sm
from scipy.optimize import curve_fit

rng = np.random.default_rng(3); N, TROLL, DRAWS = 150, 0.05, 1200
ms = np.linspace(-22.5, 22.5, 61); E = []
for m in ms:
    errs = []
    for _ in range(DRAWS):
        v = sm.gen_votes(m, m, TROLL, N, rng)
        b = sm.DIRS[v].mean(axis=0)
        errs.append((np.degrees(np.arctan2(b[1], b[0])) - m + 180) % 360 - 180)
    E.append(np.mean(errs))
E = np.array(E)

def rms_swept(sig):
    if sig < 1e-6: return float(np.sqrt(np.mean(E**2)))
    k = np.arange(-90, 90.1, 0.75); phi = np.exp(-0.5*(k/sig)**2); phi /= phi.sum()
    out = []
    for m in ms:
        mm = ((m + k + 22.5) % 45) - 22.5
        out.append(np.sum(np.interp(mm, ms, E)*phi))
    return float(np.sqrt(np.mean(np.array(out)**2)))

SIG_TH, T_C, L = 8.60, 0.9, 2.0
A0 = rms_swept(0)
xs = np.linspace(0.2, 4.0, 20)
A = np.array([rms_swept(SIG_TH*np.sqrt(T_C)*x/L)/A0 for x in xs])
xq, = curve_fit(lambda x, q: np.exp(-(x/q)**2), xs, A, p0=[3.0])[0]
M = lambda u: (1+np.sin(u))/np.cos(u)
print(f"open-loop x_q = {xq:.3f}; loop-corrected ~ {xq/np.sqrt(M(0.7)):.2f}; target 0.989")
