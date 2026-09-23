"""
this study Phase 2 -- structural decomposition fit on EXISTING circle data.
2026-07-18

Model (mechanistic, each term independently motivated):

  RMSE^2(v, tau) = F^2  +  Q0^2 * exp(-2 x^2 / x_q^2)  +  sig_th^2 * T_eff * x^2 * M(u)

  x = v*sqrt(tau_sec)   [dither-to-quantum ratio variable; the collapse variable]
  u = v*tau_sec / L     [delay-gain product of the pursuit feedback loop]
  M(u) = (1+sin u)/cos u   [stationary variance amplification of the delayed
                            OU process dy/dt = -a y(t-tau) + noise, a=v/L;
                            Kuechler & Mensch 1992; diverges at u=pi/2]
  F      : v,tau-independent floor (geometric pursuit offset + residual)
  Q0     : zero-dither quantization scallop amplitude (low-v plateau)
  x_q    : dither-linearization scale (Gaussian-dither attenuation of the
           quantizer sawtooth: first-harmonic amplitude ~ exp(-2 pi^2 s^2/D^2),
           s = dither std prop. to x  ->  exp(-(x/x_q)^2) on amplitude)
  T_eff  : effective integration time of uncorrected noise (theory anchor:
           2*k_c*WIN = 1.8 s from measured k_c=3, WIN=0.3)
  sig_th : 0.1501 rad -- FIXED, measured independently (theta_autocorr E4)

Free parameters: F, Q0, x_q, T_eff  (4 params, 76 fit points after excluding
u > 1.2 where the stationary M(u) formula is invalid near instability).

Success criteria (pre-registered here):
  (1) fit R^2 >= 0.95 on the included points
  (2) fitted T_eff within factor ~2 of the 1.8 s anchor
  (3) predicted x*(tau) reproduces beta ~ 0.51 +- 0.05 via the M(u) correction
  (4) predicted instability onset v_crit(tau=1.0) ~ pi*L/(2*tau) ~ 3.1 m/s
      coincides with the observed tau=1000 blowup between v=3 and v=4
"""
import os
import json, numpy as np
from scipy import optimize, stats

DATA = os.environ.get("ENGINE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine", "crowd_control_engine"))
L = 2.0
SIG_TH = np.radians(8.60)   # 0.1501 rad, measured (E4), FIXED
U_MAX_FIT = 1.2             # exclude near-instability points from fit

d = json.load(open(DATA, encoding='utf-8'))
res = d['results']['circle']
SPEEDS = d['params']['speeds']

V, TAU, RMSE = [], [], []
for tau_ms, td in res['data'].items():
    for v, r in zip(SPEEDS, td['0.05']['rmses']):
        V.append(v); TAU.append(int(tau_ms)/1000.0); RMSE.append(r)
V = np.array(V); TAU = np.array(TAU); RMSE = np.array(RMSE)
X = V*np.sqrt(TAU)
U = V*TAU/L

mask = U < U_MAX_FIT
print(f"points total={len(V)}, fit={mask.sum()}, excluded (u>={U_MAX_FIT})={(~mask).sum()}")
print("excluded points:", [(v, t, r) for v,t,r in zip(V[~mask], TAU[~mask], RMSE[~mask])])

def M(u):
    u = np.minimum(u, 1.55)   # clamp just below pi/2 for numerics
    return (1+np.sin(u))/np.cos(u)

def model_rmse(params, v, tau):
    F, Q0, xq, Teff = params
    x = v*np.sqrt(tau); u = v*tau/L
    r2 = F**2 + Q0**2*np.exp(-2*(x/xq)**2) + SIG_TH**2*Teff*x**2*M(u)
    return np.sqrt(r2)

def loss(params):
    return model_rmse(params, V[mask], TAU[mask]) - RMSE[mask]

p0 = [0.20, 0.40, 1.0, 1.8]
fit = optimize.least_squares(loss, p0, bounds=([0.0,0.0,0.1,0.05],[1.0,1.0,5.0,20.0]))
F, Q0, xq, Teff = fit.x
pred = model_rmse(fit.x, V[mask], TAU[mask])
ss_res = np.sum((RMSE[mask]-pred)**2)
ss_tot = np.sum((RMSE[mask]-RMSE[mask].mean())**2)
r2 = 1-ss_res/ss_tot

print("\n=== Fitted parameters ===")
print(f"F     = {F:.4f} m      (floor; obs RMSE_min mean 0.244)")
print(f"Q0    = {Q0:.4f} m      (zero-dither scallop; obs low-v plateau ~0.42 -> plateau = sqrt(F^2+Q0^2) = {np.hypot(F,Q0):.3f})")
print(f"x_q   = {xq:.4f} m/s^.5 (dither-linearization scale)")
print(f"T_eff = {Teff:.4f} s    (theory anchor 2*k_c*WIN = 1.8 s; ratio = {Teff/1.8:.2f})")
print(f"R^2   = {r2:.4f} on {mask.sum()} points")

# residual diagnostics per tau
print("\nper-tau mean |resid| (m):")
for t in sorted(set(TAU)):
    m2 = mask & (TAU==t)
    resid = np.abs(model_rmse(fit.x, V[m2], TAU[m2]) - RMSE[m2])
    print(f"  tau={t:.3f}: {resid.mean():.4f}  (n={m2.sum()})")

# === predicted v*(tau), x*(tau), beta ===
print("\n=== Predicted optimum vs observed ===")
taus = np.array([0.1,0.2,0.3,0.433,0.6,1.0])
vstar_obs = [res['data'][k]['0.05']['vstar'] for k in ['100','200','300','433','600','1000']]
vgrid = np.linspace(0.05, 6.0, 3000)
vstar_pred = []
for t in taus:
    rm = model_rmse(fit.x, vgrid, np.full_like(vgrid, t))
    vstar_pred.append(vgrid[np.argmin(rm)])
vstar_pred = np.array(vstar_pred)
for t, vp, vo in zip(taus, vstar_pred, vstar_obs):
    print(f"  tau={t:.3f}: v*_pred={vp:.3f}  v*_obs={vo:.3f}  x*_pred={vp*np.sqrt(t):.3f}  x*_obs={vo*np.sqrt(t):.3f}")

sl, ic, r, p, se = stats.linregress(np.log(taus), np.log(vstar_pred))
print(f"\npredicted beta = {-sl:.3f}  (observed circle beta = 0.511)")
sl2, *_ = stats.linregress(np.log(taus), np.log(vstar_obs))
print(f"observed  beta = {-sl2:.3f}")

# ablation: remove M(u) coupling (set M=const=M(u at tau=0.1 ref)) -> beta should be exactly 0.5
vstar_noM = []
for t in taus:
    x = vgrid*np.sqrt(t)
    r2v = F**2 + Q0**2*np.exp(-2*(x/xq)**2) + SIG_TH**2*Teff*x**2*1.0
    vstar_noM.append(vgrid[np.argmin(np.sqrt(r2v))])
slA, *_ = stats.linregress(np.log(taus), np.log(vstar_noM))
print(f"ablation (M==1, pure-x model) beta = {-slA:.3f}   [must be 0.500 exactly -- x* is tau-free]")

# instability onset check
print("\n=== Instability onset (tau=1.0) ===")
print(f"v_crit theory = pi*L/(2*tau) = {np.pi*L/2:.3f} m/s")
print("observed tau=1000 rmses at v=2.5,3,4,5:",
      [res['data']['1000']['0.05']['rmses'][i] for i in [9,10,11,12]])

# collapse quality of the model itself: R(x) master + M-induced spread
print("\n=== Model-implied collapse imperfection ===")
for t in [0.1, 0.433, 1.0]:
    x_at = 1.6
    v_at = x_at/np.sqrt(t)
    print(f"  at x={x_at}, tau={t}: model RMSE = {model_rmse(fit.x, np.array([v_at]), np.array([t]))[0]:.3f}")
print("(spread at fixed x is the tau-dependent M(u) contribution -- explains why R2_collapse=0.956 < 1)")
