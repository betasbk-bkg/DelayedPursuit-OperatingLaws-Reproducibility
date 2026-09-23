# -*- coding: utf-8 -*-
"""UNIFIED CLOSURE TEST: can ONE reduced cost function J, built only from
primitives + two independently measured scalars, predict ALL THREE constants
that the paper currently reports as fitted/characterized?

  (a) corner margin      measured  u*/cos(a/2) = 0.499 +- 0.011
  (b) smooth margin      measured  ellipse u* = 0.562
  (c) dither invariant   measured  circle(8-dir) x* = v* sqrt(tau) = 1.347

Reduced model = the engine loop with the 150-agent vote mixture REPLACED by its
two measured moments: sigma_theta = 8.60 deg, correlation time T_c = k_c*W = 0.9 s
(AR(1)).  Everything else primitive: DT, W, DELAY_F, SMOOTH, L, Delta.
ZERO fitted constants.

If the minimizers reproduce (a),(b),(c), the theory is a computational closure
in the d2 sense: constants are not fitted, they are computed."""
import os, json, time
import numpy as np

DT=1/60; W=0.3; DELAY_F=26; SMOOTH=0.2; L=2.0
VI=int(W/DT); tau_tot=DELAY_F*DT+W/2+DT/SMOOTH
SIG=np.radians(8.60); TC=3*W; PHI=np.exp(-W/TC)
DIRS=np.array([[np.cos(a),np.sin(a)] for a in np.linspace(0,2*np.pi,8,endpoint=False)])

class Poly:
    def __init__(s, pts):
        s.seg=[(pts[i],pts[i+1]) for i in range(len(pts)-1)]
        s.len=[np.linalg.norm(b-a) for a,b in s.seg]
        s.cum=np.concatenate([[0],np.cumsum(s.len)]); s.circ=s.cum[-1]
        s.A=np.array([a for a,b in s.seg]); s.V=np.array([b-a for a,b in s.seg])
        s.L2=np.einsum('ij,ij->i',s.V,s.V)
    def closest(s,p):
        t=np.clip(np.einsum('ij,ij->i',p-s.A,s.V)/s.L2,0,1)
        Q=s.A+t[:,None]*s.V; d=np.linalg.norm(p-Q,axis=1); i=int(np.argmin(d))
        return s.cum[i]+t[i]*s.len[i], d[i]
    def at(s,arc):
        arc=arc%s.circ; i=min(np.searchsorted(s.cum,arc)-1,len(s.seg)-1); i=max(i,0)
        return s.A[i]+np.clip((arc-s.cum[i])/s.len[i],0,1)*s.V[i]

def regular(n, side):
    R=side/(2*np.sin(np.pi/n)); a=np.linspace(0,2*np.pi,n,endpoint=False)
    P=np.array([[R*np.cos(x),R*np.sin(x)] for x in a]); return Poly(np.vstack([P,P[0]]))
def ellipse(a_,b_,n=600):
    t=np.linspace(0,2*np.pi,n,endpoint=False)
    P=np.column_stack([a_*np.cos(t),b_*np.sin(t)]); return Poly(np.vstack([P,P[0]]))
def circle(R,n=600):
    t=np.linspace(0,2*np.pi,n,endpoint=False)
    P=np.column_stack([R*np.cos(t),R*np.sin(t)]); return Poly(np.vstack([P,P[0]]))

def rms(path, v, quant, sigma, seed, dur=60.0):
    rng=np.random.default_rng(seed)
    s0=rng.uniform(0,path.circ); pos=path.at(s0).copy()
    d0=path.at(s0+0.2)-pos; d0/=np.linalg.norm(d0)
    vel=d0*v; cmd=d0.copy(); nz=0.0
    hist=[pos.copy()]; errs=[]; nf=int(dur/DT)
    for f in range(nf):
        if f%VI==0:
            dp=hist[max(0,len(hist)-1-DELAY_F)]
            s,_=path.closest(dp); tg=path.at(s+L); d=tg-dp; n=np.linalg.norm(d)
            if n>1e-12:
                th=np.arctan2(d[1],d[0])
                if sigma>0:
                    nz=PHI*nz+np.sqrt(1-PHI**2)*rng.normal(0,sigma); th+=nz
                c=np.array([np.cos(th),np.sin(th)])
                if quant: c=DIRS[np.argmax(DIRS@c)]
                cmd=c
        vel+=SMOOTH*(cmd*v-vel); pos=pos+vel*DT; hist.append(pos.copy())
        _,dd=path.closest(pos); errs.append(dd)
    e=np.array(errs[nf//5:]); return float(np.sqrt(np.mean(e**2)))

def argmin_v(path, vgrid, quant, sigma, mc=5):
    J=[np.mean([rms(path,v,quant,sigma,700+11*k) for k in range(mc)]) for v in vgrid]
    J=np.array(J); i=int(np.argmin(J)); vs=vgrid[i]
    if 1<=i<=len(vgrid)-2:
        c=np.polyfit(vgrid[i-1:i+2],J[i-1:i+2],2)
        if c[0]>0 and vgrid[i-1]<=-c[1]/(2*c[0])<=vgrid[i+1]: vs=-c[1]/(2*c[0])
    return float(vs), J.tolist()

if __name__=="__main__":
    t0=time.time(); out={}
    print("UNIFIED CLOSURE (no fitted constants; sigma_theta & T_c measured)",flush=True)

    # (a) corner margin: polygons, quantized + measured noise
    print("\n(a) corner margin  [target u*/cos = 0.499 +- 0.011]",flush=True)
    ca_res={}
    for a_deg,n in [(120,3),(90,4),(60,6)]:
        p=regular(n,12.0); ca=np.cos(np.radians(a_deg/2))
        vg=np.arange(0.34,0.80,0.02)*ca*L/tau_tot
        v,_=argmin_v(p,vg,True,SIG)
        u=v*tau_tot/L; ca_res[a_deg]=u/ca
        print(f"   alpha={a_deg}: u*={u:.3f}  u*/cos={u/ca:.3f}  [{time.time()-t0:.0f}s]",flush=True)
    out["corner"]=ca_res

    # (b) smooth margin: ellipse 12x6, quantized + measured noise
    print("\n(b) smooth margin  [target ellipse u* = 0.562]",flush=True)
    p=ellipse(12.0,6.0)
    vg=np.arange(0.40,0.85,0.025)*L/tau_tot
    v,_=argmin_v(p,vg,True,SIG)
    out["ellipse_u"]=v*tau_tot/L
    print(f"   ellipse: u*={out['ellipse_u']:.3f}  [{time.time()-t0:.0f}s]",flush=True)

    # (c) dither invariant: circle R=10, quantized + measured noise
    print("\n(c) dither invariant  [target x* = v* sqrt(tau) = 1.347]",flush=True)
    p=circle(10.0)
    vg=np.arange(1.2,3.2,0.1)
    v,_=argmin_v(p,vg,True,SIG)
    out["circle_xstar"]=v*np.sqrt(DELAY_F*DT)
    print(f"   circle: v*={v:.3f}  x*={out['circle_xstar']:.3f}  [{time.time()-t0:.0f}s]",flush=True)

    json.dump(out,open("phase12_unified_closure_results_2026-07-21.json","w"),indent=2,default=float)
    print("\nUNIFIED CLOSURE VERDICT: compare (a)~0.499 (b)~0.562 (c)~1.347",flush=True)
