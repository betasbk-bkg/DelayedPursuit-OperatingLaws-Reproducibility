# -*- coding: utf-8 -*-
"""DECISIVE CLOSURE TEST: is the corner margin the minimizer of a cost function
computed from primitives with NO fitted constants?

Reduced model (150 agents -> 2 independently MEASURED scalars):
  primitives : DT=1/60, W=0.3, DELAY_F=26, SMOOTH=0.2, L=2.0, Delta=45deg
  measured   : sigma_theta = 8.60 deg, k_c = 3  => T_c = k_c*W = 0.9 s   [ref 1, E4]
  geometry   : regular polygon, exterior angle alpha, side s
  NOTHING FITTED.

Aggregate direction noise modeled as AR(1) across vote windows with the measured
std and correlation time (phi = exp(-W/T_c)), optionally quantized to 8 dirs.

J(u) = E[ RMS lateral error ]  at v = u*cos(alpha/2)*L/tau_tot.
Claim of closure: argmin_u J(u) reproduces the measured u*/cos(alpha/2) ~ 0.50
WITHOUT any parameter tuned to the v* data.
"""
import os, json, time
import numpy as np

DT=1/60; W=0.3; DELAY_F=26; SMOOTH=0.2; L=2.0
VI=int(W/DT); tau_tot=DELAY_F*DT+W/2+DT/SMOOTH
SIG=np.radians(8.60); TC=3*W; PHI=np.exp(-W/TC)
DIRS=np.array([[np.cos(a),np.sin(a)] for a in np.linspace(0,2*np.pi,8,endpoint=False)])

def polygon(n, side):
    R=side/(2*np.sin(np.pi/n))
    ang=np.linspace(0,2*np.pi,n,endpoint=False)
    V=np.array([[R*np.cos(a),R*np.sin(a)] for a in ang])
    return np.vstack([V,V[0]])

def make_path(V):
    seg=[(V[i],V[i+1]) for i in range(len(V)-1)]
    lens=[np.linalg.norm(b-a) for a,b in seg]
    cum=np.concatenate([[0],np.cumsum(lens)]); circ=cum[-1]
    return seg,lens,cum,circ

def closest(p,seg,lens,cum):
    bd=1e18; bs=0.0
    for i,(a,b) in enumerate(seg):
        v=b-a; l2=v@v
        t=np.clip((p-a)@v/l2,0,1); q=a+t*v; d=np.linalg.norm(p-q)
        if d<bd: bd=d; bs=cum[i]+t*lens[i]
    return bs,bd

def at(s,seg,lens,cum,circ):
    s=s%circ
    for i,(a,b) in enumerate(seg):
        if s<=cum[i+1]+1e-9:
            return a+np.clip((s-cum[i])/lens[i],0,1)*(b-a)
    return seg[-1][1]

def run(alpha_deg, n_sides, side, u, sigma, quantize, seed, dur=60.0):
    rng=np.random.default_rng(seed)
    V=polygon(n_sides,side); seg,lens,cum,circ=make_path(V)
    ca=np.cos(np.radians(alpha_deg/2)); v=u*ca*L/tau_tot
    start=rng.uniform(0,circ)
    pos=at(start,seg,lens,cum,circ).copy()
    nxt=at(start+0.1,seg,lens,cum,circ); d0=nxt-pos; d0/=np.linalg.norm(d0)
    vel=d0*v; cmd=d0.copy(); noise=rng.normal(0,SIG) if sigma>0 else 0.0
    hist=[pos.copy()]; errs=[]
    nf=int(dur/DT)
    for f in range(nf):
        if f%VI==0:
            dp=hist[max(0,len(hist)-1-DELAY_F)]
            s,_=closest(dp,seg,lens,cum)
            tgt=at(s+L,seg,lens,cum,circ); d=tgt-dp; nn=np.linalg.norm(d)
            if nn>1e-12:
                th=np.arctan2(d[1],d[0])
                if sigma>0:
                    noise=PHI*noise+np.sqrt(1-PHI**2)*rng.normal(0,sigma)
                    th+=noise
                c=np.array([np.cos(th),np.sin(th)])
                if quantize:
                    c=DIRS[np.argmax(DIRS@c)]
                cmd=c
        vel+=SMOOTH*(cmd*v-vel); pos=pos+vel*DT; hist.append(pos.copy())
        _,dd=closest(pos,seg,lens,cum); errs.append(dd)
    e=np.array(errs[nf//5:])
    return float(np.sqrt(np.mean(e**2)))

def ustar(alpha_deg,n_sides,side,sigma,quantize,mc=6):
    us=np.arange(0.34,0.78,0.02); J=[]
    for u in us:
        J.append(np.mean([run(alpha_deg,n_sides,side,u,sigma,quantize,900+13*k) for k in range(mc)]))
    J=np.array(J); i=int(np.argmin(J)); ub=us[i]
    if 1<=i<=len(us)-2:
        c=np.polyfit(us[i-1:i+2],J[i-1:i+2],2)
        if c[0]>0 and us[i-1]<=-c[1]/(2*c[0])<=us[i+1]: ub=-c[1]/(2*c[0])
    return float(ub), J.tolist(), us.tolist()

if __name__=="__main__":
    t0=time.time(); out={}
    cases=[(120,3),(90,4),(60,6)]
    print("REDUCED-MODEL CLOSURE TEST  (all inputs primitive or independently measured)")
    print(f"sigma_theta={np.degrees(SIG):.2f} deg  T_c={TC:.2f}s  phi={PHI:.3f}")
    print(f"{'alpha':>6} {'cfg':>22} {'u*':>7} {'u*/cos':>8}",flush=True)
    for quant in [False,True]:
        for sig_on in [False,True]:
            tag=f"{'quant' if quant else 'cont '}/{'noise' if sig_on else 'clean'}"
            row={}
            for a,n in cases:
                u,_,_=ustar(a,n,12.0, SIG if sig_on else 0.0, quant)
                ca=np.cos(np.radians(a/2)); row[a]=(u,u/ca)
                print(f"{a:>6} {tag:>22} {u:>7.3f} {u/ca:>8.3f}  [{time.time()-t0:.0f}s]",flush=True)
            out[tag]=row
    json.dump({k:{str(a):v for a,v in r.items()} for k,r in out.items()},
              open("phase11_Ju_closure_results_2026-07-21.json","w"),indent=2)
    print("\nCLOSURE holds if quant/noise row gives u*/cos ~ 0.50 (measured 0.499+-0.011)",flush=True)
