# -*- coding: utf-8 -*-
"""CORRECTED REDUCTION (insight from phase12 failure).
The 150-agent vector mean is NOT a grid-quantized direction: individual votes
are quantized, but their vector average is quasi-continuous with (i) a
deterministic SNAP bias toward the nearest grid direction and (ii) a small
residual fluctuation. Measuring that transfer function from the vote model
(no v* data involved) gives the correct reduction:

    aggregate_angle = ideal + snap(m) + eta,   m = ideal mod Delta (signed),
    snap(.) and std(eta) measured directly from gen_votes.

This replaces phase12's incorrect 'AR(1) noise then quantize' model, which
systematically undershot all three constants.  Zero fitted parameters."""
import os, sys, json, time
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("ENGINE_DIR", os.path.join(HERE,"..","engine","crowd_control_engine")))
import simulation_main as sm

DT=1/60; W=0.3; DELAY_F=26; SMOOTH=0.2; L=2.0
VI=int(W/DT); tau_tot=DELAY_F*DT+W/2+DT/SMOOTH
N,TROLL=150,0.05

# ---- measure the vote-mixture transfer function (from the vote model only) ----
def measure_transfer(draws=800, seed=5):
    rng=np.random.default_rng(seed)
    ms=np.linspace(-22.5,22.5,31); E=[]; S=[]
    for m in ms:
        errs=[]
        for _ in range(draws):
            v=sm.gen_votes(m,m,TROLL,N,rng); b=sm.DIRS[v].mean(axis=0)
            errs.append((np.degrees(np.arctan2(b[1],b[0]))-m+180)%360-180)
        errs=np.array(errs); E.append(errs.mean()); S.append(errs.std())
    return ms,np.array(E),np.array(S)

class Path:
    def __init__(s,pts):
        s.A=np.array(pts[:-1],float); s.V=np.array(pts[1:],float)-s.A
        s.len=np.linalg.norm(s.V,axis=1); s.L2=np.einsum('ij,ij->i',s.V,s.V)
        s.cum=np.concatenate([[0],np.cumsum(s.len)]); s.circ=s.cum[-1]
    def closest(s,p):
        t=np.clip(np.einsum('ij,ij->i',p-s.A,s.V)/s.L2,0,1)
        Q=s.A+t[:,None]*s.V; d=np.linalg.norm(p-Q,axis=1); i=int(np.argmin(d))
        return s.cum[i]+t[i]*s.len[i], d[i]
    def at(s,a):
        a=a%s.circ; i=max(min(np.searchsorted(s.cum,a)-1,len(s.A)-1),0)
        return s.A[i]+np.clip((a-s.cum[i])/s.len[i],0,1)*s.V[i]

def regular(n,side):
    R=side/(2*np.sin(np.pi/n)); a=np.linspace(0,2*np.pi,n,endpoint=False)
    P=[[R*np.cos(x),R*np.sin(x)] for x in a]; return Path(P+[P[0]])
def ellipse(a_,b_,n=600):
    t=np.linspace(0,2*np.pi,n,endpoint=False)
    P=[[a_*np.cos(x),b_*np.sin(x)] for x in t]; return Path(P+[P[0]])
def circle(R,n=600):
    t=np.linspace(0,2*np.pi,n,endpoint=False)
    P=[[R*np.cos(x),R*np.sin(x)] for x in t]; return Path(P+[P[0]])

MS,EE,SS=None,None,None
def agg_angle(th_deg,rng):
    """reduced aggregate: ideal + measured snap + measured residual fluctuation"""
    m=((th_deg+22.5)%45)-22.5
    e=np.interp(m,MS,EE); s=np.interp(m,MS,SS)
    return th_deg+e+rng.normal(0,s)

def rms(path,v,seed,dur=60.0):
    rng=np.random.default_rng(seed)
    s0=rng.uniform(0,path.circ); pos=path.at(s0).copy()
    d0=path.at(s0+0.2)-pos; d0/=np.linalg.norm(d0)
    vel=d0*v; cmd=d0.copy(); hist=[pos.copy()]; errs=[]; nf=int(dur/DT)
    for f in range(nf):
        if f%VI==0:
            dp=hist[max(0,len(hist)-1-DELAY_F)]
            s,_=path.closest(dp); tg=path.at(s+L); d=tg-dp; n=np.linalg.norm(d)
            if n>1e-12:
                th=np.degrees(np.arctan2(d[1],d[0]))
                ta=np.radians(agg_angle(th,rng)); cmd=np.array([np.cos(ta),np.sin(ta)])
        vel+=SMOOTH*(cmd*v-vel); pos=pos+vel*DT; hist.append(pos.copy())
        _,dd=path.closest(pos); errs.append(dd)
    e=np.array(errs[nf//5:]); return float(np.sqrt(np.mean(e**2)))

def argmin_v(path,vg,mc=10):
    J=np.array([np.mean([rms(path,v,3000+29*k) for k in range(mc)]) for v in vg])
    i=int(np.argmin(J)); vs=vg[i]
    if 1<=i<=len(vg)-2:
        c=np.polyfit(vg[i-1:i+2],J[i-1:i+2],2)
        if c[0]>0 and vg[i-1]<=-c[1]/(2*c[0])<=vg[i+1]: vs=-c[1]/(2*c[0])
    return float(vs)

if __name__=="__main__":
    t0=time.time()
    MS,EE,SS=measure_transfer()
    print(f"transfer measured: max|E|={np.abs(EE).max():.1f}deg, residual std~{SS.mean():.2f}deg [{time.time()-t0:.0f}s]",flush=True)
    out={}
    print("\n(a) corner margin [target 0.499]",flush=True)
    cr={}
    for a_deg,n in [(120,3),(90,4),(60,6)]:
        p=regular(n,12.0); ca=np.cos(np.radians(a_deg/2))
        vg=np.arange(0.36,0.72,0.02)*ca*L/tau_tot
        v=argmin_v(p,vg); u=v*tau_tot/L; cr[a_deg]=u/ca
        print(f"   alpha={a_deg}: u/cos={u/ca:.3f} [{time.time()-t0:.0f}s]",flush=True)
    out["corner"]=cr
    print("\n(b) ellipse [target 0.562]",flush=True)
    v=argmin_v(ellipse(12.,6.),np.arange(0.42,0.80,0.02)*L/tau_tot)
    out["ellipse"]=v*tau_tot/L; print(f"   u*={out['ellipse']:.3f} [{time.time()-t0:.0f}s]",flush=True)
    print("\n(c) circle dither [target x*=1.347]",flush=True)
    v=argmin_v(circle(10.),np.arange(1.4,3.0,0.08))
    out["circle_xstar"]=v*np.sqrt(DELAY_F*DT)
    print(f"   v*={v:.3f} x*={out['circle_xstar']:.3f} [{time.time()-t0:.0f}s]",flush=True)
    json.dump(out,open("phase14_correct_reduction_results_2026-07-21.json","w"),indent=2,default=float)
    print("\nDONE",flush=True)
