# -*- coding: utf-8 -*-
"""POWER TEST: is the reduced-model closure test resolvable at all?
Single condition (square, alpha=90, quantized, measured noise), MC=40,
fine u grid. Report J(u) with standard errors so we can see whether the
minimum is statistically locatable to the precision needed (+-0.02 in u)
to test against the measured 0.499 +- 0.011."""
import os, json, time
import numpy as np
DT=1/60; W=0.3; DELAY_F=26; SMOOTH=0.2; L=2.0
VI=int(W/DT); tau_tot=DELAY_F*DT+W/2+DT/SMOOTH
SIG=np.radians(8.60); TC=3*W; PHI=np.exp(-W/TC)
DIRS=np.array([[np.cos(a),np.sin(a)] for a in np.linspace(0,2*np.pi,8,endpoint=False)])

class Poly:
    def __init__(s,pts):
        s.A=np.array([pts[i] for i in range(len(pts)-1)])
        s.V=np.array([pts[i+1]-pts[i] for i in range(len(pts)-1)])
        s.len=np.linalg.norm(s.V,axis=1); s.L2=np.einsum('ij,ij->i',s.V,s.V)
        s.cum=np.concatenate([[0],np.cumsum(s.len)]); s.circ=s.cum[-1]
    def closest(s,p):
        t=np.clip(np.einsum('ij,ij->i',p-s.A,s.V)/s.L2,0,1)
        Q=s.A+t[:,None]*s.V; d=np.linalg.norm(p-Q,axis=1); i=int(np.argmin(d))
        return s.cum[i]+t[i]*s.len[i], d[i]
    def at(s,arc):
        arc=arc%s.circ; i=max(min(np.searchsorted(s.cum,arc)-1,len(s.A)-1),0)
        return s.A[i]+np.clip((arc-s.cum[i])/s.len[i],0,1)*s.V[i]

def square(side):
    h=side/2; P=np.array([[h,-h],[h,h],[-h,h],[-h,-h],[h,-h]],float); return Poly(P)

def rms(path,v,seed,dur=60.0):
    rng=np.random.default_rng(seed)
    s0=rng.uniform(0,path.circ); pos=path.at(s0).copy()
    d0=path.at(s0+0.2)-pos; d0/=np.linalg.norm(d0)
    vel=d0*v; cmd=d0.copy(); nz=0.0; hist=[pos.copy()]; errs=[]; nf=int(dur/DT)
    for f in range(nf):
        if f%VI==0:
            dp=hist[max(0,len(hist)-1-DELAY_F)]
            s,_=path.closest(dp); tg=path.at(s+L); d=tg-dp; n=np.linalg.norm(d)
            if n>1e-12:
                th=np.arctan2(d[1],d[0])
                nz=PHI*nz+np.sqrt(1-PHI**2)*rng.normal(0,SIG); th+=nz
                c=np.array([np.cos(th),np.sin(th)]); cmd=DIRS[np.argmax(DIRS@c)]
        vel+=SMOOTH*(cmd*v-vel); pos=pos+vel*DT; hist.append(pos.copy())
        _,dd=path.closest(pos); errs.append(dd)
    e=np.array(errs[nf//5:]); return float(np.sqrt(np.mean(e**2)))

if __name__=="__main__":
    t0=time.time(); MC=40
    p=square(20.0); ca=np.cos(np.radians(45))
    us=np.arange(0.38,0.72,0.02)
    print(f"POWER TEST square alpha=90, quantized, measured noise, MC={MC}")
    print(f"{'u':>6} {'J mean':>9} {'SE':>8} {'rel.SE':>8}",flush=True)
    means=[];ses=[]
    for u in us:
        v=u*ca*L/tau_tot
        vals=np.array([rms(p,v,5000+17*k) for k in range(MC)])
        m=vals.mean(); se=vals.std(ddof=1)/np.sqrt(MC)
        means.append(m); ses.append(se)
        print(f"{u:>6.2f} {m:>9.5f} {se:>8.5f} {100*se/m:>7.2f}%",flush=True)
    means=np.array(means); ses=np.array(ses)
    i=int(np.argmin(means)); print(f"\nargmin u = {us[i]:.3f}  (measured target 0.499)")
    lo=means[i]+ses[i]
    band=us[means<=lo]
    print(f"within 1 SE of min: u in [{band.min():.2f},{band.max():.2f}]")
    json.dump({"u":us.tolist(),"J":means.tolist(),"SE":ses.tolist()},
              open("phase13_power_results_2026-07-21.json","w"),indent=2)
    print(f"done [{time.time()-t0:.0f}s]",flush=True)
