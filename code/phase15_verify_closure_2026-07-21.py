# -*- coding: utf-8 -*-
"""VERIFICATION of the corrected-reduction closure (phase14).
Higher MC, different seed base, finer grids, SEs reported, and an independent
re-measurement of the transfer function with a different RNG seed.
Targets (measured in the full 150-agent system, never used to tune this model):
  ellipse margin u* = 0.562 ;  circle dither x* = 1.347 ;  corner u*/cos = 0.499"""
import os, sys, json, time
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("ENGINE_DIR", os.path.join(HERE,"..","engine","crowd_control_engine")))
import simulation_main as sm

DT=1/60; W=0.3; DELAY_F=26; SMOOTH=0.2; L=2.0
VI=int(W/DT); tau_tot=DELAY_F*DT+W/2+DT/SMOOTH
N,TROLL=150,0.05

def measure_transfer(draws=1500, seed=91):
    rng=np.random.default_rng(seed)
    ms=np.linspace(-22.5,22.5,37); E=[];S=[]
    for m in ms:
        e=[]
        for _ in range(draws):
            v=sm.gen_votes(m,m,TROLL,N,rng); b=sm.DIRS[v].mean(axis=0)
            e.append((np.degrees(np.arctan2(b[1],b[0]))-m+180)%360-180)
        e=np.array(e); E.append(e.mean()); S.append(e.std())
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

def ellipse(a_,b_,n=800):
    t=np.linspace(0,2*np.pi,n,endpoint=False)
    P=[[a_*np.cos(x),b_*np.sin(x)] for x in t]; return Path(P+[P[0]])
def circle(R,n=800):
    t=np.linspace(0,2*np.pi,n,endpoint=False)
    P=[[R*np.cos(x),R*np.sin(x)] for x in t]; return Path(P+[P[0]])
def regular(n,side):
    R=side/(2*np.sin(np.pi/n)); a=np.linspace(0,2*np.pi,n,endpoint=False)
    P=[[R*np.cos(x),R*np.sin(x)] for x in a]; return Path(P+[P[0]])

MS=EE=SS=None
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
                m=((th+22.5)%45)-22.5
                th2=th+np.interp(m,MS,EE)+rng.normal(0,np.interp(m,MS,SS))
                ta=np.radians(th2); cmd=np.array([np.cos(ta),np.sin(ta)])
        vel+=SMOOTH*(cmd*v-vel); pos=pos+vel*DT; hist.append(pos.copy())
        _,dd=path.closest(pos); errs.append(dd)
    e=np.array(errs[nf//5:]); return float(np.sqrt(np.mean(e**2)))

def profile(path,vg,mc,base):
    M=[];S=[]
    for v in vg:
        x=np.array([rms(path,v,base+37*k) for k in range(mc)])
        M.append(x.mean()); S.append(x.std(ddof=1)/np.sqrt(mc))
    M=np.array(M);S=np.array(S); i=int(np.argmin(M)); vs=vg[i]
    if 1<=i<=len(vg)-2:
        c=np.polyfit(vg[i-1:i+2],M[i-1:i+2],2)
        if c[0]>0 and vg[i-1]<=-c[1]/(2*c[0])<=vg[i+1]: vs=-c[1]/(2*c[0])
    band=vg[M<=M[i]+S[i]]
    return float(vs),(float(band.min()),float(band.max())),M.tolist(),S.tolist()

if __name__=="__main__":
    t0=time.time()
    MS,EE,SS=measure_transfer()
    print(f"transfer re-measured (indep. seed): max|E|={np.abs(EE).max():.1f}deg residual={SS.mean():.2f}deg [{time.time()-t0:.0f}s]",flush=True)
    out={}
    # (c) circle dither x*
    vg=np.arange(1.5,2.9,0.05)
    vs,band,M,S=profile(circle(10.),vg,25,7777)
    xs=vs*np.sqrt(DELAY_F*DT); xb=(band[0]*np.sqrt(DELAY_F*DT),band[1]*np.sqrt(DELAY_F*DT))
    out["circle_xstar"]={"pred":xs,"band":xb,"target":1.347}
    print(f"(c) circle x* = {xs:.3f}  1SE band [{xb[0]:.3f},{xb[1]:.3f}]  target 1.347  err {100*(xs-1.347)/1.347:+.1f}% [{time.time()-t0:.0f}s]",flush=True)
    # (b) ellipse margin
    vg=np.arange(0.44,0.78,0.015)*L/tau_tot
    vs,band,M,S=profile(ellipse(12.,6.),vg,25,4242)
    u=vs*tau_tot/L; ub=(band[0]*tau_tot/L,band[1]*tau_tot/L)
    out["ellipse_u"]={"pred":u,"band":ub,"target":0.562}
    print(f"(b) ellipse u* = {u:.3f}  1SE band [{ub[0]:.3f},{ub[1]:.3f}]  target 0.562  err {100*(u-0.562)/0.562:+.1f}% [{time.time()-t0:.0f}s]",flush=True)
    # (a) corner margins
    cr={}
    for a_deg,n in [(90,4),(60,6),(120,3)]:
        ca=np.cos(np.radians(a_deg/2))
        vg=np.arange(0.38,0.70,0.015)*ca*L/tau_tot
        vs,band,M,S=profile(regular(n,12.),vg,20,1515+a_deg)
        u=vs*tau_tot/L/ca; ub=(band[0]*tau_tot/L/ca,band[1]*tau_tot/L/ca)
        cr[a_deg]={"pred":u,"band":ub}
        print(f"(a) alpha={a_deg}: u/cos = {u:.3f}  1SE band [{ub[0]:.3f},{ub[1]:.3f}]  target 0.499 [{time.time()-t0:.0f}s]",flush=True)
    out["corner"]=cr
    json.dump(out,open("phase15_verify_closure_results_2026-07-21.json","w"),indent=2,default=float)
    print("\nVERIFIED",flush=True)
