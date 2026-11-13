import numpy as np
from models import chain, truss, power_and_grads, mirror_maps, asymmetry
from sqp_driver import multistart
KB=(1e4,5e5)
for name in ("chain","truss"):
    mdl=chain() if name=="chain" else truss(); mk,mm=mirror_maps(mdl)
    for lab,k0 in (("mirror-averaged k0",0.5*(mdl["k0"]+mdl["k0"][mk])),("uniform k0",np.full(mdl["n_k"],mdl["k0"].mean()))):
        md=dict(mdl); md["k0"]=k0; mload=np.full(mdl["n_m"],0.1); mload[mdl["load_mass"]]=5.0; mload*=1; 
        k,m,P,h,nm,allP=multistart(md,False,True,KB,(0.1,5.0),[("initial",None,None)],n_random=50,n_sym=100)
        print("%s masses only, %s: best P %.4e from %s, asym m %.3f; masses>1kg at %s"%(name,lab,P,nm,asymmetry(m,mm),[(i,round(v,2)) for i,v in enumerate(m) if v>1]),flush=True)
