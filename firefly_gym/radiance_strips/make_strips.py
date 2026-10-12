import sys, numpy as np
N=int(sys.argv[1]) if len(sys.argv)>1 else 5
panes=[np.array([[0.300047248602,21.4879055023,20.9150009155],[0.300047248602,21.4879055023,19.2549991608],
                 [0.332718759775,23.8276767731,19.2549991608],[0.332718759775,23.8276767731,20.9150009155]]),
       np.array([[0.278405874968,19.9380569458,20.9150009155],[0.278405874968,19.9380569458,19.2549991608],
                 [0.297813296318,21.3279209137,19.2549991608],[0.297813296318,21.3279209137,20.9150009155]])]
ztop,zbot=20.9150009155,19.2549991608
edges=np.linspace(ztop,zbot,N+1)
out=["# Exterior window split into %d horizontal strips (strip 1 = top), each a Klems receiver/sender"%N]
for k in range(N):
    zt,zb=edges[k],edges[k+1]
    out.append(f"\n#@rfluxmtx h=kf u=Z\n\nvoid light strip{k+1}\n0\n0\n3 1 1 1\n")
    for p,P in enumerate(panes):
        v=P.copy(); v[[0,3],2]=zt; v[[1,2],2]=zb
        out.append(f"strip{k+1} polygon strip{k+1}.pane{p+1}\n0\n0\n12\n"+"\n".join("  %.9f %.9f %.9f"%tuple(r) for r in v)+"\n")
    # also write single-strip sender file for the daylight matrix
    with open(f"strips/strip{k+1}.rad","w") as f:
        f.write("#@rfluxmtx h=kf u=Z\n\nvoid light strip%d\n0\n0\n3 1 1 1\n\n"%(k+1))
        for p,P in enumerate(panes):
            v=P.copy(); v[[0,3],2]=zt; v[[1,2],2]=zb
            f.write(f"strip{k+1} polygon strip{k+1}.pane{p+1}\n0\n0\n12\n"+"\n".join("  %.9f %.9f %.9f"%tuple(r) for r in v)+"\n\n")
open(f"strips/window_strips{N}.rad","w").write("\n".join(out))
print(edges)
