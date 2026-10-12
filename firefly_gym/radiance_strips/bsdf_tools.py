import os, re, subprocess, numpy as np

RINGS = [(0, 5, 1), (5, 15, 8), (15, 25, 16), (25, 35, 20), (35, 45, 24),
         (45, 55, 24), (55, 65, 24), (65, 75, 16), (75, 90, 12)]
ENV = dict(os.environ)          # Radiance tools must be on PATH


def lambdas():
    L = []
    for lo, hi, n in RINGS:
        L += [np.pi * (np.sin(np.radians(hi))**2 - np.sin(np.radians(lo))**2) / n] * n
    return np.array(L)


LAM = lambdas()


def read_xml(path, wl="Visible"):
    s = open(path).read()
    out = {}
    for b in re.findall(r"<WavelengthData>(.*?)</WavelengthData>", s, re.S):
        w = re.search(r"<Wavelength[^>]*>(.*?)</Wavelength>", b).group(1)
        d = re.search(r"<WavelengthDataDirection>(.*?)</WavelengthDataDirection>", b).group(1).strip()
        a = np.array(re.search(r"<ScatteringData>(.*?)</ScatteringData>", b, re.S).group(1)
                     .replace(",", " ").split(), float).reshape(145, 145)
        if w == wl:
            out[d] = a
    return out


def to_A(bsdf):
    """transfer matrix (rmtxop convention): A[out, in] = BSDF[out, in] * Lambda[in]"""
    return bsdf * LAM[None, :]


def to_bsdf(A):
    return A / LAM[None, :]


def layer(ext, inn, P=None):
    """Combine two layers given as dicts of BSDF blocks ('Transmission Front' ...).
    ext = exterior layer (front faces outdoors), inn = interior layer.
    P: optional permutation applied to directions travelling between the layers."""
    I = np.eye(145)
    Q = I if P is None else I[P]
    Tf1, Tb1, Rf1, Rb1 = (to_A(ext[k]) for k in ("Transmission Front", "Transmission Back", "Reflection Front", "Reflection Back"))
    Tf2, Tb2, Rf2, Rb2 = (to_A(inn[k]) for k in ("Transmission Front", "Transmission Back", "Reflection Front", "Reflection Back"))
    Tf2, Rf2 = Tf2 @ Q, Q.T @ Rf2 @ Q
    Tb2, Rb2_ = Q.T @ Tb2, Rb2
    Tf2 = Tf2
    Kf = np.linalg.inv(I - Rb1 @ Rf2)
    Kb = np.linalg.inv(I - Rf2 @ Rb1)
    out = {
        "Transmission Front": Tf2 @ Kf @ Tf1,
        "Reflection Front": Rf1 + Tb1 @ Rf2 @ Kf @ Tf1,
        "Transmission Back": Tb1 @ Kb @ Tb2,
        "Reflection Back": Rb2_ + (Tf2 @ Rb1 @ Kb @ Tb2),
    }
    return {k: to_bsdf(v) for k, v in out.items()}


def hemi(bsdf):
    """hemispherical value per incident direction"""
    return (bsdf * LAM[:, None]).sum(0)


def run_genbsdf(tag, geom_files, dim, c=2000, ropt="-ab 5 -ad 64 -lw 1e-4 -as 0", cwd="."):
    out = os.path.join(cwd, f"{tag}.xml")
    if os.path.exists(out) and os.path.getsize(out) > 1000:
        return out
    cmd = (f"genBSDF -n 2 -c {c} -r '{ropt}' +forward +backward -geom meter "
           f"-dim {' '.join(map(str, dim))} {' '.join(geom_files)} > {out}")
    subprocess.run(cmd, shell=True, cwd=cwd, env=ENV, check=True, stderr=subprocess.DEVNULL)
    return out


def make_blind(tag, angle, depth=0.08, pitch=0.072, rho=0.65, spec=0.02, rough=0.15,
               curv=None, nslats=40, width=3.0, zoff=0.012, margin=14, cwd="."):
    """External venetian blind layer only, +Z into room, geometry at Z<0.
    genblinds: positive angle tilts the (after transform) exterior edge UP."""
    H = nslats * pitch
    mat = os.path.join(cwd, f"{tag}_mat.rad")
    open(mat, "w").write(f"void plastic slat_mat\n0\n0\n5 {rho} {rho} {rho} {spec} {rough}\n")
    cflag = f"-r {curv}" if curv else ""
    geo = os.path.join(cwd, f"{tag}_geo.rad")
    subprocess.run(f"genblinds slat_mat slat {depth} {width} {H} {nslats} {angle} {cflag} "
                   f"| xform -rz 90 -rx -90 | xform -t {width} 0 {-zoff} > {geo}",
                   shell=True, env=ENV, check=True)
    # sample the central part of the blind to avoid edge slats
    dim = (width/2 - 0.5, width/2 + 0.5, margin * pitch, H - margin * pitch, -(zoff + depth + 0.01), 0)
    return run_genbsdf(tag, [os.path.basename(mat), os.path.basename(geo)], dim, cwd=cwd)
