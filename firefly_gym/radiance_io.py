"""Minimal readers for the Radiance files used in Implementation_Radiance/office_with_results.

* read_rmtx  -> Radiance matrix files (.mtx / .vec) written by rmtxop / rxfluxmtx / rtrace
               returns ndarray (NROWS, NCOLS, NCOMP)
* read_hsr   -> flat (non run-length-encoded) hyperspectral pictures written by rxfluxmtx -fc
               (FORMAT=Radiance_spectra, NCOMP spectral mantissas + 1 shared exponent per pixel)

Spectral channel order in this model is 780 nm -> 380 nm in 20 bands of 20 nm (-cs 20).
"""
from __future__ import annotations

import numpy as np

# Photopic weights (lm/W) that turn the 20 spectral channels (780 -> 380 nm) into luminous
# quantities, i.e. what `rmtxop -c Y` does. They sum to 179 lm/W (Radiance WHTEFFICACY).
# Fitted against Innsbruck_day_noblinds_wp_ill.mtx: median relative error 2e-8.
PHOTOPIC_W = np.array([
    0.0, 0.00546064, 0.01640711, 0.06824996, 0.2704784,
    1.04861362, 3.43107111, 8.52951904, 16.26562456, 24.68874767,
    31.34506471, 33.19978749, 29.12728375, 18.01060318, 7.70781288,
    3.35158179, 1.40653636, 0.45888169, 0.0628152, 0.00548247,
])
BAND_CENTERS_NM = 780 - 20 * (np.arange(20) + 0.5)  # 770, 750, ..., 390


def _split_header(data: bytes):
    i = data.index(b"\n\n")
    hdr = data[:i].decode("latin1")
    h = {}
    for line in hdr.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            h[k.strip()] = v.strip()
    return h, data[i + 2:]


def read_rmtx(path, dtype=np.float32) -> np.ndarray:
    with open(path, "rb") as f:
        h, body = _split_header(f.read())
    ncomp = int(h.get("NCOMP", 3))
    fmt = h.get("FORMAT", "ascii")
    n_exp = int(h["NROWS"]) * int(h["NCOLS"]) * ncomp if ("NROWS" in h and "NCOLS" in h) else None
    if fmt in ("float", "double"):
        size = 4 if fmt == "float" else 8
        dt = ("<" if h.get("BigEndian", "0") == "0" else ">") + ("f4" if size == 4 else "f8")
        nbytes = n_exp * size if n_exp else len(body) // size * size
        a = np.frombuffer(body[:nbytes], dtype=dt)  # newer rxfluxmtx may append padding
    else:
        a = np.array(body.split(), dtype=np.float64)
    if "NROWS" in h and "NCOLS" in h:
        nr, nc = int(h["NROWS"]), int(h["NCOLS"])
        a = a[: nr * nc * ncomp].reshape(nr, nc, ncomp)
    else:  # e.g. rtrace -I+ output: one value per sensor
        a = a.reshape(-1, 1, ncomp)
    return a.astype(dtype)


def read_hsr(path, ncomp: int = 20) -> np.ndarray:
    """Return (H, W, ncomp) float array of a flat Radiance_spectra picture."""
    with open(path, "rb") as f:
        h, rest = _split_header(f.read())
    j = rest.index(b"\n")
    res = rest[:j].decode().split()          # e.g. "-Y 192 +X 192"
    H, W = int(res[1]), int(res[3])
    n = H * W * (ncomp + 1)
    img = np.frombuffer(rest[j + 1: j + 1 + n], np.uint8).reshape(H, W, ncomp + 1)
    m = img[..., :ncomp].astype(np.float64)
    e = img[..., ncomp].astype(np.int32)
    scale = np.where(e > 0, np.ldexp(1.0, e - 136), 0.0)
    return (m + 0.5) * scale[..., None] * (e[..., None] > 0)


def to_photopic(x: np.ndarray) -> np.ndarray:
    """Contract the last (spectral) axis with the photopic weights -> lux or cd/m2."""
    return x @ PHOTOPIC_W
