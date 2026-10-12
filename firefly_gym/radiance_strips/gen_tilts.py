"""Generate the system BSDF (external venetian blind + glazing) for a range of slat tilts.
Blind layer: genBSDF on a calibrated slat model; glazing: measured Bauing_ILM_Lamelle_XXX.xml;
layers combined with the Klems matrix method. Tilt convention = the measured '+75' file."""
import re, numpy as np, bsdf_tools as bt, subprocess
PITCH, RHO, CURV, DEPTH = 0.070, 0.68, -0.08, 0.08
X = bt.read_xml('Bauing_ILM_Lamelle_XXX.xml'); M = bt.read_xml('Bauing_ILM_Lamelle_+75.xml')
tmpl = open('Bauing_ILM_Lamelle_+75.xml').read()

def write_xml(path, blocks, name):
    s = tmpl.replace('Bauing_ILM_Lamelle_+75', name)
    def repl(m):
        b = m.group(0)
        w = re.search(r'<Wavelength[^>]*>(.*?)</Wavelength>', b).group(1)
        d = re.search(r'<WavelengthDataDirection>(.*?)</WavelengthDataDirection>', b).group(1).strip()
        if w != 'Visible':
            return ''          # drop solar blocks (not modelled)
        a = np.clip(blocks[d], 0, None)
        txt = '\n'.join(', '.join('%.6f' % v for v in row) for row in a)
        return re.sub(r'<ScatteringData>.*?</ScatteringData>', '<ScatteringData>\n' + txt + '\n</ScatteringData>', b, flags=re.S)
    s = re.sub(r'<WavelengthData>.*?</WavelengthData>', repl, s, flags=re.S)
    open(path, 'w').write(s)

rows = []
for tilt in [0, 15, 30, 45, 60, 75]:
    tag = f'blind_t{tilt:02d}'
    B = bt.read_xml(bt.make_blind(tag, -tilt, depth=DEPTH, pitch=PITCH, rho=RHO, curv=CURV))
    S = bt.layer(B, X)
    write_xml(f'Bauing_model_tilt{tilt:02d}.xml', S, f'Bauing_model_tilt{tilt:02d}')
    h = {k: bt.hemi(S[k])[0] for k in S}
    rows.append((tilt, h['Transmission Front'], h['Reflection Front'], h['Reflection Back']))
    print(f'tilt {tilt:2d}: Tvis,n {h["Transmission Front"]:.3f}  Rf {h["Reflection Front"]:.3f}  Rb {h["Reflection Back"]:.3f}', flush=True)
hm = {k: bt.hemi(M[k])[0] for k in M}
print(f'measured +75: Tvis,n {hm["Transmission Front"]:.3f}  Rf {hm["Reflection Front"]:.3f}  Rb {hm["Reflection Back"]:.3f}')
