#!/usr/bin/env python3
"""T7F: the 'lowest index wins' rule, read from the --write-region-fields VTU.
Overlapping boxes (nx = 8): control A = [1/8,1/2]x[1/8,3/8] (index 1),
B = [3/8,7/8]x[1/8,3/8] (index 2); tracking A' = [3/8,7/8]x[1/2,7/8] (1),
B' = [1/8,1/2]x[1/2,7/8] (2).  Elements in A and B must carry index 1.
usage: t7f_region_index.py <file.vtu>"""
import base64, re, struct, sys
import numpy as np
s = open(sys.argv[1]).read()
def arrays(section):
    out = {}
    for m in re.finditer(r'<DataArray type="(\w+)"(?: Name="([^"]*)")?[^>]*format="binary">\s*(\S+)\s*</DataArray>', section):
        typ, name, txt = m.group(1), m.group(2) or "Points", m.group(3)
        nb = struct.unpack("<I", base64.b64decode(txt[:8]))[0]
        raw = base64.b64decode(txt[8:])[:nb]
        dt = {"Float32": "<f4", "Int32": "<i4", "UInt16": "<u2", "UInt8": "u1", "Float64": "<f8"}[typ]
        out[name] = np.frombuffer(raw, dtype=dt)
    return out
pts = arrays(re.search(r"<Points>(.*?)</Points>", s, re.S).group(1))["Points"].reshape(-1, 3)
cells = arrays(re.search(r"<Cells>(.*?)</Cells>", s, re.S).group(1))
cd = arrays(re.search(r"<CellData[^>]*>(.*?)</CellData>", s, re.S).group(1))
conn, off = cells["connectivity"], cells["offsets"]
start = np.concatenate([[0], off[:-1]])
cen = np.array([pts[conn[a:b]].mean(0) for a, b in zip(start, off)])
def inside(c, b): return b[0] < c[0] < b[1] and b[2] < c[1] < b[3]
ctrl = [(0.125, 0.5, 0.125, 0.375), (0.375, 0.875, 0.125, 0.375)]
trk = [(0.375, 0.875, 0.5, 0.875), (0.125, 0.5, 0.5, 0.875)]
def expected(c, boxes):
    for k, b in enumerate(boxes):
        if inside(c, b): return k + 1
    return 0
ok = True
for name, boxes in (("control_region", ctrl), ("tracking_region", trk)):
    got = cd[name].astype(int)
    exp = np.array([expected(c, boxes) for c in cen])
    both = np.array([inside(c, boxes[0]) and inside(c, boxes[1]) for c in cen])
    good = (got == exp).all()
    ok &= good and both.sum() > 0 and (got[both] == 1).all()
    print(f"T7F_REGION {name}: cells={len(got)} counts(0/1/2)={[int((got == k).sum()) for k in (0, 1, 2)]} "
          f"overlap_cells={int(both.sum())} index_in_overlap={sorted(int(v) for v in set(got[both]))} matches_rule={good}")
print("T7F_REGION_RESULT", "PASS" if ok else "FAIL")
