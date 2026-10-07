#!/usr/bin/env python3
import os, sys, hashlib
W = sys.argv[1]
ops = ["M", "K", "M_track", "M_ctrl_whole", "M_ctrl_box", "M_Gamma", "N_selected", "coords"]
ext = lambda op: "txt" if op == "coords" else "coo"
for ctl in ("distributed", "neumann", "dirichlet"):
    for op in ops:
        h = {}
        for b in ("ex20", "ex21off", "ex21w0"):
            p = f"{W}/{b}_{ctl}_{op}.{ext(op)}"
            h[b] = hashlib.sha256(open(p, "rb").read()).hexdigest()[:12] if os.path.exists(p) else "MISSING"
        same_off = h["ex20"] == h["ex21off"]; same_w0 = h["ex20"] == h["ex21w0"]
        print(f"PARITY_DUMP control={ctl:<11} op={op:<12} ex20={h['ex20']} ex21_off={'BYTE-IDENTICAL' if same_off else h['ex21off']} ex21_w0={'BYTE-IDENTICAL' if same_w0 else h['ex21w0']}")
