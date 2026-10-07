#!/usr/bin/env python3
"""Vector extraction helper: parse a pdftocairo SVG page, accumulate the
transform stack, and return every stroked path as page-space polylines with its
stroke colour.  No rasterisation, no OCR."""
import re
import xml.etree.ElementTree as ET

NS = "{http://www.w3.org/2000/svg}"


def matmul(a, b):
    # affine [a b c d e f] as in SVG matrix(a,b,c,d,e,f)
    return [a[0]*b[0] + a[2]*b[1], a[1]*b[0] + a[3]*b[1],
            a[0]*b[2] + a[2]*b[3], a[1]*b[2] + a[3]*b[3],
            a[0]*b[4] + a[2]*b[5] + a[4], a[1]*b[4] + a[3]*b[5] + a[5]]


def parse_transform(t):
    m = [1, 0, 0, 1, 0, 0]
    if not t:
        return m
    for name, args in re.findall(r"(\w+)\(([^)]*)\)", t):
        v = [float(x) for x in re.split(r"[ ,]+", args.strip()) if x]
        if name == "matrix":
            n = v
        elif name == "translate":
            n = [1, 0, 0, 1, v[0], v[1] if len(v) > 1 else 0]
        elif name == "scale":
            n = [v[0], 0, 0, v[1] if len(v) > 1 else v[0], 0, 0]
        else:
            raise ValueError(name)
        m = matmul(m, n)
    return m


def apply(m, x, y):
    return m[0]*x + m[2]*y + m[4], m[1]*x + m[3]*y + m[5]


def subpaths(d):
    out, cur = [], []
    for op, x, y in re.findall(r"([MLC])\s*([-\d.eE]+)\s+([-\d.eE]+)", d):
        if op == "M":
            if len(cur) > 1:
                out.append(cur)
            cur = [(float(x), float(y))]
        else:
            cur.append((float(x), float(y)))
    if len(cur) > 1:
        out.append(cur)
    if re.search(r"[CQAHVZ]", d.replace("e", "")):
        pass
    return out


def stroked_paths(svg_file):
    tree = ET.parse(svg_file)
    res = []

    def walk(el, m):
        m = matmul(m, parse_transform(el.get("transform")))
        if el.tag == NS + "path" and el.get("stroke") not in (None, "none"):
            for sp in subpaths(el.get("d", "")):
                res.append({"stroke": el.get("stroke"),
                            "width": float(el.get("stroke-width", "1")),
                            "pts": [apply(m, x, y) for x, y in sp]})
        for c in el:
            if c.tag != NS + "defs":
                walk(c, m)
    walk(tree.getroot(), [1, 0, 0, 1, 0, 0])
    return res
