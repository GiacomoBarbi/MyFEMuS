#!/usr/bin/env python3
"""Sort the report's '## N.' sections by N, renumber them 1..K and remap every '§N' reference.
usage: t13_report_order.py <report.md>"""
import re, sys
p = sys.argv[1]
s = open(p).read()
parts = re.split(r"(?m)^(?=## \d+\. )", s)
head, secs = parts[0], parts[1:]
num = lambda sec: int(re.match(r"## (\d+)\.", sec).group(1))
secs.sort(key=num)
old = [num(x) for x in secs]
assert len(set(old)) == len(old), "duplicate section numbers"
remap = {o: i + 1 for i, o in enumerate(old)}
out = []
for sec in secs:
    o = num(sec)
    sec = re.sub(r"^## \d+\.", f"## {remap[o]}.", sec, count=1)
    out.append(sec if sec.endswith("\n\n") else sec.rstrip("\n") + "\n\n")
body = "".join(out)
# references: §N, §N–M, (§N, §M) -- the request's own section numbers are written "request §N" and kept
def fix(m):
    n = int(m.group(1))
    return "§" + str(remap.get(n, n))
body = re.sub(r"(?<!request )§(\d+)", fix, body)
open(p, "w").write(head + body.rstrip("\n") + "\n")
print("sections:", " ".join(f"{o}->{remap[o]}" for o in old))
