#!/usr/bin/env python3
"""T14 x T13: variable wind on face UNIONS -> ../../t14_multiface.csv (aggregated from the T14 tables).
Rows: the union gate runs (Neumann opp / adj / three, exact neumann-var-opp-3d; Dirichlet opp / adj with L2,
H1-space, H1-time, combined; non-unit adjacent), the explicit B_D / G1D union checks (C + C^T = B_w - D_w on
I u C with the independent reference), byte-identical permutation under the generic wind, and the union MPI
cases.  PASS iff every source row passes."""
import csv, os
D = os.path.dirname(os.path.abspath(__file__)) + "/../.."
rows = []
rd = lambda f: list(csv.DictReader(open(f"{D}/{f}")))
for r in rd("t14_gates_neumann.csv"):
    if r["group"] in ("opp", "adj", "thr", "ex2"):
        rows.append({"source": "gates_neumann", "case": f"{r['group']}_{r['wind']}_{r['scheme']}_n{r['n']}",
                     "check": "G1W identity / G13N union / FD / negatives",
                     "value": f"G1W={r['G1W_identity']} union={r['G13N_union']} fd={r['fd_shell']} drop={r['G13N_neg_drop']} missdiv={r['fd_neg_missing_div']}",
                     "verdict": r["verdict"]})
for r in rd("t14_gates_dirichlet.csv"):
    if r["group"] in ("opp", "adj", "nu"):
        rows.append({"source": "gates_dirichlet", "case": f"{r['group']}_{r['wind']}_{r['reg']}_{r['scheme']}_n{r['n']}",
                     "check": "G1W_DIRICHLET / B_D^T / FD / core",
                     "value": f"G1W={r['G1W_dirichlet']} BDt={r['G3D_BD_transpose']} fd={r['G4D_fd_shell']} res={r['G6D_true_res']}",
                     "verdict": r["verdict"]})
for r in rd("t14/t14_bd.csv"):
    if "+" in r["case"]:
        rows.append({"source": "bd_explicit", "case": r["case"], "check": r["check"], "value": r["value"], "verdict": r["verdict"]})
for r in rd("t14_permutation.csv"):
    rows.append({"source": "permutation", "case": f"{r['kind']} {r['set']} {r['order']}", "check": f"byte-identical {r['compared']}",
                 "value": r["result"], "verdict": r["verdict"]})
for r in rd("t14_mpi.csv"):
    if r["case"] in ("Nof", "Daf", "Dh1"):
        rows.append({"source": "mpi", "case": f"{r['case']} np={r['np']}", "check": "fields / J / trace set vs np = 1",
                     "value": f"y={r['y_maxrel']} u={r['u_maxrel']} J={r['J_rel']} trace={r['trace_nodes']}/{r['trace_expected']}",
                     "verdict": r["verdict"]})
with open(f"{D}/t14_multiface.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
ok = all(r["verdict"] == "PASS" for r in rows)
print("T14_MULTIFACE_RESULT", "PASS" if ok else "FAIL", f"({sum(r['verdict'] == 'PASS' for r in rows)}/{len(rows)})")
