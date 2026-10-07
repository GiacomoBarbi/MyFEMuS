#!/usr/bin/env python3
"""T0 -- algebraic cross-check of the Leveque-Pearson (LP, NLAA 2022, eqs. (13)-(19))
Crank-Nicolson optimality system against the ex21 DTO Crank-Nicolson KKT system.

NOT a PDE convergence test.  Three parts:

  A. generic matrices: random SPD M, random NONSYMMETRIC L (no structure), Nt = 3;
  B. the ex21 binary itself: M, L = K + C and the assembled space-time E, E^T
     dumped by `TimeOptCtrl_ex21 --exact-oc conv-sin3 --nx 2 --steps 3
     --exact-oc-dump ...` (t0_dump/), restricted to the free (non-Dirichlet) DOFs;
  C. solution-level non-equivalence: both systems solved with the same data on a
     1D convection-diffusion model, Nt = 8 ... 256.

LP system (after eliminating y^0 and p^{nt} = 0), unknowns y = (y^1..y^nt),
p = (p^0..p^{nt-1}), rows n = 0..nt-1:

    adjoint  Mbar (y^n + y^{n+1}) + (L+)^T p^n + (L-)^T p^{n+1} = Mbar (yhat^n + yhat^{n+1})
    state    L- y^n + L+ y^{n+1} - Mbeta (p^n + p^{n+1})      = tau/2 (f^n + f^{n+1})
    [Lam11 Lam12; Lam21 -Lam22] [y; p] = [b1; b2]
    L+ = tau/2 L + M,  L- = tau/2 L - M,  Mbar = tau/2 M,  Mbeta = tau/(2 beta) M.

ex21 system (Solver.hpp; dt-scaled multiplier lambda, eta = alpha_t = 1, N = M_omega = M,
alpha_u = beta), unknowns Y = (y_1..y_N), Lambda = (lambda_1..lambda_N):

    [ E                 (1/alpha_u) I(x)M ] [Y]        [ rhs_y ]
    [ -W(x)M            E^T               ] [Lambda] = [ -W(x)M Yd ]
    E = blockdiag(M/dt + L/2) + blocksubdiag(-M/dt + L/2),  W = diag(1,..,1,1/2),
    rhs_y,n = (F_{n-1} + F_n)/2 + [n=1] (M/dt - L/2) y_0,   u = -lambda/alpha_u.

Every claimed equality is reported as a relative Frobenius norm; every claimed
difference as the norm of the difference.  Exit status 0 iff all PASS.
"""
import os
import sys
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla

HERE = os.path.dirname(os.path.abspath(__file__))
TOL = 1e-13
results = []


def rel(a, b):
    nb = np.linalg.norm(b)
    return np.linalg.norm(a - b) / (nb if nb > 0 else 1.0)


def check(label, value, expect_zero=True, tol=TOL):
    ok = (value <= tol) if expect_zero else (value > 1e3 * tol)
    results.append((label, value, expect_zero, ok))
    tag = "PASS" if ok else "FAIL"
    kind = "== 0" if expect_zero else "!= 0"
    print(f"  [{tag}] {label:78s} {value:.3e}  (expected {kind})")


def blocks(nt, m, entries):
    """Block matrix from {(row, col): block}."""
    A = np.zeros((nt * m, nt * m))
    for (i, j), B in entries.items():
        if 0 <= i < nt and 0 <= j < nt:
            A[i * m:(i + 1) * m, j * m:(j + 1) * m] += B
    return A


def lp_system(M, L, tau, beta, nt):
    m = M.shape[0]
    Lp, Lm = tau / 2 * L + M, tau / 2 * L - M
    Mbar, Mb = tau / 2 * M, tau / (2 * beta) * M
    L11, L12, L21, L22 = {}, {}, {}, {}
    for n in range(nt):
        # y index j <-> y^{j+1};  p index k <-> p^k
        L11[(n, n)] = Mbar             # y^{n+1}
        L11[(n, n - 1)] = Mbar         # y^n (n >= 1; y^0 is data)
        L12[(n, n)] = Lp.T             # p^n
        L12[(n, n + 1)] = Lm.T         # p^{n+1} (p^{nt} = 0 dropped)
        L21[(n, n)] = Lp               # y^{n+1}
        L21[(n, n - 1)] = Lm           # y^n
        L22[(n, n)] = Mb               # p^n
        L22[(n, n + 1)] = Mb           # p^{n+1}
    return (blocks(nt, m, L11), blocks(nt, m, L12), blocks(nt, m, L21),
            blocks(nt, m, L22))


def lp_rhs(M, L, tau, nt, y0, F, Yd):
    """F, Yd: lists of nt+1 nodal vectors (index 0..nt)."""
    Lm, Mbar = tau / 2 * L - M, tau / 2 * M
    b1 = np.concatenate([Mbar @ (Yd[n] + Yd[n + 1]) for n in range(nt)])
    b2 = np.concatenate([tau / 2 * (F[n] + F[n + 1]) for n in range(nt)])
    m = M.shape[0]
    b1[:m] -= Mbar @ y0
    b2[:m] -= Lm @ y0
    return b1, b2


def ex21_system(M, L, dt, alpha_u, nt):
    m = M.shape[0]
    E = blocks(nt, m, {**{(n, n): M / dt + L / 2 for n in range(nt)},
                       **{(n, n - 1): -M / dt + L / 2 for n in range(1, nt)}})
    w = np.ones(nt); w[-1] = 0.5
    Wtrack = np.kron(np.diag(w), M)
    S = np.kron(np.eye(nt), M) / alpha_u
    return E, Wtrack, S, w


def ex21_rhs(M, L, dt, nt, y0, F, Yd, w):
    m = M.shape[0]
    ry = np.concatenate([(F[n - 1] + F[n]) / 2 for n in range(1, nt + 1)])
    ry[:m] += (M / dt - L / 2) @ y0
    rp = np.concatenate([-w[n - 1] * (M @ Yd[n]) for n in range(1, nt + 1)])
    return ry, rp


def structural_checks(M, L, tau, beta, nt, label, E_binary=None, ET_binary=None):
    m = M.shape[0]
    L11, L12, L21, L22 = lp_system(M, L, tau, beta, nt)
    E, Wtrack, S, w = ex21_system(M, L, tau, beta, nt)
    if E_binary is not None:
        check(f"{label}: ex21 binary E == blockdiag(M/dt+L/2)+subdiag(-M/dt+L/2)", rel(E_binary, E))
        check(f"{label}: ex21 binary E^T (assembled adjoint) == (binary E)^T", rel(ET_binary, E_binary.T))
        E = E_binary
    ET = E.T
    Lp, Lm = tau / 2 * L + M, tau / 2 * L - M
    # 2. state operator
    check(f"{label}: (2) LP Lambda21 == dt * E", rel(L21, tau * E))
    # 3. adjoint operator
    check(f"{label}: (3) LP Lambda12 == Lambda21^T", rel(L12, L21.T))
    check(f"{label}: (3) LP Lambda12 == dt * E^T (positional p <-> -lambda)", rel(L12, tau * ET))
    # 1. state rows under the control identification ubar_n = (p^n + p^{n+1})/(2 beta)
    Avg = np.eye(nt) / 2 + np.diag(np.ones(nt - 1), 1) / 2      # interval n <- (p^n + p^{n+1})/2
    check(f"{label}: (1) LP Lambda22 == (dt/beta) (Avg (x) M)", rel(L22, tau / beta * np.kron(Avg, M)))
    rng = np.random.default_rng(5)
    y, p = rng.standard_normal(nt * m), rng.standard_normal(nt * m)
    ubar = np.kron(Avg, np.eye(m)) @ p / beta                   # LP interval-average control
    lp_state = L21 @ y - L22 @ p                                # LP state rows (LHS)
    ex_state = tau * (E @ y - np.kron(np.eye(nt), M) @ ubar)    # dt * (E Y - eta N u), u := ubar
    check(f"{label}: (1) LP state rows == dt * ex21 state rows (u_n := ubar_n)", rel(lp_state, ex_state))
    # 4. tracking coupling
    Lbar = tau / 2 * (np.eye(nt) + np.diag(np.ones(nt - 1), -1))
    check(f"{label}: (4) LP Lambda11 == (dt/2)(I + subdiag) (x) M  [interval-averaged]", rel(L11, np.kron(Lbar, M)))
    check(f"{label}: (4) ex21 tracking == dt diag(1,..,1,1/2) (x) M  [nodal trapezoid]",
          rel(tau * Wtrack, np.kron(tau * np.diag(w), M)))
    off = lambda A, k: sum(np.linalg.norm(A[i * m:(i + 1) * m, (i + k) * m:(i + k + 1) * m])
                           for i in range(nt) if 0 <= i + k < nt)
    check(f"{label}: (4) LP Lambda11 strictly-lower block norm (bidiagonal)", off(L11, -1), expect_zero=False)
    check(f"{label}: (4) ex21 tracking strictly-lower block norm (diagonal)", off(Wtrack, -1))
    check(f"{label}: (4) ||Lambda11 - dt W(x)M|| / ||Lambda11||", rel(L11, tau * Wtrack), expect_zero=False)
    # 5. control coupling
    check(f"{label}: (5) LP Lambda22 strictly-upper block norm (bidiagonal)", off(L22, +1), expect_zero=False)
    check(f"{label}: (5) ex21 control coupling strictly-upper block norm (diagonal)", off(S, +1))
    check(f"{label}: (5) ||Lambda22 - dt (1/alpha_u) I(x)M|| / ||Lambda22||", rel(L22, tau * S), expect_zero=False)
    # LP symmetrisation (16)-(17)
    T1 = np.kron(np.eye(nt) + np.diag(np.ones(nt - 1), 1), np.eye(m))
    T2 = T1.T
    A_, C_, B_ = T1 @ L11, T2 @ L22, T2 @ L21
    check(f"{label}: LP (17) A = T1 Lambda11 symmetric", rel(A_, A_.T))
    check(f"{label}: LP (17) C = T2 Lambda22 symmetric", rel(C_, C_.T))
    check(f"{label}: LP (17) B = T2 Lambda21 == (T1 Lambda12)^T", rel(B_, (T1 @ L12).T))
    return L11, L12, L21, L22, E, Wtrack, S, w


def full_rhs_check(M, L, tau, beta, nt, label):
    m = M.shape[0]
    rng = np.random.default_rng(11)
    y0 = rng.standard_normal(m)
    F = [rng.standard_normal(m) for _ in range(nt + 1)]
    Yd = [rng.standard_normal(m) for _ in range(nt + 1)]
    E, Wtrack, S, w = ex21_system(M, L, tau, beta, nt)
    b1, b2 = lp_rhs(M, L, tau, nt, y0, F, Yd)
    ry, rp = ex21_rhs(M, L, tau, nt, y0, F, Yd, w)
    check(f"{label}: (1) LP state rhs b2 == dt * ex21 rhs_y (incl. y_0 term, trapezoid source)", rel(b2, tau * ry))
    check(f"{label}: (4) LP adjoint rhs b1 vs dt * ex21 tracking rhs (sign-aligned)", rel(b1, -tau * rp), expect_zero=False)


def solve_both(M, L, tau, beta, nt, y0, F, Yd):
    L11, L12, L21, L22 = lp_system(M, L, tau, beta, nt)
    b1, b2 = lp_rhs(M, L, tau, nt, y0, F, Yd)
    K_lp = np.block([[L11, L12], [L21, -L22]])
    s_lp = np.linalg.solve(K_lp, np.concatenate([b1, b2]))
    E, Wtrack, S, w = ex21_system(M, L, tau, beta, nt)
    ry, rp = ex21_rhs(M, L, tau, nt, y0, F, Yd, w)
    K_ex = np.block([[E, S], [-Wtrack, E.T]])
    s_ex = np.linalg.solve(K_ex, np.concatenate([ry, rp]))
    n = nt * M.shape[0]
    return s_lp[:n], s_lp[n:], s_ex[:n], s_ex[n:]


def solve_both_sparse(M, L, tau, beta, nt, y0, F, Yd):
    """Part C: the same two systems, assembled with sparse Kronecker products
    (identical block formulas to lp_system / ex21_system)."""
    M, L = sps.csr_matrix(M), sps.csr_matrix(L)
    Lp, Lm = tau / 2 * L + M, tau / 2 * L - M
    I = sps.identity(nt, format="csr")
    Sub, Sup = sps.eye(nt, k=-1, format="csr"), sps.eye(nt, k=1, format="csr")
    L11 = sps.kron(I + Sub, tau / 2 * M)
    L12 = sps.kron(I, Lp.T) + sps.kron(Sup, Lm.T)
    L21 = sps.kron(I, Lp) + sps.kron(Sub, Lm)
    L22 = sps.kron(I + Sup, tau / (2 * beta) * M)
    b1, b2 = lp_rhs(M.toarray(), L.toarray(), tau, nt, y0, F, Yd)
    K_lp = sps.bmat([[L11, L12], [L21, -L22]], format="csc")
    s_lp = spla.spsolve(K_lp, np.concatenate([b1, b2]))
    E = sps.kron(I, M / tau + L / 2) + sps.kron(Sub, -M / tau + L / 2)
    w = np.ones(nt); w[-1] = 0.5
    Wt = sps.kron(sps.diags(w), M)
    S = sps.kron(I, M / beta)
    ry, rp = ex21_rhs(M.toarray(), L.toarray(), tau, nt, y0, F, Yd, w)
    K_ex = sps.bmat([[E, S], [-Wt, E.T]], format="csc")
    s_ex = spla.spsolve(K_ex, np.concatenate([ry, rp]))
    n = nt * M.shape[0]
    return s_lp[:n], s_lp[n:], s_ex[:n], s_ex[n:]


def main():
    print("T0 -- LP (NLAA 2022, (13)-(19)) vs ex21 DTO Crank-Nicolson KKT\n")
    # ------------------------------------------------------------ A
    print("A. generic matrices (m = 4, Nt = 3, dt = 0.37, beta = 0.013; L random, nonsymmetric)")
    rng = np.random.default_rng(2022)
    m, nt, tau, beta = 4, 3, 0.37, 0.013
    G = rng.standard_normal((m, m))
    M = G @ G.T + m * np.eye(m)
    L = rng.standard_normal((m, m)) + 2 * np.eye(m)
    print(f"  ||L - L^T||_F / ||L||_F = {rel(L, L.T):.3e}  (nonsymmetric)")
    structural_checks(M, L, tau, beta, nt, "A")
    full_rhs_check(M, L, tau, beta, nt, "A")
    y0 = rng.standard_normal(m)
    F = [rng.standard_normal(m) for _ in range(nt + 1)]
    Yd = [rng.standard_normal(m) for _ in range(nt + 1)]
    ylp, plp, yex, lex = solve_both(M, L, tau, beta, nt, y0, F, Yd)
    check("A: (6) solutions: ||y_LP - y_ex21|| / ||y_ex21|| (same data)", rel(ylp, yex), expect_zero=False)
    check("A: (6) solutions: ||p_LP + lambda_ex21|| / ||lambda_ex21|| (positional)", rel(plp, -lex), expect_zero=False)

    # ------------------------------------------------------------ B
    print("\nB. the ex21 binary (conv-sin3, nx = 2 Q2, Nt = 3, dt = 1/3, nu = 1/4, v = (1, 1/2), alpha_u = 1e-2)")
    for scheme in ("cn", "be"):
        pre = os.path.join(HERE, "t0_dump", f"ex21_nx2_nt3_{scheme}")
        def load(tag):
            with open(f"{pre}_{tag}.txt") as fh:
                r, c = map(int, fh.readline().split())
                A = np.zeros((r, c))
                for line in fh:
                    i, j, v = line.split()
                    A[int(i), int(j)] = float(v)
            return A
        Mfull, Lfull, Efull, ETfull = load("M"), load("L"), load("E"), load("ET")
        Nfull, Tfull = load("N"), load("Mtrack")
        nfull = Mfull.shape[0]
        free = np.where(np.abs(np.diag(Mfull)) > 0)[0]
        fixed = np.setdiff1d(np.arange(nfull), free)
        print(f"  [{scheme}] spatial DOFs {nfull}, free {len(free)}, Dirichlet {len(fixed)}")
        check(f"B[{scheme}]: N (control mass) == M  [full-domain control]", rel(Nfull, Mfull))
        check(f"B[{scheme}]: M_omega (tracking mass) == M  [full-domain tracking]", rel(Tfull, Mfull))
        check(f"B[{scheme}]: L is nonsymmetric on the free DOFs: ||L - L^T||/||L||",
              rel(Lfull[np.ix_(free, free)], Lfull[np.ix_(free, free)].T), expect_zero=False)
        dt, nt3 = 1.0 / 3.0, 3
        idx = np.concatenate([n * nfull + free for n in range(nt3)])
        Ef, ETf = Efull[np.ix_(idx, idx)], ETfull[np.ix_(idx, idx)]
        Mf, Lf = Mfull[np.ix_(free, free)], Lfull[np.ix_(free, free)]
        check(f"B[{scheme}]: ex21 binary E^T == (binary E)^T  (full, incl. Dirichlet rows)", rel(ETfull, Efull.T))
        if scheme == "cn":
            structural_checks(Mf, Lf, dt, 1e-2, nt3, "B[cn]", E_binary=Ef, ET_binary=ETf)
        else:
            Ebe = blocks(nt3, len(free), {**{(n, n): Mf / dt + Lf for n in range(nt3)},
                                         **{(n, n - 1): -Mf / dt for n in range(1, nt3)}})
            check("B[be]: ex21 binary E == blockdiag(M/dt + L) + subdiag(-M/dt)", rel(Ef, Ebe))

    # ------------------------------------------------------------ C
    print("\nC. solution-level comparison, 1D model y_t - nu y_xx + v y_x on (0,1), FD, m = 63,"
          " nu = 0.25, v = 1, beta = 1e-2, T = 1, smooth data")
    mm, nu, v = 63, 0.25, 1.0
    hx = 1.0 / (mm + 1)
    xs = np.linspace(hx, 1 - hx, mm)
    K = nu / hx**2 * (2 * np.eye(mm) - np.eye(mm, k=1) - np.eye(mm, k=-1))
    C = v / (2 * hx) * (np.eye(mm, k=1) - np.eye(mm, k=-1))
    M1, L1 = np.eye(mm), K + C
    print("   Nt    dt        ||y_LP - y_ex21||/||y||  ratio   ||y_LP(T)-y_ex21(T)||/||y(T)||")
    prev = None
    rows = []
    for nt in (8, 16, 32, 64, 128, 256):
        dt = 1.0 / nt
        tt = np.linspace(0, 1, nt + 1)
        y0 = np.sin(np.pi * xs)
        F = [np.cos(2 * t) * np.sin(np.pi * xs) for t in tt]
        Yd = [np.sin(2 * np.pi * xs) * (1 + t) for t in tt]
        ylp, plp, yex, lex = solve_both_sparse(M1, L1, dt, 1e-2, nt, y0, F, Yd)
        d = rel(ylp, yex)
        dT = rel(ylp[-mm:], yex[-mm:])
        ratio = "" if prev is None else f"{prev / d:6.2f}"
        print(f"  {nt:4d}  {dt:.5f}   {d:.4e}             {ratio:>6s}  {dT:.4e}")
        rows.append((nt, d, dT))
        prev = d
    order = np.log2(rows[-2][1] / rows[-1][1])
    check("C: (6) y_LP != y_ex21 at Nt = 8 (relative difference)", rows[0][1], expect_zero=False)
    print(f"  observed order of ||y_LP - y_ex21|| in dt (last pair): {order:.3f}")
    results.append(("C: difference -> 0 under refinement at order ~2", order, None, 1.8 < order < 2.3))
    print(f"  [{'PASS' if 1.8 < order < 2.3 else 'FAIL'}] C: difference converges to 0 at order ~2 in dt")

    nfail = sum(1 for r in results if not r[3])
    print(f"\nT0 SUMMARY: {len(results) - nfail}/{len(results)} PASS")
    print("CONCLUSION: identical state operator E (and E^T), identical state rows and state rhs under the\n"
          "interval-average control identification; DIFFERENT tracking (1,1) and control (2,2) couplings\n"
          "(LP interval-averaged bidiagonal vs ex21 nodal/interval diagonal); the solutions differ at\n"
          "O(dt^2) for the same data -> same continuous problem, different discrete KKT systems.")
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
