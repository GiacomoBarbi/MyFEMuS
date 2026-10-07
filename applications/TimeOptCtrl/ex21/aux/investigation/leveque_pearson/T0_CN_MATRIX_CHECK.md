# T0 — algebraic check of the Leveque–Pearson CN system against ex21's CN KKT system

**Result: PASS, 49/49 checks.**

The two Crank–Nicolson optimality systems have the **same state operator**. After
matching the state rows by the interval-average control, they also have the **same
state rows and the same state right-hand side**. They differ in the **tracking
(1,1) block** and the **control (2,2) block**. For the same data the two
discrete solutions are different, and the difference goes to zero at O(dt²).

This turns §§5–6 of `LEVEQUE_PEARSON_SOURCE_AUDIT.md` from an inference [B] into a
measured result [A]. It also sharpens one statement there (see "Correction to the
audit" below).

Files:
- `t0_cn_matrix_check.py` — the check;
- `logs/t0_cn_matrix_check.log` — its output;
- `t0_dump/` — the operators dumped by the ex21 binary.

## 1. What is compared

**LP (Leveque–Pearson, NLAA 2022, eqs. (13)–(15), as printed; checked against the
PDF text).** Notation:
- `L± = τ/2 L ± M`, `M̄ = τ/2 M`, `M_β = τ/(2β) M`;
- unknowns `y = (y¹…y^{nt})` and `p = (p⁰…p^{nt−1})`;
- conditions `p^{nt} = 0` and `y⁰` given.

The rows for `n = 0…nt−1` are:

    adjoint  M̄(yⁿ + yⁿ⁺¹) + (L⁺)ᵀpⁿ + (L⁻)ᵀpⁿ⁺¹ = M̄(ŷⁿ + ŷⁿ⁺¹)
    state    L⁻yⁿ + L⁺yⁿ⁺¹ − M_β(pⁿ + pⁿ⁺¹)    = τ/2 (fⁿ + fⁿ⁺¹)
    [Λ11 Λ12; Λ21 −Λ22] [y; p] = [b1; b2]

**ex21** (`Solver.hpp`; dt-scaled multiplier λ; `η = α_track = 1`;
`N = M_ω = M`; `α_u = β`):

    [ E          (1/α_u) I⊗M ] [Y]   [ rhs_y      ]
    [ −W⊗M       Eᵀ          ] [Λ] = [ −W⊗M Y_d   ]
    E = blockdiag(M/dt + L/2) + blocksubdiag(−M/dt + L/2),   W = diag(1,…,1,½)
    rhs_y,n = ½(F_{n−1} + F_n) + [n = 1](M/dt − L/2) y₀,     u = −λ/α_u

## 2. Three parts

| part | matrices | purpose |
|---|---|---|
| A | random SPD `M` (4×4) and a random **nonsymmetric** `L` with no structure (`‖L−Lᵀ‖/‖L‖ = 0.49`); `Nt = 3`, `dt = 0.37`, `β = 0.013` | the identities are algebraic; they do not depend on `L`'s structure |
| B | `M`, `L = K + C`, `N`, `M_ω` and the **assembled space-time `E`, `Eᵀ` written by the ex21 binary** (`--exact-oc conv-sin3 --nx 2 --steps 3 --exact-oc-dump`, CN and BE). Restricted to the 9 free DOFs (16 Dirichlet DOFs removed); `L` is nonsymmetric (`0.32`) | ties the model in part A to the operator ex21 actually solves |
| C | 1D finite-difference convection–diffusion model (`m = 63`, `ν = 1/4`, `v = 1`, `β = 1e-2`, `T = 1`, smooth data), `Nt = 8…256`, both systems solved directly | non-equivalence at the level of the solution, and its decay rate |

## 3. Results

Values are relative Frobenius norms (`‖X − Y‖/‖Y‖`). Part A is listed first,
then part B (CN).

| # | claim | A | B (ex21 binary, CN) | verdict |
|---|---|---|---|---|
| – | ex21 binary `E` = `blockdiag(M/dt+L/2) + subdiag(−M/dt+L/2)` | – | 4.8e-18 | equal |
| – | ex21 binary `Eᵀ` (assembled adjoint) = `(binary E)ᵀ` | – | 0 | equal |
| – | ex21 binary BE `E` = `blockdiag(M/dt+L) + subdiag(−M/dt)` | – | 3.0e-18 | equal |
| – | ex21 `N = M` and `M_ω = M` (full-domain control and tracking) | – | 0, 0 | equal |
| 2 | `Λ21 = dt·E` (state operator) | 9.3e-17 | 8.6e-17 | **equal** |
| 3 | `Λ12 = Λ21ᵀ` | 0 | 0 | equal |
| 3 | `Λ12 = dt·Eᵀ` (with the positional identification `p ↔ −λ`) | 9.3e-17 | 8.6e-17 | **equal** |
| 1 | `Λ22 = (dt/β)(Avg ⊗ M)`, where `Avg` maps p to the interval average `(pⁿ+pⁿ⁺¹)/2` | 0 | 0 | equal |
| 1 | LP state rows = `dt ×` ex21 state rows when `u_n := ū_n = (pⁿ+pⁿ⁺¹)/(2β)` (random y, p) | 1.8e-16 | 1.6e-16 | **equal** |
| 1 | LP state rhs `b2` = `dt × rhs_y` (includes the `y₀` term; trapezoidal source in both) | 2.6e-16 | – | **equal** |
| 4 | `Λ11 = (dt/2)(I + subdiag) ⊗ M` (interval-averaged tracking) | 0 | 0 | as stated |
| 4 | ex21 tracking = `dt·diag(1,…,1,½) ⊗ M` (nodal trapezoid weights) | 0 | 0 | as stated |
| 4 | strictly-lower block norm: LP `Λ11` / ex21 tracking | 10.8 / **0** | 0.055 / **0** | LP bidiagonal, ex21 diagonal |
| 4 | `‖Λ11 − dt W⊗M‖/‖Λ11‖` | **0.667** | **0.667** | **different** |
| 4 | LP adjoint rhs `b1` vs `dt ×` ex21 tracking rhs (sign-aligned) | **0.59** | – | **different** |
| 5 | strictly-upper block norm: LP `Λ22` / ex21 control coupling | 830 / **0** | 5.5 / **0** | LP bidiagonal, ex21 diagonal |
| 5 | `‖Λ22 − (dt/α_u) I⊗M‖/‖Λ22‖` | **0.645** | **0.645** | **different** |
| – | LP (17): `T1Λ11` symmetric, `T2Λ22` symmetric, `T2Λ21 = (T1Λ12)ᵀ` | 0, 0, 0 | 0, 0, 0 | LP's symmetrisation reproduced |
| 6 | solutions for the same data: `‖y_LP − y_ex21‖/‖y‖`, `‖p_LP + λ_ex21‖/‖λ‖` | **0.39**, **1.5** | – | **different** |

**Part C.** Same data and the same 1D model. The first column is
`‖y_LP − y_ex21‖/‖y‖` over all time levels; the second is the same quantity at
`t = T`.

| Nt | over all time levels | ratio to previous row | at t = T |
|---|---|---|---|
| 8 | 6.77e-2 | – | 9.76e-2 |
| 16 | 1.59e-2 | 4.25 | 2.99e-2 |
| 32 | 3.80e-3 | 4.19 | 7.94e-3 |
| 64 | 9.28e-4 | 4.09 | 2.02e-3 |
| 128 | 2.30e-4 | 4.04 | 5.06e-4 |
| 256 | 5.71e-5 | 4.02 | 1.27e-4 |

The difference between the two discrete solutions is **not round-off**. It
converges to zero at order **2.01**, which is consistent with two second-order
discretisations of the same continuous problem.

## 4. Conclusion

1. **What matches exactly.**
   - The state operator `Λ21 = dt·E`.
   - The adjoint operator `Λ12 = Λ21ᵀ = dt·Eᵀ`. These are the same matrices once
     LP's `p = (p⁰…p^{nt−1})` and ex21's `−Λ = −(λ₁…λ_N)` are identified
     **positionally**.
   - The state rows and the state rhs, including the `y₀` term and the
     trapezoidal source, once LP's interval average of its nodal control is
     identified with ex21's interval control. This needs `β = α_u`, `η = 1`,
     `N = M`.

   All of these hold to round-off on generic matrices and on the operators of
   the ex21 binary.
2. **What differs.**
   - The **tracking block**: LP is lower-bidiagonal and interval-averaged,
     `M̄(yⁿ + yⁿ⁺¹)`. ex21 is block-diagonal with nodal trapezoid weights
     `(1,…,1,½)`.
   - The **control block**: LP is upper-bidiagonal, `M_β(pⁿ + pⁿ⁺¹)`. ex21 is
     block-diagonal, one interval control per interval.
   - The **tracking right-hand side** (interval average vs nodal).
   - The **meaning of the time index** of the adjoint unknown. In LP, `pⁿ` is
     nodal (`t_n`) and `p^{nt} = 0` is imposed. In ex21, `λ_n` is an interval
     quantity whose terminal condition is `λ_{N+1} = 0`.

   For the same data the solutions differ (39 % in `y` at `Nt = 3`). The
   difference decays as O(dt²).
3. **Therefore** the two methods discretise **the same continuous optimal-control
   problem** but do **not** produce the same discrete KKT system.
   - No exact transformation was found that maps one onto the other, and none
     was forced.
   - The solution-level test (item 6, part C) rules out any transformation that
     preserves the state: such a transformation would give identical `y` for
     identical data.

## 5. Correction to the audit

Audit §6 described the adjoint-operator pattern as equal. That is now confirmed,
and in a stronger form:
- with the positional identification `p ↔ −λ`, LP's `Λ12` **is** `dt·Eᵀ`, entry
  for entry;
- the whole difference sits in the two "mass-type" blocks (tracking and control)
  and in the tracking rhs;
- LP's `p^{nt} = 0` and ex21's `λ_{N+1} = 0` are the same algebraic condition on
  the last unknown. They are interpreted at different times (`t_{nt}` versus the
  interval beyond `t_N`).

The audit's remark that the terminal conditions "differ" therefore concerns
**interpretation and the coupling blocks**, not the adjoint operator.
