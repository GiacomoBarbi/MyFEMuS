# ex21 — zero-convection baseline (Phase 1)

**Claim tested.** ex21 with `v = 0` reproduces ex20.

**Binary.** Final ex21 build
`femusbin-control/applications/TimeOptCtrl/ex21/TimeOptCtrl_ex21`,
sha256 `c5f89f0e…df19e606` (`logs/final_binary.sha256`). Every result below
comes from `scripts/run_all_gates.sh` on that single binary (rerun after the
Mathematics 2025 Example 3 and Example 4 additions — `source_regions`, the
benchmark presets; every number reproduced identically to the earlier binaries
`e5412ac1…` and `256681ba…`). ex20 reference
binary sha256 `90ee8fe0…7dfe7` (unchanged since before ex21 existed).
Serial, `OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=1`.

Two meanings of "v = 0" are tested separately:

* **ex21 default** — no ex21 option. This is the path every ex20 user gets.
* **ex21 `--convection on --vx 0 --vy 0`** — the convection machinery is
  switched on with a zero velocity: the assembled `C` is zero, but every adjoint
  product goes through `MatMultTranspose`, the backward sweep uses transposed
  blocks and the spatial sweep solves use GMRES instead of CG. This is the
  stronger test of the new code paths.

## 1. ex20's own 61-case golden suite, run with the ex21 binary

```
python3 aux/cn_stage1/be_nonregression.py \
   --golden aux/cn_stage1/baseline/refactor_golden_openblas \
   --golden aux/cn_stage1/baseline/cli_golden_openblas \
   --exe .../ex21/TimeOptCtrl_ex21          (run from ex20/, sanitised env)
BE NON-REGRESSION (time_scheme metadata normalised): 61/61 cases identical
```

The goldens were produced by ex20 and cover distributed (box / whole-domain /
multi-region), Neumann and Dirichlet control, all five preconditioners, the
verification batteries, the literature and manufactured benchmarks, and an MPI
`np = 2` case. **ex21 default is bit-identical to ex20 on all 61**
(`logs/ex21_default_vs_ex20_goldens.log`).

## 2. Direct ex20 ↔ ex21 comparison incl. Crank–Nicolson

`scripts/zero_convection_baseline.py`; common options
`--nx 8 --nz 8 --steps 16 --target-type sin_x_minus_t --alpha-u 1e-3
--no-paraview`, same mesh, Q2, scheme, objective, target, control region,
`α_u`, solver options. Max relative differences over all time slots and DOFs
(`results/zero_convection_baseline.csv`, which also stores both command lines):

| case | variant | state | adjoint | control | `J_total` | outer its | inner y / p its | true residual | bitwise |
|---|---|---|---|---|---|---|---|---|---|
| distributed BE ADDITIVE_GAMG | default | 0 | 0 | 0 | 0 | 14 = 14 | 381/401 = | 1.30e-11 = | ✔ |
| | conv on, v=0 | 0 | 0 | 0 | 0 | 14 = 14 | = | = | ✔ |
| distributed CN ADDITIVE_GAMG | default | 0 | 0 | 0 | 0 | 14 = 14 | 892/949 = | 9.11e-12 = | ✔ |
| | conv on, v=0 | 0 | 0 | 0 | 0 | = | = | = | ✔ |
| distributed BE SCHUR_MATCHING | default | 0 | 0 | 0 | 0 | 2 = 2 | 481/14 = | 3.23e-12 = | ✔ |
| | conv on, v=0 | 0 | 0 | 0 | 0 | = | = | = | ✔ |
| distributed CN SCHUR_MATCHING | default | 0 | 0 | 0 | 0 | 3 = 3 | 1739/21 = | 4.20e-13 = | ✔ |
| | conv on, v=0 | 0 | 0 | 0 | 0 | = | = | = | ✔ |
| distributed BE SCHUR_SWEEP | default | 0 | 0 | 0 | 0 | 2 = 2 | 18/14 = | 2.43e-15 = | ✔ |
| | conv on, v=0 | 1.9e-15 | 1.3e-15 | 7.8e-16 | 1.3e-16 | 2 = 2 | 18/14 = | 2.40e-15 | roundoff |
| distributed CN SCHUR_SWEEP | default | 0 | 0 | 0 | 0 | 2 = 2 | 18/14 = | 1.20e-14 = | ✔ |
| | conv on, v=0 | 1.8e-15 | 1.1e-15 | 1.1e-15 | 1.2e-16 | 2 = 2 | 18/14 = | 1.20e-14 = | roundoff |
| Neumann BE ADDITIVE_GAMG | default | 0 | 0 | 0 | 0 | 16 = 16 | 450/453 = | 1.17e-11 = | ✔ |
| Neumann CN SCHUR_BOUNDARY_REDUCED | default | 0 | 0 | 0 | 0 | 2 = 2 | 6/2 = | 2.51e-15 = | ✔ |
| Dirichlet CN SCHUR_DIRICHLET_REDUCED | default | 0 | 0 | 0 | 0 | 2 = 2 | 6/2 = | 5.64e-15 = | ✔ |

* Neumann and Dirichlet control are compared on the default path only: ex21
  refuses `--convection on` for boundary control (see
  `EX21_SOLVER_NONSYMMETRY_AUDIT.md`), so there is no convection variant.
* With the convection machinery active and `v = 0`, ADDITIVE_GAMG and
  SCHUR_MATCHING stay **bit-identical** (for a symmetric AIJ matrix the serial
  `MatMultTranspose` happens to give the same bits here). SCHUR_SWEEP differs
  at **≤ 1.9e-15**, from GMRES replacing CG inside the spatial sweep solves;
  the outer and inner iteration counts and the residual level are unchanged.

## 3. Verdict

ex21 with `v = 0` reproduces ex20: **bit for bit** on the default path (61/61
goldens + 9 direct cases, BE and CN), and to **roundoff (≤ 2e-15)** with the
convection code paths switched on. Phase 2 was allowed to proceed.

## 4. The one ex21 change that can alter a default-path result

`Solver.cpp`, CN + homogeneous-Dirichlet **state** BC: the initial state is
restricted to zero trace before the CN right-hand side is formed (see
`EX21_CONVECTION_VALIDATION.md` §K). No ex20 case reaches it — ex20 can set the
Dirichlet state BC only through its benchmarks, which reject CN — and none of
the 61 goldens changed. When `y₀` vanishes on `∂Ω` it is a no-op.
