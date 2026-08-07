# Documentation audit baseline (WP-00)

**Project:** metric-flow-x (repository name: blood-flow)  
**Audit date (UTC):** 2026-08-07  
**Scope:** repository-verified static facts and command attempts only. This document is a baseline, not a proposal to change implementation or scientific claims.

## Inputs and limitations

The supplied audit context and plan were not present at the requested paths:

- `/var/folders/8z/hlb6vc015qjggytkxn84m6_c0000gn/T/pi-worktree-698521ca-0/context.md` — unavailable.
- `/var/folders/8z/hlb6vc015qjggytkxn84m6_c0000gn/T/pi-worktree-698521ca-0/plan.md` — unavailable.

(The worktree uses the equivalent `/private/var/...` prefix; neither file was found.) No supplied audit plan was therefore available. Findings below are limited to files in this checkout and the commands recorded here. The uploaded TeX material containing the requested `\who_i` blocker is also unavailable; the repository contains `latex/blood_flow.tex` and defines `\who`, but no `\who_i` occurrence was found.

## Repository snapshot and hash inventory

Commands were run from the repository root.

| Fact | Evidence |
|---|---|
| HEAD | `6e9714aba11f12e785ad9281fa0a4e1972fd0b18` |
| HEAD tree | `bf23df68e58a5eb5ecc6dc9eec42a02c12ec556a` |
| HEAD subject | `Merge remote-tracking branch 'origin/main'` |
| tracked paths | 213 |
| worktree at start | clean (`git status --short` produced no output) |
| `git ls-files` SHA-256 | `daddd0959bb0c20bc2bb134a7f5e8d99db08bf17d9078a1c276ed68d2b5b344e` |
| `git ls-tree -r --name-only HEAD` SHA-256 | `daddd0959bb0c20bc2bb134a7f5e8d99db08bf17d9078a1c276ed68d2b5b344e` |
| `git ls-files -s` SHA-256 | `5c38c2562d5053d0207b174f7373f38c20cccaa05c8003ec72557f5e0e2a9494` |

The audit files are the only intended repository changes. Generated `build/` and Doxygen output were used for command attempts and are not part of this change.

## Static implementation facts

- `source/blood_flow_system.cc` and `include/blood_flow_system.h` instantiate **SUNDIALS IDA**, not ARKode (`SUNDIALS::IDA<VectorType>` and `solve_dae`).
- The unknown layout is documented in source comments and code as `FESystem(FE_DGQ(fe_degree), 2)` for cell area/velocity, `FE_DGQ(1), 2` for trace area/velocity, followed by terminal-capacitor pressures. This is an HDG-type monolithic vector with cell, trace, and capacitor blocks; trace rows are algebraic for IDA.
- The direct path selects PETSc `SparseDirectMUMPS` when PETSc is active, or `TrilinosWrappers::SolverDirect` otherwise. The iterative path uses `LA::SolverGMRES` with `LA::MPI::PreconditionILU`.
- An earlier audit snapshot recorded stale README claims relative to those facts: it said `Blood Flow DG/ARKode Solver`, named ARKode as the time integrator, named UMFPACK, and claimed an exact Jacobian. These are historical audit findings, not claims made by the current README, and were recorded rather than edited.
- The source still exposes parameter-file sections named `ARKOde parameters` in older/sample files while current IDA files use `IDA parameters`; this is a documentation/configuration consistency issue, not changed here.
- Licensing, funding, mathematical decisions, and tutorial claims were not changed.

## Parameter inventory

`find parameters -type f` reports 76 files: 26 `*.prm*` files, 2 VTK meshes, 3 notebooks, 1 TXT file, 3 CSV files, and 41 PNG files. There is one application source (`apps/blood_flow.cc`). There are 51 unique `set` keys across `parameters/**/*.prm*`.

The parameter inventory includes mesh path, finite-element degree, refinement, flux/stability controls, physical constants (`rho`, `mu`, `E`, `h_wall`, `a0`, `a_d`, `p_d`, `p0`, `r0`, `L`, `m`), boundary/RCR data (`R1`, `R2`, `C`, `P_out`), initial/final time and tolerances, output controls, direct/iterative selection, and solver settings. Existing files include both `ARKOde parameters` and `IDA parameters` subsections; no parameter-file migration was attempted.

`parameters/Test_cases.txt` is present and contains five manufactured/sinusoidal test-case descriptions, common values, numerical settings, and a note that boundary oscillations persist in some sinusoidal tests. It is reference data, not a proof of a benchmark result.

## VTK topology inventory

The tracked runtime meshes are legacy ASCII unstructured-grid files with line cells (VTK cell type 3):

| File | Points | Cells | Cell types | Topology |
|---|---:|---:|---:|---|
| `parameters/single_vessel.vtk` | 2 | 1 | 1 | one line, one vessel |
| `parameters/aortic.vtk` | 4 | 3 | 3 | three lines: one inlet segment bifurcating into two branches |

Both include `CELL_DATA` vessel/physical arrays and `POINT_DATA` boundary/RCR arrays. Additional tracked notebook VTK assets include a 37-vessel network (`notebooks/37_vessel_network.vtk`, 38 points/37 cells) and 56-artery assets (`notebooks/56_adnr*.vtk`, 78 points/77 cells). These assets do not establish a 57-artery benchmark.

## CMake, build, and test inventory

### Configure and build

Command: `cmake -S . -B build`  
**PASS (exit 0).** deal.II 9.8.0-rc1 was found at `/Applications/deal.II.app/Contents/Resources/Libraries`; CMake configured 12 tests and generated build files.

Command: `cmake --build build`  
**PASS (exit 0).** `test_library` and `blood_flow` built successfully. The compiler emitted 10 `-Wunused-variable` warnings for `a_hat_extractor` and `u_hat_extractor` in `source/blood_flow_system.cc`.

### CTest inventory

The configured 12 tests are:

1. `tests/template.debug`
2. `tests/test_constant.debug`
3. `tests/test_jacobian_boundary.debug`
4. `tests/test_jacobian_cell.debug`
5. `tests/test_jacobian_eta.debug`
6. `tests/test_jacobian_interior.debug`
7. `tests/test_jacobian_junction.debug`
8. `tests/test_mass_integration.debug`
9. `tests/test_mass_integration_velocity.debug`
10. `tests/test_mass_ones.debug`
11. `tests/test_trace_residual.debug`
12. `tests/test_vtk.debug`

Command: `ctest --test-dir build --output-on-failure`  
**FAIL (exit 8):** 11/12 passed; `tests/test_jacobian_cell.debug` failed its expected-output DIFF. Exact reported diagnostic:

```text
tests/test_jacobian_cell.debug: DIFF failed. ------ First 20 lines of numdiff output:
----------------
##3       #:3   <== 2
##3       #:3   ==> 18
@ Absolute error = 1.6000000000e+1, Relative error = 8.0000000000e+0

92% tests passed, 1 tests failed out of 12
The following tests FAILED:
  4 - tests/test_jacobian_cell.debug (Failed)
Errors while running CTest
```

The test built and ran before the expected-output comparison failed. No test output or source was modified to make it pass.

## Documentation and Doxygen/Sphinx inventory

`doc/Doxyfile` exists and requests HTML output from `./source`, with `GENERATE_XML = NO`, while `doc/conf.py` configures Breathe/Exhale to consume `build/docs/doxygen/xml`.

Command: `doxygen doc/Doxyfile`  
**PASS (exit 0), with warnings.** Exact relevant warnings:

```text
warning: Tag 'DOT_MULTI_TARGETS' at line 2914 of file 'doc/Doxyfile' has become obsolete.
warning: source './doc/images' is not a readable file or directory... skipping.
error: Tag file './doc/deal.tag' does not exist or is not a file. Skipping it...
Doxygen version used: 1.17.0
```

Doxygen generated HTML, but did not generate the XML tree expected by the Sphinx configuration.

Command: `sphinx-build -b html doc doc/_build/html`  
**FAIL (exit 2).** Exact blocker:

```text
Configuration error:
Exhale: the specified folder [.../build/docs/doxygen/xml] does not exist.  Has Doxygen been run?
```

The installed Sphinx executable was Sphinx 7.4.7. `python3 -m sphinx --version` independently failed with `No module named sphinx` because that interpreter is not the interpreter owning the executable. No documentation build configuration was changed.

## LaTeX inventory

Tracked source files include `latex/blood_flow.tex` and `latex/blood_flow.bib`. The uploaded TeX artifact requested for the `\who_i` check was not present, so its status is **unavailable**, not passed or failed.

Command (outputs isolated under `/tmp`): `(cd latex && latexmk -pdf -interaction=nonstopmode -outdir=/tmp/metric-flow-x-latex-audit2 blood_flow.tex)`  
**PASS (exit 0).** pdfTeX/BibTeX completed and wrote a 14-page PDF. The run emitted a persistent overfull box warning (`7.71951pt too wide` at line 746) and hyperref warnings about math tokens in PDF strings. The first run also reported undefined references before the BibTeX/rerun cycle; the final output was produced. This is a build result, not a claim that the manuscript is publication-clean.

## Benchmark and reference-data evidence

- `parameters/Test_cases.txt` is the repository's plain-text reference/test-case summary.
- Tracked names/content establish 37-artery and 56-artery material (`parameters/benchmark-parameters/37arteries_network`, `parameters/benchmark-parameters/56_ADNR`, `NumData/37-arteries`, and `NumData/56-arteries`).
- `git grep -n -i -E '57[ -]?arter|57 arteries|57-arter' HEAD -- ':(exclude)*.png'` produced no matches (exit 1). There is therefore **no proven 57-artery benchmark in this repository**. The presence of files with incidental digits, or a 56-artery VTK, is not evidence for 57 arteries.

## Blockers and non-goals

Blockers recorded for follow-up are: missing supplied context/plan; unavailable uploaded TeX artifact at `\who_i`; Sphinx/Exhale XML-path mismatch; one CTest expected-output failure; Doxygen missing optional `doc/deal.tag` and `doc/images`; and the absence of a proven 57-artery benchmark. WP-00 deliberately does not repair any blocker, update README/tutorial wording, alter source behavior, change license/funding/math decisions, or add benchmark claims.
