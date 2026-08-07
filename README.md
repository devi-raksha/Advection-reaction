# Blood-flow solver

This repository contains a C++/deal.II implementation of a one-dimensional blood-flow model on vessel networks embedded in three-dimensional space. The application currently assembles an HDG-type (hybridized) monolithic system: cell area/velocity unknowns, face trace unknowns, and (for RCR terminals) capacitor-pressure unknowns are held in one distributed vector. SUNDIALS IDA advances the resulting differential-algebraic system.

This README describes the implementation that is present in the repository. It is not a validation report or a statement that the mathematical choices in the manuscript are final.

## Current implementation

- **Geometry:** 1D line cells read from legacy VTK files, embedded in `spacedim = 3`.
- **Cell fields:** cross-sectional area `A` and mean axial velocity `U`, represented with discontinuous Galerkin finite elements.
- **Hybridized fields:** face traces `A_hat` and `U_hat`; terminal RCR capacitor pressures are appended to the global unknown vector.
- **Time integration:** SUNDIALS **IDA** (`SUNDIALS::IDA`), with cell and capacitor rows treated as differential and trace rows as algebraic.
- **Numerical fluxes:** the configured string must be one of `HLL`, `HLL_HDG`, or `LAX_FRIEDRICHS`.
- **Newton linear solves:** with `Use direct solver = true`, PETSc uses `SparseDirectMUMPS` when PETSc is selected and Trilinos uses `TrilinosWrappers::SolverDirect` otherwise. With it set to `false`, the implementation uses GMRES with an ILU preconditioner.

The source contains analytical Jacobian assembly for the IDA linearization, but this README makes no claim that the Jacobian or the model has been formally validated.

## Prerequisites

A configured deal.II installation (the top-level CMake file requests deal.II 9.5.0 or newer), CMake 3.23 or newer, a C++ compiler, and the MPI/linear-algebra support provided by deal.II are required. The selected deal.II build must provide PETSc or Trilinos; the source rejects configurations with neither backend. Doxygen and the Python packages in `doc/requirements.txt` are only needed to build the developer documentation.

## Configure and build

From the repository root:

```bash
cmake -S . -B build
cmake --build build
```

CMake creates the `blood_flow` executable from `apps/blood_flow.cc`, builds the shared `test_library`, mirrors `parameters/` into `build/parameters/`, and configures the tests. Parameter templates ending in `.prm.in` are expanded while configuring; for example, `parameters/aortic.prm.in` becomes `build/parameters/aortic.prm` with the source-tree path substituted into the mesh setting.

The configure/build commands have been used in the repository audit. They still depend on a local deal.II installation and its available solver backends.

## Run

The executable requires an existing parameter-file path for a normal run. After a successful configure, the configured aortic example can be invoked as:

```bash
./build/blood_flow build/parameters/aortic.prm
```

Command-line modes are:

```text
./build/blood_flow --help
./build/blood_flow --print-parameters
./build/blood_flow --validate-parameters build/parameters/aortic.prm
```

`--print-parameters` prints the schema registered by the existing
`BloodFlowSystem` and its existing parameter defaults. `--validate-parameters`
parses the given file against that schema without running the simulation or
writing a `last_used_parameters.prm` file. Missing parameter files are rejected
without creating a fallback file. Runtime execution is environment- and
mesh-dependent and has not been asserted here as a completed benchmark run.

See [Configuration](doc/configuration.md) for the registered parameter groups and [Outputs](doc/outputs.md) for the files written by the current implementation.

## Tests

CTest discovers the test executables configured by deal.II's `DEAL_II_PICKUP_TESTS()` macro. Run the configured build-tree tests with:

```bash
ctest --test-dir build --output-on-failure
```

The current configured discovery includes `template`, `test_constant`, Jacobian boundary/cell/eta/interior/junction tests, mass integration tests, `test_mass_ones`, `test_trace_residual`, and `test_vtk` (the generated test names have a `.debug` suffix in this build). The audit run configured 12 tests: 11 passed and `tests/test_jacobian_cell.debug` failed its expected-output comparison. No source or expected output was changed to hide that result.

## Repository map

- `apps/`: application entry points; currently `blood_flow.cc`.
- `include/`, `source/`: the `BloodFlowSystem` implementation, parameter handling, assembly, solvers, and VTK utilities.
- `parameters/`: input meshes, parameter files/templates, and example/reference assets. The configured copies used by a build are under `build/parameters/`.
- `tests/`: deal.II test sources and expected output files.
- `doc/`: this documentation skeleton and the Doxygen/Sphinx configuration.
- `latex/`: the repository's mathematical manuscript source. It is not modified by this documentation work.
- `scripts/`: helper scripts, including documentation serving/building and test/formatting helpers.

## Scope and status

The repository contains example inputs and reference assets for several network sizes, including 37-artery material and the ADAN56 benchmark, which represents 56 anatomical arteries with 77 computational vessel segments. Mathematical choices, benchmark provenance, and canonical publication status remain separate review topics; see `doc/development/documentation-audit.md` and `doc/development/model-decisions.md`.

## Citing

If you use this software, please cite it as **Blood-flow solver**. The
canonical bibliography is `bibliography/references.bib`,
and its metadata policy and unresolved-key list are documented in
[`bibliography/README.md`](bibliography/README.md). The documentation reference
page is [References](doc/references.md). This repository does not claim that
all citations resolve when source keys are unavailable.

## License

This project is licensed under the MIT License (see `LICENSE.md`).
