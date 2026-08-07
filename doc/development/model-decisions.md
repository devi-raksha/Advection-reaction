# WP-01: Blocking maintainer decisions

**Status:** Unresolved; all blockers below require a maintainer decision before the affected benchmark, model, or publication material is treated as canonical.

**Scope of this record:** WP-01 records evidence and decisions needed only. It does not change source code, legal or funding text, model claims, benchmark data, or tutorials.

**Issue tracking:** GitHub Issues are disabled for this repository. No issue links were opened.

## B-01 — Dataset topology and benchmark identity

- **Severity:** High (benchmark identity and reproducibility)
- **Evidence:**
  - `parameters/benchmark-parameters/37arteries_network/37_vessel_network.prm` references `notebooks/37_vessel_network.vtk`.
  - `parameters/benchmark-parameters/56_ADNR/56_adnr.prm` references `notebooks/56_adnr_new.vtk`.
  - The corresponding mesh/data locations are `NumData/37-arteries/` and `NumData/56-arteries/`.
  - The recorded topology tuples are: bifurcation **4/3/1/3**; 37 network **38/37/15/17**; `56_adnr` **78/77/30/32 with taper**; `56_adnr_new` **78/77/30/32 without taper**.
  - No proven 57-artery dataset has been identified.
- **Decision needed:** Confirm the meaning and ordering of each topology tuple, select the canonical 37 and 56 input files, decide whether taper is part of the `56_adnr` benchmark definition, and decide whether a 57-artery result may be named or shown at all.
- **Prohibited assumptions:** Do not infer tuple component meanings; do not treat similarly named VTK files as equivalent; do not treat a 37- or 56-artery dataset as a 57-artery dataset; do not claim taper/no-taper equivalence.
- **Status:** **UNRESOLVED / BLOCKING**.

## B-02 — Grant and funding attribution

- **Severity:** High (funding accuracy and compliance)
- **Evidence:**
  - The grant identifier under review is **101172493**.
  - The exact fact recorded for WP-01 is that 101172493 is **dealii-X, Horizon Europe/EuroHPC JU**, and is **not ERC/H2020**.
  - No exact grant-contract text or authoritative contract excerpt is included in the evidence available for this record; no repository file has been approved as the canonical contract source.
- **Decision needed:** Provide and approve the exact contract text (including the official project name, programme, funder, and required acknowledgement wording) and identify the canonical source to cite.
- **Prohibited assumptions:** Do not relabel the grant as ERC or H2020; do not reconstruct contractual wording from a project name or grant number; do not publish an acknowledgement until the exact contract text is verified.
- **Status:** **UNRESOLVED / BLOCKING**.

## B-03 — Mixed license and copied-file notices

- **Severity:** High (legal provenance and distribution)
- **Evidence:**
  - Root project license is MIT in `LICENSE.md:1`.
  - `apps/blood_flow.cc:3` carries `SPDX-License-Identifier: LGPL-2.1-or-later`.
  - `tests/tests.h:3` carries `SPDX-License-Identifier: LGPL-2.1-or-later`; `tests/tests.h:8-11` also refers to deal.II dual licensing and its external `LICENSE.md`/`CONTRIBUTING.md`.
  - The copied ParsedTools/FSI files `include/constants.h`, `include/function.h`, `source/constants.cc`, and `source/function.cc` carry FSI-suite / GNU LGPL-3.0-or-later notices in their file headers.
- **Decision needed:** Establish provenance and license treatment for each copied or adapted file, confirm which notices must remain, determine the project-level license/notice presentation, and obtain any required upstream attribution or permission review.
- **Prohibited assumptions:** Do not infer that the MIT root license supersedes an SPDX/header notice; do not remove, normalize, or rewrite LGPL/deal.II/FSI-suite notices; do not assert that copied files are relicensed merely because they are in this repository.
- **Status:** **UNRESOLVED / BLOCKING**.

## B-04 — Mathematical and numerical model choices

- **Severity:** Critical (scientific correctness and claims)
- **Evidence:**
  - `latex/blood_flow.tex:70-73` defines density and the viscous-friction coefficient; `source/blood_flow_system.cc:70-74` exposes density, viscosity/profile friction, and tube-law parameters.
  - `latex/blood_flow.tex:112-121` defines pressure and wave-speed derivatives; `source/blood_flow_system.cc:1595-1601` and `source/blood_flow_system.cc:1696-1698` implement HLL Jacobian/wave-speed derivative paths.
  - `latex/blood_flow.tex:282-325` documents HLL wave speeds and flux branches; `source/blood_flow_system.cc:1552-1601` implements HLL residual/Jacobian paths.
  - `latex/blood_flow.tex:908-939` documents consistent initialization and differential rates; `source/blood_flow_system.cc:1166-1245` implements initial-solution/trace initialization.
  - `source/blood_flow_system.cc:1022-1023` records the global cell/trace layout decision; `latex/blood_flow.tex:768-835` describes the monolithic DAE block structure.
- **Decision needed:** Resolve and record the canonical choices for friction density, the `m`/square-root law, `gamma`, HLL speeds and derivatives, monolithic versus condensation formulation, initial rates, and the proof/verification status of each choice.
- **Prohibited assumptions:** Do not select a parameterization from a single `.prm` file as the scientific model; do not treat an implementation or manuscript equation as proof of correctness; do not infer HLL derivative validity from compilation; do not call the system monolithic or condensed without maintainer agreement; do not claim validated initial rates.
- **Status:** **UNRESOLVED / BLOCKING**.

## B-05 — Canonical TeX source and publication state

- **Severity:** High (publication integrity)
- **Evidence:**
  - The repository TeX source is `latex/blood_flow.tex`.
  - The uploaded TeX material is newer than the repository copy, but has a `\\who_i` compile blocker and unfinished analysis.
  - The repository copy contains the current manuscript material but is not thereby established as the canonical publication source.
- **Decision needed:** Identify the canonical TeX source, decide whether/how the newer upload is imported, resolve the `\\who_i` compile blocker, and define the acceptance gate for the unfinished analysis before publication claims are made.
- **Prohibited assumptions:** Do not silently overwrite `latex/blood_flow.tex`; do not treat a non-compiling upload as publication-ready; do not treat the repository copy as canonical solely because it compiles or is tracked; do not report unfinished analysis as completed.
- **Status:** **UNRESOLVED / BLOCKING**.

## B-06 — Provenance for image-only 37/56 results

- **Severity:** High (reproducibility and evidence quality)
- **Evidence:**
  - 37-network result images are present under `parameters/benchmark-parameters/37arteries_network/` (for example `Aortic_arcII_P.png` and `filnal_time_pressure_image.png`).
  - 37-artery result PDFs are present under `NumData/37-arteries/`.
  - 56-artery result PDFs are present under `NumData/56-arteries/`.
  - The relevant run configuration is `parameters/benchmark-parameters/56_ADNR/56_adnr.prm`; its VTK input path points to `notebooks/56_adnr_new.vtk`.
  - These image/PDF artifacts do not, by themselves, establish raw-data provenance, exact run parameters, mesh identity, or post-processing steps.
- **Decision needed:** Require raw output and provenance (input mesh, parameter file/version, solver run, post-processing, and checksums) for every reported 37/56 result, or explicitly label each result as a demonstration image rather than validated data.
- **Prohibited assumptions:** Do not infer numerical provenance from a filename or image; do not present image-only results as reproducible benchmark evidence; do not attribute a result to `56_adnr` versus `56_adnr_new` without confirming the input mesh and taper choice.
- **Status:** **UNRESOLVED / BLOCKING**.

## Proposed maintainer message

> **WP-01: record blocking maintainer decisions.** Please resolve B-01 through B-06 before benchmark, model, funding, license, or publication claims proceed. Confirm dataset topology and canonical inputs (including taper and the absence of a proven 57-artery dataset); provide the exact 101172493 dealii-X Horizon Europe/EuroHPC JU contract wording; approve provenance and notice treatment for MIT/LGPL/deal.II/FSI-suite files; select and verify the model choices and proof status; establish the canonical TeX source and publication gate; and require raw provenance or demonstration labels for image-only 37/56 results. GitHub Issues are disabled, so no issue links were opened.
