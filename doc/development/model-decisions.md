# WP-01: Blocking maintainer decisions

**Overall status:** B-01 identity is resolved, but its source/license/raw-reference provenance remains pending; B-02 through B-06 remain unresolved and require maintainer decisions before the affected benchmark, model, funding, license, or publication material is treated as canonical.


**Issue tracking:** GitHub Issues are disabled for this repository. No issue links were opened.

## B-01 — Dataset topology and benchmark identity

- **Severity:** High (benchmark identity and reproducibility)
- **Evidence:**
  - `parameters/benchmark-parameters/37arteries_network/37_vessel_network.prm` references `notebooks/37_vessel_network.vtk`.
  - `parameters/benchmark-parameters/56_ADNR/56_adnr.prm` references `notebooks/56_adnr_new.vtk`.
  - The corresponding mesh/data locations are `NumData/37-arteries/` and `NumData/56-arteries/`.
  - `notebooks/56_adnr_new.vtk` validates as 78 points and 77 `VTK_LINE` cells, with vessel IDs 0 through 76 and terminal boundary IDs 1 through 31.
  - The selected VTK contains cell arrays `vessel_id`, `a0`, `a_d`, `E`, `h_wall`, `p_d`, `p0`, `L`, and `r_d`, and point arrays `boundary_id`, `R1`, `R2`, `C`, and `P_out`.
  - `notebooks/56_adnr.vtk` has the same points/connectivity and common arrays but additionally contains `r_in` and `r_out`; the variants are not treated as equivalent.
  - **Maintainer clarification:** ADAN56 denotes **56 anatomical arteries represented by 77 computational vessel segments**.
- **Resolved decision (identity):** The ADAN56 identity is resolved for this work unit. `notebooks/56_adnr_new.vtk` is the selected tutorial input because it is the mesh referenced by the existing 56_ADNR parameter source; its exact arrays and connectivity were checked before copying. The tutorial must use the ADAN56 name and must not assert a different anatomical artery count.
- **Status:** **IDENTITY RESOLVED; SOURCE/LICENSE/RAW-REFERENCE PROVENANCE PENDING**.

## B-02 — Grant and funding attribution

- **Severity:** High (funding accuracy and compliance)
- **Evidence:**
  - The grant identifier under review is **101172493**.
  - The exact fact recorded for WP-01 is that 101172493 is **dealii-X, Horizon Europe/EuroHPC JU**, and is **not ERC/H2020**.
  - No exact grant-contract text or authoritative contract excerpt is included in the evidence available for this record; no repository file has been approved as the canonical contract source.
- **Decision needed:** Provide and approve the exact contract text (including the official project name, programme, funder, and required acknowledgement wording) and identify the canonical source to cite.
- **Status:** **UNRESOLVED / BLOCKING**.

## B-03 — Mixed license and copied-file notices

- **Severity:** High (legal provenance and distribution)
- **Evidence:**
  - Root project license is MIT in `LICENSE.md:1`.
  - `apps/metric_flow_x.cc:3` carries `SPDX-License-Identifier: LGPL-2.1-or-later`.
  - `tests/tests.h:3` carries `SPDX-License-Identifier: LGPL-2.1-or-later`; `tests/tests.h:8-11` also refers to deal.II dual licensing and its external `LICENSE.md`/`CONTRIBUTING.md`.
  - The copied ParsedTools/FSI files `include/constants.h`, `include/function.h`, `source/constants.cc`, and `source/function.cc` carry FSI-suite / GNU LGPL-3.0-or-later notices in their file headers.
- **Decision needed:** Establish provenance and license treatment for each copied or adapted file, confirm which notices must remain, determine the project-level license/notice presentation, and obtain any required upstream attribution or permission review.
- **Status:** **UNRESOLVED / BLOCKING**.

## B-04 — Mathematical and numerical model choices

- **Evidence:**
  - `latex/metric_flow.tex:70-73` defines density and the viscous-friction coefficient; `source/metric_flow_system.cc:70-74` exposes density, viscosity/profile friction, and tube-law parameters.
  - `latex/metric_flow.tex:112-121` defines pressure and wave-speed derivatives; `source/metric_flow_system.cc:1595-1601` and `source/metric_flow_system.cc:1696-1698` implement HLL Jacobian/wave-speed derivative paths.
  - `latex/metric_flow.tex:282-325` documents HLL wave speeds and flux branches; `source/metric_flow_system.cc:1552-1601` implements HLL residual/Jacobian paths.
  - `latex/metric_flow.tex:908-939` documents consistent initialization and differential rates; `source/metric_flow_system.cc:1166-1245` implements initial-solution/trace initialization.
  - `source/metric_flow_system.cc:1022-1023` records the global cell/trace layout decision; `latex/metric_flow.tex:768-835` describes the monolithic DAE block structure.
- **Status:** **UNRESOLVED / BLOCKING**.

## B-05 — Canonical TeX source and publication state

- **Severity:** High (publication integrity)
- **Evidence:**
  - The repository TeX source is `latex/metric_flow.tex`.
  - The uploaded TeX material is newer than the repository copy, but has a `\\who_i` compile blocker and unfinished analysis.
  - The repository copy contains the current manuscript material but is not thereby established as the canonical publication source.
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
- **Status:** **UNRESOLVED / BLOCKING**.

## Proposed maintainer message
