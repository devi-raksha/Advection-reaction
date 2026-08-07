# Result provenance

> **Status — evidence inventory, not runtime acceptance.** This page separates
> repository-measured facts, immutable inputs, and missing maintainer evidence. It
> does not turn reference tables, images, or input validation into a claim that the
> solver completed or that a benchmark was accepted.

## How to read this page

- **Measured facts** are properties observed directly in the checked-out files or
  produced by a checked-in validator/collector.
- **Immutable inputs** are tracked source assets, byte-for-byte copies, or derived
  records whose content and checksum are fixed for review. An immutable input is
  not automatically a runtime result.
- **Missing maintainer evidence** is the provenance, decision, or runtime evidence
  still required before a result can be called reproducible or accepted.

## Single-vessel MMS TXT

### Measured facts

- `tutorials/01_single_vessel_mms/reference/convergence.txt` is present and is
  parsed by `tools/collect_convergence.py` into the adjacent JSON record.
- The collector preserves the displayed error and rate strings. The JSON records
  three polynomial-degree tables and the TXT SHA-256
  `7789933cabd07a015fbbd50d8ff14d293f0831a6f436bfd64b308d76e7e1d3a5`.
- The repository does not contain a clean end-to-end run of the MMS templates.
  The numerical cells in the TXT are therefore reference-report values, not values
  measured by the current checkout.

### Immutable inputs

- The TXT report is the immutable reference input; its `Cycle`, `DoFs`, errors, and
  rates are not retyped or replaced with a different mesh-count convention.
- `reference/convergence.json` is the checked-in TXT-derived representation, and
  `reference/mms_expressions.json` records source-derived symbolic expressions.
- `p1.prm.in`, `p2.prm.in`, `p3.prm.in`, and
  `parameters/single_vessel.vtk` are preparation inputs. The templates retain a
  parser-backed zero RHS and are not claimed runnable MMS cases.

### Missing maintainer evidence

The source report's acquisition/run provenance, a clean build and run for all
profiles and degrees, raw runtime errors, and an approved numerical acceptance
comparison are missing. No MMS runtime acceptance is claimed.

## Y-junction VTK

### Measured facts

- `tools/validate_vtk_network.py` reports `valid: true` for
  `tutorials/02_y_junction/network.vtk`: 4 points, 3 `VTK_LINE` cells, and 3
  vessels.
- The measured boundary records contain inlet ID 0, junction ID 255, and terminal
  IDs 1 and 2. The copied asset's SHA-256 is
  `4536f94764b94aa4f34fdaac83cd01e4a8e1211f727e7bf72a2f978ba4ca3b58`.

### Immutable inputs

- `tutorials/02_y_junction/network.vtk` is documented as a byte-for-byte copy of
  `notebooks/bifurcation_network.vtk`.
- `tutorials/02_y_junction/provenance.json` and `topology.json` preserve the source
  path, digest, validator result, connectivity, vessel IDs, and boundary IDs.
- `parameters.prm.in` and the generated `network.svg` are tutorial preparation and
  visualization assets, not solver-result evidence.

### Missing maintainer evidence

A successful clean runtime, raw residual columns, quantitative residual results,
and an approved physical or benchmark comparison are not present. Input validation
alone does not establish runtime acceptance.

## 37-segment VTK

### Measured facts

- `tools/validate_vtk_network.py` reports `valid: true` for
  `tutorials/03_37_arteries/network.vtk`: 38 points, 37 `VTK_LINE` cells, and 37
  vessel IDs.
- Terminal boundary IDs are 1 through 16; the active VTK has no `r_in`/`r_out`
  taper arrays. This page therefore uses **37-segment network**, not a tapering or
  anatomical benchmark claim.
- `tutorials/03_37_arteries/reference/provenance.yml` records the copied source
  digest `28c6d138903f4e5cca31d78f57ed6adfd31448fe81216babee9650f30b82bcc1`.

### Immutable inputs

- `tutorials/03_37_arteries/network.vtk` is a byte-for-byte copy of
  `notebooks/37_vessel_network.vtk`.
- `topology.json`, `reference/provenance.yml`, `parameters.prm.in`, and the
  deterministic `network.svg` are tracked input/provenance or visualization
  artifacts.
- `tools/analyze_periodicity.py` is the checked-in diagnostic. It compares two
  complete final periods only when the caller supplies a period and explicitly
  named trace columns.

### Missing maintainer evidence

No raw reference curves or machine-readable measurement tables are present. The
periodicity acceptance threshold remains unset, and no clean solver run, run
configuration record, sampling metadata, units, or external benchmark provenance
has been supplied. This tutorial is a reproducible demonstration input, not a
validated benchmark.

## Image-only 37/56 results

### Measured facts

- Image/PDF result artifacts are present under `NumData/37-arteries/` and
  `NumData/56-arteries/`; related image assets also exist under the benchmark
  parameter directories.
- These files can establish that artifacts are tracked. Their pixels do not expose
  the raw numerical samples, sampling grids, uncertainty, or complete execution
  history needed to reproduce a result.

### Immutable inputs

- The tracked image/PDF files are immutable repository artifacts in their current
  form.
- The repository also tracks benchmark parameter files, including
  `parameters/benchmark-parameters/56_ADNR/56_adnr.prm`, but a nearby parameter
  file does not prove that it produced a particular image or identify the exact
  mesh/version used for it.

### Missing maintainer evidence

Raw outputs, exact input mesh and parameter/version mapping, solver/build details,
sampling locations and cycle selection, units, post-processing scripts, checksums,
and literature or external reference sources are missing. The image-only 37/56
artifacts must not be used as quantitative validation evidence until B-06 is
resolved; they remain provenance pointers or demonstration images.

## ADAN56 benchmark input (B-01 identity resolved)

### Measured facts

- The maintainer clarification identifies ADAN56 as **56 anatomical arteries
  represented by 77 computational vessel segments**.
- `notebooks/56_adnr_new.vtk` validates as 78 points, 77 `VTK_LINE` cells, and
  vessel IDs 0 through 76, with terminal boundary IDs 1 through 31.
- Its exact cell arrays are `vessel_id`, `a0`, `a_d`, `E`, `h_wall`, `p_d`, `p0`,
  `L`, and `r_d`; its exact point arrays are `boundary_id`, `R1`, `R2`, `C`, and
  `P_out`.
- `notebooks/56_adnr.vtk` has matching points/connectivity and common arrays but
  also contains `r_in` and `r_out`. The selected tutorial variant is the `new`
  file referenced by the existing 56_ADNR parameter source; variant equivalence
  is not inferred.

### Immutable inputs

- `tutorials/04_adan56/network.vtk` is a byte-for-byte copy of
  `notebooks/56_adnr_new.vtk`, with its SHA-256 and validation facts recorded in
  `tutorials/04_adan56/reference/provenance.yml`.
- `tutorials/04_adan56/parameters.prm.in` is a source-backed adaptation of
  `parameters/benchmark-parameters/56_ADNR/56_adnr.prm` using a portable mesh path.
- `tutorials/04_adan56/topology.json` and `network.svg` are deterministic records
  generated from the validated VTK connectivity.

### Missing maintainer evidence

B-01 identity is resolved for the ADAN56 tutorial, but source attribution and
license/raw-data provenance for the underlying network remain pending. Raw solver
outputs, a run receipt, literature linkage, raw reference curves, and an approved
quantitative threshold are also missing. Existing image-only 56 results remain
demonstration/reference material, not validation data. No runtime success or
benchmark validation claim is made.

## Proposed maintainer message

**WP-11: add ADAN56 benchmark tutorial.** Add the source-backed ADAN56 input for 56
anatomical arteries represented by 77 computational vessel segments, with its exact
VTK arrays/topology and the existing 56_ADNR parameter source. Keep source/license
provenance, raw reference data, literature linkage, runtime receipts, and any
quantitative acceptance threshold visibly pending. This work does not alter code,
legal/funding/license text, or model decisions.
