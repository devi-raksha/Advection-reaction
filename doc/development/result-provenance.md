# Result provenance

## How to read this page

- **Measured facts** are properties observed directly in the checked-out files or
  produced by a checked-in validator/collector.
- **Immutable inputs** are tracked source assets, byte-for-byte copies, or derived
  records whose content and checksum are fixed for review.

## Single-vessel MMS TXT

### Measured facts

- `tutorials/01_single_vessel_mms/reference/convergence.txt` is present and is
  parsed by `tools/collect_convergence.py` into the adjacent JSON record.
- The collector preserves the displayed error and rate strings. The JSON records
  three polynomial-degree tables and the TXT SHA-256
  `7789933cabd07a015fbbd50d8ff14d293f0831a6f436bfd64b308d76e7e1d3a5`.
- The numerical cells in the TXT are reference-report values.

### Immutable inputs

- The TXT report is the immutable reference input; its `Cycle`, `DoFs`, errors, and
  rates are not retyped or replaced with a different mesh-count convention.
- `reference/convergence.json` is the checked-in TXT-derived representation, and
  `reference/mms_expressions.json` records source-derived symbolic expressions.
- `p1.prm.in`, `p2.prm.in`, `p3.prm.in`, and
  `parameters/single_vessel.vtk` are preparation inputs.

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
  visualization assets.

## 37-segment VTK

### Measured facts

- `tools/validate_vtk_network.py` reports `valid: true` for
  `tutorials/03_37_arteries/network.vtk`: 38 points, 37 `VTK_LINE` cells, and 37
  vessel IDs.
- Terminal boundary IDs are 1 through 16; the active VTK has no `r_in`/`r_out`
  taper arrays. This page therefore uses **37-segment network**.
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

## Image-only 37/56 results

### Measured facts

- Image/PDF result artifacts are present under `NumData/37-arteries/` and
  `NumData/56-arteries/`; related image assets also exist under the benchmark
  parameter directories.
- These files are tracked result artifacts.

### Immutable inputs

- The tracked image/PDF files are immutable repository artifacts in their current
  form.
- The repository also tracks benchmark parameter files, including
  `parameters/benchmark-parameters/56_ADNR/56_adnr.prm`, but a nearby parameter
  mesh/version is associated with it.

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
  file referenced by the existing 56_ADNR parameter source.

### Immutable inputs

- `tutorials/04_adan56/network.vtk` is a byte-for-byte copy of
  `notebooks/56_adnr_new.vtk`.
- `tutorials/04_adan56/parameters.prm.in` is a source-backed adaptation of
  `parameters/benchmark-parameters/56_ADNR/56_adnr.prm` using a portable mesh path.
- `tutorials/04_adan56/topology.json` and `network.svg` are deterministic records
  generated from the validated VTK connectivity.

## Proposed maintainer message

**WP-11: add ADAN56 benchmark tutorial.** Add the source-backed ADAN56 input for 56
anatomical arteries represented by 77 computational vessel segments, with its exact
VTK arrays/topology and the existing 56_ADNR parameter source. Keep source/license
provenance, raw reference data, literature linkage, and runtime receipts.
