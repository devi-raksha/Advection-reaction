# WP-07 manufactured-solution blocker record

## Evidence delivered

- `tools/generate_mms_expressions.py` derives exact mass and momentum source
  terms with SymPy from the volume equations assembled in
  `source/blood_flow_system.cc`.
- `tools/collect_convergence.py` parses the supplied report once and renders
  Markdown or TeX without retyping values.
- `tutorials/01_single_vessel_mms/reference/convergence.txt` is a byte-for-byte
  copy of the immutable report. Its SHA-256 is recorded in the adjacent JSON.

## Remaining blocker (source-backed)

The current `FunctionParser` instances are constructed with the constants
`rho`, `mu`, `xi`, `m`, and `Rt` (see `source/blood_flow_system.cc`). The
implemented pressure law additionally reads vessel-specific `E`, `h_wall`,
`a_d`, `p0`, and `p_d` from VTK data. A user `RHS expression` cannot currently
refer to those vessel values. Therefore the generated expressions preserve
those symbols and document FunctionParser syntax, but the p1/p2/p3 templates
are not claimed runnable MMS cases; their safe baseline RHS remains zero.

The executable already writes per-vessel CSV probe output and prints error
summaries, but no source-backed Verification mode or structured convergence
JSON output was added. A clean build-and-run of the templates is also not
available in this checkout. Exposing vessel constants (or adding a narrowly
specified manufactured-source interface) and then validating a clean run are
required before claiming reproducibility from the immutable reference numbers.

No local run is used to replace, adjust, or infer any Cycle, DoF, error, or rate
value in the reference report.
