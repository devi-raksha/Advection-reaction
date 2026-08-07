# WP-07 manufactured-solution blocker record

## Evidence delivered

- `tools/generate_mms_expressions.py` derives exact mass and momentum source
  terms with SymPy from the volume equations assembled in
  `source/metric_flow_system.cc`.
- `tools/collect_convergence.py` parses the supplied report once and renders
  Markdown or TeX without retyping values.
- `tutorials/01_single_vessel_mms/reference/convergence.txt` is a byte-for-byte
  copy of the immutable report. Its SHA-256 is recorded in the adjacent JSON.
