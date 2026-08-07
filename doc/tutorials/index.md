# Tutorials

These tutorials describe repository-backed inputs and diagnostics. They are
reproducibility documentation, not runtime acceptance reports or claims of model
validation.

## Available tutorials

- [Single-vessel manufactured solution](single-vessel-mms.md) — source-backed MMS
  inputs and the immutable TXT-derived reference table.
- [Y-junction](../../tutorials/02_y_junction/README.md) — a validated three-vessel
  network input, topology record, and residual-diagnostic preparation.
- [37-segment arterial network](../../tutorials/03_37_arteries/README.md) — a
  validated network input and periodicity-diagnostic demonstration; raw comparison
  curves are not available.
- [ADAN56 benchmark](../../tutorials/04_adan56/README.md) — a source-backed
  56-anatomical-artery input represented by 77 computational vessel segments;
  runtime and quantitative validation remain pending.

## Fourth tutorial status

The maintainer clarified the ADAN56 identity as 56 anatomical arteries represented
computationally by 77 vessel segments. The tutorial records that identity and the
selected `56_adnr_new.vtk` input, while keeping missing source/license details,
raw reference data, and runtime provenance visibly pending.

See [result provenance](../development/result-provenance.md) for the evidence status
of these inputs and result artifacts.
