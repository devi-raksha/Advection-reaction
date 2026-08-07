# WP-14 licensing status

**Status:** **BLOCKED — B-03 unresolved**

**Scope:** licensing inventory and provenance audit only. This record does not
relicense files, rewrite notices, or make a distribution decision.

## Executive finding

The repository contains a root MIT notice, but it also contains explicit
LGPL/deal.II and FSI-suite notices. Those notices cannot be treated as
which files were copied, adapted, or newly authored. The project must not make

The proposed maintainer message is:

> **WP-14: record licensing conflict pending approval.**

## Evidence inventory

The following is a transcription of repository evidence. It records what the
files say; it is not a legal conclusion.

| File or notice | Observed declaration | Provenance evidence and current treatment |
|---|---|---|
| `apps/metric_flow_x.cc:3` | `SPDX-License-Identifier: LGPL-2.1-or-later`; deal.II authors, 2024–2025 | Application header says it is a blood-flow example built on deal.II. Whether this header describes the whole file, an adapted upstream file, or an approved project contribution is unresolved. |
| `tests/tests.h:3` | `SPDX-License-Identifier: LGPL-2.1-or-later` | Header identifies deal.II and says source is dual licensed under Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later, with external deal.II license/contribution references. The applicable provenance and notice bundle are unresolved. |
| `include/constants.h` | FSI-suite / ParsedTools; GNU LGPL version 3.0 or later; copyright Luca Heltai 2022 | Header says it is part of the FSI-suite platform and refers to the FSI-suite `LICENSE`. No copied upstream license file or source revision is present in this checkout. |
| `include/function.h` | FSI-suite / ParsedTools; GNU LGPL version 3.0 or later; copyright Luca Heltai 2022 | Same FSI-suite notice pattern as `include/constants.h`; source revision and adaptation history are unresolved. |
| `source/constants.cc` | FSI-suite / ParsedTools; GNU LGPL version 3.0 or later; copyright Luca Heltai 2022 | Implementation corresponding to `include/constants.h`; the header points to an external FSI-suite license, not this repository’s `LICENSE.md`. |
| `source/function.cc` | FSI-suite / ParsedTools; GNU LGPL version 3.0 or later; copyright Luca Heltai 2022 | Implementation corresponding to `include/function.h`; source revision, modifications, and required attribution are unresolved. |
| `tests/template.cc`, `tests/test_*.cc` with deal.II headers | Apache-2.0 WITH LLVM-exception wording | These test files contain deal.II-derived notice text. Each file’s exact origin, modification status, and required notice are not yet verified. |
| `scripts/indent` | deal.II LGPL version 2.1 or later wording | Script header identifies deal.II. Its provenance and redistribution obligations require the same upstream review. |

The textual phrase “license” in a file is not by itself a license grant. The
inventory distinguishes explicit SPDX identifiers and recognizable upstream
for files with no notice.

## Decisions required before unblocking B-03

The following decisions must be recorded by the maintainers and, where
appropriate, confirmed by legal counsel or the relevant upstream maintainers.

1. **File-by-file provenance.** For every row in the inventory, identify the
   upstream repository/project (deal.II, FSI-suite/ParsedTools, or this
   project), upstream path, source revision or release, date obtained, and
   whether the file is copied verbatim, adapted, or newly authored. Preserve
   copyright and attribution evidence rather than inferring authorship from a
   current directory.
2. **Applicable license and notice set.** Confirm the exact license expression
   for each copied or adapted file, including the deal.II dual-license wording
   and the FSI-suite LGPL-3.0-or-later wording. Confirm which full license
   texts, copyright notices, attribution notices, modification notices, and
   source-offer/relinkability obligations must accompany a source or binary
   distribution.
3. **Compatibility and distribution model.** Obtain an explicit legal review
   of the coexistence of the root MIT notice, LGPL-2.1-or-later files,
   LGPL-3.0-or-later files, and Apache-2.0 WITH LLVM-exception files. Decide
   whether the root MIT notice is limited to original project material and
   what a combined distribution must say. Do not assume that “MIT” in
   `README.md` relicenses third-party files.
4. **Upstream permission and attribution.** Confirm whether the current
   copies/adaptations satisfy deal.II and FSI-suite attribution conditions and
   whether any permission, contributor approval, or upstream notice must be
   obtained. Record the authoritative upstream license locations and copies
   (or URLs/revisions) used for the decision.
5. **Header-change authority.** Decide whether any file header, SPDX identifier,
   Name the maintainer/legal approver. Until that decision is recorded, all
   existing notices must remain byte-for-byte untouched.
6. **Release gate.** Define the evidence that closes B-03: an approved
   file-by-file inventory, provenance links/checksums or upstream revisions,
   approved notice/license bundle, reviewed project-level wording, and a
   maintainer/legal sign-off. A clean output from the report tool is not a
   substitute for that sign-off.


- The MIT text in `LICENSE.md` does **not** automatically replace an SPDX or
  upstream notice in a source file.
- A file’s presence in this repository does **not** prove that it was
  relicensed, authored by the current maintainers, or modified with upstream
  permission.
- `LGPL-2.1-or-later` and `LGPL-3.0-or-later` are not interchangeable labels.
- The deal.II dual-license text and the FSI-suite LGPL notice must not be
  shortened, normalized, or removed while provenance is unresolved.
- No release, package, binary, or README/Citation license assertion should be
  approved on the basis of this blocked audit alone.

## File inventory and provenance checklist

Use this checklist to complete the audit. “Observed” means that the current
checkout contains the corresponding notice; “verified” requires an upstream
source, revision, and maintainer/legal review.

| Path/group | Observed notice | Upstream source/revision | Copied or adapted? | Copyright/attribution verified | Full license/notice retained | Maintainer/legal decision |
|---|---|---|---|---|---|---|
| `LICENSE.md` | MIT | `[ ]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` |
| `apps/metric_flow_x.cc` | LGPL-2.1-or-later; deal.II | `[ ]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` |
| `tests/tests.h` | LGPL-2.1-or-later; deal.II dual-license text | `[ ]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` |
| `include/constants.h` | FSI-suite/ParsedTools LGPL-3.0-or-later | `[ ]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` |
| `include/function.h` | FSI-suite/ParsedTools LGPL-3.0-or-later | `[ ]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` |
| `source/constants.cc` | FSI-suite/ParsedTools LGPL-3.0-or-later | `[ ]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` |
| `source/function.cc` | FSI-suite/ParsedTools LGPL-3.0-or-later | `[ ]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` |
| `include/vtk_utils.h` | deal.II Apache-2.0 WITH LLVM-exception | `[ ]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` |
| `tests/template.cc`, `tests/test_*.cc` | deal.II Apache-2.0 WITH LLVM-exception text | `[ ]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` |
| `scripts/indent` | deal.II LGPL-2.1-or-later text | `[ ]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` |
| All remaining tracked source, scripts, docs, data, and generated artifacts | No license conclusion without review | `[ ]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` |

For each unchecked row, attach (1) an upstream URL and immutable revision or a
written new-authorship statement, (2) a checksum or equivalent identity for
the copied source, (3) a list of local modifications, and (4) the approved
license/notice treatment. Keep this record separate from any later change to

## Read-only report tool

`tools/check_licenses.py` scans the repository’s application, include, source,
test, script, and top-level metadata files. It reports explicit SPDX/textual
notices and exits nonzero when it finds a file-level notice that conflicts with
the root MIT declaration or when one file contains multiple recognized license
families. It is intentionally read-only: it never changes a header, writes a
license file, normalizes SPDX text, or assigns a license to a file with no
notice.

Example:

```console
$ python tools/check_licenses.py
# expected while B-03 is unresolved: report includes conflicts and exits 1
```

The report is an audit aid only. A conflict is expected at this blocked stage
and is not permission to “fix” a notice mechanically.

## Resolution status

- **B-03:** **UNRESOLVED / BLOCKING**.
- **WP-14:** audit package complete only when this document and the read-only
  report tool are present; legal/provenance approval is intentionally not
  `CITATION.cff` license fields were not changed by WP-14.

**Proposed maintainer message:** `WP-14: record licensing conflict pending approval.`
