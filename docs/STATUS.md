# Research status — nine-page polish, September 21, 2026

## Current revision

The paper has a nine-page main text and 32 pages in total, including 31 cited references and full proofs. The introduction, model setup (Section 3), and experiments (Section 6) have been substantially expanded. The exact arbitrary-schedule boundary, sharp near-boundary law, noise extension, and all reported numerical results are preserved. See REVISION_V3.md for the changes and precision corrections.

The 212 tests were rerun for this revision. All recorded result files and the scientific computation/test code match the parent revision byte-for-byte. Full experiment reproduction was performed for the preceding research revision, not repeated during this editorial pass; its audit remains in reproduction_audit.json. The new preservation audit states that distinction explicitly.

The local ICLR style now matches the official Git blob byte-for-byte. The build enforces a nine-page main text, the pinned style, and absence of unresolved citations/references or overflowing boxes. All final PDF pages were rendered and visually inspected.

## Scientific scope

The main theorem covers all measurable precommitted schedules in the symmetric shared-optimum quadratic family. It does not cover arbitrary feedback policies, arbitrary Hessians, or general neural networks. The exact-SGD library uses known population moments and uncharged search. Negative finite-data, nonlinear, and paid-pilot results remain part of the paper. No new numerical outcomes are claimed for this revision.

## Remaining author responsibilities

No independent human proof review, exhaustive novelty verification, formal proof-assistant verification, remote CI run, conference submission, or acceptance is claimed. Human authors must review and take responsibility for the paper, settle authorship and consent, and complete the venue submission requirements. Those responsibilities do not imply that a stated main proof is missing from the manuscript.

## Delivery

This revision is supplied as a PDF, complete source/results ZIP, anonymous supplement, and Git bundle on research/nine-page-polish. No GitHub update was attempted or claimed for this editorial pass. The bundle preserves the earlier repository history. Historical publication-attempt records refer only to the preceding revision, not to current connector capabilities.
