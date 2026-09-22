# Nine-page manuscript revision

Revision date: September 21, 2026. Parent research revision: `dc32104a8c79dbcf98684f6dd812c0e12852092a`.

## Deliverable

The main text now occupies **exactly nine pages**, ending with Section 7. The complete paper is **32 pages**: AI-use and reproducibility statements and 31 references begin on page 10; full appendices begin on page 12. The requested extra space was used for explanation, not larger type or altered margins.

## Introduction

The introduction now explains why beating uniform sampling, the reverse source order, or an exposure-matched mixture does not establish a benefit over the best static mixture. It motivates the common-target model, the role of orthogonal error, and the difference between existence of a better schedule and worthwhile compute savings. Contributions and the hierarchy of evidence are explicit, including the negative nonlinear result.

## Section 3: model and comparator

Three subsections explain what the scheduler controls, the geometry and initial-error ensemble, and the meaning of the benchmark. Added details include source-cost normalization, the fixed-optimizer convention, order of matrix products, population versus realized residual information, target-risk geometry, and a concrete ensemble whose mean, covariance, and second moment differ. The exact boundary requires symmetry after cost normalization; it is not an arbitrary unequal-source-cost theorem. The static optimality argument and distinction between candidate risk and optimal curriculum risk are retained.

## Section 6: experiments

Four subsections distinguish formula checks, exact expected SGD, stochastic comparisons, and paid-information/out-of-model tests. The full SGD condition grid, update and example counts, source sampling conventions, and static-risk bracketing are now in the main text. Candidate-library selection and the unchanged flow-derived rule answer different questions. Counts summarize a fixed configuration grid rather than a probability of practical success. Monte Carlo checks validate the implementation, not a simultaneous confidence claim. Paid-pilot and nonlinear failures remain visible.

## Precision corrections

The boundary figure now says that the **best possible gain** is one after the stopping budget; a bad schedule can still have gain below one. Exact-SGD descriptive medians use the **upper static bracket**, whereas the more-than-one-percent counts use the **lower bracket**. This distinction is now explicit in the main text, captions, appendix, and table-generation script. No numerical values were changed.

The bibliography is ordered by author key; duplicate arXiv identifiers in published venue fields were removed, while the source links were retained. The verified NeurIPS 2025 venue for Shukor et al. and ICML 2026 venue for Dai and Zheng were recorded. There are still 31 cited references, not an inflated reference count.

## Verification performed for this revision

The 212-test suite was rerun successfully. All 37 tracked files under results/ and 13 scientific-code/test files are byte-identical to the parent revision; the detailed check is in v3_preservation_audit.json. The only experiment-script edit changes an explanatory table caption. The full numerical experiment suites were not rerun for this editorial revision. Their prior same-environment reproduction audit is preserved and is not presented as new execution.

The official ICLR 2027 style is now byte-identical to Git blob f61ad7efce0855557694078c0945e6c33feb8236. The build checks this pin, the nine-page main-text marker, unresolved citations/references, and overflowing boxes. All 32 PDF pages were rendered with pdftoppm and reviewed in page/contact-sheet views. The main text has no font-size, margin, line-spacing, or negative-space overrides. The two header/author-line patches say that this is a draft rather than claiming submission.

## Status

This is a completed manuscript-polishing revision, not a new theorem or new experimental campaign. The written proofs and scientific scope are unchanged apart from explanatory clarification. Human author sign-off, independent scientific review, and actual submission are not claimed. Use the anonymous supplement rather than the identity-bearing full archive or Git bundle for double-blind review.
