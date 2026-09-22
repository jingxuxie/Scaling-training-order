# Author-facing submission checklist

This is a completed submission-format revision, not evidence of an actual submission or acceptance.

## Checked in this revision

The main text is exactly nine pages; the complete PDF is 32 pages. AI-use and reproducibility statements and references begin on page 10, and appendices begin on page 12. All 31 reference entries are cited. Compilation has no unresolved references/citations or overflowing boxes. The final pages were rendered and visually inspected. The paper has no author-identifying repository URL in its text or author metadata. A separate anonymous supplementary ZIP excludes Git history and publishing records.

Current ICLR 2027 guidance checked on September 21, 2026:
https://iclr.cc/Conferences/2027/AuthorGuidelines
https://iclr.cc/Conferences/2027/AIPolicyForAuthors

The initial main-text limit is nine pages. References, appendices after the bibliography, the required AI-use statement, and the optional reproducibility statement are excluded under these guidelines. This local audit is not a conference-issued compliance certification.

## Official style provenance

Source: https://github.com/ICLR/Master-Template/blob/master/iclr2027/iclr2027_conference.sty
Git blob SHA-1: f61ad7efce0855557694078c0945e6c33feb8236
Local file length: 9025 bytes.

The bundled style is now byte-identical to that fetched official source; it is not the earlier comment-stripped transcription. scripts/build_paper.py verifies its Git-blob hash. No margins, font sizes, or spacing parameters are overridden to reach nine pages. In main.tex only, two text patches plus a running-header assignment identify the artifact as a prepared draft rather than falsely asserting that it has been submitted. For an actual submission, authors can remove the etoolbox patch block and the explicit fancyhead assignment to restore the official review wording, then rerun the page-count check.

## Before actual submission

Human authors must verify scientific claims and citations, agree on authorship and consent, assess novelty and adequacy of evidence, and complete the relevant eligibility, conflict, and registration information. The AI-use statement must remain accurate. No OpenReview action has been taken. Use the anonymous supplementary archive for review; the full source archive and Git bundle deliberately retain author/repository provenance. High test coverage is not independent scientific review.
