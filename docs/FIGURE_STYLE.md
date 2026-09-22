# ICLR 2027 template and figure update

The project was extracted from `Scaling-training-order-v3.zip`. The style,
bibliography style, `natbib.sty`, and `fancyhdr.sty` were copied from the supplied
local `iclr-2027-style-files` directory without modification. The conference
style was already identical to that template. Custom draft header and author
line patches were removed to use the template's standard anonymous review
layout. The header is template text; no submission was performed.

All 11 figures are generated from the existing saved result files. The eight
figures included in the manuscript use vector PDF assets. The other three
ancillary plots receive the same style.

- Only the left and bottom axes remain, with outward ticks and light horizontal
  guides.
- Serif labels and STIX mathematical notation match the paper typography.
- Distinct colors, dashes, and markers distinguish curves; legends are unboxed
  and placed above the plotting area so they do not hide observations.
- Each exact stopping-budget cross matches its curve's color. SGD plots use
  color for batch size and line/marker style for source sampling convention.
- Reference lines are neutral gray, uncertainty intervals are retained, and
  all plotted numerical values are unchanged. Gain-one points have enough
  space above the bottom axis; the ancillary uncertainty-window plot uses
  data-based vertical limits that include every point.
- PDF fonts are embedded TrueType; PNG companions are exported at 300 dpi.

The reusable style is `experiments/figure_style.py`. To regenerate presentation
assets from saved results without rerunning experiments or rewriting summaries:

```bash
python experiments/report.py
python experiments/report_v2.py --figures-only
python scripts/build_paper.py
```

Use the plotting dependencies in `requirements-lock.txt`. This formatting pass
used Python 3.12.14, NumPy 2.3.5, pandas 2.2.3, Matplotlib 3.10.8, and the installed
TeX Live 2025 distribution. Dependencies for this local run are isolated in
the workspace's `.figure-deps` directory.

The build checks the pinned conference style, the nine-page main-text marker,
unresolved references/citations, and overflowing boxes. All 32 compiled pages
were rendered for visual review, along with all 11 standalone figures. The
current audit records preservation checks against the original ZIP. Scientific
experiments and proof tests were not rerun for this presentation-only update.
