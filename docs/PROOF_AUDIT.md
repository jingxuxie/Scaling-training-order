# Internal proof audit — September 21, 2026

This audit is AI-assisted and records checks performed during development. It is not independent peer review, a proof-assistant verification, or an exhaustive novelty search.

## New arbitrary-schedule proof

The evolving vector from the slow basis direction satisfies W = y^2 - x^2 and W' = -2 d W + 2(a-d)x^2 in the general mirror family. Off-diagonal signs cancel exactly. The integrated lower bound holds for absolutely continuous solutions under bounded measurable controls, so it does not assume finitely many phases. Adding a common ridge merely shifts both diagonals.

The transition is invertible and has fixed determinant from Liouville's formula. The Gram identity is exact in two dimensions. Dropping the squared inner product and minimizing over the enlarged feasible interval t >= exp(-2dB) gives a valid lower bound. It is attained by balanced mixing above the threshold. At epsilon=0 the unconstrained determinant bound can be zero; the separate energy bound is retained.

The necessity proof explicitly computes the first derivative of the inner product of the two columns. Its weight is proportional to cosh(c(B-t)); the chosen half-budget amplitude ratio cancels the weighted integral. Sign-conjugation yields the required odd/even parity. The slow-column increase has a strictly positive second-order coefficient K. The risk expansion is uniform locally for fixed geometry/B, not uniformly over degenerating source parameters. The bounded feasible amplitude is respected.

For the 1/8 law, choosing eta^2 = b xi/(2K) changes the slow-column norm to the unconstrained optimum to second-order error. The column inner product contributes only cubic error. The exact Gram lower envelope and feasible witness sandwich both optimal two-phase and arbitrary-schedule risks. No existence of an optimal arbitrary control is assumed; infima suffice.

## Additive noise

Each Brownian increment has isotropic covariance. With a deterministic schedule, its suffix transition is an admissible path with epsilon=1. The no-benefit theorem therefore bounds every suffix contribution. The conclusion does not extend automatically to path-dependent feedback. Positive limiting balanced noise risk yields a gain ceiling converging to one.

## Exact Gaussian SGD and static envelopes

Both same-example fourth moments and distinct-example cross terms are included. Per-example domain mixing is quadratic in the mixture weight; per-batch domain mixing is affine. The label-noise injection is affine and independent of current error. Moment maps are positive on symmetric PSD matrices.

The nuclear-norm map bound follows from positive/negative decomposition and rank-one equality. The trace dual matrix is affine or matrix convex, hence bounded by endpoint maximum eigenvalues on each cell. The curvature formula retains the extra second derivative of the per-example map and the derivative of the noise injection. Ordered factors are not assumed to commute. Scalar interpolation supplies h^2/8. Values are computed in float64; no directed-rounding certification is claimed. Every final relative bracket width is below 0.001 on the fixed grid.

## Retained results checked for compatibility

The common-diagonal static comparison uses scalar spectral Jensen, not operator convexity of the matrix exponential. Golden–Thompson applies generically to two phases only. The earlier two-phase cone bound remains true but is weaker than the new all-schedule boundary in the mirror family. The explicit preparation threshold is renamed epsilon_prep to avoid confusion with the optimal-schedule boundary.

The capacity result includes the entire unlearned tail and one shared schedule across all blocks. The pilot comparison uses the no-pilot full-budget competitor and does not refund rejected pilots. The small nonlinear test uses a heuristic input-geometry switch fraction, not a computed neural-Hessian schedule; negative outcomes are retained.

## Evidence

212 tests pass, including general mirror geometries outside the rank-one normalization, arbitrary multi-phase inequalities, exact moment Monte Carlo comparisons, per-example moment validation, and static bracket tests. Independent high-precision calculations check 150 boundary signs and 45 near-boundary coefficients. All 21 raw and four derived CSVs and both numerical summary JSONs reproduce byte-for-byte in the same environment. None of these checks proves novelty or correctness for all inputs by computation alone.
