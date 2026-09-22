# Literature distinction and open review questions

Primary-source checks include optimal data-mixture scaling (Shukor et al.; Ye et al.; Dai and Zheng), dynamic mixtures (Aioli, RegMix-D, Olmix), local Lie-bracket order prediction (Sweeney), optimal-control curricula (Mori et al.; Gu et al.), and switched-system background (Liberzon). Full bibliographic entries are in `references.json` and cited in the paper. This is a targeted audit, not proof that no overlapping theorem exists.

The candidate new contribution is the exact finite-horizon necessary-and-sufficient threshold over all measurable controls in the specified common-diagonal mirror family, with the matching near-threshold 1/8 gain law and the suffix-noise consequence. The proof uses an indefinite quadratic-form differential identity plus a determinant constraint and an explicitly orthogonalized second-order perturbation.

Classical components are credited: scalar spectral Jensen, Golden–Thompson, determinant evolution, Gaussian fourth-moment closure, matrix exponential perturbation bounds, and scalar interpolation. The exact SGD closure is an evaluation tool, not claimed as a new learning theory. The CPU experiments test the model's consequences and failure modes.

An independent review should challenge whether an equivalent finite-horizon weighted-Gram result already exists in switched-system control, whether the model's symmetry and precommitted-schedule assumption leave a sufficiently meaningful learning question, and whether the empirical scope supports a theory-first submission. These questions are not resolved by counting citations or tests.
