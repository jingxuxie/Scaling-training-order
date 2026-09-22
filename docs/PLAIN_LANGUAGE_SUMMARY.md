# When does training order actually help?

Suppose two data sources teach the same task. Each source is useful for correcting different mistakes. A fixed mixture uses the same source proportions throughout training. A curriculum changes those proportions over time.

The fair question is not whether a particular ordering beats a 50–50 baseline. It is whether any ordering beats the **best fixed mixture**, retuned for that training budget.

## The central result

We solve this question exactly in a small quadratic learning model. The two sources are symmetric. The main initial error points between their preferred directions, but there may also be error in the perpendicular direction. We call the amount of that second error `epsilon`.

For a source angle `phi` and training work `B`, a curriculum can help exactly when

`epsilon < exp(-2 * B * cos(phi))`.

Otherwise balanced static mixing is optimal even among schedules allowed to change weights arbitrarily often. Below the boundary, just two mixed phases are enough for a strict improvement.

For example, with a 45-degree source angle and orthogonal error mass 0.01, static mixing is exactly optimal after training work about 3.256. This is a model-specific, dimensionless work value, not a universal number of SGD steps.

## Why the boundary exists

A first source can move the main residual error toward a direction that a second source learns quickly. This preparation is useful when the residual direction is sufficiently well known.

But the same transformation also acts on the error in the other direction. We prove that switching cannot reduce that slow error faster than balanced mixing does. The two error directions also cannot be compressed independently: the total change in area is fixed by the training budget. Combining these facts gives the exact boundary.

The schedule must be fixed before drawing the initial error. An adaptive controller that observes each realized residual and chooses a different path is outside the theorem. This matters: uncertainty about the residual is not the same as observing it exactly.

## Strictly better is not always meaningfully better

Just below the boundary, the best gain is very small. If the orthogonal error is a relative fraction `xi` below its critical value, the best relative gain is approximately `xi^2 / 8`. A 10% relative distance below the boundary has a leading gain of only 0.125%. Such a benefit may not cover the cost of learning the geometry or planning the schedule.

With a perfectly known direction, preparation can instead yield a rapidly growing ratio of losses. That still does not imply a similarly large saving in compute. At 45 degrees, the explicit construction has a 6.1-fold loss advantage at work 8, but its asymptotic compute-to-target advantage is about 1.17-fold.

## What happens with noisy training?

We analyzed a separate additive-noise model and exact expected SGD in Gaussian linear regression. Noise limits what ordering can achieve. It also matters whether a training batch chooses one domain for the entire batch or samples domains independently for each example. The latter is our primary comparison and generally gives more modest gains.

A small oracle schedule library improves by over 1% in 186 of 432 per-example-mixing settings, using a lower bound covering every static mixture weight. Its overall median gain is only 1.0033, using the best evaluated static risk (the upper bracket) as the descriptive numerator. Applying the original flow preparation rule unchanged is worse overall. Earlier finite-data and small nonlinear tests remain mixed or negative.

These experiments support the mathematical mechanism in some conditions, not a universal instruction to use curricula. They also do not establish that an oracle schedule search is worth paying for: the search has access to known population moments.

## What the paper contributes

The paper provides an exact answer to a carefully stated comparison problem, a proof that covers all switching schedules, a sharp measure of how useful the favorable regime can be, and tests that distinguish ideal learning dynamics from noisy training. It also shows that under common spectral-comparability assumptions, optimizing order cannot improve the power-law scaling exponent.

All stated main proofs are written and all reported experiments have been executed. The project is ready for human author review and submission preparation; it has not been independently reviewed or submitted.

## Nine-page revision

The latest manuscript expands the introduction and Sections 3 and 6 to give nine main-text pages. Scientific results are unchanged; the added material explains the assumptions, comparison metrics, and limits of the evidence.
