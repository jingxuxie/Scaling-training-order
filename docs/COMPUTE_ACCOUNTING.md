# Compute and information accounting

- **Population flow:** B is integrated compute-normalized gradient-flow duration. Source costs can be absorbed into generators, but the exact family uses equal costs. The optimizer is held fixed. B is not an empirically calibrated wall-clock or FLOP count.
- **Capacity:** C = J B is a declared normalized model-work convention, with J blocks chosen before training. The entire unlearned tail is part of risk. No capacity growth is included.
- **Exact/perturbed/spectral searches:** Oracle geometry and numerical planning are available to the optimizer. These plots compare training loss at fixed B, not end-to-end model-development cost.
- **Paid pilot:** D n coordinate observations cost D n q. A declared planning price h is also charged. A candidate trains for B - D n q - h. Its certificate compares with the best static mixture using the entire original B. The two h values use matched observation draws. Prices q and h are declared parameters, not measured universal hardware prices. The experiment does not decide whether to buy the pilot.
- **SGD:** Each setting has equal steps, learning rate and batch size across methods. The static comparator is hindsight best on the evaluation seed over an 11-point grid, with tuning cost uncharged. It is intentionally stronger than a deployable tuned baseline but not a total-development-cost comparison.
- **Nonlinear:** Both layers of a width-16 tanh network train in NumPy. Each method receives equal steps and batch size. Five static weights provide a hindsight baseline. The transferred switch fraction uses proxy budget 4 for 200 steps and 8 for 800 steps; at learning rate 0.02 the latter is not the actual integrated step size (16). It is a heuristic stress test, not a direct flow discretization.
- **Repeated seeds:** Methods share initialization and data draws within a seed. Unadjusted bootstrap intervals summarize paired median ratios across independent seeds in each condition. They are not population guarantees over future tasks.

## Exact moment extension

The new Gaussian SGD sweep is separate from the retained finite-dataset sweep. Its inputs are exact population covariances and an initial second moment. Each comparison fixes step size eta, steps K, and minibatch size m: B = K eta and processed examples = K m. Oracle schedule search and static-envelope evaluation are uncharged. A gain therefore measures training-dynamics improvement, not end-to-end development savings. Per-example and per-batch source sampling are recorded as separate experimental conditions. The moment closure itself is an analytical evaluation tool, not a trained neural model.

The arbitrary-schedule theorem also has no switch cost. Its no-benefit result remains valid when additional nonnegative costs restrict the feasible class; its existence result below the boundary need not survive such costs. Near-boundary gain is deliberately quantified to make this distinction visible.
