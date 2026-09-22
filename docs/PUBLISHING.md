# Importing the nine-page revision

The revision branch is `research/nine-page-polish`. It descends from the complete all-schedule revision and the original repository initialization. No history has been rewritten. This editorial pass is delivered by Git bundle; no current remote update is claimed.

To inspect and then publish the branch from a new folder:

```bash
git clone -b research/nine-page-polish Scaling-training-order-v3.bundle training-order-review
cd training-order-review
git remote set-url origin https://github.com/jingxuxie/Scaling-training-order.git
git push -u origin research/nine-page-polish
```

To import into an existing clone without overwriting main:

```bash
git fetch /path/to/Scaling-training-order-v3.bundle \
  research/nine-page-polish:research/nine-page-polish
git switch research/nine-page-polish
git push -u origin research/nine-page-polish
```

Review the branch before merging; neither sequence forces or overwrites main. The anonymous supplement excludes identifying Git history and these publishing instructions. Historical connection/push records in this repository concern the previous revision only.
