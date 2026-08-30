# DA3408 — AI Operations — Module 1 Assignment

**Kirthan S (DA24B009)**

Submission repository for Module 1 (Experiment Management & Reproducibility). This README tells you
where to find each question's answer.

## Where to look

| Question | Location | Contents |
|----------|----------|----------|
| **Q1** — Technical Debt Diagnosis | [`Q1/`](Q1/) | `writeup_q1.tex` / `writeup_q1.pdf` — category identification (a/b/c) + mitigation |
| **Q2** — MLflow Experiment Comparison | [`Q2/`](Q2/) | `train.py` (MLP on MNIST), `comparison.png` (6-run table), `writeup_q2.tex`/`.pdf` (analysis + log code) |
| **Q4** — End-to-End Reproducibility Drill | separate repo (below) | partner reproducibility capstone |

## Q4 repository (capstone)

Q4 is completed in a separate private repository, shared with the assigned partner:

**https://github.com/kirthans/da3408-mnist-mlp-repro**

Partner A's contribution (DVC-versioned dataset, MLflow run with seed/`git_commit`/artifact, model
registered to Staging) is in that repo's commit history; Partner B's reproduction and note are
committed there as well.

## Reproducing Q2 locally

```bash
# from a Python 3.9 environment with mlflow, scikit-learn, pandas, numpy
python Q2/train.py
mlflow ui --backend-store-uri file:./mlruns   # view the mnist-mlp experiment
```
