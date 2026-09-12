# Differential Experience Architecture

**Differential Experience Architecture (DEA): deformation laws, susceptibility, and counterfactual answerability in controlled virtual worlds.**

This repository provides a reproducible computational testbed for the theory of Differential Experience Architecture. It implements graded perturbations of the six Experience Architecture conditions — Difference (D), Availability (A), Orientation (O), Integration (I), Temporality (T), and Answerability (R) — and evaluates how those perturbations deform a probabilistic reachable-response field.

The core local model is

\[
\Delta_A(\varepsilon) = D_{KL}(P_\varepsilon\|P_0)
\approx \frac12\varepsilon^T \Chi_A \varepsilon,
\]

where \(\Chi_A\) is the Answerability Susceptibility Matrix. Diagonal entries represent condition-specific susceptibility and off-diagonal entries represent local interaction/nonseparability between experiential conditions.

## What is tested

The workflow executes the following computational checks:

1. local deformation-law verification;
2. single-condition susceptibility recovery;
3. paired-condition cross-susceptibility recovery;
4. separable/null architecture test;
5. coupled-architecture test;
6. diagonal baseline versus full interaction model on held-out perturbations;
7. finite-difference step-size convergence;
8. Hessian symmetry;
9. positive-semidefinite/eigenvalue checks;
10. mutual-constraint ablation;
11. sampling-noise robustness;
12. determinant/rank/eigenvalue diagnostics;
13. simulated controlled-virtual-world perturbations;
14. matched conventional-metric comparison.

The repository does **not** claim human-subject validation. Real participant VR data, gaze/controller logs, and physiological signals are not included.

## Reproduce

```bash
python -m pip install -r requirements.txt
python scripts/run_all.py
pytest -q
```

Generated files are written to `results/` and `figures/`.

## GitHub Actions

Every push and pull request runs the complete test suite and reproduction script. The workflow uploads the generated results and figures as a downloadable artifact.

## Repository structure

```text
dea/
  core.py
  experiments.py
scripts/
  run_all.py
tests/
  test_core.py
  test_experiments.py
results/
figures/
.github/workflows/
  reproduce.yml
```

## License

Apache License 2.0.

Copyright (C) 2026 Mohammad Amir Khusru Akhtar
