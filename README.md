# Differential Experience Architecture

**Differential Experience Architecture (DEA): deformation laws, susceptibility, and counterfactual answerability in controlled virtual worlds.**

This repository provides a reproducible computational testbed for Differential Experience Architecture (DEA). It implements graded perturbations of the six Experience Architecture conditions—Difference (D), Availability (A), Orientation (O), Integration (I), Temporality (T), and Answerability (R)—and evaluates how those perturbations deform a probabilistic reachable-response field.

The core local model is

$$
\Delta_A(\varepsilon)
=
D_{\mathrm{KL}}\!\left(P_\varepsilon \,\Vert\, P_0\right)
\approx
\frac{1}{2}\,
\varepsilon^{\top}
\boldsymbol{\chi}_A
\varepsilon.
$$

Here, $\boldsymbol{\chi}_A$ is the **Answerability Susceptibility Matrix**. Its diagonal entries represent condition-specific local susceptibility, while its off-diagonal entries quantify local second-order interaction or nonseparability between experiential conditions in the specified intervention coordinates.

## Computational Validation

The reproducible workflow evaluates:

1. local deformation-law verification;
2. single-condition susceptibility recovery;
3. paired-condition cross-susceptibility recovery;
4. separable/null architecture behavior;
5. coupled-architecture behavior;
6. diagonal-only versus full interaction models on held-out perturbations;
7. finite-difference step-size convergence;
8. Hessian symmetry;
9. positive-semidefinite and eigenvalue diagnostics;
10. coupling ablation;
11. finite-sample noise robustness; and
12. determinant, rank, and eigenvalue diagnostics.

The computational experiments use controlled synthetic response architectures. They are designed to test whether the numerical implementation recovers the mathematical behavior predicted by the DEA construction under known separable, coupled, and ablated conditions.

The repository does not claim human-subject validation. Human-participant virtual-reality data, gaze or controller logs, and physiological measurements are not included. Controlled virtual worlds with human participants constitute a future empirical validation stage.

## Reproduction

```bash
python -m pip install -r requirements.txt
python scripts/run_all.py
pytest -q
```

Generated computational outputs are written to `results/` and `figures/`.

## GitHub Actions

The GitHub Actions workflow executes the automated test suite and reproduction pipeline. Generated numerical results and figures are preserved as workflow artifacts, providing an independently executable record of the computational validation.

## Repository Structure

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

.github/
  workflows/
    reproduce.yml
```

## Reproducibility Scope

The repository provides the computational implementation supporting the synthetic validation of DEA, including susceptibility estimation, separability and coupling tests, interaction recovery, predictive comparison, numerical convergence checks, ablation, and finite-sample robustness analysis.

The computational results should be interpreted as consistency evidence for the proposed mathematical construction rather than evidence about human experiential organization.

## Citation

Akhtar, M. A. K., & Hans, A. (2026). *Differential Experience Architecture: Deformation Laws of Counterfactual Answerability in Controlled Virtual Worlds* (Version V1). Zenodo. https://doi.org/10.5281/zenodo.22726990

## License

Apache License 2.0.

Copyright (C) 2026 Mohammad Amir Khusru Akhtar
