# Differential Experience Architecture computational validation

## Separable architecture

- Hessian/Fisher Frobenius error: `4.980e-08`
- Symmetry error: `0.000e+00`
- Minimum eigenvalue: `3.657e-01`
- Rank: `6`
- Interaction fraction: `0.000000`
- Determinant ratio: `1.000000`
- Full-model RMSE: `1.021e-04`
- Diagonal-model RMSE: `1.021e-04`
- Relative full-vs-diagonal improvement: `0.00%`
- Sampling-noise KL MAE (n=20,000): `6.943e-04`

## Coupled architecture

- Hessian/Fisher Frobenius error: `4.650e-08`
- Symmetry error: `0.000e+00`
- Minimum eigenvalue: `1.536e-01`
- Rank: `6`
- Interaction fraction: `0.476172`
- Determinant ratio: `0.370840`
- Full-model RMSE: `1.891e-04`
- Diagonal-model RMSE: `1.108e-03`
- Relative full-vs-diagonal improvement: `82.94%`
- Sampling-noise KL MAE (n=20,000): `5.694e-04`

## Ablated architecture

- Hessian/Fisher Frobenius error: `6.076e-08`
- Symmetry error: `0.000e+00`
- Minimum eigenvalue: `3.657e-01`
- Rank: `6`
- Interaction fraction: `0.000000`
- Determinant ratio: `1.000000`
- Full-model RMSE: `1.438e-04`
- Diagonal-model RMSE: `1.438e-04`
- Relative full-vs-diagonal improvement: `-0.00%`
- Sampling-noise KL MAE (n=20,000): `7.253e-04`

## Scope

These results validate the mathematical/synthetic test harness only. They are not human-subject VR validation.
