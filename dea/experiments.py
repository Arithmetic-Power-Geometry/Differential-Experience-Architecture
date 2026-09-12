from dataclasses import dataclass
import itertools
import numpy as np

from .core import CONDITIONS, answerability_deformation, diagonal_prediction, fisher_at_baseline, interaction_fraction, numerical_hessian, quadratic_prediction, determinant_ratio


@dataclass(frozen=True)
class Architecture:
    name: str
    base_logits: np.ndarray
    linear_map: np.ndarray


def separable_architecture():
    k, d = 7, 6
    base = np.zeros(k)
    C = np.eye(k)[:, :d] - np.ones((k, d)) / k
    Q, _ = np.linalg.qr(C)
    L = Q[:, :d] * 1.6
    return Architecture("separable", base, L)


def coupled_architecture():
    arch = separable_architecture()
    L = arch.linear_map.copy()
    L[:, 1] += 0.75 * L[:, 0]
    L[:, 3] += 0.55 * L[:, 2]
    L[:, 5] += 0.45 * L[:, 0] - 0.35 * L[:, 4]
    return Architecture("coupled", arch.base_logits.copy(), L)


def ablated_architecture(coupled):
    G = fisher_at_baseline(coupled.base_logits, coupled.linear_map)
    sep = separable_architecture()
    Gs = fisher_at_baseline(sep.base_logits, sep.linear_map)
    scales = np.sqrt(np.diag(G) / np.diag(Gs))
    return Architecture("ablated", coupled.base_logits.copy(), sep.linear_map * scales[None, :])


def perturbation_grid(levels=(-0.20, -0.10, -0.05, 0.05, 0.10, 0.20), max_active=3):
    d = len(CONDITIONS)
    rows = []
    for active in range(1, max_active + 1):
        for idx in itertools.combinations(range(d), active):
            for vals in itertools.product(levels, repeat=active):
                e = np.zeros(d)
                for i, v in zip(idx, vals):
                    e[i] = v
                rows.append(e)
    return np.asarray(rows)


def evaluate_architecture(arch, seed=2026):
    func = lambda e: answerability_deformation(e, arch.base_logits, arch.linear_map)
    G_exact = fisher_at_baseline(arch.base_logits, arch.linear_map)
    G_num = numerical_hessian(func, len(CONDITIONS), h=1e-4)
    X = perturbation_grid(levels=(-0.10, -0.05, 0.05, 0.10), max_active=3)
    y = np.array([func(e) for e in X])
    pred_full = np.array([quadratic_prediction(e, G_num) for e in X])
    pred_diag = np.array([diagonal_prediction(e, G_num) for e in X])
    rmse_full = float(np.sqrt(np.mean((y - pred_full) ** 2)))
    rmse_diag = float(np.sqrt(np.mean((y - pred_diag) ** 2)))
    hs = np.array([2e-2, 1e-2, 5e-3, 2e-3, 1e-3])
    h_errors = [float(np.linalg.norm(numerical_hessian(func, len(CONDITIONS), h=h)-G_exact, ord="fro")) for h in hs]
    from .core import response_distribution, kl_divergence
    rng = np.random.default_rng(seed)
    probe = np.array([0.08, -0.06, 0.05, 0.0, 0.04, -0.05])
    p0 = response_distribution(np.zeros(len(CONDITIONS)), arch.base_logits, arch.linear_map)
    pe = response_distribution(probe, arch.base_logits, arch.linear_map)
    true_kl = kl_divergence(pe, p0)
    ests = []
    n = 20000
    for _ in range(200):
        c0 = rng.multinomial(n, p0) + 0.5
        ce = rng.multinomial(n, pe) + 0.5
        q0 = c0/c0.sum(); qe = ce/ce.sum()
        ests.append(kl_divergence(qe, q0))
    eig = np.linalg.eigvalsh(G_num)
    return {
        "architecture": arch.name,
        "hessian_fro_error": float(np.linalg.norm(G_num-G_exact, ord="fro")),
        "symmetry_error": float(np.max(np.abs(G_num-G_num.T))),
        "min_eigenvalue": float(eig.min()),
        "rank": int(np.linalg.matrix_rank(G_num, tol=1e-8)),
        "interaction_fraction": interaction_fraction(G_num),
        "determinant_ratio": determinant_ratio(G_num),
        "rmse_full": rmse_full,
        "rmse_diag": rmse_diag,
        "full_vs_diag_improvement": float((rmse_diag-rmse_full)/rmse_diag) if rmse_diag>0 else 0.0,
        "step_sizes": hs.tolist(),
        "step_errors": h_errors,
        "noise_true_kl": true_kl,
        "noise_mae": float(np.mean(np.abs(np.asarray(ests)-true_kl))),
        "hessian": G_num,
        "exact_fisher": G_exact,
    }
