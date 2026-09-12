import numpy as np

CONDITIONS = ("D", "A", "O", "I", "T", "R")


def softmax(z):
    z = np.asarray(z, dtype=float)
    z = z - np.max(z)
    e = np.exp(z)
    return e / e.sum()


def kl_divergence(p, q):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if np.any(p <= 0) or np.any(q <= 0):
        raise ValueError("KL inputs must be strictly positive")
    return float(np.sum(p * np.log(p / q)))


def response_distribution(eps, base_logits, linear_map, quadratic_maps=None):
    """Smooth probabilistic reachable-response field."""
    eps = np.asarray(eps, dtype=float)
    logits = np.asarray(base_logits, dtype=float) + np.asarray(linear_map, dtype=float) @ eps
    if quadratic_maps is not None:
        q = np.asarray(quadratic_maps, dtype=float)
        logits = logits + 0.5 * np.einsum("i,kij,j->k", eps, q, eps)
    return softmax(logits)


def answerability_deformation(eps, base_logits, linear_map, quadratic_maps=None):
    p0 = response_distribution(np.zeros(len(eps)), base_logits, linear_map, quadratic_maps)
    pe = response_distribution(eps, base_logits, linear_map, quadratic_maps)
    return kl_divergence(pe, p0)


def fisher_at_baseline(base_logits, linear_map):
    p0 = softmax(base_logits)
    L = np.asarray(linear_map, dtype=float)
    mean = p0 @ L
    centered = L - mean
    return centered.T @ (p0[:, None] * centered)


def numerical_hessian(func, d, h=1e-4):
    H = np.zeros((d, d), dtype=float)
    z = np.zeros(d)
    f0 = func(z)
    for i in range(d):
        ei = np.zeros(d); ei[i] = h
        H[i, i] = (func(ei) - 2 * f0 + func(-ei)) / (h * h)
        for j in range(i + 1, d):
            ej = np.zeros(d); ej[j] = h
            val = (func(ei + ej) - func(ei - ej) - func(-ei + ej) + func(-ei - ej)) / (4 * h * h)
            H[i, j] = H[j, i] = val
    return H


def quadratic_prediction(eps, H):
    eps = np.asarray(eps, dtype=float)
    return float(0.5 * eps @ H @ eps)


def diagonal_prediction(eps, H):
    eps = np.asarray(eps, dtype=float)
    return float(0.5 * np.sum(np.diag(H) * eps * eps))


def interaction_fraction(H, atol=1e-15):
    H = np.asarray(H, dtype=float)
    off = H - np.diag(np.diag(H))
    denom = np.linalg.norm(H, ord="fro")
    return 0.0 if denom <= atol else float(np.linalg.norm(off, ord="fro") / denom)


def determinant_ratio(H, atol=1e-12):
    H = np.asarray(H, dtype=float)
    d = np.diag(H)
    if np.any(d <= atol):
        return np.nan
    return float(np.linalg.det(H) / np.prod(d))
