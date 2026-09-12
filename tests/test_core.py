import numpy as np
from dea.core import softmax, kl_divergence, numerical_hessian, fisher_at_baseline, answerability_deformation
from dea.experiments import separable_architecture, coupled_architecture


def test_softmax_and_kl():
    p = softmax([0,0,0])
    assert np.allclose(p, [1/3]*3)
    assert abs(kl_divergence(p,p)) < 1e-12


def test_hessian_matches_fisher_separable():
    a = separable_architecture()
    f = lambda e: answerability_deformation(e, a.base_logits, a.linear_map)
    H = numerical_hessian(f, 6, 1e-4)
    G = fisher_at_baseline(a.base_logits, a.linear_map)
    assert np.linalg.norm(H-G, ord='fro') < 1e-5


def test_hessian_matches_fisher_coupled():
    a = coupled_architecture()
    f = lambda e: answerability_deformation(e, a.base_logits, a.linear_map)
    H = numerical_hessian(f, 6, 1e-4)
    G = fisher_at_baseline(a.base_logits, a.linear_map)
    assert np.linalg.norm(H-G, ord='fro') < 1e-5
    assert np.max(np.abs(H-H.T)) < 1e-12
    assert np.linalg.eigvalsh(H).min() > -1e-6
