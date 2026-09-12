from dea.experiments import separable_architecture, coupled_architecture, ablated_architecture, evaluate_architecture


def test_separable_has_negligible_cross_terms():
    r = evaluate_architecture(separable_architecture())
    assert r['interaction_fraction'] < 1e-4


def test_coupled_has_cross_terms_and_full_model_advantage():
    r = evaluate_architecture(coupled_architecture())
    assert r['interaction_fraction'] > 0.20
    assert r['rmse_full'] < 0.35 * r['rmse_diag']


def test_ablation_removes_interaction():
    c = coupled_architecture()
    r = evaluate_architecture(ablated_architecture(c))
    assert r['interaction_fraction'] < 1e-4
