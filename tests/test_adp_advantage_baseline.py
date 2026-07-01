"""Tests for the per-epoch advantage baseline (fix (c)) on the ADP value fns.

The baseline is a pure target reparameterisation: fit on y - b(t), predict adds
b(t) back. predict() must therefore return a full cost-to-go estimate regardless
of whether the baseline is enabled, so the action-generator Q reconstruction
Q(s,a) = C(s,a) + V'(s_post) stays valid for BOTH XGBoost and the neural VF.
"""
import numpy as np
import pytest

from env.mdp import State
from agents.fn.value_fn import XGBoostValueFn, NeuralValueFn


def _make_states(n_states, n_assets, rng):
    states = []
    for k in range(n_states):
        s = State(
            d=rng.uniform(0, 1, n_assets),
            h=np.zeros(n_assets),
            ell=np.zeros(n_assets),
            r=np.zeros(n_assets),
            n_fail=np.zeros(n_assets),
        )
        s.t = int(rng.integers(0, 120))
        states.append(s)
    return states


def test_baseline_disabled_matches_plain_fit_xgb():
    """With baseline OFF, fit_targets must be identical to the old fit(X, y)."""
    rng = np.random.default_rng(0)
    states = _make_states(300, 5, rng)
    y = rng.uniform(1e8, 1e10, len(states))

    vf_plain = XGBoostValueFn(xgb_params={'n_estimators': 20})
    vf_plain.fit(vf_plain._feats(states), y)
    p_plain = vf_plain.predict(states)

    vf_ft = XGBoostValueFn(xgb_params={'n_estimators': 20})
    assert not vf_ft._baseline_enabled
    vf_ft.fit_targets(states, y)
    p_ft = vf_ft.predict(states)

    np.testing.assert_allclose(p_plain, p_ft, rtol=1e-6)
    assert vf_ft._baseline is None  # baseline never built when disabled


@pytest.mark.parametrize("VF", [XGBoostValueFn, NeuralValueFn])
def test_baseline_predict_is_full_cost_estimate(VF):
    """predict() returns model_advantage + b(t); on the training set it should
    recover the target scale (a valid cost-to-go), not the centered advantage."""
    rng = np.random.default_rng(1)
    states = _make_states(400, 5, rng)
    # target with a strong per-epoch trend + per-state noise
    t = np.array([s.t for s in states])
    y = 1e10 * (1.0 - t / 120.0) + rng.normal(0, 1e8, len(states))

    if VF is XGBoostValueFn:
        vf = VF(xgb_params={'n_estimators': 50})
    else:
        vf = VF(hidden_dims=(64, 64))
    vf.set_baseline_enabled(True)
    vf.fit_targets(states, y)

    assert vf._baseline is not None
    preds = vf.predict(states)
    # predictions live in the cost-to-go space (same order of magnitude as y),
    # not the centered-advantage space (~1e8).
    assert preds.mean() > 1e9
    # and they track the per-epoch trend (corr with y)
    corr = np.corrcoef(preds, y)[0, 1]
    assert corr > 0.7


def test_baseline_reduces_target_variance_xgb():
    """The advantage residual the model fits has far lower variance than the raw
    cost-to-go target (the whole point of fix (c))."""
    rng = np.random.default_rng(2)
    states = _make_states(500, 5, rng)
    t = np.array([s.t for s in states])
    y = 1e10 * (1.0 - t / 120.0) + rng.normal(0, 5e7, len(states))

    vf = XGBoostValueFn(xgb_params={'n_estimators': 10})
    vf.set_baseline_enabled(True)
    vf._fit_baseline(t, y)
    adv = y - vf._baseline_for(states)
    assert adv.std() < 0.5 * y.std()


@pytest.mark.parametrize("VF", [XGBoostValueFn, NeuralValueFn])
def test_baseline_survives_save_load(VF, tmp_path):
    rng = np.random.default_rng(3)
    states = _make_states(200, 5, rng)
    y = rng.uniform(1e8, 1e10, len(states))
    if VF is XGBoostValueFn:
        vf = VF(xgb_params={'n_estimators': 20})
    else:
        vf = VF(hidden_dims=(32, 32))
    vf.set_baseline_enabled(True)
    vf.fit_targets(states, y)
    p_before = vf.predict(states)

    path = str(tmp_path / "vf")
    vf.save(path)
    if VF is XGBoostValueFn:
        vf2 = VF(xgb_params={'n_estimators': 20})
    else:
        vf2 = VF(hidden_dims=(32, 32))
    vf2.load(path)
    assert vf2._baseline_enabled
    np.testing.assert_allclose(p_before, vf2.predict(states), rtol=1e-5)
