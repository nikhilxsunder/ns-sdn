# Copyright (c) 2026 Nikhil Sunder
# SPDX-License-Identifier: MIT
"""Behavioral tests: the two alignments, the published configuration, pickling, the flags."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from ns_sdn.exceptions import DimensionError, SpecificationError
from ns_sdn.spectral_decomposition import NSSDN, NSSDNResult, Penalties, Stage
from ns_sdn.spectral_state import SpectralState, _wasserstein_1d

SHORT = (Stage(1e-2, 40),)


def chirp(n: int = 400, seed: int = 0) -> np.ndarray:
    """The design-notebook sanity series: ramped amplitude, drifting frequency, linear trend."""
    t = np.arange(n, dtype=np.float64)
    amplitude = 0.3 + 0.25 / (1.0 + np.exp(-(t - n / 2) / 30.0))
    omega = 0.03 + 0.00005 * t
    y = 0.001 * (t - n / 2) + amplitude * np.sin(np.cumsum(omega))
    return y + 0.02 * np.random.default_rng(seed).standard_normal(n)


@pytest.fixture(scope="module")
def filtered() -> NSSDNResult:
    return NSSDN(chirp(), k=4, state_dim=24, schedule=(Stage(1e-2, 200),), seed=0).fit()


@pytest.fixture(scope="module")
def predicted() -> NSSDNResult:
    return NSSDN(
        chirp(), k=4, state_dim=24, alignment="predicted", schedule=(Stage(1e-2, 200),), seed=0
    ).fit()


def test_published_configuration_parameter_count() -> None:
    model = NSSDN(chirp(50), seed=0)
    assert (model.k, model.state_dim, model.transition, model.alignment) == (
        16,
        128,
        "gru",
        "filtered",
    )
    assert model._build_network().n_parameters == 58_849
    assert [(s.learning_rate, s.steps) for s in model.schedule] == [(1e-3, 1200), (4e-4, 1800)]
    assert model.schedule[0].penalties == Penalties(amplitude=2e-4, frequency=8e-4)
    assert model.schedule[1].penalties == Penalties(amplitude=3e-4, frequency=1.5e-3)


def test_filtered_rows_and_one_step_protocol(filtered: NSSDNResult) -> None:
    n = filtered.endog.shape[0]
    assert filtered.emission.shape == (n,) and filtered.amplitude.shape == (n, 4)
    assert filtered.observed.shape == (n,) and filtered.nobs == n - 1
    assert np.allclose(filtered.fittedvalues, filtered.emission[:-1])
    assert np.allclose(filtered.resid, filtered.endog[1:] - filtered.fittedvalues)
    full = np.concatenate([filtered.endog, chirp(100, seed=2)])
    holdout = filtered.predict(full)[-100:]
    assert holdout.shape == (100,)
    assert np.allclose(holdout, filtered.decompose(full)["emission"][-101:-1])
    assert np.sqrt(np.mean(filtered.reconstruction_resid**2)) < np.sqrt(filtered.sigma2)


def test_predicted_rows_use_only_data_through_t_minus_one(predicted: NSSDNResult) -> None:
    n = predicted.endog.shape[0]
    assert predicted.emission.shape == (n - 1,) and predicted.observed.shape == (n - 1,)
    assert np.allclose(predicted.fittedvalues, predicted.emission)
    y = chirp(seed=1)
    base = predicted.predict(y)
    bumped = y.copy()
    bumped[250] += 1.0
    shifted = predicted.predict(bumped)
    assert np.allclose(base[:250], shifted[:250])
    assert not np.isclose(base[250], shifted[250])
    one = predicted.forecast(1)
    assert np.allclose(
        one, predicted.predict(np.concatenate([predicted.endog, [0.0]]))[-1:], atol=1e-6
    )


def test_paths_reproduce_the_emission(filtered: NSSDNResult, predicted: NSSDNResult) -> None:
    for res in (filtered, predicted):
        assert res.r_squared > 0.95
        assert np.allclose(res.trend + res.components.sum(axis=1), res.emission, atol=1e-5)
        assert np.allclose(res.oscillation, res.components.sum(axis=1), atol=1e-5)
        assert np.allclose(res.effective_amplitude, np.abs(res.mix) * res.amplitude)
        assert np.allclose(res.amplitude, res.amplitude_raw * res.gate)
        assert np.all(res.frequency > 0.0) and np.all((res.gate > 0.0) & (res.gate < 1.0))
        assert np.isclose(res.power_share.sum(), 1.0)


def test_scaled_paths_are_the_training_units(filtered: NSSDNResult) -> None:
    paths = filtered.decompose(scaled=True)
    assert np.allclose(paths["observed"], filtered.scale.transform(filtered.endog))
    assert np.allclose(
        paths["trend"] * filtered.scale.slope + filtered.scale.offset, filtered.trend
    )
    assert np.allclose(paths["amplitude"] * filtered.scale.slope, filtered.amplitude)
    assert np.allclose(paths["frequency"], filtered.frequency)
    assert paths["observed"].min() >= -1.0 - 1e-9 and paths["observed"].max() <= 1.0 + 1e-9


def test_forecast_shapes(filtered: NSSDNResult) -> None:
    assert filtered.forecast(1).shape == (1,) and filtered.forecast(12).shape == (12,)


def test_pickle_roundtrip(filtered: NSSDNResult) -> None:
    clone = pickle.loads(pickle.dumps(filtered))
    assert np.allclose(clone.forecast(5), filtered.forecast(5))
    assert np.allclose(clone.predict(filtered.endog), filtered.fittedvalues)
    assert clone.n_params == filtered.n_params and clone.alignment == "filtered"


def test_rendering_and_frames(filtered: NSSDNResult) -> None:
    text = str(filtered)
    assert "NSSDN(k=4) Results" in text and "component 1" in text and "Filtered alignment" in text
    frame = filtered.to_pandas()
    assert frame.shape == (400, 4 + 6 * 4)
    assert "component_4" in frame.columns and "phase_offset_1" in frame.columns
    table = filtered.compare(filtered, criterion="bic")
    assert table.columns[3] == "BIC"


def test_ordered_frequencies_are_strictly_increasing() -> None:
    res = NSSDN(chirp(200), k=3, state_dim=8, ordered_frequency=True, schedule=SHORT, seed=0).fit()
    assert np.all(np.diff(res.frequency, axis=1) > 0.0)
    assert res.frequency.max() < np.pi


def test_timeaware_transition_and_irregular_grid() -> None:
    y = chirp(150)
    time = np.cumsum(np.random.default_rng(0).uniform(0.5, 1.5, size=150))
    for alignment in ("filtered", "predicted"):
        res = NSSDN(
            y,
            time=time,
            k=2,
            state_dim=8,
            alignment=alignment,
            transition="timeaware",
            schedule=SHORT,
            seed=0,
        ).fit()
        assert res.fittedvalues.shape == (149,)
        assert res.forecast(3, time=time[-1] + np.array([1.0, 2.0, 3.5])).shape == (3,)


def test_exogenous_columns_are_required_at_forecast_time() -> None:
    y = chirp(150)
    x = np.random.default_rng(0).standard_normal((150, 2))
    res = NSSDN(y, exog=x, k=2, state_dim=8, schedule=SHORT, seed=0).fit()
    with pytest.raises(DimensionError):
        res.forecast(3)
    assert res.forecast(3, exog=np.zeros((3, 2))).shape == (3,)
    with pytest.raises(DimensionError):
        res.predict(y)


def test_penalties_and_objectives_are_wired() -> None:
    weights = Penalties(
        amplitude=1e-3,
        frequency=1e-3,
        phase_curvature=1e-4,
        trend=1e-4,
        sparsity=1e-3,
        spectrum=1e-4,
    )
    res = NSSDN(
        chirp(120), k=2, state_dim=8, objective="huber", penalties=weights, schedule=SHORT, seed=0
    ).fit()
    assert set(res.history[-1].parts) == {
        "prediction",
        "amplitude",
        "frequency",
        "phase_curvature",
        "trend",
        "sparsity",
        "spectrum",
    }


def test_specification_errors() -> None:
    y = chirp(50)
    with pytest.raises(SpecificationError):
        NSSDN(y, k=0)
    with pytest.raises(SpecificationError):
        NSSDN(y, alignment="smoothed")
    with pytest.raises(SpecificationError):
        NSSDN(y, objective="mae")
    with pytest.raises(SpecificationError):
        Stage(0.0, 10)
    with pytest.raises(SpecificationError):
        Penalties(amplitude=-1.0)
    with pytest.raises(DimensionError):
        NSSDN(np.ones((10, 2)))


def test_spectral_state(filtered: NSSDNResult) -> None:
    state = filtered.spectral_state()
    assert isinstance(state, SpectralState)
    assert state.nobs == filtered.emission.shape[0]
    assert np.allclose(state.weight.sum(axis=1), 1.0)
    assert np.all(state.velocity >= 0.0)
    assert np.all((state.effective_count >= 1.0 - 1e-9) & (state.effective_count <= 4.0 + 1e-9))
    assert 0.0 <= state.disentanglement_score <= 1.0
    assert "Spectral state" in str(state)


def test_wasserstein_closed_form() -> None:
    a = np.array([0.1, 0.5])
    w = np.array([0.5, 0.5])
    assert _wasserstein_1d(a, w, a, w) == 0.0
    assert np.isclose(_wasserstein_1d(a, w, a + 0.2, w), 0.2)
    assert np.isclose(_wasserstein_1d(a, np.array([1.0, 0.0]), a, np.array([0.0, 1.0])), 0.4)
