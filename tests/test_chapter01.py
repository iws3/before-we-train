"""Tests for Chapter 1. Run with:  pytest -q

Script files start with a digit, so they cannot be imported with a normal `import`;
we load them by path instead.
"""
import importlib.util
import math
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CH1 = os.path.join(ROOT, "chapter-01")
sys.path.insert(0, CH1)

import common  # noqa: E402


def load(name):
    path = os.path.join(CH1, name)
    spec = importlib.util.spec_from_file_location(name[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---- 1.1 ------------------------------------------------------------------ #
def test_bigram_probabilities_match_the_post():
    m = load("1_1_1_bigram_counter.py")
    clean = m.train(m.CLEAN)
    assert m.prob(clean, "the", "cat") == pytest.approx(2 / 3)
    assert m.prob(clean, "the", "dog") == pytest.approx(1 / 3)
    spam = m.train(m.CLEAN + [m.SPAM] * 3)
    assert m.prob(spam, "the", "free") == pytest.approx(0.5)
    assert m.prob(spam, "the", "cat") == pytest.approx(1 / 3)


def test_exercise_1_needs_28_copies():
    assert load("1_1_1_bigram_counter.py").spam_copies_needed(0.9) == 28


def test_years_of_reading():
    m = load("1_1_3_scale_and_fertility.py")
    assert round(m.years_of_reading(15e12)) == 85_616


def test_filter_tradeoff_numbers():
    m = load("1_1_4_filter_tradeoff.py")
    n0, q0, h0, _ = m.measure(m.DOCS, 0.0)
    n1, q1, h1, s1 = m.measure(m.DOCS, 0.5)
    assert (n0, n1) == (10, 6)
    assert q0 == pytest.approx(0.545) and q1 == pytest.approx(0.7667, abs=1e-3)
    assert h0 == pytest.approx(0.950, abs=1e-3) and h1 == pytest.approx(0.451, abs=1e-3)
    assert s1["Yoruba"] == 0


# ---- 1.2 ------------------------------------------------------------------ #
def test_cross_entropy_identity_and_kl_nonnegative():
    rng = np.random.default_rng(0)
    for _ in range(200):
        k = int(rng.integers(2, 30))
        p, q = rng.random(k) + 1e-3, rng.random(k) + 1e-3
        p, q = p / p.sum(), q / q.sum()
        assert common.cross_entropy(p, q) == pytest.approx(common.entropy(p) + common.kl_divergence(p, q))
        assert common.kl_divergence(p, q) >= -1e-12


def test_ln_vocab():
    assert math.log(32_000) == pytest.approx(10.37, abs=0.01)
    assert math.log(50_257) == pytest.approx(10.82, abs=0.01)
    assert math.log(128_256) == pytest.approx(11.76, abs=0.01)
    m = load("1_2_5_ln_vocab_check.py")
    assert m.starting_loss(1000, 0.0) == pytest.approx(math.log(1000))


def test_separable_spam_matches_theory():
    """In the theory's setting (separable spam, unigram), observed ratio ~ 1/(1-eps)."""
    m = load("1_2_4_spam_pollution_experiment.py")
    text = common.synthetic_text(200_000)
    r = m.experiment(text, order=0, every=4, spam_lines=m.separable_spam_lines())
    assert r["ratio"] == pytest.approx(r["bound"], rel=0.05)


def test_overlapping_spam_is_below_the_bound():
    m = load("1_2_4_spam_pollution_experiment.py")
    r = m.experiment(common.synthetic_text(200_000), order=0, every=4, spam_lines=None)
    assert 1.0 < r["ratio"] < r["bound"]


def test_spam_injection_and_removal():
    text = common.synthetic_text(50_000)
    polluted, eps = common.inject_spam(text, every=4)
    assert 0 < eps < 1
    assert common.remove_spam(polluted).strip() == text.strip()


# ---- 1.3 ------------------------------------------------------------------ #
def test_perplexity_worked_example():
    m = load("1_3_1_perplexity_and_bpb.py")
    probs = np.array([0.5, 0.25, 0.1, 0.8])
    L = m.loss_from_probs(probs)
    assert L == pytest.approx(1.1513, abs=1e-4)
    assert math.exp(L) == pytest.approx(3.1623, abs=1e-3)
    assert m.bits_per_byte(1000, 2.3, 4200) == pytest.approx(0.790, abs=1e-3)


def test_softmax_gradient_matches_finite_differences():
    m = load("1_3_2_softmax_gradient_check.py")
    rng = np.random.default_rng(0)
    for _ in range(50):
        z = rng.normal(0, 2, 6)
        y = int(rng.integers(0, 6))
        assert np.allclose(m.analytic_grad(z, y), m.numeric_grad(z, y), atol=1e-6)


def test_chinchilla_table_values():
    E, mt, dt = common.scaling_terms(8e9, 15e12)
    assert (mt, dt) == (pytest.approx(0.175, abs=1e-3), pytest.approx(0.084, abs=1e-3))
    assert common.scaling_loss(8e9, 15e12) == pytest.approx(1.95, abs=0.01)
    assert common.scaling_loss(405e9, 15.6e12) == pytest.approx(1.82, abs=0.01)
    assert 2 ** -0.28 == pytest.approx(0.824, abs=1e-3)


def test_closed_form_optimum_matches_numeric():
    m = load("1_3_4_compute_optimal_and_inference.py")
    for c in common.SCALING_CONSTANTS:
        N, D = m.optimum_closed_form(5.88e23, c)
        Nn, Dn, _ = m.optimum_numeric(5.88e23, c)
        assert Nn == pytest.approx(N, rel=0.01)
        assert 6 * N * D == pytest.approx(5.88e23)


def test_power_law_fit_recovers_parameters():
    m = load("1_3_5_mini_scaling_experiment.py")
    D = np.array([1e4, 3e4, 1e5, 3e5, 1e6, 3e6])
    L = 1.5 + 40 / D ** 0.35
    E, B, beta = m.fit_power_law(D, L)
    assert beta == pytest.approx(0.35, abs=0.05) and E == pytest.approx(1.5, abs=0.05)


def test_exercise_solutions():
    sys.path.insert(0, os.path.join(CH1, "exercises"))
    import solutions
    assert solutions.exercise_1() == 28
    assert solutions.exercise_2()[0] == pytest.approx(1 / 0.92)
    assert solutions.exercise_3() == pytest.approx(11.09, abs=0.01)
    assert solutions.exercise_4() == pytest.approx(9.56e13, rel=0.01)


# ---- every script runs end to end ----------------------------------------- #
@pytest.mark.parametrize("script", sorted(f for f in os.listdir(CH1) if f[0] == "1" and f.endswith(".py")))
def test_script_runs(script, tmp_path):
    text_file = tmp_path / "sample.txt"
    text_file.write_text(common.synthetic_text(60_000))
    args = [sys.executable, os.path.join(CH1, script)]
    if script in ("1_2_4_spam_pollution_experiment.py", "1_3_5_mini_scaling_experiment.py"):
        args += ["--text", str(text_file)]
    r = subprocess.run(args, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip()
