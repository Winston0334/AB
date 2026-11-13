"""Checks that the hand-written estimators give the textbook answers.

  * fixed-effects OLS (within transformation) = OLS with one dummy per player
  * the memory-light Within design = the direct fixed-effects estimator
  * fixed-effects 2SLS = 2SLS with player dummies, and recovers a true effect that OLS misses
  * HMM forward algorithm and mixture-of-Markov-chains likelihoods = brute-force sums over hidden paths

Run:  python tests/test_estimators.py
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from segments import HMM, MixtureMarkov  # noqa: E402
from tilt import Within, fe_ols  # noqa: E402


def panel(seed: int = 0):
    """Players with fixed effects a_i; d is endogenous (correlated with the error), z is a valid instrument."""
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(400), rng.integers(1, 15, 400))
    a = rng.normal(size=400)[g]
    z, x1, e = rng.normal(size=len(g)), rng.normal(size=len(g)) + a, rng.normal(size=len(g)) + 0.5 * a
    d = 0.8 * z + 0.6 * e + rng.normal(size=len(g))
    y = 1.0 * x1 - 2.0 * d + 3.0 * a + e
    return g, z, x1, d, y


def dummy_ols(y, X, g):
    keep = np.bincount(g)[g] >= 2                                # the FE estimator drops singletons
    D = (g[keep, None] == np.unique(g[keep])[None, :]).astype(float)
    return np.linalg.lstsq(np.column_stack([X[keep], D]), y[keep], rcond=None)[0][: X.shape[1]], keep, D


def test_fixed_effects_ols() -> None:
    g, z, x1, d, y = panel()
    res = fe_ols(y, np.column_stack([x1, d]), ["x1", "d"], g)
    b, *_ = dummy_ols(y, np.column_stack([x1, d]), g)
    assert np.allclose(res["coef"].est.to_numpy(), b, atol=1e-8), (res["coef"], b)


def test_within_matches_fe_ols() -> None:
    g, z, x1, d, y = panel(1)
    W = Within([("x1", lambda: x1), ("d", lambda: d), ("z", lambda: z)], {"y": y}, g, np.ones(len(y), bool))
    a = W.ols("y", ["x1", "d"])["coef"]
    b = fe_ols(y, np.column_stack([x1, d]), ["x1", "d"], g)["coef"]
    assert np.allclose(a.est, b.est, rtol=1e-5) and np.allclose(a.se, b.se, rtol=1e-4), (a, b)


def test_fixed_effects_2sls() -> None:
    g, z, x1, d, y = panel(2)
    W = Within([("x1", lambda: x1), ("d", lambda: d), ("z", lambda: z)], {"y": y}, g, np.ones(len(y), bool))
    iv = W.iv("y", "d", ["z"], ["x1"])["coef"].set_index("term")
    direct = fe_ols(y, x1[:, None], ["x1"], g, endog=d, Z=z[:, None], endog_name="d")["coef"].set_index("term")
    _, keep, D = dummy_ols(y, np.column_stack([x1, d]), g)
    Wm, Xa = np.column_stack([z[keep], x1[keep], D]), np.column_stack([d[keep], x1[keep], D])
    Xhat = Wm @ np.linalg.lstsq(Wm, Xa, rcond=None)[0]
    manual = np.linalg.lstsq(Xhat, y[keep], rcond=None)[0][0]
    ols_d = fe_ols(y, np.column_stack([x1, d]), ["x1", "d"], g)["coef"].set_index("term").est["d"]
    assert abs(iv.est["d"] - manual) < 1e-4 and abs(direct.est["d"] - manual) < 1e-8, (iv.est["d"], direct.est["d"], manual)
    assert abs(manual + 2.0) < 3 * iv.se["d"], manual          # IV recovers the true -2 ...
    assert abs(ols_d + 2.0) > 10 * iv.se["d"], ols_d           # ... which OLS misses


def test_hmm_likelihood() -> None:
    rng = np.random.default_rng(3)
    m = HMM(3)
    m.pi = rng.dirichlet(np.ones(3))
    m.A = rng.dirichlet(np.ones(3), size=3)
    m.B = rng.dirichlet(np.ones(3), size=3)
    X = rng.integers(0, 3, size=(5, 4))
    for x, ll in zip(X, m.loglik_per_seq(X)):
        total = sum(m.pi[h[0]] * m.B[h[0], x[0]] * np.prod([m.A[h[t - 1], h[t]] * m.B[h[t], x[t]] for t in range(1, 4)])
                    for h in itertools.product(range(3), repeat=4))
        assert np.isclose(ll, np.log(total)), (ll, np.log(total))


def test_mixture_likelihood() -> None:
    rng = np.random.default_rng(4)
    m = MixtureMarkov(2)
    m.w = np.array([0.3, 0.7])
    m.pi = rng.dirichlet(np.ones(6), size=2)
    m.A = rng.dirichlet(np.ones(3), size=(2, 3))
    X = np.column_stack([rng.integers(1, 3, 6), rng.integers(0, 3, (6, 5))])   # week 0 is always active
    for x, ll in zip(X, m.loglik_per_seq(X)):
        pair = (x[0] - 1) * 3 + x[1]
        total = sum(m.w[k] * m.pi[k, pair] * np.prod([m.A[k, x[t], x[t + 1]] for t in range(1, len(x) - 1)])
                    for k in range(2))
        assert np.isclose(ll, np.log(total)), (ll, np.log(total))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all estimator checks passed")
