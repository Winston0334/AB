"""Part 3: is non-Markov behaviour habit, or a mix of player types?

Part 2 found strong duration dependence: the longer a player has been in a state, the less likely
they are to leave it. Two explanations fit that pattern:

  habit          each player's own transition probabilities change with time in the state
  heterogeneity  players differ; each type is a plain Markov chain, and the mix in a state shifts
                 toward the "stayers" over time

This script fits two latent-type models to each player's weekly state sequence (weeks 0-51, states
Inactive / Casual / Engaged as in Part 2), both by EM written here in numpy:

  mixture of Markov chains  each player belongs to one of K types for all weeks; within a type,
                            transitions are a plain Markov chain with no duration dependence
  hidden Markov model       K hidden states with Markov dynamics that emit the observed state;
                            a player can move between hidden states

They are compared with the semi-Markov chain from Part 2 on (a) in-sample BIC, (b) log-likelihood on
a held-out cohort, and (c) how well they reproduce the share of players active each week.

Outputs: reports/tables/segments_*.csv, reports/figures/segments_*.png

Example:
    python src/segments.py --panel data/panel --out reports
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from transitions import (DUR_CAP, GRID, INK, INK2, SERIES, STATES, SURFACE, load, pct, plt,
                         prevalence_check, style, tenure_band)

N_WEEKS = 52          # weeks 0..51 are observed for every cohort player
ALPHA = 1e-3          # tiny Dirichlet smoothing so no probability is exactly zero
TINY = 1e-300


# ----------------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------------
def sequences(df: pd.DataFrame) -> tuple[pd.Index, np.ndarray, np.ndarray]:
    d = df[df.week < N_WEEKS]
    wide = d.pivot(index="player", columns="week", values="state")
    cohort = d.groupby("player").cohort_month.first().reindex(wide.index).to_numpy()
    return wide.index, wide.to_numpy().astype(np.int64), cohort


def compress(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unique sequences, their counts, and the index mapping players to them."""
    U, inv, cnt = np.unique(X, axis=0, return_inverse=True, return_counts=True)
    return U, cnt.astype(float), inv.ravel()


def transition_counts(X: np.ndarray) -> np.ndarray:
    """(N, 3, 3) counts of week-to-week transitions per sequence."""
    pair = X[:, :-1] * 3 + X[:, 1:]
    C = np.zeros((len(X), 9))
    for v in range(9):
        C[:, v] = (pair == v).sum(axis=1)
    return C.reshape(-1, 3, 3)


# ----------------------------------------------------------------------------------------------
# Mixture of Markov chains
# ----------------------------------------------------------------------------------------------
class MixtureMarkov:
    """K types. Type k has a distribution pi[k] over the (week-0, week-1) state pair and one 3x3
    transition matrix A[k] for every week from week 1 on, with no duration dependence.

    Week 0 is the onboarding week (active by construction), so it gets its own joint start
    distribution instead of being forced through the same matrix as later weeks."""

    kind = "mixture"
    PAIRS = [(a, b) for a in (1, 2) for b in range(3)]           # 6 possible (week0, week1) pairs

    def __init__(self, K: int):
        self.K = K

    @property
    def n_params(self) -> int:
        return (self.K - 1) + self.K * 5 + self.K * 6

    @staticmethod
    def _pair(X: np.ndarray) -> np.ndarray:
        return (X[:, 0] - 1) * 3 + X[:, 1]

    def _stats(self, X: np.ndarray):
        return self._pair(X), transition_counts(X[:, 1:])

    def _log_components(self, pair: np.ndarray, C: np.ndarray) -> np.ndarray:
        return (np.log(self.w)[None, :] + np.log(self.pi[:, pair]).T
                + np.einsum("nab,kab->nk", C, np.log(self.A)))

    def _m_step(self, R: np.ndarray, pair: np.ndarray, C: np.ndarray, wts: np.ndarray) -> None:
        Rw = R * wts[:, None]
        self.w = Rw.sum(0) / Rw.sum()
        self.pi = Rw.T @ np.eye(6)[pair] + ALPHA
        self.pi /= self.pi.sum(1, keepdims=True)
        A = np.einsum("nk,nab->kab", Rw, C) + ALPHA
        self.A = A / A.sum(2, keepdims=True)

    def fit(self, X: np.ndarray, wts: np.ndarray, n_starts: int = 6, max_iter: int = 2000,
            tol: float = 1e-9, seed: int = 0) -> "MixtureMarkov":
        pair, C = self._stats(X)
        rng = np.random.default_rng(seed)
        best = None
        for _ in range(n_starts):
            self._m_step(rng.dirichlet(np.ones(self.K), size=len(X)), pair, C, wts)
            prev = -np.inf
            for it in range(max_iter):
                L = self._log_components(pair, C)
                m = L.max(1, keepdims=True)
                ll_i = m[:, 0] + np.log(np.exp(L - m).sum(1))
                ll = float((wts * ll_i).sum())
                self._m_step(np.exp(L - ll_i[:, None]), pair, C, wts)
                if ll - prev < tol * abs(ll):
                    break
                prev = ll
            if best is None or ll > best[0]:
                best = (ll, self.w.copy(), self.pi.copy(), self.A.copy(), it)
        self.loglik, self.w, self.pi, self.A, self.n_iter = best
        self._order()
        return self

    def _order(self) -> None:
        """Order types from least to most engaged (long-run share of active weeks)."""
        o = np.argsort([stationary(A)[1:].sum() for A in self.A])
        self.w, self.pi, self.A = self.w[o], self.pi[o], self.A[o]

    def posterior(self, X: np.ndarray) -> np.ndarray:
        """Type posterior from however many weeks X has (at least 1)."""
        if X.shape[1] == 1:   # only week 0 seen: marginalise the pair over week 1
            p0 = self.pi.reshape(self.K, 2, 3).sum(2)            # (K, week0 in {Casual, Engaged})
            L = np.log(self.w)[None] + np.log(p0[:, X[:, 0] - 1]).T
        else:
            L = self._log_components(*self._stats(X))
        L -= L.max(1, keepdims=True)
        R = np.exp(L)
        return R / R.sum(1, keepdims=True)

    def loglik_per_seq(self, X: np.ndarray) -> np.ndarray:
        L = self._log_components(*self._stats(X))
        m = L.max(1, keepdims=True)
        return m[:, 0] + np.log(np.exp(L - m).sum(1))

    def predict_from_week1(self, X: np.ndarray, horizon: int) -> np.ndarray:
        """Expected state shares for weeks 1..horizon given each player's weeks 0 and 1."""
        post = self.posterior(X[:, :2])
        s1 = X[:, 1]
        out = np.zeros((horizon, 3))
        P = np.repeat(np.eye(3)[None], self.K, axis=0)          # A_k^(t-1)
        for t in range(horizon):
            out[t] = np.einsum("nk,knj->j", post, P[:, s1, :]) / len(X)
            P = P @ self.A
        return out

    def get_state(self) -> dict:
        return {"w": self.w, "pi": self.pi, "A": self.A, "loglik": self.loglik}

    def set_state(self, d: dict) -> "MixtureMarkov":
        self.w, self.pi, self.A, self.loglik = d["w"], d["pi"], d["A"], float(d["loglik"])
        return self


def stationary(A: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eig(A.T)
    v = np.real(vecs[:, np.argmin(np.abs(vals - 1))])
    return v / v.sum()


# ----------------------------------------------------------------------------------------------
# Hidden Markov model with categorical emissions
# ----------------------------------------------------------------------------------------------
class HMM:
    """K hidden states; pi (K), transitions A (K,K), emissions B (K,3) over observed states."""

    kind = "hmm"

    def __init__(self, K: int):
        self.K = K

    @property
    def n_params(self) -> int:
        return (self.K - 1) + self.K * (self.K - 1) + self.K * 2

    def _fb(self, X: np.ndarray):
        """Scaled forward-backward. Arrays are time-major (T, N, K) so each step is contiguous."""
        T, N = X.shape[1], X.shape[0]
        E = self.B.T[X.T]                                        # (T,N,K) emission probs
        alpha = np.empty((T, N, self.K))
        c = np.empty((T, N))
        a = self.pi[None] * E[0]
        c[0] = a.sum(1)
        alpha[0] = a / c[0, :, None]
        for t in range(1, T):
            a = (alpha[t - 1] @ self.A) * E[t]
            c[t] = a.sum(1)
            alpha[t] = a / c[t, :, None]
        beta = np.empty((T, N, self.K))
        beta[-1] = 1.0
        for t in range(T - 2, -1, -1):
            beta[t] = ((E[t + 1] * beta[t + 1]) @ self.A.T) / c[t + 1, :, None]
        return E, alpha, beta, c

    def _em_step(self, X: np.ndarray, wts: np.ndarray, onehot: np.ndarray) -> float:
        E, alpha, beta, c = self._fb(X)
        ll = float((wts * np.log(c).sum(0)).sum())
        gamma = alpha * beta
        gamma /= gamma.sum(2, keepdims=True)
        gw = gamma * wts[None, :, None]
        left = (alpha[:-1] * wts[None, :, None]).reshape(-1, self.K)
        right = (E[1:] * beta[1:] / c[1:, :, None]).reshape(-1, self.K)
        self.pi = gw[0].sum(0) + ALPHA
        self.pi /= self.pi.sum()
        self.A = self.A * (left.T @ right) + ALPHA
        self.A /= self.A.sum(1, keepdims=True)
        self.B = gw.reshape(-1, self.K).T @ onehot + ALPHA
        self.B /= self.B.sum(1, keepdims=True)
        return ll

    def fit(self, X: np.ndarray, wts: np.ndarray, n_starts: int = 6, max_iter: int = 1500,
            tol: float = 1e-7, seed: int = 0, burn_in: int = 40) -> "HMM":
        """Each random start runs `burn_in` EM iterations; only the best start is run to convergence."""
        rng = np.random.default_rng(seed)
        onehot = np.eye(3)[X.T].reshape(-1, 3)                   # time-major, matches gamma
        cands = []
        for _ in range(n_starts):
            self.pi = rng.dirichlet(np.ones(self.K))
            self.A = rng.dirichlet(np.ones(self.K), size=self.K) + 4 * np.eye(self.K)
            self.A /= self.A.sum(1, keepdims=True)
            self.B = rng.dirichlet(np.ones(3), size=self.K)
            for _ in range(burn_in):
                ll = self._em_step(X, wts, onehot)
            cands.append((ll, self.pi.copy(), self.A.copy(), self.B.copy()))
        _, self.pi, self.A, self.B = max(cands, key=lambda c: c[0])
        prev = -np.inf
        for it in range(max_iter):
            ll = self._em_step(X, wts, onehot)
            if ll - prev < tol * abs(ll):
                break
            prev = ll
        self.loglik, self.n_iter = ll, burn_in + it
        self._order()
        return self

    def _order(self) -> None:
        o = np.argsort(1 - self.B[:, 0])                         # by P(active) emitted
        self.pi, self.A, self.B = self.pi[o], self.A[o][:, o], self.B[o]

    def loglik_per_seq(self, X: np.ndarray) -> np.ndarray:
        return np.log(self._fb(X)[3]).sum(0)

    def get_state(self) -> dict:
        return {"pi": self.pi, "A": self.A, "B": self.B, "loglik": self.loglik}

    def set_state(self, d: dict) -> "HMM":
        self.pi, self.A, self.B, self.loglik = d["pi"], d["A"], d["B"], float(d["loglik"])
        return self

    def predict_from_week1(self, X: np.ndarray, horizon: int) -> np.ndarray:
        _, alpha, _, _ = self._fb(X[:, :2])
        h = alpha[1]                                             # filtered hidden dist at week 1
        out = np.zeros((horizon, 3))
        out[0] = np.bincount(X[:, 1], minlength=3) / len(X)      # week 1 is observed
        for t in range(1, horizon):
            h = h @ self.A
            out[t] = (h @ self.B).mean(0)
        return out


# ----------------------------------------------------------------------------------------------
# Semi-Markov chain from Part 2, as a sequence likelihood
# ----------------------------------------------------------------------------------------------
def _run_lengths(X: np.ndarray) -> np.ndarray:
    D = np.ones_like(X)
    for t in range(1, X.shape[1]):
        D[:, t] = np.where(X[:, t] == X[:, t - 1], D[:, t - 1] + 1, 1)
    return np.minimum(D, DUR_CAP)


class SemiMarkov:
    """P(next | state, weeks in state, weeks since joining), fit by counting with back-off to
    (state, weeks in state) and (state) where cells are thin. Week 0 has its own tenure band."""

    kind = "semi-markov"

    def __init__(self, min_count: int = 20):
        self.min_count = min_count

    def _keys(self, X: np.ndarray):
        N, T = X.shape
        D = _run_lengths(X)
        band = np.array([-1] + [tenure_band(t) for t in range(1, T)])
        s, d, b = X[:, :-1].ravel(), D[:, :-1].ravel(), np.tile(band[:-1], N)
        return s, d, b, X[:, 1:].ravel()

    def fit(self, X: np.ndarray, wts: np.ndarray) -> "SemiMarkov":
        s, d, b, nxt = self._keys(X)
        w = np.repeat(wts, X.shape[1] - 1)
        df = pd.DataFrame({"s": s, "d": d, "b": b, "n": nxt, "w": w})
        self.tables = []
        for keys in (["s", "d", "b"], ["s", "d"], ["s"]):
            c = df.groupby(keys + ["n"]).w.sum().unstack(fill_value=0).reindex(columns=range(3), fill_value=0)
            self.tables.append((keys, c))
        self.n_params = 2 * sum(int((c.sum(axis=1) >= self.min_count).sum()) for _, c in self.tables[:2]) + 6
        self.p0 = np.bincount(X[:, 0], weights=wts, minlength=3) / wts.sum()
        return self

    def _probs(self, s, d, b) -> np.ndarray:
        out = np.full((len(s), 3), np.nan)
        for keys, c in self.tables:
            ok = c[c.sum(axis=1) >= self.min_count] if len(keys) > 1 else c
            p = (ok + ALPHA).div((ok + ALPHA).sum(axis=1), axis=0)
            idx = pd.MultiIndex.from_arrays([{"s": s, "d": d, "b": b}[k] for k in keys]) if len(keys) > 1 \
                else pd.Index(s)
            vals = p.reindex(idx).to_numpy()
            fill = np.isnan(out[:, 0]) & ~np.isnan(vals[:, 0])
            out[fill] = vals[fill]
        return out

    def loglik_per_seq(self, X: np.ndarray) -> np.ndarray:
        s, d, b, nxt = self._keys(X)
        p = self._probs(s, d, b)[np.arange(len(s)), nxt]
        ll = np.log(p).reshape(len(X), -1).sum(1)
        return ll + np.log(self.p0[X[:, 0]])


# ----------------------------------------------------------------------------------------------
# What each model implies for the duration curves, regime names, early identification
# ----------------------------------------------------------------------------------------------
def implied_duration_curves(U: np.ndarray, w: np.ndarray, mix: MixtureMarkov, hmm: HMM,
                            max_dur: int = 12) -> pd.DataFrame:
    """For every week t >= 1: the observed P(active at t+1), and each model's one-step-ahead
    prediction using only weeks 0..t, averaged by (current state, weeks in current state).

    A model that explains duration dependence reproduces the observed rise or fall with duration.
    In the mixture, any such pattern can only come from learning which fixed type a player is."""
    N, T = U.shape
    D = np.minimum(_run_lengths(U), max_dur)
    pair = (U[:, 0] - 1) * 3 + U[:, 1]
    L = np.log(mix.w)[None] + np.log(mix.pi[:, pair]).T
    logA = np.log(mix.A)
    act_from = 1 - mix.A[:, :, 0]                                # (K,3) P(active next | type, state)
    _, alpha, _, _ = hmm._fb(U)
    p_hmm = alpha @ hmm.A @ (1 - hmm.B[:, 0])                    # (T,N)
    frames = []
    for t in range(1, T - 1):
        if t >= 2:
            L = L + logA[:, U[:, t - 1], U[:, t]].T
        R = np.exp(L - L.max(1, keepdims=True))
        R /= R.sum(1, keepdims=True)
        frames.append(pd.DataFrame({"state": U[:, t], "dur": D[:, t], "w": w,
                                    "observed": (U[:, t + 1] > 0).astype(float),
                                    "mixture": (R * act_from[:, U[:, t]].T).sum(1), "hmm": p_hmm[t]}))
    f = pd.concat(frames, ignore_index=True)
    for c in ("observed", "mixture", "hmm"):
        f[c] = f[c] * f.w
    g = f.groupby(["state", "dur"]).sum().reset_index()
    for c in ("observed", "mixture", "hmm"):
        g[c] = g[c] / g.w
    g["state"] = g.state.map(dict(enumerate(STATES)))
    return g.rename(columns={"w": "weeks"})


def regime_names(hmm: HMM) -> list[str]:
    """Readable names for 5- or 6-regime fits. The start regime is Onboarding; the rest, from least
    to most active, are Gone, Lapsed, Casual regular, then the engaged regime(s). With six regimes
    the two engaged regimes are told apart by persistence: Core stays, Burst does not."""
    names = [f"regime {k}" for k in range(hmm.K)]
    if hmm.K not in (5, 6):
        return names
    start = int(np.argmax(hmm.pi))
    rest = sorted((k for k in range(hmm.K) if k != start), key=lambda k: 1 - hmm.B[k, 0])
    names[start] = "Onboarding"
    for k, n in zip(rest[:3], ["Gone", "Lapsed", "Casual regular"]):
        names[k] = n
    engaged = sorted(rest[3:], key=lambda k: hmm.A[k, k])
    for k, n in zip(engaged, ["Burst", "Core"] if len(engaged) == 2 else ["Core"]):
        names[k] = n
    return names


def auc(score: np.ndarray, y: np.ndarray) -> float:
    """Mann-Whitney AUC with tied scores counted as half."""
    from scipy.stats import rankdata
    r = rankdata(score)
    n1 = y.sum()
    n0 = len(y) - n1
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def p_active_window(hmm: HMM, X: np.ndarray, m: int, start: int = 12, end: int = 15) -> np.ndarray:
    """P(at least one active week in weeks start..end | weeks 0..m), from the HMM filter."""
    _, alpha, _, _ = hmm._fb(X[:, :m + 1])
    v = alpha[m] @ np.linalg.matrix_power(hmm.A, start - m)
    for wk in range(start, end + 1):
        v = v * hmm.B[:, 0]
        if wk < end:
            v = v @ hmm.A
    return 1 - v.sum(1)


# ----------------------------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------------------------
def curve_mae(pred_active: np.ndarray, obs_active: np.ndarray) -> float:
    return float(np.abs(pred_active[1:] - obs_active[1:]).mean())


def run(panel: Path, out: Path, k_max_mix: int = 10, k_max_hmm: int = 8, n_starts: int = 4,
        budget_s: float | None = None, hmm_k: int = 6) -> dict:
    tables, figs = out / "tables", out / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:5.0f}s] {m}", flush=True)  # noqa: E731

    df = load(panel)
    players, X, cohort = sequences(df)
    H = N_WEEKS - 1                                              # weeks 1..51
    test_month = sorted(set(cohort))[-1]
    is_test = cohort == test_month
    obs_all = np.array([(X[:, t] > 0).mean() for t in range(1, N_WEEKS)])
    obs_test = np.array([(X[is_test, t] > 0).mean() for t in range(1, N_WEEKS)])
    log(f"{len(X):,} players x {N_WEEKS} weeks; holdout cohort {test_month} ({is_test.sum():,} players)")

    # Step-4 semi-Markov chain curves, re-run on the same 52-week window.
    dfw = df[df.week < N_WEEKS]
    tw = dfw[(dfw.week >= 1) & dfw.next_state.notna()]
    sm_all = prevalence_check(dfw, tw, horizon=H)
    sm_hold = prevalence_check(dfw[dfw.cohort_month == test_month], tw[tw.cohort_month != test_month], horizon=H)
    act = lambda p, m: (p[f"{m}_Casual"] + p[f"{m}_Engaged"]).to_numpy()  # noqa: E731

    U_tr, w_tr, _ = compress(X[~is_test])
    U_all, w_all, inv_all = compress(X)
    X_te = X[is_test]
    n_all = len(X)

    rows, models = [], {}
    sm_model = SemiMarkov().fit(U_tr, w_tr)
    sm_full = SemiMarkov().fit(U_all, w_all)
    ll_full = float((w_all * sm_full.loglik_per_seq(U_all)).sum())
    rows.append({"model": "semi-Markov chain (Part 2)", "kind": "semi-markov", "K": None,
                 "n_params": sm_full.n_params, "loglik": ll_full,
                 "bic": -2 * ll_full + sm_full.n_params * np.log(n_all),
                 "holdout_ll_per_player": float(sm_model.loglik_per_seq(X_te).mean()),
                 "curve_mae_in_sample": curve_mae(act(sm_all, "duration_tenure"), obs_all),
                 "curve_mae_holdout": curve_mae(act(sm_hold, "duration_tenure"), obs_test)})
    log("semi-Markov chain done")

    cache = out / "tables" / "segments_fits"
    cache.mkdir(parents=True, exist_ok=True)

    def fitted(cls, K: int, split: str, U: np.ndarray, w: np.ndarray):
        f = cache / f"{cls.kind}_K{K}_{split}_s{n_starts}.npz"
        if f.exists():
            return cls(K).set_state(dict(np.load(f)))
        if budget_s is not None and time.time() - t0 > budget_s:
            raise TimeoutError("time budget used; rerun the same command to continue")
        m = cls(K).fit(U, w, n_starts=n_starts)
        np.savez(f, **m.get_state())
        return m

    for cls, ks in ((MixtureMarkov, range(1, k_max_mix + 1)), (HMM, range(2, k_max_hmm + 1))):
        for K in ks:
            m_tr = fitted(cls, K, "train", U_tr, w_tr)
            m_all = fitted(cls, K, "all", U_all, w_all)
            p_in = m_all.predict_from_week1(X, H)
            p_ho = m_tr.predict_from_week1(X_te, H)
            name = "plain Markov chain" if (cls is MixtureMarkov and K == 1) else f"{cls.kind} K={K}"
            rows.append({"model": name, "kind": cls.kind, "K": K, "n_params": m_all.n_params,
                         "loglik": m_all.loglik, "bic": -2 * m_all.loglik + m_all.n_params * np.log(n_all),
                         "holdout_ll_per_player": float(m_tr.loglik_per_seq(X_te).mean()),
                         "curve_mae_in_sample": curve_mae(p_in[:, 1:].sum(1), obs_all),
                         "curve_mae_holdout": curve_mae(p_ho[:, 1:].sum(1), obs_test)})
            models[(cls.kind, K)] = {"all": m_all, "train": m_tr, "pred_in": p_in, "pred_ho": p_ho}
            log(f"{name}: BIC {rows[-1]['bic']:,.0f}, holdout LL/player {rows[-1]['holdout_ll_per_player']:.3f}, "
                f"curve MAE {rows[-1]['curve_mae_in_sample'] * 100:.2f} / {rows[-1]['curve_mae_holdout'] * 100:.2f} pp")
    comp = pd.DataFrame(rows)
    comp.to_csv(tables / "segments_model_comparison.csv", index=False)

    best_mix = int(comp[comp.kind == "mixture"].sort_values("bic").K.iloc[0])
    best_hmm = int(comp[comp.kind == "hmm"].sort_values("bic").K.iloc[0])
    mix = models[("mixture", k_max_mix)]["all"]                  # the most flexible fixed-type model
    hmm = models[("hmm", hmm_k)]["all"]                          # the hidden Markov model we interpret
    names = regime_names(hmm)

    # Holdout curves for the chart.
    curves = pd.DataFrame({"week": np.arange(1, N_WEEKS), "observed": obs_test,
                           "plain_markov": models[("mixture", 1)]["pred_ho"][:, 1:].sum(1),
                           f"mixture_K{k_max_mix}": models[("mixture", k_max_mix)]["pred_ho"][:, 1:].sum(1),
                           f"hmm_K{hmm_k}": models[("hmm", hmm_k)]["pred_ho"][:, 1:].sum(1),
                           f"hmm_K{best_hmm}": models[("hmm", best_hmm)]["pred_ho"][:, 1:].sum(1),
                           "semi_markov": act(sm_hold, "duration_tenure")})
    curves.to_csv(tables / "segments_curves_holdout.csv", index=False)

    # Parameters.
    params = []
    for k in range(mix.K):
        st = stationary(mix.A[k])
        for a in range(3):
            params.append({"type": k, "weight": mix.w[k], "p_week0_engaged": mix.pi[k, 3:].sum(),
                           "p_active_week1": mix.pi[k, [1, 2, 4, 5]].sum(), "from": STATES[a],
                           **{f"to_{STATES[b]}": mix.A[k, a, b] for b in range(3)},
                           "stationary_active": st[1:].sum()})
    pd.DataFrame(params).to_csv(tables / "segments_mixture_params.csv", index=False)
    hp = pd.DataFrame(hmm.A, columns=[f"to {n}" for n in names])
    hp.insert(0, "regime", names)
    hp.insert(1, "initial", hmm.pi)
    for o in range(3):
        hp[f"emits {STATES[o]}"] = hmm.B[:, o]
    hp["expected_weeks_in_regime"] = 1 / (1 - np.diag(hmm.A))
    hp.to_csv(tables / "segments_hmm_params.csv", index=False)

    # Duration curves implied by each model vs. observed.
    dur = implied_duration_curves(U_all, w_all, mix, hmm)
    dur.to_csv(tables / "segments_duration_implied.csv", index=False)
    dur_err = pd.DataFrame([{"state": st, "model": m,
                             "weighted_mae_pp": float((np.abs(g[m] - g.observed) * g.weeks).sum() / g.weeks.sum() * 100)}
                            for st, g in dur.groupby("state") for m in ("mixture", "hmm")])
    dur_err.to_csv(tables / "segments_duration_error.csv", index=False)
    log("implied duration curves done")

    # Regime shares by week (smoothed posteriors).
    _, alpha, beta, _ = hmm._fb(U_all)
    gamma = alpha * beta
    gamma /= gamma.sum(2, keepdims=True)
    shares = (gamma * w_all[None, :, None]).sum(1) / w_all.sum()   # (T,K)
    regimes = pd.DataFrame(shares, columns=names)
    regimes.insert(0, "week", range(N_WEEKS))
    regimes.to_csv(tables / "segments_regime_shares.csv", index=False)

    # Where players go when onboarding ends: most likely regime in week 1, profiled by week 0.
    exit_regime = np.array(names)[gamma[1].argmax(1)][inv_all]
    G = df[df.week < N_WEEKS].pivot(index="player", columns="week", values="n_games").reindex(players).to_numpy()
    w0 = df[df.week == 0].set_index("player").reindex(players)
    prof = pd.DataFrame({
        "regime_after_onboarding": exit_regime, "week0_games": G[:, 0],
        "week0_engaged": (w0.state == 2).to_numpy(),
        "week0_score_vs_expected_pg": (w0.perf_vs_expected / w0.n_games).to_numpy(),
        "week0_bullet_share": w0.share_bullet.to_numpy(), "week0_tournament_share": w0.share_tournament.to_numpy(),
        "week0_3plus_streak": (w0.max_session_loss_streak >= 3).to_numpy(),
        "active_weeks_1_51": (X[:, 1:] > 0).sum(1), "active_in_weeks_12_15": (X[:, 12:16] > 0).any(1)})
    profiles = prof.groupby("regime_after_onboarding").agg(
        players=("week0_games", "size"), week0_games_median=("week0_games", "median"),
        week0_engaged=("week0_engaged", "mean"), week0_score_vs_expected_pg=("week0_score_vs_expected_pg", "mean"),
        week0_bullet_share=("week0_bullet_share", "mean"), week0_tournament_share=("week0_tournament_share", "mean"),
        week0_3plus_streak=("week0_3plus_streak", "mean"), active_weeks_mean=("active_weeks_1_51", "mean"),
        active_in_weeks_12_15=("active_in_weeks_12_15", "mean"))
    profiles.insert(1, "share", profiles.players / profiles.players.sum())
    order = ["Gone", "Lapsed", "Casual regular", "Burst", "Core", "Onboarding"]
    profiles = profiles.reindex([n for n in order if n in profiles.index] or profiles.index)
    profiles.to_csv(tables / "segments_onboarding_exit_profiles.csv")

    # How early can we tell who will still be playing in weeks 12-15? (held-out cohort, model fit
    # on earlier cohorts) vs. a simple baseline: games played so far.
    hmm_tr = models[("hmm", hmm_k)]["train"]
    y = (X_te[:, 12:16] > 0).any(1)
    G_te = G[is_test]
    early = []
    for m in (0, 1, 2, 4, 8):
        p = p_active_window(hmm_tr, X_te, m)
        games = G_te[:, :m + 1].sum(1)
        top = p >= np.quantile(p, 0.9)
        early.append({"weeks_observed": f"0-{m}", "auc_hmm": auc(p, y), "auc_games_so_far": auc(games, y),
                      "auc_active_weeks_so_far": auc((X_te[:, :m + 1] > 0).sum(1), y),
                      "base_rate": float(y.mean()), "top10pct_rate_hmm": float(y[top].mean()),
                      "mean_predicted": float(p.mean())})
    early = pd.DataFrame(early)
    early.to_csv(tables / "segments_early_identification.csv", index=False)

    figures(figs, curves, k_max_mix, hmm_k, best_hmm, dur, regimes, names)
    log("done")
    return {"comp": comp, "best_mix": best_mix, "best_hmm": best_hmm, "hmm_k": hmm_k, "mix": mix, "dur_err": dur_err,
            "hmm": hmm, "names": names, "hp": hp, "curves": curves, "dur": dur, "regimes": regimes,
            "profiles": profiles, "early": early}


def figures(figs: Path, curves: pd.DataFrame, k_mix: int, hmm_k: int, best_hmm: int,
            dur: pd.DataFrame, regimes: pd.DataFrame, names: list[str]) -> None:
    four = SERIES + ["#eda100", "#e87ba4"]
    # 1. Holdout curves.
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=200)
    ax.plot(curves.week, curves.observed, color=INK, linewidth=2.5, label="observed (held-out cohort)")
    lines = [(f"mixture_K{k_mix}", f"{k_mix} fixed player types (mixture)"),
             (f"hmm_K{hmm_k}", f"hidden Markov model, {hmm_k} regimes"),
             (f"hmm_K{best_hmm}", f"hidden Markov model, {best_hmm} regimes"),
             ("semi_markov", "semi-Markov chain (Part 2)")]
    if best_hmm == hmm_k:
        lines.pop(2)
    for i, (col, name) in enumerate(lines):
        ax.plot(curves.week, curves[col], color=four[i], linewidth=2, linestyle=(0, (4, 2)), label=name)
    style(ax, "Share active each week, held-out cohort: data vs. models fit on earlier cohorts")
    pct(ax)
    ax.set_ylim(0, None)
    ax.set_xlabel("weeks since first game", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper right")
    fig.tight_layout()
    fig.savefig(figs / "segments_fit.png", facecolor=SURFACE)
    plt.close(fig)

    # 2. Duration curves: observed vs. what each model implies.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=200)
    for ax, st, title in zip(axes, ["Engaged", "Inactive"],
                             ["Currently Engaged", "Currently Inactive"]):
        d = dur[(dur.state == st) & (dur.weeks >= 200)]
        ax.plot(d.dur, d.observed, color=INK, linewidth=2.5, marker="o", markersize=4, label="observed")
        ax.plot(d.dur, d.mixture, color=SERIES[0], linewidth=2, linestyle=(0, (4, 2)),
                label=f"implied by {k_mix} fixed types")
        ax.plot(d.dur, d.hmm, color=SERIES[1], linewidth=2, linestyle=(0, (4, 2)),
                label=f"implied by HMM ({hmm_k} regimes)")
        style(ax, title)
        pct(ax)
        ax.set_ylim(0, 1 if st == "Engaged" else None)
        ax.set_xlabel("weeks already in this state (12 = 12 or more)", color=INK2, fontsize=9)
    axes[0].set_ylabel("chance of playing next week", color=INK2, fontsize=9)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower right")
    fig.suptitle("Duration dependence: observed vs. what each model implies (one-step-ahead, past data only)",
                 x=0.01, ha="left", color=INK, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(figs / "segments_duration.png", facecolor=SURFACE)
    plt.close(fig)

    # 3. Regime shares over time (Gone omitted: it is everyone else).
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=200)
    r = regimes[regimes.week >= 1]
    show = [n for n in ["Lapsed", "Casual regular", "Burst", "Core", "Onboarding"] if n in names] or \
        [n for n in names if n != names[0]]
    for i, n in enumerate(show):
        ax.plot(r.week, r[n], color=four[i % len(four)], linewidth=2, label=n)
    style(ax, "Share of new players in each hidden regime, by week (log scale; the rest are Gone)")
    ax.set_yscale("log")
    ticks = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
    ax.set_yticks(ticks, [f"{t * 100:g}%" for t in ticks])
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.set_ylim(0.0005, 0.35)
    ax.set_xlabel("weeks since first game", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper right", ncol=3)
    fig.tight_layout()
    fig.savefig(figs / "segments_regimes.png", facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--panel", type=Path, default=Path("data/panel"))
    p.add_argument("--out", type=Path, default=Path("reports"))
    p.add_argument("--k-max-mix", type=int, default=10, help="largest number of fixed types")
    p.add_argument("--k-max-hmm", type=int, default=8, help="largest number of hidden states")
    p.add_argument("--hmm-k", type=int, default=6, help="number of hidden regimes to interpret")
    p.add_argument("--starts", type=int, default=4, help="EM random restarts per model")
    p.add_argument("--time-budget", type=float, default=None,
                   help="seconds; stop before fitting a new model after this and exit with code 3. "
                        "Fitted models are cached, so rerunning continues where it stopped.")
    args = p.parse_args()
    try:
        r = run(args.panel, args.out, args.k_max_mix, args.k_max_hmm, args.starts, args.time_budget, args.hmm_k)
    except TimeoutError as e:
        print(e)
        raise SystemExit(3)
    pd.set_option("display.width", 220, "display.max_columns", 30)
    c = r["comp"].copy()
    c["curve_mae_in_sample"] *= 100
    c["curve_mae_holdout"] *= 100
    print("\nmodel comparison (curve MAE in pp)\n", c.round(3).to_string(index=False))
    print(f"\nlowest BIC: mixture K={r['best_mix']}, HMM K={r['best_hmm']}; interpreting HMM K={r['hmm_k']}")
    print("\nHMM regimes\n", r["hp"].round(3).to_string(index=False))
    print("\nimplied duration curve error (weeks-weighted)\n", r["dur_err"].round(2).to_string(index=False))
    d = r["dur"]
    for st in ("Engaged", "Inactive", "Casual"):
        print(f"\nduration curve, currently {st}\n",
              d[d.state == st][["dur", "weeks", "observed", "mixture", "hmm"]].round(4).to_string(index=False))
    print("\nregime shares\n", r["regimes"].iloc[[0, 1, 2, 4, 8, 12, 26, 51]].round(4).to_string(index=False))
    print("\nprofiles by regime after onboarding\n", r["profiles"].round(3).to_string())
    print("\nearly identification (held-out cohort)\n", r["early"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
