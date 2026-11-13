"""Part 4: do losing streaks cause players to stop? Within-player (fixed-effects) causal tests.

Part 2 found that weeks with a 3+ losing streak are followed by more dropout, after controls.
That comparison is across players and weeks, so it can be confounded: players differ in
skill, motivation and schedule, and a bad day can cause both losses and quitting.

This script asks a narrower, better-identified question at the game level:

    Take the same player, at the same point in a session, entering a game with the same losing
    streak, against an opponent of the same relative strength. Does LOSING this game rather than
    winning it change the chance that they stop?

  * Player fixed effects remove everything stable about the player (skill, motivation, habits).
  * Conditioning on the entering streak removes selection into long streaks: both arms share the
    same history up to the game, and only its result differs.
  * Conditioning on the pre-game expected score (Elo formula from both ratings) leaves the
    result driven by luck and short-run form.
  * A balance test checks that, given these controls, losses are not more common after
    pre-game signs of fatigue.
  * An instrumental-variable design uses only the part of the result explained by opponent
    strength (matchmaking), removing short-run form altogether.

Outcomes after each game: stop the session (no game within 30 min), quit for a week (no game in
the next 7 days) and quit for a month (30 days). Games too close to the end of the data to observe
the outcome are dropped for that outcome.

A second part re-estimates Part 2's weekly result (3+ streak -> inactive next week) with player
fixed effects.

Outputs: data/analysis/tilt_*.parquet, reports/tables/tilt_*.csv, reports/figures/tilt_*.png

Example:
    python src/tilt.py --panel data/panel --out reports
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

from transitions import GRID, INK, INK2, SERIES, SURFACE, load, plt, style

STREAK_BINS = ["0", "1", "2", "3", "4+"]       # losses in a row in this session, entering the game
OUTCOMES = {"stop_session": "stops the session", "quit7": "no game in the next 7 days",
            "quit30": "no game in the next 30 days"}
SAMPLES = {"cohort": "new-player cohort (all their games)", "all20": "20% random sample of all players"}


# ----------------------------------------------------------------------------------------------
# 1. Game-level analysis table (DuckDB)
# ----------------------------------------------------------------------------------------------
def build_games(panel: Path, dest: Path, sample: str) -> Path:
    """One row per player-game with outcomes, the entering streak and pre-game controls."""
    dest.mkdir(parents=True, exist_ok=True)
    f = dest / f"tilt_{sample}.parquet"
    if f.exists():
        return f
    where = "p.in_cohort" if sample == "cohort" else "md5_number(g.player) % 5 = 0"
    con = duckdb.connect()
    con.execute("SET TimeZone = 'UTC'")
    con.execute(f"""
    COPY (
    WITH b AS (SELECT max(ts) AS data_end FROM read_parquet('{(panel / 'player_games.parquet').as_posix()}')),
    g AS (
        SELECT g.*, p.in_cohort, p.n_games AS lifetime_games
        FROM read_parquet('{(panel / 'player_games.parquet').as_posix()}') g
        JOIN read_parquet('{(panel / 'players.parquet').as_posix()}') p USING (player)
        WHERE {where}
    ),
    s AS (
        SELECT g.*,
            row_number() OVER w AS pos_in_session,
            date_diff('second', first_value(ts) OVER w, ts) / 60.0 AS min_into_session,
            coalesce(lag(session_loss_streak) OVER w, 0) AS entering_streak,
            date_diff('second', ts, b.data_end) AS secs_to_end
        FROM g, b
        WINDOW w AS (PARTITION BY player, session_no ORDER BY game_no)
    )
    SELECT player, game_no::INT AS game_no, session_no, pos_in_session::INT AS pos_in_session,
        min_into_session::FLOAT AS min_into_session, entering_streak::SMALLINT AS entering_streak,
        score::FLOAT AS score, expected_score::FLOAT AS expected_score, elo, opp_elo, speed, is_tournament,
        (gap_s / 60.0)::FLOAT AS mins_since_prev,
        hour(ts)::TINYINT AS hour_utc, dayofweek(ts)::TINYINT AS dow,
        in_cohort, lifetime_games::INT AS lifetime_games,
        CASE WHEN session_end_censored THEN NULL ELSE session_end END AS stop_session,
        CASE WHEN next_gap_s IS NOT NULL AND next_gap_s <= 7 * 86400 THEN false
             WHEN secs_to_end > 7 * 86400 THEN true END AS quit7,
        CASE WHEN next_gap_s IS NOT NULL AND next_gap_s <= 30 * 86400 THEN false
             WHEN secs_to_end > 30 * 86400 THEN true END AS quit30
    FROM s
    WHERE NOT abandoned AND expected_score IS NOT NULL
    ) TO '{f.as_posix()}' (FORMAT parquet, COMPRESSION zstd)
    """)
    return f


# ----------------------------------------------------------------------------------------------
# 2. Fixed-effects regression with player-clustered standard errors (written out in numpy)
# ----------------------------------------------------------------------------------------------
def demean(M: np.ndarray, g: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Subtract group means column by column, in place (the within transformation)."""
    for j in range(M.shape[1]):
        M[:, j] -= (np.bincount(g, weights=M[:, j], minlength=len(counts)) / counts)[g]
    return M


def cluster_meat(S: np.ndarray, g: np.ndarray, n_groups: int) -> np.ndarray:
    """sum over clusters of (sum_i s_i)(sum_i s_i)' for score rows S."""
    G = np.column_stack([np.bincount(g, weights=S[:, j], minlength=n_groups) for j in range(S.shape[1])])
    return G.T @ G


def fe_ols(y: np.ndarray, X: np.ndarray, names: list[str], g: np.ndarray,
           endog: np.ndarray | None = None, Z: np.ndarray | None = None, endog_name: str = "") -> dict:
    """OLS or 2SLS with player fixed effects absorbed (within transformation).

    Standard errors are clustered by player with the usual G/(G-1) * (N-1)/(N-K) small-sample
    factor (the fixed effects are nested in the clusters, so they are not counted in K).
    For 2SLS, `endog` (n,) is instrumented by Z (n, m); X holds the included controls."""
    keep = np.isfinite(y) & np.isfinite(X).all(1)
    if endog is not None:
        keep &= np.isfinite(endog) & np.isfinite(Z).all(1)
    gg = np.unique(g[keep], return_inverse=True)[1]
    counts = np.bincount(gg)
    multi = counts[gg] >= 2                                      # singletons carry no within variation
    keep_idx = np.flatnonzero(keep)[multi]
    gg = np.unique(gg[multi], return_inverse=True)[1]
    counts = np.bincount(gg).astype(float)
    y = demean(y[keep_idx].astype(float)[:, None], gg, counts)[:, 0]
    X = demean(X[keep_idx].astype(float), gg, counts)
    varies = X.std(0) > 1e-9                                     # drop controls with no within variation
    X, names = X[:, varies], [n for n, v in zip(names, varies) if v]
    G_n, N = len(counts), len(y)
    out = {"n": N, "n_players": G_n}
    if endog is None:
        XtX = X.T @ X
        beta = np.linalg.solve(XtX, X.T @ y)
        u = y - X @ beta
        B = np.linalg.inv(XtX)
        meat = cluster_meat(X * u[:, None], gg, G_n)
        coef_names = names
    else:
        D = demean(endog[keep_idx].astype(float)[:, None], gg, counts)[:, 0]
        Zm = demean(Z[keep_idx].astype(float), gg, counts)
        W = np.column_stack([Zm, X])                             # all instruments
        WtW_inv = np.linalg.inv(W.T @ W)
        pi = WtW_inv @ (W.T @ D)
        Dhat = W @ pi
        m = Zm.shape[1]
        Vfs = WtW_inv @ cluster_meat(W * (D - Dhat)[:, None], gg, G_n) @ WtW_inv
        out["first_stage_F"] = float(pi[:m] @ np.linalg.solve(Vfs[:m, :m], pi[:m]) / m)
        out["first_stage_coef"] = float(pi[0])
        Xh = np.column_stack([Dhat, X])
        XtX = Xh.T @ Xh
        B = np.linalg.inv(XtX)
        beta = B @ (Xh.T @ y)
        u = y - np.column_stack([D, X]) @ beta                   # residuals use the actual regressor
        meat = cluster_meat(Xh * u[:, None], gg, G_n)
        coef_names = [endog_name] + names
    K = len(beta)
    V = G_n / (G_n - 1) * (N - 1) / (N - K) * B @ meat @ B
    se = np.sqrt(np.diag(V))
    out["coef"] = pd.DataFrame({"term": coef_names, "est": beta, "se": se,
                                "ci_lo": beta - 1.96 * se, "ci_hi": beta + 1.96 * se,
                                "p": 2 * stats.norm.sf(np.abs(beta / se))})
    out["V"] = pd.DataFrame(V, index=coef_names, columns=coef_names)
    return out


def ols_pooled(y: np.ndarray, X: np.ndarray, names: list[str], g: np.ndarray) -> dict:
    """Pooled OLS (no fixed effects), clustered by player."""
    keep = np.isfinite(y) & np.isfinite(X).all(1)
    y, X, gg = y[keep], X[keep].astype(float), np.unique(g[keep], return_inverse=True)[1]
    X = np.column_stack([np.ones(len(X)), X])
    names = ["const"] + names
    B = np.linalg.inv(X.T @ X)
    beta = B @ (X.T @ y)
    u = y - X @ beta
    G_n, N, K = int(gg.max()) + 1, len(y), len(beta)
    V = G_n / (G_n - 1) * (N - 1) / (N - K) * B @ cluster_meat(X * u[:, None], gg, G_n) @ B
    se = np.sqrt(np.diag(V))
    return {"n": N, "n_players": G_n, "V": pd.DataFrame(V, index=names, columns=names),
            "coef": pd.DataFrame({"term": names, "est": beta, "se": se, "ci_lo": beta - 1.96 * se,
                                  "ci_hi": beta + 1.96 * se, "p": 2 * stats.norm.sf(np.abs(beta / se))})}


# ----------------------------------------------------------------------------------------------
# 3. Memory-light fixed-effects design: demean once, solve every model from M'M
# ----------------------------------------------------------------------------------------------
class Within:
    """Player-demeaned regressors for one estimation sample, stored as float32.

    All models on the sample are subsets of the columns, so their coefficients come from the
    cross-product matrix M'M (accumulated in float64). Residuals and cluster scores are computed
    one column at a time, so memory stays at roughly one copy of the design."""

    CHUNK = 250_000

    def __init__(self, cols: list[tuple[str, callable]], ys: dict[str, np.ndarray], g: np.ndarray,
                 rows: np.ndarray):
        keep = rows.copy()
        for v in ys.values():
            keep &= np.isfinite(v)
        gg = np.unique(g[keep], return_inverse=True)[1]
        multi = np.bincount(gg)[gg] >= 2                           # singletons carry no within variation
        self.rows = np.flatnonzero(keep)[multi]
        self.g = np.unique(gg[multi], return_inverse=True)[1]
        self.counts = np.bincount(self.g).astype(float)
        self.n, self.G = len(self.rows), len(self.counts)
        self.names = [n for n, _ in cols]
        self.M = np.empty((self.n, len(cols)), np.float32)
        for j, (_, fn) in enumerate(cols):
            self.M[:, j] = self._demean(np.asarray(fn(), dtype=float)[self.rows])
        self.Y = {k: self._demean(np.asarray(v, dtype=float)[self.rows]) for k, v in ys.items()}
        k = len(cols)
        self.MM = np.zeros((k, k))
        self.My = {y: np.zeros(k) for y in self.Y}
        for c in range(0, self.n, self.CHUNK):
            Mc = self.M[c:c + self.CHUNK].astype(float)
            self.MM += Mc.T @ Mc
            for y, v in self.Y.items():
                self.My[y] += Mc.T @ v[c:c + self.CHUNK]

    def _demean(self, v: np.ndarray) -> np.ndarray:
        return v - (np.bincount(self.g, weights=v, minlength=self.G) / self.counts)[self.g]

    def _ix(self, xs: list[str]) -> list[int]:
        return [self.names.index(x) for x in xs]

    def _dot(self, ix: list[int], beta: np.ndarray) -> np.ndarray:
        out = np.empty(self.n)
        for c in range(0, self.n, self.CHUNK):
            out[c:c + self.CHUNK] = self.M[c:c + self.CHUNK, ix].astype(float) @ beta
        return out

    def _cluster_sums(self, vecs: list[np.ndarray], u: np.ndarray) -> np.ndarray:
        return np.column_stack([np.bincount(self.g, weights=v * u, minlength=self.G) for v in vecs])

    def _finish(self, names, beta, B, S) -> dict:
        K = len(beta)
        V = self.G / (self.G - 1) * (self.n - 1) / (self.n - K) * B @ (S.T @ S) @ B
        se = np.sqrt(np.diag(V))
        return {"n": self.n, "n_players": self.G, "V": pd.DataFrame(V, index=names, columns=names),
                "coef": pd.DataFrame({"term": names, "est": beta, "se": se, "ci_lo": beta - 1.96 * se,
                                      "ci_hi": beta + 1.96 * se, "p": 2 * stats.norm.sf(np.abs(beta / se))})}

    def _drop_constant(self, xs: list[str]) -> list[str]:
        return [x for x in xs if self.MM[self.names.index(x), self.names.index(x)] > 1e-6 * self.n]

    def ols(self, y: str, xs: list[str]) -> dict:
        xs = self._drop_constant(xs)
        ix = self._ix(xs)
        B = np.linalg.inv(self.MM[np.ix_(ix, ix)])
        beta = B @ self.My[y][ix]
        u = self.Y[y] - self._dot(ix, beta)
        S = self._cluster_sums([self.M[:, j] for j in ix], u)
        return self._finish(xs, beta, B, S)

    def iv(self, y: str, endog: str, instr: list[str], exog: list[str]) -> dict:
        """2SLS from the cross-products: endog instrumented by instr, exog included."""
        exog = self._drop_constant(exog)
        wi, xi, di = self._ix(instr + exog), self._ix(exog), self.names.index(endog)
        Gww = self.MM[np.ix_(wi, wi)]
        Gww_inv = np.linalg.inv(Gww)
        pi = Gww_inv @ self.MM[wi, di]                            # first stage
        Dhat = self._dot(wi, pi)
        D = self.M[:, di].astype(float)
        m = len(instr)
        Sfs = self._cluster_sums([self.M[:, j] for j in wi], D - Dhat)
        n, K = self.n, len(wi)
        Vfs = self.G / (self.G - 1) * (n - 1) / (n - K) * Gww_inv @ (Sfs.T @ Sfs) @ Gww_inv
        F = float(pi[:m] @ np.linalg.solve(Vfs[:m, :m], pi[:m]) / m)
        Gwx = self.MM[np.ix_(wi, xi)]
        A = np.block([[np.atleast_2d(pi @ Gww @ pi), (pi @ Gwx)[None, :]],
                      [(pi @ Gwx)[:, None], self.MM[np.ix_(xi, xi)]]])
        b = np.concatenate([[pi @ self.My[y][wi]], self.My[y][xi]])
        B = np.linalg.inv(A)
        beta = B @ b
        u = self.Y[y] - beta[0] * D - self._dot(xi, beta[1:])    # residuals use the actual regressor
        S = self._cluster_sums([Dhat] + [self.M[:, j] for j in xi], u)
        res = self._finish([endog] + exog, beta, B, S)
        res["first_stage_F"], res["first_stage_coef"] = F, float(pi[0])
        return res


def prepare(f: Path) -> pd.DataFrame:
    cols = ["player", "game_no", "pos_in_session", "min_into_session", "entering_streak", "score",
            "expected_score", "elo", "speed", "is_tournament", "mins_since_prev", "hour_utc", "dow"] + list(OUTCOMES)
    d = pd.read_parquet(f, columns=cols)
    d["pid"] = pd.factorize(d.pop("player"))[0]
    d["streak_bin"] = np.minimum(d.entering_streak, 4).astype(np.int8)
    for c in OUTCOMES:
        d[c] = d[c].astype(float)
    return d


EXPERIENCE = [(1, 10, "games 1-10"), (11, 50, "11-50"), (51, 200, "51-200"), (201, 10**9, "201+")]  # game number


def columns(d: pd.DataFrame) -> dict[str, list[tuple[str, callable]]]:
    """Named regressor builders, grouped. Built lazily so only one float64 column exists at a time.

    Controls are all pre-game. Hour-of-day (UTC) dummies combined with player fixed effects stand
    in for local time of day, since each player's time zone is fixed."""
    loss = lambda: (d.score == 0).to_numpy(float)                 # noqa: E731
    draw = lambda: (d.score == 0.5).to_numpy(float)               # noqa: E731
    sb = d.streak_bin.to_numpy()
    e = d.expected_score.to_numpy(float)
    G: dict[str, list] = {}
    G["result"] = [("loss", loss), ("draw", draw)]
    G["result_by_streak"] = []
    for k, lab in enumerate(STREAK_BINS):
        G["result_by_streak"] += [(f"loss | entering {lab}", lambda k=k: loss() * (sb == k)),
                                  (f"draw | entering {lab}", lambda k=k: draw() * (sb == k))]
    G["streak"] = [(f"entering streak {STREAK_BINS[k]}", lambda k=k: (sb == k).astype(float)) for k in range(1, 5)]
    pos = np.minimum(d.pos_in_session.to_numpy(), 20)
    G["session"] = [(f"game {k}+ in session", lambda k=k: (pos >= k).astype(float)) for k in (2, 3, 4, 6, 11)]
    G["session"] += [("log minutes into session", lambda: np.log1p(d.min_into_session.to_numpy(float)))]
    G["player_state"] = [("log games played so far", lambda: np.log1p(d.game_no.to_numpy(float))),
                         ("own rating (per 100)", lambda: (d.elo.to_numpy(float) - 1500) / 100)]
    G["expected"] = [("expected score", lambda: e)] + \
        [(f"expected score spline {kn}", lambda kn=kn: np.maximum(e - kn, 0)) for kn in (0.2, 0.35, 0.5, 0.65, 0.8)]
    G["game_type"] = [(f"speed {s}", lambda s=s: (d.speed == s).to_numpy(float)) for s in ("bullet", "rapid", "classical")]
    G["game_type"] += [("tournament", lambda: d.is_tournament.to_numpy(float))]
    G["time"] = [(f"hour {h}", lambda h=h: (d.hour_utc == h).to_numpy(float)) for h in range(1, 24)]
    G["time"] += [(f"dow {w}", lambda w=w: (d.dow == w).to_numpy(float)) for w in range(1, 7)]
    G["points"] = [("points", lambda: d.score.to_numpy(float))]
    gn = d.game_no.to_numpy()
    G["experience"] = []
    for i, (lo, hi, lab) in enumerate(EXPERIENCE):
        inb = lambda lo=lo, hi=hi: ((gn >= lo) & (gn <= hi)).astype(float)  # noqa: E731
        G["experience"] += [(f"loss | {lab}", lambda f=inb: loss() * f()), (f"draw | {lab}", lambda f=inb: draw() * f()),
                            (f"points | {lab}", lambda f=inb: d.score.to_numpy(float) * f()),
                            (f"expected | {lab}", lambda f=inb: e * f())]
        if i < len(EXPERIENCE) - 1:
            G["experience"] += [(f"bin | {lab}", inb)]
    return G


def names_of(G: dict, *groups: str) -> list[str]:
    return [n for grp in groups for n, _ in G[grp]]


CONTROL_GROUPS = ("streak", "session", "player_state", "expected", "game_type", "time")
CONTROL_GROUPS_NO_EXPECTED = ("streak", "session", "player_state", "game_type", "time")


# ----------------------------------------------------------------------------------------------
# 4. Analyses
# ----------------------------------------------------------------------------------------------
def common_rows(d: pd.DataFrame) -> np.ndarray:
    """Games where all three outcomes are observed, so every outcome uses the same games."""
    return np.isfinite(d[list(OUTCOMES)].to_numpy()).all(1)


def lincom(res: dict, weights: dict[str, float]) -> tuple[float, float]:
    c = res["coef"].set_index("term")
    w = pd.Series(weights)
    est = float((c.est.reindex(w.index) * w).sum())
    V = res["V"].loc[w.index, w.index].to_numpy()
    return est, float(np.sqrt(w.to_numpy() @ V @ w.to_numpy()))


def row(res: dict, term: str, **extra) -> dict:
    r = res["coef"].set_index("term").loc[term]
    return {**extra, "est": r.est, "se": r.se, "ci_lo": r.ci_lo, "ci_hi": r.ci_hi, "p": r.p,
            "n": res["n"], "n_players": res["n_players"]}


def main_models(d: pd.DataFrame, sample: str, log) -> dict[str, pd.DataFrame]:
    """Design ladder, overall effect, IV on all games and (cohort) experience heterogeneity,
    all from one demeaned design on the common sample."""
    G = columns(d)
    rows_ok = common_rows(d)
    groups = ["result", "result_by_streak", *CONTROL_GROUPS, "points"]
    if sample == "cohort":
        groups.append("experience")
    W = Within([c for grp in groups for c in G[grp]], {o: d[o].to_numpy() for o in OUTCOMES},
               d.pid.to_numpy(), rows_ok)
    log(f"[{sample}] design built: {W.n:,} games, {W.G:,} players, {len(W.names)} columns")
    sbw = d.streak_bin[rows_ok].value_counts(normalize=True)
    tilt_w = {f"loss | entering {STREAK_BINS[k]}": sbw.get(k, 0) / sbw[sbw.index >= 2].sum() for k in (2, 3, 4)}
    tilt_w["loss | entering 0"] = -1.0
    ladder, overall, ivr, het = [], [], [], []
    by_streak = names_of(G, "result_by_streak")
    controls_all = names_of(G, *CONTROL_GROUPS)
    controls_noexp = names_of(G, *CONTROL_GROUPS_NO_EXPECTED)
    is_win = (d.score.to_numpy() == 1)
    for out in OUTCOMES:
        y = d[out].to_numpy()
        # 1. pooled (no FE), streak dummies only
        Xp = np.column_stack([fn() for _, fn in G["result_by_streak"] + G["streak"]])[rows_ok]
        res1 = ols_pooled(y[rows_ok], Xp, by_streak + names_of(G, "streak"), d.pid.to_numpy()[rows_ok])
        del Xp
        res2 = W.ols(out, by_streak + names_of(G, "streak"))
        res3 = W.ols(out, by_streak + controls_all)
        for spec, res in (("1 pooled, no controls", res1), ("2 player FE", res2), ("3 player FE + controls", res3)):
            for lab in STREAK_BINS:
                ladder.append(row(res, f"loss | entering {lab}", sample=sample, outcome=out, spec=spec,
                                  entering_streak=lab))
            est, se = lincom(res, tilt_w)
            ladder.append({"sample": sample, "outcome": out, "spec": spec, "entering_streak": "2+ minus 0",
                           "est": est, "se": se, "ci_lo": est - 1.96 * se, "ci_hi": est + 1.96 * se,
                           "p": 2 * stats.norm.sf(abs(est / se)), "n": res["n"], "n_players": res["n_players"]})
        base = float(np.nanmean(y[rows_ok & is_win]))
        Xp = np.column_stack([fn() for _, fn in G["result"]])[rows_ok]
        for spec, res in (("pooled, no controls", ols_pooled(y[rows_ok], Xp, ["loss", "draw"], d.pid.to_numpy()[rows_ok])),
                          ("player FE + controls", W.ols(out, ["loss", "draw"] + controls_all))):
            r = row(res, "loss", sample=sample, outcome=out, spec=spec)
            overall.append({**r, "base_rate_after_win": base, "relative": r["est"] / base})
        del Xp
        # IV on all games, and the matching OLS (per point; reported as loss vs win = -1 x)
        resiv = W.iv(out, "points", ["expected score"], controls_noexp)
        resols = W.ols(out, ["points"] + controls_all)
        ivr.append(iv_row(resiv, resols, sample, out, "all games"))
        if sample == "cohort":
            labs = [lab for *_, lab in EXPERIENCE]
            bins = [f"bin | {lab}" for lab in labs[:-1]]
            reso = W.ols(out, [f"loss | {lab}" for lab in labs] + [f"draw | {lab}" for lab in labs] + bins + controls_all)
            for lo, hi, lab in EXPERIENCE:
                m = (d.game_no >= lo) & (d.game_no <= hi)
                het.append({**row(reso, f"loss | {lab}", sample=sample, outcome=out, experience=lab),
                            "games": int((m & rows_ok).sum()),
                            "base_rate_after_win": float(np.nanmean(y[rows_ok & is_win & m.to_numpy()]))})
        log(f"[{sample}] models done for {out}")
    out = {"ladder": pd.DataFrame(ladder), "overall": pd.DataFrame(overall), "iv": pd.DataFrame(ivr)}
    if het:
        out["het"] = pd.DataFrame(het)
    del W
    return out


def iv_row(resiv: dict, resols: dict, sample: str, out: str, subset: str) -> dict:
    r = resiv["coef"].set_index("term").loc["points"]
    o = resols["coef"].set_index("term").loc["points"]
    return {"sample": sample, "outcome": out, "subset": subset,
            "iv_loss_effect": -r.est, "iv_se": r.se, "iv_ci_lo": -r.est - 1.96 * r.se, "iv_ci_hi": -r.est + 1.96 * r.se,
            "first_stage_F": resiv["first_stage_F"], "first_stage_coef": resiv["first_stage_coef"],
            "ols_fe_loss_effect": -o.est, "ols_fe_se": o.se, "n": resiv["n"], "n_players": resiv["n_players"]}


def iv_subsets(d: pd.DataFrame, sample: str, log) -> pd.DataFrame:
    """IV vs OLS within arena games (pairing by the tournament system) and casual games (players
    pick opponents from seeks), and (cohort) within each experience bin."""
    G = columns(d)
    rows_ok = common_rows(d)
    groups = ["points", *CONTROL_GROUPS]
    subsets = {"arena tournaments (automatic pairing)": d.is_tournament.to_numpy(),
               "casual pairing (players choose opponents)": ~d.is_tournament.to_numpy()}
    if sample == "cohort":
        for lo, hi, lab in EXPERIENCE:
            subsets[f"experience: {lab}"] = ((d.game_no >= lo) & (d.game_no <= hi)).to_numpy()
    rows = []
    for sub, m in subsets.items():
        W = Within([c for grp in groups for c in G[grp]], {o: d[o].to_numpy() for o in OUTCOMES},
                   d.pid.to_numpy(), rows_ok & m)
        for out in OUTCOMES:
            rows.append(iv_row(W.iv(out, "points", ["expected score"], names_of(G, *CONTROL_GROUPS_NO_EXPECTED)),
                               W.ols(out, ["points"] + names_of(G, *CONTROL_GROUPS)), sample, out, sub))
        del W
        log(f"[{sample}] IV subset done: {sub}")
    return pd.DataFrame(rows)


def balance(d: pd.DataFrame, sample: str) -> pd.DataFrame:
    """Is a loss, given expected score, entering streak, player FE and the other controls, more
    likely after pre-game signs of fatigue or rustiness? Regress each pre-game covariate on the
    result; if results are as good as random, the loss coefficient is about 0."""
    G = columns(d)
    checks = {
        "log minutes since previous game": (np.log1p(d.mins_since_prev.to_numpy(float)), []),
        "log minutes into session": (np.log1p(d.min_into_session.to_numpy(float)), ["log minutes into session"]),
        "position in session (capped at 20)": (np.minimum(d.pos_in_session, 20).to_numpy(float),
                                               [n for n, _ in G["session"] if n.startswith("game ")]),
        "own rating (per 100)": ((d.elo.to_numpy(float) - 1500) / 100, ["own rating (per 100)"]),
    }
    W = Within([c for grp in ("result", *CONTROL_GROUPS) for c in G[grp]], {k: v for k, (v, _) in checks.items()},
               d.pid.to_numpy(), common_rows(d))
    rows = []
    for name, (v, drop) in checks.items():
        xs = ["loss", "draw"] + [x for x in names_of(G, *CONTROL_GROUPS) if x not in drop]
        r = row(W.ols(name, xs), "loss", sample=sample, covariate=name)
        r["in_sd_units"] = r["est"] / float(np.nanstd(v[common_rows(d)]))
        rows.append(r)
    del W
    return pd.DataFrame(rows)


def weekly_fe(panel: Path) -> pd.DataFrame:
    """Part 2's weekly result, re-estimated as a linear probability model with and without
    player fixed effects: P(inactive next week | active this week)."""
    df = load(panel)
    t = df[(df.week >= 1) & df.next_state.notna() & (df.state > 0)].copy()
    t["inactive_next"] = (t.next_state == 0).astype(float)
    cov = ["streak3", "log_games", "perf_pg", "rating_change_100", "share_bullet", "share_tournament",
           "lost_on_time_share", "log_wis", "log_tenure", "log_cum_games"]
    t = t.dropna(subset=cov)
    g = pd.factorize(t.player)[0]
    rows = []
    for origin, sub in [("Casual", t.state == 1), ("Engaged", t.state == 2), ("Any active", t.state > 0)]:
        s = t[sub]
        X = s[cov].to_numpy(float)
        if origin == "Any active":
            X = np.column_stack([X, (s.state == 2).to_numpy(float)])
            names = cov + ["engaged this week"]
        else:
            names = cov
        y = s.inactive_next.to_numpy()
        for spec, gg in (("pooled", None), ("player FE", g[sub.to_numpy()])):
            res = ols_pooled(y, X, names, g[sub.to_numpy()]) if gg is None else fe_ols(y, X, names, gg)
            r = res["coef"].set_index("term").loc["streak3"]
            rows.append({"origin": origin, "spec": spec, "est": r.est, "se": r.se, "ci_lo": r.ci_lo,
                         "ci_hi": r.ci_hi, "p": r.p, "n_weeks": res["n"], "n_players": res["n_players"],
                         "base_rate": float(y.mean())})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------
# 5. Figures
# ----------------------------------------------------------------------------------------------
def figures(figs: Path, ladder: pd.DataFrame, het: pd.DataFrame | None, ivt: pd.DataFrame,
            overall: pd.DataFrame) -> None:
    # 1. Loss effect by entering streak, three designs, per sample.
    specs = ["1 pooled, no controls", "2 player FE", "3 player FE + controls"]
    labels = ["pooled, no controls", "+ player fixed effects", "+ pre-game controls (main)"]
    for sample in ladder["sample"].unique():
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.9), dpi=200)
        for ax, out in zip(axes, OUTCOMES):
            for i, (spec, lab) in enumerate(zip(specs, labels)):
                d = ladder[(ladder["sample"] == sample) & (ladder.outcome == out) & (ladder.spec == spec)
                           & ladder.entering_streak.isin(STREAK_BINS)]
                x = np.arange(len(STREAK_BINS)) + (i - 1) * 0.22
                ax.errorbar(x, d.est * 100, yerr=[(d.est - d.ci_lo) * 100, (d.ci_hi - d.est) * 100], fmt="o",
                            color=SERIES[i], ecolor=SERIES[i], elinewidth=1.4, markersize=5, capsize=0, label=lab)
            ax.axhline(0, color=INK2, linewidth=0.8)
            ax.set_xticks(range(len(STREAK_BINS)), STREAK_BINS)
            style(ax, f"P({OUTCOMES[out]})")
            ax.set_xlabel("losses in a row before this game (this session)", color=INK2, fontsize=8)
        axes[0].set_ylabel("effect of losing vs winning (pp)", color=INK2, fontsize=9)
        axes[0].legend(frameon=False, fontsize=7.5, labelcolor=INK2, loc="upper right")
        fig.suptitle(f"Effect of losing a game, by the losing streak entering it — {SAMPLES[sample]} (95% CI)",
                     x=0.01, ha="left", color=INK, fontsize=12, fontweight="bold")
        fig.tight_layout()
        fig.savefig(figs / f"tilt_by_streak_{sample}.png", facecolor=SURFACE)
        plt.close(fig)

    # 2. How the estimate changes as the design tightens.
    steps = [("pooled", "pooled, no controls"), ("player FE + controls", "player FE + pre-game controls"),
             ("IV all", "IV: opponent strength, all games"),
             ("IV arena", "IV: arena games (automatic pairing)")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), dpi=200, sharey=True)
    for ax, out in zip(axes, ["stop_session", "quit7"]):
        for i, sample in enumerate(overall["sample"].unique()):
            vals = []
            for key, _ in steps:
                if key.startswith("IV"):
                    sub = "all games" if key == "IV all" else "arena tournaments (automatic pairing)"
                    r = ivt[(ivt["sample"] == sample) & (ivt.outcome == out) & (ivt.subset == sub)].iloc[0]
                    vals.append((r.iv_loss_effect, r.iv_ci_lo, r.iv_ci_hi))
                else:
                    sp = "pooled, no controls" if key == "pooled" else "player FE + controls"
                    r = overall[(overall["sample"] == sample) & (overall.outcome == out) & (overall.spec == sp)].iloc[0]
                    vals.append((r.est, r.ci_lo, r.ci_hi))
            v = np.array(vals) * 100
            y = np.arange(len(steps)) + (i - 0.5) * 0.25
            ax.errorbar(v[:, 0], y, xerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt="o", color=SERIES[i],
                        ecolor=SERIES[i], elinewidth=1.5, markersize=6, label=SAMPLES[sample])
        ax.axvline(0, color=INK2, linewidth=0.8)
        ax.set_yticks(range(len(steps)), [lab for _, lab in steps])
        style(ax, f"P({OUTCOMES[out]})")
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_xlabel("effect of losing vs winning (pp)", color=INK2, fontsize=9)
    axes[0].invert_yaxis()                                       # shared y-axis: flip once
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, fontsize=8, labelcolor=INK2, loc="lower center", ncol=2)
    fig.suptitle("The effect of a loss under progressively stricter designs (95% CI)",
                 x=0.01, ha="left", color=INK, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(figs / "tilt_designs.png", facecolor=SURFACE)
    plt.close(fig)

    # 3. New players: effect by experience, OLS-FE vs IV.
    if het is not None:
        h = het[het.outcome == "quit7"].reset_index(drop=True)
        iv_e = ivt[(ivt.outcome == "quit7") & ivt.subset.str.startswith("experience")].reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(8, 3.8), dpi=200)
        x = np.arange(len(h))
        ax.errorbar(x - 0.12, h.est * 100, yerr=[(h.est - h.ci_lo) * 100, (h.ci_hi - h.est) * 100], fmt="o",
                    color=SERIES[0], ecolor=SERIES[0], elinewidth=1.5, markersize=6, label="player FE + controls")
        ax.errorbar(x + 0.12, iv_e.iv_loss_effect * 100,
                    yerr=[(iv_e.iv_loss_effect - iv_e.iv_ci_lo) * 100, (iv_e.iv_ci_hi - iv_e.iv_loss_effect) * 100],
                    fmt="s", color=SERIES[1], ecolor=SERIES[1], elinewidth=1.5, markersize=6,
                    label="IV: opponent strength")
        ax.axhline(0, color=INK2, linewidth=0.8)
        ax.set_xticks(x, [f"{e}\n(rate after a win: {b:.1%})" for e, b in zip(h.experience, h.base_rate_after_win)])
        style(ax, "New players: effect of a loss on not playing for a week (pp, 95% CI)")
        ax.set_xlabel("which of the player's games this was (1st-10th, 11th-50th, ...)", color=INK2, fontsize=9)
        ax.set_ylabel("losing vs winning (pp)", color=INK2, fontsize=9)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper right")
        fig.tight_layout()
        fig.savefig(figs / "tilt_experience.png", facecolor=SURFACE)
        plt.close(fig)


# ----------------------------------------------------------------------------------------------
def run(panel: Path, out: Path, analysis_dir: Path, samples: list[str], budget_s: float | None = None) -> dict:
    """Each (sample, part) result is cached in reports/tables/tilt_parts, so a run stopped by the
    time budget continues where it left off."""
    tables, figs, parts = out / "tables", out / "figures", out / "tables" / "tilt_parts"
    for p in (tables, figs, parts):
        p.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:5.0f}s] {m}", flush=True)  # noqa: E731

    def cached(key: str, fn) -> dict[str, pd.DataFrame]:
        files = sorted(parts.glob(f"{key}__*.csv"))
        if files:
            return {f.stem.split("__", 1)[1]: pd.read_csv(f) for f in files}
        if budget_s is not None and time.time() - t0 > budget_s:
            raise TimeoutError("time budget used; rerun the same command to continue")
        res = fn()
        res = res if isinstance(res, dict) else {"result": res}
        for k, v in res.items():
            v.to_csv(parts / f"{key}__{k}.csv", index=False)
        return res

    collected: dict[str, list[pd.DataFrame]] = {}
    for sample in samples:
        f = build_games(panel, analysis_dir, sample)
        d = None

        def data():
            nonlocal d
            if d is None:
                d = prepare(f)
                log(f"[{sample}] {len(d):,} player-games, {d.pid.nunique():,} players")
            return d

        pieces = {
            "desc": lambda: data().loc[common_rows(data())].groupby("streak_bin").agg(
                games=("score", "size"), loss_rate=("score", lambda s: (s == 0).mean()),
                **{f"{o}_rate": (o, "mean") for o in OUTCOMES}).reset_index().assign(sample=sample),
            "main": lambda: main_models(data(), sample, log),
            "balance": lambda: balance(data(), sample),
            "ivsub": lambda: iv_subsets(data(), sample, log),
        }
        for key, fn in pieces.items():
            res = cached(f"{sample}_{key}", fn)
            for k, v in res.items():
                name = {"result": key}.get(k, k)
                name = {"desc": "desc", "balance": "balance", "ivsub": "iv"}.get(name, name)
                collected.setdefault(name, []).append(v)
        del d
    collected.setdefault("weekly", []).append(cached("weekly", lambda: weekly_fe(panel))["result"])
    r = {k: pd.concat(v, ignore_index=True) for k, v in collected.items()}
    for k, v in r.items():
        v.to_csv(tables / f"tilt_{k}.csv", index=False)
    figures(figs, r["ladder"], r.get("het"), r["iv"], r["overall"])
    log("done")
    return r


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--panel", type=Path, default=Path("data/panel"))
    p.add_argument("--out", type=Path, default=Path("reports"))
    p.add_argument("--analysis-dir", type=Path, default=Path("data/analysis"))
    p.add_argument("--samples", nargs="+", default=list(SAMPLES), choices=list(SAMPLES))
    p.add_argument("--build-only", action="store_true", help="only write the game-level tables")
    p.add_argument("--time-budget", type=float, default=None,
                   help="seconds; stop before starting a new piece after this and exit with code 3. "
                        "Finished pieces are cached, so rerunning continues where it stopped.")
    args = p.parse_args()
    if args.build_only:
        for s in args.samples:
            print(build_games(args.panel, args.analysis_dir, s))
        return
    try:
        r = run(args.panel, args.out, args.analysis_dir, args.samples, args.time_budget)
    except TimeoutError as e:
        print(e)
        raise SystemExit(3)
    pd.set_option("display.width", 220, "display.max_columns", 30, "display.max_rows", 300)
    print("\ndescriptives\n", r["desc"].round(4).to_string(index=False))
    L = r["ladder"].copy()
    for c in ("est", "ci_lo", "ci_hi"):
        L[c] *= 100
    print("\neffect of losing vs winning (pp)\n",
          L.pivot_table(index=["sample", "outcome", "entering_streak"], columns="spec", values="est").round(2))
    print("\nmain spec with CIs (pp)\n", L[L.spec.str.startswith("3")][
        ["sample", "outcome", "entering_streak", "est", "ci_lo", "ci_hi", "p", "n", "n_players"]].round(3).to_string(index=False))
    print("\noverall\n", r["overall"].round(4).to_string(index=False))
    print("\nbalance\n", r["balance"].round(4).to_string(index=False))
    print("\nIV\n", r["iv"].round(4).to_string(index=False))
    if "het" in r:
        print("\nexperience\n", r["het"].round(4).to_string(index=False))
    print("\nweekly\n", r["weekly"].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
