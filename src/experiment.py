"""Part 5: size an A/B test of "protected first matches" before running it.

Part 4 found that losses early in a player's life raise the chance they stay away, and that the
instrument behind the estimate (opponent strength) is what a matchmaker controls. The obvious
experiment is to protect newcomers' first matches. Before spending an experiment on it, this
script asks how big the effect could plausibly be, and whether a test could detect it.

  1. Natural experiment. Every newcomer starts at 1500, so the strength of their first opponent is
     a draw from whoever is available. In arena tournaments the system assigns that opponent; in
     casual games the newcomer picks one, so the draw can reflect the player. Balance on later
     skill (rating after 30 games) decides which source is usable.
  2. Implied effect of the policy on the primary metric, with its confidence interval, plus who would
     absorb the losses a protective matchmaker shifts (two-sided interference).
  3. Power: baseline, variance, sample size and duration at the real enrollment rate, with and
     without regression adjustment.
  4. Simulated A/A and A/B tests replayed on the real newcomers: false positives under peeking,
     a group-sequential boundary, empirical vs analytic power, sample-ratio-mismatch detection.

Outputs: data/analysis/experiment_first_games.parquet, reports/tables/experiment_*.csv,
reports/figures/experiment_*.png

Example:
    python src/experiment.py --panel data/panel --out reports
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

from transitions import GRID, INK, INK2, SERIES, SURFACE, pct, plt, style

DAY = 86400
METRICS = {
    "return_d1_7": "plays again on days 1-7 after the first game",
    "active_week2": "plays in days 7-13 (week 2)",
    "came_back": "ever plays again after day 0",
    "games_14d": "games in the first 14 days",
}
PRIMARY = "return_d1_7"


# ----------------------------------------------------------------------------------------------
# 1. Data: one row per new player
# ----------------------------------------------------------------------------------------------
def build_first_games(panel: Path, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    f = dest / "experiment_first_games.parquet"
    if f.exists():
        return f
    pg = (panel / "player_games.parquet").as_posix()
    pl = (panel / "players.parquet").as_posix()
    con = duckdb.connect()
    con.execute("SET TimeZone = 'UTC'")
    con.execute(f"""
    COPY (
    WITH c AS (SELECT player, first_ts FROM read_parquet('{pl}') WHERE in_cohort),
    g AS (SELECT g.*, c.first_ts, date_diff('second', c.first_ts, g.ts) AS t FROM read_parquet('{pg}') g JOIN c USING (player)),
    outcomes AS (
        SELECT player,
            bool_or(t >= {DAY} AND t < 8 * {DAY}) AS return_d1_7,
            bool_or(t >= 7 * {DAY} AND t < 14 * {DAY}) AS active_week2,
            bool_or(t >= {DAY}) AS came_back,
            count(*) FILTER (WHERE t < 14 * {DAY}) AS games_14d,
            count(*) FILTER (WHERE t < {DAY}) AS games_day0,
            max(CASE WHEN game_no = 30 THEN elo + coalesce(rating_diff, 0) END) AS rating_after_30,
            count(*) AS games_total
        FROM g GROUP BY player
    ),
    first AS (SELECT * FROM g WHERE game_no = 1),
    opp AS (
        SELECT o.game_id, o.player AS opponent, o.game_no AS opp_game_no, p.first_ts AS opp_first_ts
        FROM read_parquet('{pg}') o JOIN read_parquet('{pl}') p USING (player)
    )
    SELECT f.player, f.first_ts, f.expected_score, f.score, f.elo, f.opp_elo, f.speed, f.is_tournament,
           hour(f.ts) AS hour_utc, dayofweek(f.ts) AS dow, f.lost_on_time,
           o.opp_game_no, date_diff('day', o.opp_first_ts, f.ts) AS opp_days_since_first,
           oc.* EXCLUDE (player)
    FROM first f
    JOIN outcomes oc USING (player)
    LEFT JOIN opp o ON o.game_id = f.game_id AND o.opponent = f.opponent
    ORDER BY f.first_ts
    ) TO '{f.as_posix()}' (FORMAT parquet, COMPRESSION zstd)
    """)
    return f


def load(f: Path) -> pd.DataFrame:
    d = pd.read_parquet(f)
    for m in ("return_d1_7", "active_week2", "came_back"):
        d[m] = d[m].astype(float)
    d["games_14d"] = d.games_14d.astype(float)
    d["points"] = d.score.astype(float)
    d["day"] = (d.first_ts - d.first_ts.min()).dt.total_seconds() // DAY
    return d


def pre_covariates(d: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Covariates fixed before the matchmaker picks the opponent: the player's choice of speed and
    of arena vs casual play, the hour and day of the week, and the calendar day of enrollment."""
    cols, names = [], []
    for s in ("bullet", "rapid", "classical"):
        cols.append((d.speed == s).to_numpy(float)); names.append(f"speed {s}")
    cols.append(d.is_tournament.to_numpy(float)); names.append("arena")
    for h in range(1, 24):
        cols.append((d.hour_utc == h).to_numpy(float)); names.append(f"hour {h}")
    for w in range(1, 7):
        cols.append((d.dow == w).to_numpy(float)); names.append(f"dow {w}")
    for m in sorted(d.first_ts.dt.month.unique())[1:]:
        cols.append((d.first_ts.dt.month == m).to_numpy(float)); names.append(f"month {m}")
    X = np.column_stack(cols)
    keep = X.std(0) > 0                                          # drop columns constant in this subset
    return X[:, keep], [n for n, k in zip(names, keep) if k]


# ----------------------------------------------------------------------------------------------
# 2. Regressions (heteroskedasticity-robust, one row per player)
# ----------------------------------------------------------------------------------------------
def ols(y: np.ndarray, X: np.ndarray, names: list[str]) -> pd.DataFrame:
    X = np.column_stack([np.ones(len(X)), X])
    names = ["const"] + names
    B = np.linalg.inv(X.T @ X)
    beta = B @ X.T @ y
    u = y - X @ beta
    n, k = X.shape
    V = n / (n - k) * B @ (X.T * u**2) @ X @ B                    # HC1
    se = np.sqrt(np.diag(V))
    return pd.DataFrame({"term": names, "est": beta, "se": se, "ci_lo": beta - 1.96 * se,
                         "ci_hi": beta + 1.96 * se, "p": 2 * stats.norm.sf(np.abs(beta / se))}).set_index("term")


def tsls(y: np.ndarray, endog: np.ndarray, z: np.ndarray, X: np.ndarray, names: list[str]) -> tuple[pd.DataFrame, float]:
    W = np.column_stack([np.ones(len(X)), z, X])
    pi = np.linalg.lstsq(W, endog, rcond=None)[0]
    v = endog - W @ pi
    Bw = np.linalg.inv(W.T @ W)
    Vfs = len(W) / (len(W) - W.shape[1]) * Bw @ (W.T * v**2) @ W @ Bw
    F = pi[1] ** 2 / Vfs[1, 1]
    Xh = np.column_stack([np.ones(len(X)), W @ pi, X])
    Xa = np.column_stack([np.ones(len(X)), endog, X])
    B = np.linalg.inv(Xh.T @ Xh)
    beta = B @ Xh.T @ y
    u = y - Xa @ beta
    n, k = Xh.shape
    V = n / (n - k) * B @ (Xh.T * u**2) @ Xh @ B
    se = np.sqrt(np.diag(V))
    nm = ["const", "points"] + names
    return pd.DataFrame({"term": nm, "est": beta, "se": se, "ci_lo": beta - 1.96 * se, "ci_hi": beta + 1.96 * se,
                         "p": 2 * stats.norm.sf(np.abs(beta / se))}).set_index("term"), float(F)


SUBSETS = {"arena (opponent assigned)": lambda d: d.is_tournament.to_numpy(),
           "casual (opponent chosen)": lambda d: ~d.is_tournament.to_numpy(),
           "all first games": lambda d: np.ones(len(d), bool)}


def natural_experiment(d: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Effect of the first opponent's strength (Elo-expected score of the first game) on each metric,
    separately where the opponent is assigned (arena) and where the newcomer chooses (casual)."""
    eff, bal, fs, binned = [], [], [], []
    for sub, mfn in SUBSETS.items():
        s_ = d[mfn(d)].reset_index(drop=True)
        C, Cn = pre_covariates(s_)
        e = s_.expected_score.to_numpy(float)
        pts = s_.points.to_numpy(float)
        f = ols(pts, np.column_stack([e, C]), ["expected score"] + Cn).loc["expected score"]
        fs.append({"subset": sub, "first_stage_coef": f.est, "se": f.se, "players": len(s_)})
        for m in METRICS:
            y = s_[m].to_numpy(float)
            rf = ols(y, np.column_stack([e, C]), ["expected score"] + Cn).loc["expected score"]
            naive = ols(y, np.column_stack([pts, e, C]), ["points", "expected score"] + Cn).loc["points"]
            iv, F = tsls(y, pts, e, C, Cn)
            r = iv.loc["points"]
            eff.append({"subset": sub, "metric": m, "base_mean": float(y.mean()), "players": len(s_),
                        "reduced_form_per_0.1": rf.est / 10, "rf_ci_lo": rf.ci_lo / 10, "rf_ci_hi": rf.ci_hi / 10,
                        "rf_p": rf.p, "iv_win_vs_loss": r.est, "iv_ci_lo": r.ci_lo, "iv_ci_hi": r.ci_hi,
                        "first_stage_F": F, "ols_win_vs_loss": naive.est, "ols_ci_lo": naive.ci_lo, "ols_ci_hi": naive.ci_hi})
        # Balance: the first draw should not predict the player's eventual skill (rating after 30
        # games, which one early result barely moves) nor their choice of speed.
        for name, v, mask, drop in [
            ("rating after 30 games (per 100)", s_.rating_after_30.to_numpy(float) / 100,
             s_.rating_after_30.notna().to_numpy(), ()),
            ("bullet first game", (s_.speed == "bullet").to_numpy(float), np.ones(len(s_), bool), ("speed",)),
        ]:
            keep = [i for i, n in enumerate(Cn) if not n.startswith(drop)] if drop else list(range(len(Cn)))
            r = ols(v[mask], np.column_stack([e[mask], C[mask][:, keep]]),
                    ["expected score"] + [Cn[i] for i in keep]).loc["expected score"]
            bal.append({"subset": sub, "covariate": name, "coef_per_0.1_expected": r.est / 10,
                        "ci_lo": r.ci_lo / 10, "ci_hi": r.ci_hi / 10, "p": r.p, "n": int(mask.sum())})
        if sub != "all first games":
            q = pd.qcut(s_.expected_score, 5, duplicates="drop")
            b = s_.groupby(q, observed=True).agg(expected=("expected_score", "mean"), points=("points", "mean"),
                                                 rating_after_30=("rating_after_30", "mean"),
                                                 **{m: (m, "mean") for m in METRICS}, players=("player", "size"))
            binned.append(b.reset_index(drop=True).assign(subset=sub))
    # Arena robustness: half of arena newcomers meet another newcomer first (both at 1500), and that
    # pairing is itself associated with lower return. Separate the two: control for it, or keep only
    # first games against established players.
    a = d[d.is_tournament.to_numpy()].reset_index(drop=True)
    C, Cn = pre_covariates(a)
    e = a.expected_score.to_numpy(float)
    newopp = (a.opp_game_no <= 10).to_numpy(float)
    rob = []
    for m in METRICS:
        y = a[m].to_numpy(float)
        r1 = ols(y, np.column_stack([e, newopp, C]), ["expected score", "newcomer opponent"] + Cn)
        vet = newopp == 0
        r2 = ols(y[vet], np.column_stack([e[vet], C[vet]]), ["expected score"] + Cn).loc["expected score"]
        for spec, r in (("controlling for newcomer opponent", r1.loc["expected score"]), ("established opponents only", r2)):
            rob.append({"metric": m, "spec": spec, "term": "expected score, per 0.1", "est": r.est / 10,
                        "ci_lo": r.ci_lo / 10, "ci_hi": r.ci_hi / 10, "p": r.p})
        r = r1.loc["newcomer opponent"]
        rob.append({"metric": m, "spec": "controlling for expected score", "term": "first opponent is a newcomer",
                    "est": r.est, "ci_lo": r.ci_lo, "ci_hi": r.ci_hi, "p": r.p,
                    "rate_vs_newcomer": float(y[newopp == 1].mean()), "rate_vs_established": float(y[newopp == 0].mean())})
    return {"effects": pd.DataFrame(eff), "balance": pd.DataFrame(bal), "first_stage": pd.DataFrame(fs),
            "binned": pd.concat(binned, ignore_index=True), "robustness": pd.DataFrame(rob),
            "arena_share_newcomer_opponent": float(newopp.mean())}


# ----------------------------------------------------------------------------------------------
# 3. Policy lift and who absorbs the losses
# ----------------------------------------------------------------------------------------------
EXPERIENCE_EFFECT = {  # Part 4, IV by game number: effect of a loss on not playing for a week
    "games 1-10": (0.0350, 0.0081), "11-50": (0.0156, 0.0059), "51-200": (0.0076, 0.0038), "201+": (-0.0015, 0.0012)}


def experience_bin(n: pd.Series) -> pd.Series:
    return pd.cut(n, [0, 10, 50, 200, np.inf], labels=list(EXPERIENCE_EFFECT))


CLEAN = "arena (opponent assigned)"


def policy(d: pd.DataFrame, ne: dict, targets=(0.5, 0.6)) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Implied lift in the primary metric from raising each newcomer's first-game expected score to
    at least `target`, using the arena natural experiment (lift = slope x mean increase), and an
    optimistic translation of Part 4's within-player estimate for newcomers' first ten games.

    Opponent side: the newcomer's gain in points is the opponent's loss. The experience mix of
    opponents rated at or below the newcomer (the pool a protective matchmaker draws from), weighted
    by Part 4's effect of a loss at each experience level, gives the extra week-long absences caused
    on the other side of the board."""
    eff = ne["effects"].set_index(["subset", "metric"]).loc[(CLEAN, PRIMARY)]
    slope, lo, hi = eff["reduced_form_per_0.1"] * 10, eff.rf_ci_lo * 10, eff.rf_ci_hi * 10
    fs = float(ne["first_stage"].set_index("subset").loc["all first games", "first_stage_coef"])
    e = d.expected_score.to_numpy()
    pool = d[(d.opp_elo <= d.elo) & d.opp_game_no.notna()]
    mix = experience_bin(pool.opp_game_no).value_counts(normalize=True).reindex(list(EXPERIENCE_EFFECT)).fillna(0)
    cost_per_point = sum(mix[k] * EXPERIENCE_EFFECT[k][0] for k in mix.index)
    newbie_effect, newbie_se = EXPERIENCE_EFFECT["games 1-10"]
    rows = []
    for tgt in targets:
        de = float(np.maximum(tgt - e, 0).mean())
        share = float((e < tgt).mean())
        common = {"target_expected_score": tgt, "share_of_newcomers_changed": share, "mean_increase_expected_score": de,
                  "opponent_points_lost_per_newcomer": fs * de,
                  "opponent_extra_week_absence_pp": fs * de * cost_per_point * 100}
        rows.append({"source": "arena natural experiment", **common, "lift_pp": slope * de * 100,
                     "lift_ci_lo_pp": lo * de * 100, "lift_ci_hi_pp": hi * de * 100})
        rb = ne["robustness"].set_index(["metric", "spec", "term"]).loc[(PRIMARY, "established opponents only",
                                                                          "expected score, per 0.1")]
        rows.append({"source": "arena, established opponents only (most favourable)", **common,
                     "lift_pp": rb.est * 10 * de * 100, "lift_ci_lo_pp": rb.ci_lo * 10 * de * 100,
                     "lift_ci_hi_pp": rb.ci_hi * 10 * de * 100})
        rows.append({"source": "optimistic: Part 4 newcomer effect applied to the first game", **common,
                     "lift_pp": newbie_effect * fs * de * 100,
                     "lift_ci_lo_pp": (newbie_effect - 1.96 * newbie_se) * fs * de * 100,
                     "lift_ci_hi_pp": (newbie_effect + 1.96 * newbie_se) * fs * de * 100})
    mixdf = mix.rename("share_of_pool").reset_index().rename(columns={"index": "opponent_experience",
                                                                      "opp_game_no": "opponent_experience"})
    return pd.DataFrame(rows), mixdf


# ----------------------------------------------------------------------------------------------
# 4. Power
# ----------------------------------------------------------------------------------------------
def n_per_arm(p0: float, lift: float, alpha=0.05, power=0.8, var_reduction=0.0) -> float:
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    p1 = p0 + lift
    return z**2 * (p0 * (1 - p0) + p1 * (1 - p1)) * (1 - var_reduction) / lift**2


def power_table(d: pd.DataFrame, pol: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    p0 = float(d[PRIMARY].mean())
    per_day = len(d) / (d.day.max() + 1)
    C, Cn = pre_covariates(d)
    y = d[PRIMARY].to_numpy(float)
    X = np.column_stack([np.ones(len(C)), C])
    r2 = 1 - np.var(y - X @ np.linalg.lstsq(X, y, rcond=None)[0]) / np.var(y)
    info = {"baseline": p0, "newcomers_per_day": per_day, "r2_pre_covariates": float(r2)}
    rows = []
    special = pol[pol.target_expected_score == 0.5]
    lifts = sorted({0.0025, 0.005, 0.01, 0.015, 0.02, 0.03}
                   | {round(x / 100, 5) for x in special.lift_ci_hi_pp if x > 0}
                   | {round(x / 100, 5) for x in special.lift_pp if x > 0})
    for lift in lifts:
        for vr, lab in ((0.0, "difference in means"), (r2, "regression-adjusted")):
            n = n_per_arm(p0, lift, var_reduction=vr)
            rows.append({"lift_pp": lift * 100, "analysis": lab, "n_per_arm": n,
                         "enrollment_days_50_50": 2 * n / per_day,
                         "total_days_incl_7d_maturation": 2 * n / per_day + 8,
                         "power_with_92_days_of_newcomers": float(stats.norm.cdf(
                             lift / np.sqrt((p0 * (1 - p0) + (p0 + lift) * (1 - p0 - lift)) * (1 - vr) / (len(d) / 2))
                             - stats.norm.ppf(0.975)))})
    return pd.DataFrame(rows), info


def power_experiment_b(d: pd.DataFrame, ne: dict) -> pd.DataFrame:
    """Experiment B: in arenas, never pair a newcomer's first game with another newcomer.

    Triggered population: newcomers whose first game is in an arena (only they can be affected).
    If the observed gap were fully causal, the intent-to-treat lift is gap x share who would have met
    a newcomer; the table also shows half and a quarter of that, in case it is partly confounded."""
    a = d[d.is_tournament.to_numpy()]
    p0 = float(a[PRIMARY].mean())
    per_day = len(a) / (d.day.max() + 1)
    share = ne["arena_share_newcomer_opponent"]
    rob = ne["robustness"]
    gap = -float(rob[(rob.metric == PRIMARY) & (rob.term == "first opponent is a newcomer")].est.iloc[0])
    rows = []
    for frac, lab in ((1.0, "gap fully causal"), (0.5, "half causal"), (0.25, "a quarter causal")):
        lift = gap * share * frac
        n = n_per_arm(p0, lift)
        rows.append({"scenario": lab, "baseline_arena_newcomers": p0, "arena_newcomers_per_day": per_day,
                     "share_who_would_meet_a_newcomer": share, "observed_gap_pp": gap * 100, "itt_lift_pp": lift * 100,
                     "n_per_arm": n, "enrollment_days": 2 * n / per_day, "total_days_incl_maturation": 2 * n / per_day + 8})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------
# 5. Simulations on the real players
# ----------------------------------------------------------------------------------------------
def obf_boundaries(looks: int, alpha=0.05) -> np.ndarray:
    """Two-sided O'Brien-Fleming-type boundaries via Lan-DeMets alpha spending, equally spaced looks,
    computed by Monte Carlo on the canonical Brownian motion (exact enough for a demonstration)."""
    rng = np.random.default_rng(123)
    t = np.arange(1, looks + 1) / looks
    spend = 2 * (1 - stats.norm.cdf(stats.norm.ppf(1 - alpha / 2) / np.sqrt(t)))   # Lan-DeMets OBF, two-sided
    Z = np.cumsum(rng.standard_normal((200_000, looks)), axis=1) / np.sqrt(np.arange(1, looks + 1))
    bounds, alive = [], np.ones(len(Z), bool)
    for k in range(looks):
        target = spend[k] - (spend[k - 1] if k else 0)
        cand = np.abs(Z[alive, k])
        b = np.quantile(cand, 1 - target * len(Z) / alive.sum()) if target > 0 else np.inf
        bounds.append(b)
        alive &= np.abs(Z[:, k]) < b
    return np.array(bounds)


def z_stat(y: np.ndarray, a: np.ndarray) -> float:
    y1, y0 = y[a], y[~a]
    se = np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0))
    return (y1.mean() - y0.mean()) / se


def simulate(d: pd.DataFrame, n_sims: int = 1000, seed: int = 7, lifts=(0.01, 0.02)) -> dict:
    """A/A and A/B tests replayed on the real newcomers, in enrollment order (92 days).

    A/A: random 50/50 split; count how often |z| > 1.96 at any daily look from day 14 (naive peeking),
    at the end only (fixed horizon), or past an O'Brien-Fleming boundary at weekly looks.
    A/B: each treated non-returner is flipped to a returner with probability lift / (1 - baseline),
    which adds `lift` to the treated arm's rate; record power for the fixed-horizon and sequential
    tests and compare with the analytic formula."""
    rng = np.random.default_rng(seed)
    y = d[PRIMARY].to_numpy(float)
    p0 = y.mean()
    day = d.day.to_numpy().astype(int)
    n_days = day.max() + 1
    ends = np.searchsorted(day, np.arange(n_days), side="right")
    week_ends = ends[np.arange(6, n_days, 7)]
    if week_ends[-1] != len(y):
        week_ends = np.append(week_ends, len(y))
    bounds = obf_boundaries(len(week_ends))
    aa = {"peek_daily": 0, "fixed_horizon": 0, "obf_weekly": 0}
    ab = {lift: {"fixed_horizon": 0, "obf_weekly": 0, "estimates": []} for lift in lifts}
    for _ in range(n_sims):
        a = rng.random(len(y)) < 0.5
        zs = np.array([z_stat(y[:k], a[:k]) for k in ends[13:]])
        aa["peek_daily"] += bool((np.abs(zs) > 1.96).any())
        aa["fixed_horizon"] += bool(abs(zs[-1]) > 1.96)
        zw = np.array([z_stat(y[:k], a[:k]) for k in week_ends])
        aa["obf_weekly"] += bool((np.abs(zw) > bounds).any())
        u = rng.random(len(y))
        for lift in lifts:
            yt = np.where(a & (y == 0) & (u < lift / (1 - p0)), 1.0, y)
            ab[lift]["fixed_horizon"] += bool(abs(z_stat(yt, a)) > 1.96)
            zw = np.array([z_stat(yt[:k], a[:k]) for k in week_ends])
            ab[lift]["obf_weekly"] += bool((np.abs(zw) > bounds).any())
            ab[lift]["estimates"].append(yt[a].mean() - yt[~a].mean())
    n_arm = len(y) / 2
    abrows = []
    for lift, r in ab.items():
        sd = np.sqrt((p0 * (1 - p0) + (p0 + lift) * (1 - p0 - lift)) / n_arm)
        abrows.append({"lift_pp": lift * 100, "power_fixed_horizon": r["fixed_horizon"] / n_sims,
                       "power_obf_weekly": r["obf_weekly"] / n_sims,
                       "analytic_power": float(stats.norm.cdf(lift / sd - 1.96)),
                       "mean_estimate_pp": float(np.mean(r["estimates"]) * 100),
                       "sd_estimate_pp": float(np.std(r["estimates"]) * 100)})
    # SRM: a logging bug silently drops treated players whose first game was a loss (they return
    # less, so the bug also biases the estimate upward). Run as an A/A, so the true effect is 0.
    srm = []
    for rate in (0.02, 0.10):
        det, bias, pvals = 0, [], []
        for _ in range(200):
            a = rng.random(len(y)) < 0.5
            keep = ~(a & (d.points.to_numpy() == 0) & (rng.random(len(y)) < rate))
            n1, n0 = int((a & keep).sum()), int((~a & keep).sum())
            chi2 = (n1 - n0) ** 2 / (n1 + n0)
            pv = float(stats.chi2.sf(chi2, 1))
            det += pv < 0.001                                     # the usual strict SRM alarm threshold
            pvals.append(pv)
            bias.append(y[a & keep].mean() - y[~a & keep].mean())
        srm.append({"drop_rate_of_treated_losers": rate, "share_of_treated_arm_lost": rate * float((d.points == 0).mean()),
                    "detection_rate_p_below_0.001": det / 200, "median_p": float(np.median(pvals)),
                    "mean_bias_pp": float(np.mean(bias) * 100)})
    srm = pd.DataFrame(srm)
    return {"aa": {k: v / n_sims for k, v in aa.items()}, "ab": pd.DataFrame(abrows), "bounds": bounds.tolist(),
            "looks": len(week_ends), "srm": srm, "n_sims": n_sims, "days": int(n_days), "n": len(y)}


def peeking_curve(d: pd.DataFrame, max_looks=(1, 2, 5, 10, 20, 40, 80), n_sims: int = 1000, seed: int = 11) -> pd.DataFrame:
    """A/A false-positive rate when testing at |z| > 1.96 after each of k equally spaced looks."""
    rng = np.random.default_rng(seed)
    y = d[PRIMARY].to_numpy(float)
    rows = []
    hits = {k: 0 for k in max_looks}
    for _ in range(n_sims):
        a = rng.random(len(y)) < 0.5
        for k in max_looks:
            idx = np.linspace(len(y) / k, len(y), k).astype(int)
            hits[k] += bool(any(abs(z_stat(y[:i], a[:i])) > 1.96 for i in idx))
    for k in max_looks:
        rows.append({"looks": k, "false_positive_rate": hits[k] / n_sims})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------
# 6. Figures
# ----------------------------------------------------------------------------------------------
def figures(figs: Path, ne: dict, power: pd.DataFrame, info: dict, pol: pd.DataFrame, peek: pd.DataFrame) -> None:
    b = ne["binned"]
    panels = [("rating_after_30", "Balance: rating after 30 games", None),
              ("games_14d", "Games in the first 14 days", None),
              (PRIMARY, "Plays again on days 1-7", "pct")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.9), dpi=200)
    for ax, (col, title, fmt) in zip(axes, panels):
        for i, sub in enumerate([CLEAN, "casual (opponent chosen)"]):
            bb = b[b.subset == sub]
            ax.plot(bb.expected, bb[col], color=SERIES[i], marker="o", linewidth=2, label=sub)
        style(ax, title)
        if fmt == "pct":
            pct(ax)
        ax.set_xlabel("first game's expected score (higher = weaker opponent)", color=INK2, fontsize=8)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper right")
    fig.suptitle("Natural experiment: how strong was a newcomer's first opponent? (quintiles)",
                 x=0.01, ha="left", color=INK, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(figs / "experiment_natural.png", facecolor=SURFACE)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=200)
    for i, lab in enumerate(["difference in means", "regression-adjusted"]):
        p = power[power.analysis == lab].sort_values("lift_pp")
        ax.plot(p.lift_pp, p.total_days_incl_7d_maturation, color=SERIES[i], linewidth=2, marker="o", markersize=4,
                label=lab, linestyle="-" if i == 0 else (0, (4, 2)))
    ax.axhline(92 + 8, color=INK2, linewidth=0.8, linestyle=":")
    ax.text(ax.get_xlim()[1], 92 + 8, "one quarter of newcomers ", color=INK2, fontsize=8, ha="right", va="bottom")
    marks = pol[pol.target_expected_score == 0.5].set_index("source")
    for key, lab in [("arena natural experiment", "upper 95% bound,\nnatural experiment"),
                     ("optimistic: Part 4 newcomer effect applied to the first game", "Part 4 effect\napplied"),
                     ("arena, established opponents only (most favourable)", "most favourable\nupper bound")]:
        x = marks.loc[key, "lift_ci_hi_pp"] if key.startswith("arena") else marks.loc[key, "lift_pp"]
        if x > 0:
            ax.axvline(x, color=SERIES[2], linewidth=1, linestyle=":")
            ax.text(x * 1.04, 30, lab, color=INK2, fontsize=7.5, va="bottom")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ticks = [(14, "2 wk"), (30, "1 mo"), (100, "~3 mo"), (365, "1 yr"), (3650, "10 yr"), (36500, "100 yr")]
    ax.set_yticks([t for t, _ in ticks], [l for _, l in ticks])
    xt = [0.05, 0.1, 0.25, 0.5, 1, 2, 3]
    ax.set_xticks(xt, [f"{x:g}" for x in xt])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.yaxis.set_minor_locator(plt.NullLocator())
    style(ax, f"Days to run a 50/50 test at 80% power ({info['newcomers_per_day']:.0f} Chess960 newcomers/day)")
    ax.set_xlabel("true lift in 'plays again on days 1-7' (percentage points, log scale)", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper right")
    fig.tight_layout()
    fig.savefig(figs / "experiment_power.png", facecolor=SURFACE)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=200)
    ax.plot(peek.looks, peek.false_positive_rate, color=SERIES[1], marker="o", linewidth=2)
    ax.axhline(0.05, color=INK2, linewidth=0.8, linestyle=":")
    ax.text(peek.looks.max(), 0.055, "nominal 5%", color=INK2, fontsize=8, ha="right")
    ax.set_xscale("log")
    ax.set_xticks(peek.looks, [str(k) for k in peek.looks])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    style(ax, "A/A tests on real newcomers: false positives when peeking")
    pct(ax)
    ax.set_ylim(0, None)
    ax.set_xlabel("number of times the test is checked (stopping at the first p < 0.05)", color=INK2, fontsize=9)
    fig.tight_layout()
    fig.savefig(figs / "experiment_peeking.png", facecolor=SURFACE)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------
def run(panel: Path, out: Path, analysis_dir: Path, n_sims: int) -> dict:
    tables, figs = out / "tables", out / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:5.0f}s] {m}", flush=True)  # noqa: E731
    d = load(build_first_games(panel, analysis_dir))
    log(f"{len(d):,} new players, enrolled over {int(d.day.max()) + 1} days")
    ne = natural_experiment(d)
    pol, mix = policy(d, ne)
    power, info = power_table(d, pol)
    power_b = power_experiment_b(d, ne)
    log("natural experiment, policy and power done")
    sim = simulate(d, n_sims)
    log("A/A and A/B simulations done")
    peek = peeking_curve(d, n_sims=n_sims)
    log("peeking curve done")
    ne["effects"].to_csv(tables / "experiment_natural_effects.csv", index=False)
    ne["balance"].to_csv(tables / "experiment_balance.csv", index=False)
    ne["binned"].to_csv(tables / "experiment_binned.csv", index=False)
    pol.to_csv(tables / "experiment_policy.csv", index=False)
    mix.to_csv(tables / "experiment_opponent_pool.csv", index=False)
    power.to_csv(tables / "experiment_power.csv", index=False)
    peek.to_csv(tables / "experiment_peeking.csv", index=False)
    ne["first_stage"].to_csv(tables / "experiment_first_stage.csv", index=False)
    ne["robustness"].to_csv(tables / "experiment_arena_robustness.csv", index=False)
    power_b.to_csv(tables / "experiment_power_b.csv", index=False)
    sim["srm"].to_csv(tables / "experiment_srm.csv", index=False)
    sim["ab"].to_csv(tables / "experiment_ab_simulation.csv", index=False)
    pd.DataFrame([{**info, **{f"aa_{k}": v for k, v in sim["aa"].items()}, "n_sims": sim["n_sims"],
                   "players": sim["n"], "days": sim["days"], "weekly_looks": sim["looks"],
                   "obf_bounds": " ".join(f"{b:.2f}" for b in sim["bounds"])}]).to_csv(tables / "experiment_simulation.csv", index=False)
    figures(figs, ne, power, info, pol, peek)
    log("done")
    return {"d": d, "ne": ne, "pol": pol, "mix": mix, "power": power, "power_b": power_b, "info": info,
            "sim": sim, "peek": peek}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--panel", type=Path, default=Path("data/panel"))
    p.add_argument("--out", type=Path, default=Path("reports"))
    p.add_argument("--analysis-dir", type=Path, default=Path("data/analysis"))
    p.add_argument("--sims", type=int, default=1000, help="replications for the A/A and A/B simulations")
    p.add_argument("--build-only", action="store_true")
    args = p.parse_args()
    if args.build_only:
        print(build_first_games(args.panel, args.analysis_dir))
        return
    r = run(args.panel, args.out, args.analysis_dir, args.sims)
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print("\nnatural experiment\n", r["ne"]["effects"].round(4).to_string(index=False))
    print("\nfirst stage\n", r["ne"]["first_stage"].round(4).to_string(index=False))
    print("\nbalance\n", r["ne"]["balance"].round(4).to_string(index=False))
    print("\nbinned\n", r["ne"]["binned"].round(3).to_string(index=False))
    print("\npolicy\n", r["pol"].round(4).to_string(index=False))
    print("\nopponent pool\n", r["mix"].round(3).to_string(index=False))
    print("\ninfo", {k: round(v, 4) for k, v in r["info"].items()})
    print("\npower\n", r["power"].round(1).to_string(index=False))
    print("\nA/A", r["sim"]["aa"], "looks", r["sim"]["looks"], "bounds", np.round(r["sim"]["bounds"], 2))
    print("\nA/B\n", r["sim"]["ab"].round(4).to_string(index=False))
    print("\nSRM\n", r["sim"]["srm"].round(4).to_string(index=False))
    print("\narena robustness\n", r["ne"]["robustness"].round(4).to_string(index=False))
    print("\nexperiment B power\n", r["power_b"].round(4).to_string(index=False))
    print("\npeeking\n", r["peek"].to_string(index=False))


if __name__ == "__main__":
    main()
