"""Part 2: weekly engagement states and what moves players between them.

States per player-week (weeks since first game):
    Inactive  0 games
    Casual    1-4 games
    Engaged   5+ games      (5 is roughly the 60th percentile of active weeks)

Because every week is observed (no intermittent observation), the multi-state model is a
discrete-time Markov chain: one multinomial logit per origin state for next week's state, with
covariates measured in the current week and standard errors clustered by player. This is the
discrete-time analogue of a continuous-time msm model in R.

Outputs: reports/tables/transitions_*.csv, reports/figures/transitions_*.png

Example:
    python src/transitions.py --panel data/panel --out reports
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import duckdb
import matplotlib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

STATES = ["Inactive", "Casual", "Engaged"]
ENGAGED_MIN_GAMES = 5
DUR_CAP = 16  # durations >= this are pooled when estimating duration-dependent transitions

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]


def load(panel: Path) -> pd.DataFrame:
    con = duckdb.connect()
    con.execute("SET TimeZone = 'UTC'")  # Lichess timestamps are UTC; keep cohort months machine-independent
    df = con.execute(f"""
        SELECT w.player, strftime(p.first_ts, '%Y-%m') AS cohort_month, w.week, w.n_games,
               w.max_session_loss_streak, w.perf_vs_expected, w.rating_change,
               w.share_bullet, w.share_tournament, w.lost_on_time, w.cum_games
        FROM read_parquet('{(panel / 'player_weeks.parquet').as_posix()}') w
        JOIN read_parquet('{(panel / 'players.parquet').as_posix()}') p USING (player)
        ORDER BY w.player, w.week
    """).df()
    df["state"] = np.select([df.n_games == 0, df.n_games < ENGAGED_MIN_GAMES], [0, 1], 2).astype(np.int8)
    g = df.groupby("player", sort=False)
    df["next_state"] = g.state.shift(-1)

    # Weeks spent in the current state so far (run length, this week included).
    run_id = (df.state != g.state.shift()).cumsum()
    df["weeks_in_state"] = df.groupby(run_id).cumcount() + 1

    # Features of the current week (defined when active).
    active = df.n_games > 0
    df["log_games"] = np.log1p(df.n_games)
    df["streak3"] = (df.max_session_loss_streak >= 3).astype(float).where(active)
    df["perf_pg"] = (df.perf_vs_expected / df.n_games).where(active)
    df["rating_change_100"] = (df.rating_change / 100).where(active)
    df["lost_on_time_share"] = (df.lost_on_time / df.n_games).where(active)
    # Last active week's features, carried forward into inactive weeks.
    for c in ("streak3", "perf_pg", "log_games"):
        df["last_" + c] = df[c].where(active).groupby(df.player, sort=False).ffill()
    df["log_tenure"] = np.log(df.week.clip(lower=1))
    df["log_wis"] = np.log(df.weeks_in_state)
    df["log_cum_games"] = np.log1p(df.cum_games)
    return df


def transition_matrix(t: pd.DataFrame) -> pd.DataFrame:
    m = pd.crosstab(t.state, t.next_state.astype(int))
    m = m.reindex(index=range(3), columns=range(3), fill_value=0)
    m.index = [f"from {s}" for s in STATES]
    m.columns = [f"to {s}" for s in STATES]
    return m


def to_prob(m: pd.DataFrame) -> pd.DataFrame:
    return m.div(m.sum(axis=1), axis=0)


COVARIATES = {
    0: ["log_wis", "log_tenure", "log_cum_games", "last_log_games", "last_streak3", "last_perf_pg"],
    1: ["log_wis", "log_tenure", "log_cum_games", "log_games", "streak3", "perf_pg",
        "rating_change_100", "share_bullet", "share_tournament", "lost_on_time_share"],
    2: ["log_wis", "log_tenure", "log_cum_games", "log_games", "streak3", "perf_pg",
        "rating_change_100", "share_bullet", "share_tournament", "lost_on_time_share"],
}
BINARY = {"streak3", "last_streak3"}
LABELS = {
    "log_wis": "weeks in current state (log)",
    "log_tenure": "weeks since first game (log)",
    "log_cum_games": "games played so far (log)",
    "log_games": "games this week (log)",
    "last_log_games": "games in last active week (log)",
    "streak3": "3+ loss streak in a session this week",
    "last_streak3": "3+ loss streak in last active week",
    "perf_pg": "score above rating expectation, per game",
    "last_perf_pg": "score above expectation, last active week",
    "rating_change_100": "rating change this week (per 100)",
    "share_bullet": "share of bullet games",
    "share_tournament": "share of tournament games",
    "lost_on_time_share": "share of games lost on time",
}


def _probs(B: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Multinomial logit probabilities; B is K x (J-1), first outcome is the base."""
    eta = np.column_stack([np.zeros(len(X)), X @ B])
    eta -= eta.max(axis=1, keepdims=True)
    e = np.exp(eta)
    return e / e.sum(axis=1, keepdims=True)


def contrast(var: str, x: pd.Series) -> tuple[float, float, str]:
    """The low -> high change used to express each covariate's effect."""
    if var in BINARY:
        return 0.0, 1.0, "0 -> 1"
    if var.startswith("share_") or var.endswith("_share"):
        return 0.0, 1.0, "none -> all games"
    lo, hi = x.quantile([0.25, 0.75])
    if hi == lo:
        lo, hi = x.quantile([0.10, 0.90])
    return float(lo), float(hi), "25th -> 75th percentile"


def fit_origin(t: pd.DataFrame, origin: int) -> tuple[pd.DataFrame, object]:
    """Multinomial logit for next-week state among weeks starting in `origin`.

    Each covariate's effect is the average change in the probability of each destination when
    that covariate moves from a low to a high value (others kept as observed), with a delta-method
    95% CI that uses the player-clustered covariance of the coefficients.
    """
    cols = COVARIATES[origin]
    d = t[t.state == origin].dropna(subset=cols + ["next_state"])
    X = sm.add_constant(d[cols].astype(float), has_constant="add")
    y = d.next_state.astype(int)
    groups = pd.factorize(d.player)[0]
    res = sm.MNLogit(y, X).fit(method="newton", maxiter=100, disp=False,
                              cov_type="cluster", cov_kwds={"groups": groups})
    K, J1 = res.params.shape
    B0 = res.params.to_numpy()
    # cov_params is ordered equation by equation, i.e. column-major over B.
    V = res.cov_params().to_numpy()
    theta0 = B0.flatten(order="F")
    Xa = X.to_numpy()
    dest_codes = sorted(y.unique())  # outcome order used by MNLogit
    rows = []
    for j, var in enumerate(cols, start=1):
        lo, hi, how = contrast(var, d[var])
        X_lo, X_hi = Xa.copy(), Xa.copy()
        X_lo[:, j], X_hi[:, j] = lo, hi

        P_lo, P_hi = _probs(B0, X_lo), _probs(B0, X_hi)
        est = (P_hi - P_lo).mean(axis=0)
        # Analytic Jacobian: d p_j / d beta_m = p_j (1[j=m] - p_m) x, for equations m = 1..J-1.
        J = J1 + 1
        jac = np.empty((J, K * J1))
        for jj in range(J):
            for m in range(1, J):
                ind = 1.0 if jj == m else 0.0
                w_hi = P_hi[:, jj] * (ind - P_hi[:, m])
                w_lo = P_lo[:, jj] * (ind - P_lo[:, m])
                jac[jj, (m - 1) * K:m * K] = (w_hi @ X_hi - w_lo @ X_lo) / len(Xa)
        se = np.sqrt(np.clip(np.diag(jac @ V @ jac.T), 0, None))
        for i, dest in enumerate(dest_codes):
            z = est[i] / se[i] if se[i] > 0 else np.nan
            rows.append({"origin": STATES[origin], "destination": STATES[dest], "variable": var,
                         "label": LABELS.get(var, var), "contrast": how, "low": lo, "high": hi,
                         "ame": est[i], "se": se[i], "ci_lo": est[i] - 1.96 * se[i], "ci_hi": est[i] + 1.96 * se[i],
                         "p": 2 * stats.norm.sf(abs(z)) if np.isfinite(z) else np.nan,
                         "n_weeks": len(d), "n_players": d.player.nunique()})
    return pd.DataFrame(rows), res


def duration_curves(t: pd.DataFrame) -> pd.DataFrame:
    """P(active next week) by origin state and weeks already spent in it (the Markov check)."""
    d = t.copy()
    d["dur"] = d.weeks_in_state.clip(upper=DUR_CAP)
    d["active_next"] = (d.next_state > 0).astype(int)
    out = d.groupby(["state", "dur"]).agg(n=("active_next", "size"), active_next=("active_next", "mean")).reset_index()
    out["state"] = out.state.map(dict(enumerate(STATES)))
    return out


TENURE_BANDS = [(1, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, 32), (33, 10_000)]


def tenure_band(week: int) -> int:
    for i, (lo, hi) in enumerate(TENURE_BANDS):
        if lo <= week <= hi:
            return i
    return len(TENURE_BANDS) - 1


def _rows(t: pd.DataFrame, keys: list[str]) -> dict:
    """Empirical next-state distributions keyed by `keys` (rows with no data are left out)."""
    cnt = t.groupby(keys + ["next_state"]).size().unstack(fill_value=0).reindex(columns=[0, 1, 2], fill_value=0)
    cnt = cnt[cnt.sum(axis=1) > 0]
    return {k if isinstance(k, tuple) else (k,): v / v.sum() for k, v in zip(cnt.index, cnt.to_numpy(float))}


def prevalence_check(df: pd.DataFrame, t: pd.DataFrame, horizon: int = 52) -> pd.DataFrame:
    """Observed share in each state by week vs. three chains, all started from the observed
    week-1 distribution and propagated exactly (no simulation):
      markov          one transition matrix for everyone, every week
      duration        transition depends on (state, weeks already in state)
      duration_tenure transition depends on (state, weeks in state, weeks since first game)
    Sparse cells fall back to the next simpler model."""
    obs = (df[(df.week >= 1) & (df.week <= horizon)].groupby("week").state
           .value_counts(normalize=True).unstack(fill_value=0).reindex(columns=range(3), fill_value=0))
    D = DUR_CAP
    tt = t.assign(dur=t.weeks_in_state.clip(upper=D), tband=t.week.map(tenure_band))
    m0 = _rows(tt, ["state"])
    m1 = _rows(tt, ["state", "dur"])
    m2 = _rows(tt, ["state", "dur", "tband"])

    def row(model: str, s: int, k: int, week: int) -> np.ndarray:
        if model == "duration_tenure" and (s, k, tenure_band(week)) in m2:
            return m2[(s, k, tenure_band(week))]
        if model in ("duration", "duration_tenure") and (s, k) in m1:
            return m1[(s, k)]
        return m0[(s,)]

    w1 = df[df.week == 1].assign(dur=lambda x: x.weeks_in_state.clip(upper=D))
    start = (w1.groupby(["state", "dur"]).size() / len(w1)).to_dict()
    out = pd.DataFrame({"week": np.arange(1, horizon + 1)})
    for i, s in enumerate(STATES):
        out[f"observed_{s}"] = obs[i].reindex(out.week).to_numpy()
    for model in ("markov", "duration", "duration_tenure"):
        dist, path = dict(start), []
        for week in range(1, horizon + 1):
            shares = np.zeros(3)
            for (s, k), m in dist.items():
                shares[s] += m
            path.append(shares)
            new: dict = {}
            for (s, k), m in dist.items():
                r = row(model, s, k, week)
                for s2 in range(3):
                    if r[s2] == 0:
                        continue
                    k2 = min(k + 1, D) if s2 == s else 1
                    new[(s2, k2)] = new.get((s2, k2), 0.0) + m * r[s2]
            dist = new
        for i, s in enumerate(STATES):
            out[f"{model}_{s}"] = [v[i] for v in path]
    return out


def style(ax, title: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold", pad=12)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def pct(ax, axis: str = "y") -> None:
    f = matplotlib.ticker.PercentFormatter(1.0, decimals=0)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(f)


def figures(figs: Path, dur: pd.DataFrame, prev: pd.DataFrame, ame: pd.DataFrame) -> None:
    # 1. Markov check: P(active next week) by weeks already in the state.
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=200)
    for i, s in enumerate(STATES):
        d = dur[(dur.state == s) & (dur.n >= 200)]
        ax.plot(d.dur, d.active_next, color=SERIES[i], linewidth=2, marker="o", markersize=4, label=f"currently {s}")
    style(ax, "Chance of playing next week, by weeks already spent in the current state")
    pct(ax)
    ax.set_ylim(0, 1)
    ax.set_xlabel(f"weeks in current state ({DUR_CAP} = {DUR_CAP} or more)", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="center right")
    fig.tight_layout()
    fig.savefig(figs / "transitions_duration.png", facecolor=SURFACE)
    plt.close(fig)

    # 2. Model fit: share of cohort active by week, observed vs. the three chains.
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=200)
    active = lambda p: prev[f"{p}_Casual"] + prev[f"{p}_Engaged"]  # noqa: E731
    ax.plot(prev.week, active("observed"), color=INK, linewidth=2.5, label="observed")
    for i, (key, name) in enumerate([("markov", "plain Markov chain"),
                                     ("duration", "+ weeks in current state"),
                                     ("duration_tenure", "+ weeks in state and weeks since joining")]):
        ax.plot(prev.week, active(key), color=SERIES[i], linewidth=2, linestyle=(0, (4, 2)), label=name)
    style(ax, "Share of new players active each week: data vs. three models started at week 1")
    pct(ax)
    ax.set_ylim(0, None)
    ax.set_xlabel("weeks since first game", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper right")
    fig.tight_layout()
    fig.savefig(figs / "transitions_fit.png", facecolor=SURFACE)
    plt.close(fig)

    # 3. Effects on dropping to Inactive next week, for active origins.
    keep = ["streak3", "perf_pg", "log_games", "share_tournament", "share_bullet", "lost_on_time_share", "log_wis"]
    d = ame[(ame.destination == "Inactive") & ame.variable.isin(keep) & (ame.origin != "Inactive")]
    order = [LABELS[k] for k in keep]
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=200)
    for i, origin in enumerate(["Casual", "Engaged"]):
        dd = d[d.origin == origin].set_index("label").reindex(order)
        y = np.arange(len(order)) + (i - 0.5) * 0.28
        ax.errorbar(dd.ame, y, xerr=[dd.ame - dd.ci_lo, dd.ci_hi - dd.ame], fmt="o", color=SERIES[i],
                    ecolor=SERIES[i], elinewidth=1.5, capsize=0, markersize=6, label=f"from {origin}")
    ax.axvline(0, color=INK2, linewidth=0.8)
    ax.set_yticks(np.arange(len(order)), order)
    ax.invert_yaxis()
    style(ax, "")
    fig.suptitle("Change in chance of going inactive next week (percentage points, 95% CI)",
                 x=0.01, ha="left", color=INK, fontsize=12, fontweight="bold")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v * 100:+.0f}"))
    ax.set_xlabel("binary: 0 -> 1;  shares: none -> all games;  others: 25th -> 75th percentile",
                  color=INK2, fontsize=8)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower right")
    fig.tight_layout()
    fig.savefig(figs / "transitions_effects.png", facecolor=SURFACE)
    plt.close(fig)


def run(panel: Path, out: Path) -> dict:
    tables, figs = out / "tables", out / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = load(panel)
    t = df[(df.week >= 1) & df.next_state.notna()].copy()
    print(f"loaded {len(df):,} player-weeks, {len(t):,} transitions from week 1 on ({time.time() - t0:.0f}s)", flush=True)

    counts = transition_matrix(t)
    probs = to_prob(counts)
    counts.to_csv(tables / "transitions_counts.csv")
    probs.to_csv(tables / "transitions_matrix.csv")
    first = df[(df.week == 0) & df.next_state.notna()]
    week0 = to_prob(transition_matrix(first))
    week0.to_csv(tables / "transitions_week0.csv")

    t["tenure_band"] = pd.cut(t.week, [0, 4, 12, 26, 100], labels=["weeks 1-4", "5-12", "13-26", "27+"])
    by_tenure = pd.concat({b: to_prob(transition_matrix(g)) for b, g in t.groupby("tenure_band", observed=True)})
    by_tenure.to_csv(tables / "transitions_by_tenure.csv")

    dur = duration_curves(t)
    dur.to_csv(tables / "transitions_duration.csv", index=False)
    prev = prevalence_check(df, t)
    prev.to_csv(tables / "transitions_fit.csv", index=False)
    # Out-of-sample: estimate on the earlier cohorts, predict the last cohort's curve.
    last = sorted(df.cohort_month.unique())[-1]
    holdout = prevalence_check(df[df.cohort_month == last], t[t.cohort_month != last])
    holdout.insert(0, "holdout_cohort", last)
    holdout.to_csv(tables / "transitions_fit_holdout.csv", index=False)
    print(f"descriptives done ({time.time() - t0:.0f}s)", flush=True)

    ames, fits = [], {}
    for origin in (2, 1, 0):
        a, res = fit_origin(t, origin)
        ames.append(a)
        fits[STATES[origin]] = {"n_weeks": int(a.n_weeks.iloc[0]), "n_players": int(a.n_players.iloc[0]),
                                "pseudo_r2": float(res.prsquared), "converged": bool(res.mle_retvals["converged"])}
        print(f"fitted origin {STATES[origin]} ({time.time() - t0:.0f}s)", flush=True)
    ame = pd.concat(ames, ignore_index=True)
    ame.to_csv(tables / "transitions_marginal_effects.csv", index=False)
    pd.DataFrame(fits).T.to_csv(tables / "transitions_model_fit.csv")

    figures(figs, dur, prev, ame)
    return {"counts": counts, "probs": probs, "week0": week0, "by_tenure": by_tenure, "dur": dur,
            "prev": prev, "holdout": holdout, "ame": ame, "fits": fits}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--panel", type=Path, default=Path("data/panel"))
    p.add_argument("--out", type=Path, default=Path("reports"))
    args = p.parse_args()
    r = run(args.panel, args.out)
    pd.set_option("display.width", 220, "display.max_columns", 30, "display.max_rows", 200)
    print("\ncounts\n", r["counts"])
    print("\nweekly transition matrix (weeks 1+)\n", r["probs"].round(3))
    print("\nweek 0 -> 1\n", r["week0"].round(3))
    print("\nby tenure\n", r["by_tenure"].round(3))
    print("\nduration curves\n", r["dur"].pivot(index="dur", columns="state", values="active_next").round(3))
    for name in ("prev", "holdout"):
        pr = r[name]
        print(f"\n=== {name} ===")
        fit = pd.DataFrame({"week": pr.week, **{m: pr[f"{m}_Casual"] + pr[f"{m}_Engaged"]
                                                for m in ("observed", "markov", "duration", "duration_tenure")}})
        print("fit check (share active)\n", fit.iloc[[0, 1, 3, 7, 11, 25, 51]].round(4).to_string(index=False))
        for m in ("markov", "duration", "duration_tenure"):
            print(f"  mean abs error, weeks 2-52, {m}: {(fit[m] - fit.observed).abs().iloc[1:].mean() * 100:.2f} pp")
    print("\nmodel fit\n", pd.DataFrame(r["fits"]).T)
    a = r["ame"]
    print("\nmarginal effects (pp)\n", a.assign(ame=a.ame * 100, ci_lo=a.ci_lo * 100, ci_hi=a.ci_hi * 100)[
        ["origin", "destination", "variable", "contrast", "low", "high", "ame", "ci_lo", "ci_hi", "p"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
