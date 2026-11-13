"""Part 1: retention and engagement metrics for the new-player cohort.

Reads the Parquet tables written by build_panel.py and writes:
  reports/tables/*.csv     every number behind the report
  reports/figures/*.png    charts used in reports/01_retention.md

Example:
    python src/retention.py --panel data/panel --out reports
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# Reference palette (light surface) from the dataviz guidance: fixed categorical order.
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
DAY_S = 86400


def wilson(k: float, n: float, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def add_ci(df: pd.DataFrame, k: str, n: str, name: str) -> pd.DataFrame:
    ci = [wilson(a, b) for a, b in zip(df[k], df[n])]
    df[name] = df[k] / df[n]
    df[name + "_lo"] = [c[0] for c in ci]
    df[name + "_hi"] = [c[1] for c in ci]
    return df


def style(ax, title: str, ylabel: str | None = None) -> None:
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold", pad=12)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def pct_axis(ax) -> None:
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))


def run(panel: Path, out: Path) -> dict:
    tables, figs = out / "tables", out / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET TimeZone = 'UTC'")  # Lichess timestamps are UTC; keep dates and months machine-independent
    for t in ("player_games", "sessions", "players", "player_weeks"):
        con.execute(f"CREATE VIEW {t} AS SELECT * FROM read_parquet('{(panel / t).as_posix()}.parquet')")

    # Per-player features from day 0 and the first session, and day-N activity flags.
    con.execute(f"""
    CREATE TABLE cohort AS
    WITH c AS (SELECT player, first_ts, strftime(first_ts, '%Y-%m') AS cohort_month, n_games AS lifetime_games
               FROM players WHERE in_cohort),
    g AS (
        SELECT g.*, c.cohort_month, floor(date_diff('second', c.first_ts, g.ts) / {DAY_S})::INT AS day
        FROM player_games g JOIN c USING (player)
    ),
    first_game AS (
        SELECT player, score AS first_score, expected_score AS first_expected, speed AS first_speed,
               is_tournament AS first_tournament, lost_on_time AS first_lost_on_time
        FROM g WHERE game_no = 1
    ),
    first_session AS (
        SELECT player, count(*) AS s1_games, max(session_loss_streak) AS s1_max_loss_streak,
               arg_max(score, game_no) AS s1_last_score
        FROM g WHERE session_no = 1 GROUP BY player
    ),
    days AS (
        SELECT player,
            count(*) FILTER (WHERE day = 0) AS day0_games,
            count(DISTINCT session_no) FILTER (WHERE day = 0) AS day0_sessions,
            bool_or(day = 1) AS d1, bool_or(day = 7) AS d7, bool_or(day = 14) AS d14, bool_or(day = 30) AS d30,
            bool_or(day >= 1) AS came_back,
            bool_or(day BETWEEN 7 AND 13) AS w1, bool_or(day BETWEEN 28 AND 34) AS w4,
            bool_or(day >= 90) AS active_after_d90
        FROM g GROUP BY player
    )
    SELECT * FROM c JOIN first_game USING (player) JOIN first_session USING (player) JOIN days USING (player)
    """)
    n_cohort = con.execute("SELECT count(*) FROM cohort").fetchone()[0]

    # 1. Classic day-N retention and rolling "came back" retention, by cohort month.
    dayn = con.execute("""
        SELECT coalesce(cohort_month, 'all') AS cohort_month, count(*) AS players,
               avg(d1::INT) AS d1, avg(d7::INT) AS d7, avg(d14::INT) AS d14, avg(d30::INT) AS d30,
               avg(came_back::INT) AS came_back_after_day0, avg(w1::INT) AS active_week1,
               avg(w4::INT) AS active_week4, avg(active_after_d90::INT) AS active_after_day90,
               avg((lifetime_games = 1)::INT) AS one_game_only,
               median(lifetime_games) AS median_lifetime_games
        FROM cohort GROUP BY ROLLUP (cohort_month) ORDER BY cohort_month NULLS LAST
    """).df()
    dayn.to_csv(tables / "day_n_retention.csv", index=False)

    # 2. Weekly retention curve: share active in week k since first game.
    weekly = con.execute("""
        SELECT strftime(p.first_ts, '%Y-%m') AS cohort_month, w.week,
               count(*) AS players, sum((w.n_games > 0)::INT) AS active
        FROM player_weeks w JOIN players p USING (player)
        WHERE w.week <= 52 GROUP BY ALL ORDER BY 1, 2
    """).df()
    weekly = add_ci(weekly, "active", "players", "retention")
    weekly.to_csv(tables / "weekly_retention.csv", index=False)

    # Retention among players who came back at least once (removes one-and-done visitors).
    weekly_returners = con.execute("""
        SELECT w.week, count(*) AS players, sum((w.n_games > 0)::INT) AS active
        FROM player_weeks w JOIN cohort c USING (player)
        WHERE c.came_back AND w.week <= 52 GROUP BY 1 ORDER BY 1
    """).df()
    weekly_returners = add_ci(weekly_returners, "active", "players", "retention")
    weekly_returners.to_csv(tables / "weekly_retention_returners.csv", index=False)

    # 3. Early signals: does the first-day experience predict coming back?
    def by(col_sql: str, label: str, order: str = "1") -> pd.DataFrame:
        df = con.execute(f"""
            SELECT {col_sql} AS grp, count(*) AS players,
                   sum(came_back::INT) AS came_back, sum(w1::INT) AS w1, sum(w4::INT) AS w4,
                   avg(first_expected) AS mean_first_expected_score
            FROM cohort GROUP BY 1 ORDER BY {order}
        """).df()
        df = add_ci(df, "came_back", "players", "came_back_rate")
        df = add_ci(df, "w1", "players", "w1_rate")
        df = add_ci(df, "w4", "players", "w4_rate")
        df.insert(0, "feature", label)
        return df

    signals = pd.concat([
        by("CASE first_score WHEN 1 THEN 'won' WHEN 0.5 THEN 'drew' ELSE 'lost' END", "first game result",
           "CASE grp WHEN 'won' THEN 1 WHEN 'drew' THEN 2 ELSE 3 END"),
        by("CASE WHEN first_lost_on_time THEN 'lost on time' WHEN first_score = 0 THEN 'lost otherwise' "
           "ELSE 'did not lose' END", "how the first game was lost"),
        by("CASE WHEN day0_games = 1 THEN '1' WHEN day0_games <= 3 THEN '2-3' WHEN day0_games <= 9 THEN '4-9' "
           "ELSE '10+' END", "games on day 0",
           "CASE grp WHEN '1' THEN 1 WHEN '2-3' THEN 2 WHEN '4-9' THEN 3 ELSE 4 END"),
        by("CASE WHEN s1_max_loss_streak = 0 THEN '0' WHEN s1_max_loss_streak = 1 THEN '1' "
           "WHEN s1_max_loss_streak = 2 THEN '2' ELSE '3+' END", "longest losing streak in first session"),
        by("CASE s1_last_score WHEN 1 THEN 'won' WHEN 0.5 THEN 'drew' ELSE 'lost' END",
           "last game of first session", "CASE grp WHEN 'won' THEN 1 WHEN 'drew' THEN 2 ELSE 3 END"),
        by("first_speed", "speed of first game"),
        by("CASE WHEN first_tournament THEN 'tournament' ELSE 'casual pairing' END", "first game context"),
    ], ignore_index=True)
    signals.to_csv(tables / "early_signals.csv", index=False)

    # Same first-session-streak comparison, holding day-0 volume fixed (streaks need games).
    streak_strat = con.execute("""
        SELECT CASE WHEN s1_games <= 3 THEN '2-3' WHEN s1_games <= 6 THEN '4-6' ELSE '7+' END AS first_session_games,
               CASE WHEN s1_max_loss_streak <= 1 THEN '0-1' WHEN s1_max_loss_streak = 2 THEN '2' ELSE '3+' END AS max_loss_streak,
               count(*) AS players, sum(came_back::INT) AS came_back, sum(w1::INT) AS w1
        FROM cohort WHERE s1_games >= 2 GROUP BY ALL ORDER BY 1, 2
    """).df()
    streak_strat = add_ci(streak_strat, "came_back", "players", "came_back_rate")
    streak_strat.to_csv(tables / "first_session_streak_by_volume.csv", index=False)
    streak_mid = con.execute("""
        SELECT 'loss streak, first session (2-6 games)' AS feature,
               CASE WHEN s1_max_loss_streak <= 1 THEN '0-1' WHEN s1_max_loss_streak = 2 THEN '2' ELSE '3+' END AS grp,
               count(*) AS players, sum(came_back::INT) AS came_back
        FROM cohort WHERE s1_games BETWEEN 2 AND 6 GROUP BY ALL ORDER BY 2
    """).df()
    streak_mid = add_ci(streak_mid, "came_back", "players", "came_back_rate")

    # Players who lost their first game also tended to face stronger opponents. Compare won vs lost
    # within bands of the pre-game expected score, then average the gaps weighted by band size.
    by_strength = con.execute("""
        SELECT CASE WHEN first_expected < 0.35 THEN '1: <0.35' WHEN first_expected < 0.45 THEN '2: 0.35-0.45'
                    WHEN first_expected < 0.55 THEN '3: 0.45-0.55' WHEN first_expected < 0.65 THEN '4: 0.55-0.65'
                    ELSE '5: >=0.65' END AS expected_score_band,
               count(*) FILTER (WHERE first_score = 1) AS won_n,
               avg(came_back::INT) FILTER (WHERE first_score = 1) AS won_came_back,
               count(*) FILTER (WHERE first_score = 0) AS lost_n,
               avg(came_back::INT) FILTER (WHERE first_score = 0) AS lost_came_back
        FROM cohort WHERE first_expected IS NOT NULL GROUP BY 1 ORDER BY 1
    """).df()
    by_strength["gap"] = by_strength.won_came_back - by_strength.lost_came_back
    by_strength["se"] = (by_strength.won_came_back * (1 - by_strength.won_came_back) / by_strength.won_n
                         + by_strength.lost_came_back * (1 - by_strength.lost_came_back) / by_strength.lost_n) ** 0.5
    w = by_strength.won_n + by_strength.lost_n
    adj_gap = float((by_strength.gap * w).sum() / w.sum())
    adj_se = float(((by_strength.se * w / w.sum()) ** 2).sum() ** 0.5)
    by_strength.to_csv(tables / "first_result_by_opponent_strength.csv", index=False)

    # 4. Platform engagement (all non-bot players): average DAU, MAU and stickiness by month.
    platform = con.execute("""
        WITH d AS (SELECT CAST(ts AS DATE) AS day, count(DISTINCT player) AS dau, count(*) AS player_games
                   FROM player_games GROUP BY 1),
        m AS (SELECT strftime(ts, '%Y-%m') AS month, count(DISTINCT player) AS mau FROM player_games GROUP BY 1)
        SELECT strftime(day, '%Y-%m') AS month, avg(dau) AS avg_dau, any_value(m.mau) AS mau,
               avg(dau) / any_value(m.mau) AS stickiness, sum(player_games) / 2 AS games
        FROM d JOIN m ON strftime(d.day, '%Y-%m') = m.month GROUP BY 1 ORDER BY 1
    """).df()
    platform.to_csv(tables / "platform_engagement.csv", index=False)

    # 5. Sessions.
    sess = con.execute("""
        SELECT count(*) AS sessions, median(n_games) AS median_games, quantile_cont(n_games, 0.9) AS p90_games,
               median(span_s) / 60 AS median_minutes, quantile_cont(span_s, 0.9) / 60 AS p90_minutes,
               avg((n_games = 1)::INT) AS single_game_share
        FROM sessions WHERE NOT end_censored
    """).df()
    sess.to_csv(tables / "sessions.csv", index=False)

    # ---------- Figures ----------
    # Fig 1: weekly retention by cohort month (3 series, direct-labeled + legend).
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=200)
    months = sorted(weekly.cohort_month.unique())
    for i, m in enumerate(months):
        d = weekly[(weekly.cohort_month == m) & (weekly.week >= 1)]
        ax.plot(d.week, d.retention, color=SERIES[i], linewidth=2, label=f"first game in {m}")
    style(ax, "Share of new players active in week k after their first game")
    pct_axis(ax)
    ax.set_xlabel("weeks since first game", color=INK2, fontsize=9)
    ax.set_xlim(1, 52)
    ax.set_ylim(0, None)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper right")
    fig.tight_layout()
    fig.savefig(figs / "weekly_retention.png", facecolor=SURFACE)
    plt.close(fig)

    # Fig 2: came-back rate by first-day signals (bars with 95% CIs, one hue).
    plot_data = pd.concat([signals, streak_mid], ignore_index=True)
    panels = [("first game result", "First game"), ("games on day 0", "Games on day 0"),
              ("loss streak, first session (2-6 games)", "Loss streak, first session\n(sessions of 2-6 games)")]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.8), dpi=200, sharey=True)
    for ax, (feat, title) in zip(axes, panels):
        d = plot_data[plot_data.feature == feat].reset_index(drop=True)
        x = range(len(d))
        ax.bar(x, d.came_back_rate, color=SERIES[0], width=0.6)
        ax.errorbar(x, d.came_back_rate, yerr=[(d.came_back_rate - d.came_back_rate_lo).clip(lower=0),
                                               (d.came_back_rate_hi - d.came_back_rate).clip(lower=0)],
                    fmt="none", ecolor=INK2, elinewidth=1, capsize=3)
        for xi, v in zip(x, d.came_back_rate):
            ax.text(xi, v + 0.03, f"{v:.0%}", ha="center", color=INK, fontsize=8)
        ax.set_xticks(list(x), d.grp)
        style(ax, title)
        pct_axis(ax)
    axes[0].set_ylabel("came back after day 0", color=INK2, fontsize=9)
    axes[0].set_ylim(0, 1)
    fig.suptitle("Share of new players who came back after their first day (95% CI)", x=0.01, ha="left",
                 color=INK, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(figs / "early_signals.png", facecolor=SURFACE)
    plt.close(fig)

    # Fig 3: platform stickiness (single series).
    fig, ax = plt.subplots(figsize=(8, 3.4), dpi=200)
    ax.plot(platform.month, platform.stickiness, color=SERIES[0], linewidth=2, marker="o", markersize=4)
    style(ax, "Stickiness: average DAU / MAU, all Chess960 players")
    pct_axis(ax)
    ax.set_ylim(0, platform.stickiness.max() * 1.3)
    ax.set_xticks(range(0, len(platform), 3), platform.month.iloc[::3])
    fig.tight_layout()
    fig.savefig(figs / "stickiness.png", facecolor=SURFACE)
    plt.close(fig)

    return {"cohort": n_cohort, "dayn": dayn, "weekly": weekly, "returners": weekly_returners,
            "signals": signals, "streak_strat": streak_strat, "platform": platform, "sessions": sess,
            "by_strength": by_strength, "adj_gap": adj_gap, "adj_se": adj_se}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--panel", type=Path, default=Path("data/panel"))
    p.add_argument("--out", type=Path, default=Path("reports"))
    args = p.parse_args()
    r = run(args.panel, args.out)
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print(f"cohort: {r['cohort']:,}")
    print(r["dayn"].round(3).to_string(index=False))
    print(r["sessions"].round(2).to_string(index=False))
    cols = ["feature", "grp", "players", "came_back_rate", "came_back_rate_lo", "came_back_rate_hi",
            "w1_rate", "w4_rate", "mean_first_expected_score"]
    print(r["signals"][cols].round(3).to_string(index=False))
    print(r["streak_strat"].round(3).to_string(index=False))
    print(r["by_strength"].round(3).to_string(index=False))
    print(f"won-minus-lost came-back gap, adjusted for opponent strength: {r['adj_gap']:.3f} "
          f"(95% CI {r['adj_gap'] - 1.96 * r['adj_se']:.3f} to {r['adj_gap'] + 1.96 * r['adj_se']:.3f})")
    print(r["platform"].round(3).to_string(index=False))
    w = r["weekly"]
    print(w[w.week.isin([1, 2, 4, 8, 12, 26, 52])].pivot(index="week", columns="cohort_month",
                                                        values="retention").round(3))
    rr = r["returners"]
    print(rr[rr.week.isin([1, 2, 4, 8, 12, 26, 52])][["week", "players", "retention"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
