"""Turn parsed game headers into analysis tables with DuckDB.

Outputs (Parquet, in --out):
  player_games.parquet  one row per player per game, with sessions and loss streaks
  sessions.parquet      one row per play session
  players.parquet       one row per player, with the new-player cohort flag
  player_weeks.parquet  cohort players x weeks since first game (zero-filled), the panel
                        for the state-transition and survival models

Example:
    python src/build_panel.py --games data/games --out data/panel
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import duckdb

WEEK_S = 7 * 24 * 3600
TABLES = ("player_games", "sessions", "players", "player_weeks")


class OutOfTime(Exception):
    pass


class Runner:
    """Runs CREATE TABLE steps once. With a .duckdb file, a rerun resumes after the last finished step."""

    def __init__(self, con: duckdb.DuckDBPyConnection, budget_s: float | None):
        self.con, self.budget_s, self.t0 = con, budget_s, time.time()

    def exists(self, name: str) -> bool:
        return self.con.execute(
            "SELECT count(*) FROM duckdb_tables() WHERE table_name = ?", [name]).fetchone()[0] > 0

    def _check_budget(self) -> None:
        if self.budget_s is not None and time.time() - self.t0 > self.budget_s:
            raise OutOfTime

    def table(self, name: str, sql: str) -> None:
        if self.exists(name):
            return
        self._check_budget()
        t = time.time()
        self.con.execute(f"CREATE TABLE {name} AS {sql}")
        print(f"  built {name} in {time.time() - t:.0f}s", flush=True)

    def copy(self, name: str, dest: Path) -> None:
        if dest.exists():
            return
        self._check_budget()
        t = time.time()
        tmp = dest.with_suffix(".parquet.part")
        self.con.execute(f"COPY {name} TO '{tmp.as_posix()}' (FORMAT parquet, COMPRESSION zstd)")
        tmp.rename(dest)
        print(f"  wrote {dest.name} in {time.time() - t:.0f}s", flush=True)


def build(games_dir: Path, out: Path, gap_min: int, burn_in: int, cohort_months: int,
          require_provisional: bool, db_path: str = ":memory:",
          budget_s: float | None = None) -> duckdb.DuckDBPyConnection:
    """Build all tables. With budget_s, stops before starting a step once that many seconds
    have passed and raises OutOfTime; rerun with the same db_path to continue."""
    out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(db_path)
    con.execute("SET TimeZone = 'UTC'")  # Lichess timestamps are UTC; keep dates and months machine-independent
    con.execute("SET preserve_insertion_order = false")
    r = Runner(con, budget_s)
    gap_s = gap_min * 60

    con.execute(f"CREATE OR REPLACE VIEW games AS SELECT * FROM read_parquet('{games_dir.as_posix()}/*.parquet')")
    r.table("bounds", "SELECT min(ts) AS data_start, max(ts) AS data_end FROM games")

    # 1. Long format: one row per player per finished game. BOT accounts are dropped.
    if not r.exists("player_games") and not r.exists("player_games_ordered"):
        r.table("player_games_raw", """
    WITH clean AS (
        SELECT * FROM games
        WHERE coalesce(white_title, '') <> 'BOT' AND coalesce(black_title, '') <> 'BOT'
          AND result IN ('1-0', '0-1', '1/2-1/2')
    ), long AS (
        SELECT game_id, event, ts, lower(white) AS player, lower(black) AS opponent, 'white' AS color,
               white_elo AS elo, black_elo AS opp_elo, white_rating_diff AS rating_diff,
               (CASE result WHEN '1-0' THEN 1.0 WHEN '0-1' THEN 0.0 ELSE 0.5 END)::DOUBLE AS score,
               tc_base, tc_inc, termination
        FROM clean
        UNION ALL
        SELECT game_id, event, ts, lower(black), lower(white), 'black',
               black_elo, white_elo, black_rating_diff,
               (CASE result WHEN '0-1' THEN 1.0 WHEN '1-0' THEN 0.0 ELSE 0.5 END)::DOUBLE,
               tc_base, tc_inc, termination
        FROM clean
    )
    SELECT * EXCLUDE (event),
        -- Lichess speed buckets use estimated duration = base + 40 * increment
        CASE WHEN tc_base IS NULL THEN 'correspondence'
             WHEN tc_base + 40 * tc_inc < 30   THEN 'ultrabullet'
             WHEN tc_base + 40 * tc_inc < 180  THEN 'bullet'
             WHEN tc_base + 40 * tc_inc < 480  THEN 'blitz'
             WHEN tc_base + 40 * tc_inc < 1500 THEN 'rapid'
             ELSE 'classical' END AS speed,
        1.0 / (1.0 + pow(10, (opp_elo - elo) / 400.0)) AS expected_score,
        (event ILIKE '%arena%' OR event ILIKE '%swiss%' OR event ILIKE '%tournament%') AS is_tournament,
        (score = 0 AND termination = 'Time forfeit') AS lost_on_time,
        (termination = 'Abandoned') AS abandoned
    FROM long
    """)

    # 2. Sessions (gap > gap_min starts a new one) and loss streaks.
    #    loss_streak: consecutive losses ending at this game, across sessions.
    #    session_loss_streak: the same, reset at the start of each session.
    #    session_end: last game of its session; session_end_censored: data ends too soon to know.
    if not r.exists("player_games"):
        r.table("player_games_ordered", f"""
    WITH o AS (
        SELECT *, date_diff('second', lag(ts) OVER w, ts) AS gap_s,
                  date_diff('second', ts, lead(ts) OVER w) AS next_gap_s
        FROM player_games_raw
        WINDOW w AS (PARTITION BY player ORDER BY ts, game_id)
    )
        SELECT *,
            (sum(CASE WHEN gap_s IS NULL OR gap_s > {gap_s} THEN 1 ELSE 0 END) OVER w)::INT AS session_no,
            row_number() OVER w AS game_no,
            sum(CASE WHEN score > 0 THEN 1 ELSE 0 END) OVER w AS nonloss_grp
        FROM o
        WINDOW w AS (PARTITION BY player ORDER BY ts, game_id ROWS UNBOUNDED PRECEDING)
    """)
        r.con.execute("DROP TABLE IF EXISTS player_games_raw")
        r.table("player_games", f"""
    WITH s2 AS (
        SELECT *,
            (sum(CASE WHEN score = 0 THEN 1 ELSE 0 END) OVER (
                PARTITION BY player, nonloss_grp ORDER BY ts, game_id ROWS UNBOUNDED PRECEDING))::INT AS loss_streak,
            sum(CASE WHEN score > 0 THEN 1 ELSE 0 END) OVER (
                PARTITION BY player, session_no ORDER BY ts, game_id ROWS UNBOUNDED PRECEDING) AS s_grp
        FROM player_games_ordered
    )
    SELECT s2.* EXCLUDE (nonloss_grp, s_grp),
        (sum(CASE WHEN score = 0 THEN 1 ELSE 0 END) OVER (
            PARTITION BY player, session_no, s_grp ORDER BY ts, game_id ROWS UNBOUNDED PRECEDING))::INT AS session_loss_streak,
        (next_gap_s IS NULL OR next_gap_s > {gap_s}) AS session_end,
        (next_gap_s IS NULL AND date_diff('second', ts, b.data_end) <= {gap_s}) AS session_end_censored
    FROM s2, bounds b
    """)
    con.execute("DROP TABLE IF EXISTS player_games_ordered")

    r.table("sessions", """
    SELECT player, session_no, min(ts) AS start_ts, max(ts) AS last_game_ts,
           date_diff('second', min(ts), max(ts)) AS span_s, count(*) AS n_games,
           sum(score) AS points, sum(expected_score) AS expected_points,
           max(session_loss_streak) AS max_loss_streak,
           arg_max(score, game_no) AS last_game_score,
           bool_or(session_end_censored) AS end_censored
    FROM player_games GROUP BY player, session_no
    """)

    # 3. Players and the new-player cohort.
    #    The dump has no account-creation date, so "new" = first seen after a burn-in window
    #    (no games in the first `burn_in` months) and, optionally, a first rating of exactly 1500
    #    (the provisional starting rating on Lichess).
    provisional = "AND first_elo = 1500" if require_provisional else ""
    r.table("players", f"""
    WITH p AS (
        SELECT player, min(ts) AS first_ts, max(ts) AS last_ts, count(*) AS n_games,
               arg_min(elo, game_no) AS first_elo
        FROM player_games GROUP BY player
    )
    SELECT p.*,
        floor(date_diff('second', p.first_ts, b.data_end) / {WEEK_S})::INT AS n_full_weeks,
        (p.first_ts >= date_trunc('month', b.data_start) + to_months({burn_in})
         AND p.first_ts < date_trunc('month', b.data_start) + to_months({burn_in + cohort_months})
         {provisional}) AS in_cohort
    FROM p, bounds b
    """)

    # 4. Weekly panel: week k = [first_ts + 7k days, first_ts + 7(k+1) days), only fully observed weeks.
    r.table("player_weeks", f"""
    WITH cohort AS (
        SELECT player, first_ts, n_full_weeks FROM players WHERE in_cohort AND n_full_weeks > 0
    ), grid AS (
        SELECT player, unnest(range(0, n_full_weeks)) AS week FROM cohort
    ), agg AS (
        SELECT g.player,
            floor(date_diff('second', c.first_ts, g.ts) / {WEEK_S})::INT AS week,
            count(*) AS n_games,
            count(DISTINCT g.session_no) AS n_sessions,
            count(DISTINCT CAST(g.ts AS DATE)) AS active_days,
            sum((g.score = 1)::INT) AS wins,
            sum((g.score = 0)::INT) AS losses,
            sum((g.score = 0.5)::INT) AS draws,
            sum(g.score) AS points,
            sum(g.expected_score) AS expected_points,
            sum(coalesce(g.rating_diff, 0)) AS rating_change,
            arg_max(g.elo + coalesce(g.rating_diff, 0), g.game_no) AS rating_end,
            max(g.loss_streak) AS max_loss_streak,
            max(g.session_loss_streak) AS max_session_loss_streak,
            avg((g.speed IN ('ultrabullet', 'bullet'))::INT) AS share_bullet,
            avg((g.speed = 'blitz')::INT) AS share_blitz,
            avg((g.speed IN ('rapid', 'classical', 'correspondence'))::INT) AS share_slow,
            avg(g.is_tournament::INT) AS share_tournament,
            sum(g.lost_on_time::INT) AS lost_on_time,
            sum(g.abandoned::INT) AS abandoned
        FROM player_games g JOIN cohort c USING (player)
        GROUP BY ALL
    )
    SELECT grid.player, grid.week,
        coalesce(n_games, 0) AS n_games,
        coalesce(n_sessions, 0) AS n_sessions,
        coalesce(active_days, 0) AS active_days,
        coalesce(wins, 0)::INT AS wins, coalesce(losses, 0)::INT AS losses, coalesce(draws, 0)::INT AS draws,
        points, expected_points,
        points - expected_points AS perf_vs_expected,   -- >0: did better than ratings predicted
        coalesce(rating_change, 0)::INT AS rating_change,
        last_value(rating_end IGNORE NULLS) OVER w AS rating_end,
        max_loss_streak, max_session_loss_streak,
        share_bullet, share_blitz, share_slow, share_tournament,
        coalesce(lost_on_time, 0)::INT AS lost_on_time,
        coalesce(abandoned, 0)::INT AS abandoned,
        (sum(coalesce(n_games, 0)) OVER w)::INT AS cum_games
    FROM grid LEFT JOIN agg USING (player, week)
    WINDOW w AS (PARTITION BY grid.player ORDER BY grid.week ROWS UNBOUNDED PRECEDING)
    ORDER BY grid.player, grid.week
    """)

    for table in TABLES:
        r.copy(table, out / (table + ".parquet"))
    return con


def summarize(con: duckdb.DuckDBPyConnection) -> None:
    b = con.execute("SELECT strftime(data_start, '%Y-%m-%d %H:%M'), strftime(data_end, '%Y-%m-%d %H:%M') FROM bounds").fetchone()
    print(f"data window: {b[0]} -> {b[1]}")
    for t in TABLES:
        print(f"  {t:13s} {con.execute(f'SELECT count(*) FROM {t}').fetchone()[0]:>12,} rows")
    n = con.execute("SELECT count(*) FROM players WHERE in_cohort").fetchone()[0]
    print(f"  new-player cohort: {n:,} players")
    print("  share of cohort active by week since first game:")
    rows = con.execute("""
        SELECT week, count(*) AS observed, avg((n_games > 0)::INT) AS active
        FROM player_weeks WHERE week <= 8 GROUP BY week ORDER BY week
    """).fetchall()
    for week, observed, active in rows:
        print(f"    week {week}: {active:6.1%}  (n={observed:,})")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=Path, default=Path("data/games"))
    p.add_argument("--out", type=Path, default=Path("data/panel"))
    p.add_argument("--session-gap-min", type=int, default=30)
    p.add_argument("--burn-in-months", type=int, default=3,
                   help="months at the start used only to rule out existing players")
    p.add_argument("--cohort-months", type=int, default=3,
                   help="new players first seen in this many months after the burn-in")
    p.add_argument("--no-provisional-check", action="store_true",
                   help="do not require a first rating of exactly 1500")
    p.add_argument("--db", default=":memory:",
                   help=".duckdb file to keep the tables; reruns resume after the last finished step")
    p.add_argument("--time-budget", type=float, default=None,
                   help="seconds; do not start a new step after this, exit with code 3 so it can be rerun")
    args = p.parse_args()
    try:
        con = build(args.games, args.out, args.session_gap_min, args.burn_in_months,
                    args.cohort_months, not args.no_provisional_check, args.db, args.time_budget)
    except OutOfTime:
        print("time budget used; rerun the same command to continue")
        raise SystemExit(3)
    summarize(con)


if __name__ == "__main__":
    main()
