"""End-to-end check on a tiny hand-made PGN where every expected value is known.

Run:  python tests/test_pipeline.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import zstandard

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from build_panel import build  # noqa: E402
from parse_headers import parse_file  # noqa: E402


def pgn(site, date, time, white, black, result, welo, belo, wdiff=None, bdiff=None,
        tc="180+2", term="Normal", wtitle=None, btitle=None, event="Rated Chess960 game"):
    lines = [f'[Event "{event}"]', f'[Site "https://lichess.org/{site}"]', '[Date "????.??.??"]',
             f'[White "{white}"]', f'[Black "{black}"]', f'[Result "{result}"]',
             f'[UTCDate "{date}"]', f'[UTCTime "{time}"]', f'[WhiteElo "{welo}"]', f'[BlackElo "{belo}"]']
    if wdiff is not None:
        lines += [f'[WhiteRatingDiff "{wdiff:+d}"]', f'[BlackRatingDiff "{bdiff:+d}"]']
    if wtitle:
        lines.append(f'[WhiteTitle "{wtitle}"]')
    if btitle:
        lines.append(f'[BlackTitle "{btitle}"]')
    lines += ['[Variant "Chess960"]', f'[TimeControl "{tc}"]', f'[Termination "{term}"]', "",
              f"1. e4 {{ [%clk 0:03:00] }} e5 {result}", "", ""]
    return "\n".join(lines)


GAMES = [
    # burn-in month: veteran exists before the cohort window
    pgn("g01", "2025.01.05", "08:00:00", "Veteran", "Other", "1-0", 1800, 1700, 5, -5),
    # newbie: 3 losses in one session, then a win and a loss after a 100-minute break
    pgn("g02", "2025.02.03", "10:00:00", "Newbie", "Veteran", "0-1", 1500, 1800, -10, 2),
    pgn("g03", "2025.02.03", "10:10:00", "Veteran", "Newbie", "1-0", 1802, 1490, 2, -10),
    pgn("g04", "2025.02.03", "10:20:00", "Newbie", "Other", "0-1", 1480, 1700, -10, 3, term="Time forfeit"),
    # BOT game in the gap: must be dropped and must not merge the two sessions
    pgn("g05", "2025.02.03", "11:00:00", "Newbie", "SomeBot", "1-0", 1470, 2000, btitle="BOT"),
    pgn("g06", "2025.02.03", "12:00:00", "Newbie", "Other", "1-0", 1470, 1703, 15, -15),
    pgn("g07", "2025.02.03", "12:05:00", "Other", "Newbie", "1-0", 1688, 1485, 5, -5),
    pgn("g08", "2025.02.04", "09:00:00", "Newbie", "Veteran", "0-1", 1480, 1804, -5, 1, tc="60+0"),
    pgn("g09", "2025.02.20", "18:00:00", "Newbie", "Veteran", "1/2-1/2", 1475, 1805, 3, -1),
    # first seen in Feb but not at the provisional 1500 rating: excluded from cohort
    pgn("g10", "2025.02.10", "12:00:00", "Notnew", "Other", "1-0", 1700, 1690, 5, -5),
    # first seen in March: outside the one-month cohort window
    pgn("g11", "2025.03.10", "12:00:00", "Late", "Other", "0-1", 1500, 1700, -10, 3),
    # unfinished game and unknown rating: dropped / kept with null rating respectively
    pgn("g12", "2025.03.15", "12:00:00", "Veteran", "Other", "*", 1805, 1690),
    pgn("g13", "2025.03.20", "12:00:00", "Veteran", "Anon", "1-0", 1805, "?", 2, 0),
    # last game sets the end of the data window
    pgn("g14", "2025.03.31", "23:00:00", "Veteran", "Other", "1-0", 1807, 1690, 2, -2),
]


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        raw = tmp / "lichess_db_chess960_rated_2025-test.pgn.zst"
        raw.write_bytes(zstandard.ZstdCompressor().compress("".join(GAMES).encode()))

        kept, skipped = parse_file(raw, tmp / "games")
        assert (kept, skipped) == (14, 0), (kept, skipped)

        con = build(tmp / "games", tmp / "panel", gap_min=30, burn_in=1, cohort_months=1,
                    require_provisional=True)

        assert con.execute("SELECT count(*) FROM player_games WHERE opponent = 'somebot'").fetchone()[0] == 0
        assert con.execute("SELECT count(*) FROM player_games WHERE game_id = 'g12'").fetchone()[0] == 0

        rows = con.execute("""
            SELECT game_id, session_no, loss_streak, session_loss_streak, session_end, speed, lost_on_time
            FROM player_games WHERE player = 'newbie' ORDER BY game_no
        """).fetchall()
        got = [(r[0], r[1], r[2], r[3], r[4]) for r in rows]
        expected = [
            ("g02", 1, 1, 1, False),
            ("g03", 1, 2, 2, False),
            ("g04", 1, 3, 3, True),
            ("g06", 2, 0, 0, False),
            ("g07", 2, 1, 1, True),
            ("g08", 3, 2, 1, True),   # streak continues across sessions, session streak resets
            ("g09", 4, 0, 0, True),   # a draw ends a losing streak
        ]
        assert got == expected, got
        assert rows[2][6] is True and rows[5][5] == "bullet" and rows[0][5] == "blitz"

        cohort = {r[0] for r in con.execute("SELECT player FROM players WHERE in_cohort").fetchall()}
        assert cohort == {"newbie"}, cohort

        weeks = con.execute("""
            SELECT week, n_games, n_sessions, max_loss_streak, lost_on_time, rating_change, rating_end, cum_games
            FROM player_weeks WHERE player = 'newbie' ORDER BY week
        """).fetchall()
        assert [w[0] for w in weeks] == list(range(8)), weeks       # Feb 3 10:00 -> Mar 31 23:00 = 8 full weeks
        assert [w[1] for w in weeks] == [6, 0, 1, 0, 0, 0, 0, 0], weeks
        assert weeks[1][3] is None  # no games that week: streak is missing, not zero
        assert weeks[0][2:6] == (3, 3, 1, -25), weeks[0]
        assert weeks[0][6] == 1475 and weeks[1][6] == 1475 and weeks[2][6] == 1478, weeks
        assert weeks[-1][7] == 7

        sess = con.execute("SELECT count(*), sum(end_censored::INT) FROM sessions WHERE player = 'newbie'").fetchone()
        assert sess == (4, 0), sess
        print("all checks passed")


if __name__ == "__main__":
    main()
