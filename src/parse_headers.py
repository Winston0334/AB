"""Stream .pgn.zst dumps and keep only the PGN headers, one Parquet file per month.

Moves are never parsed, so a month of Chess960 takes about a minute.

Example:
    python src/parse_headers.py data/raw/*.pgn.zst --out data/games
"""
from __future__ import annotations

import argparse
import io
import re
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import zstandard

HEADER = re.compile(r'^\[(\w+) "(.*)"\]\s*$')

SCHEMA = pa.schema([
    ("game_id", pa.string()),
    ("event", pa.string()),
    ("ts", pa.timestamp("s", tz="UTC")),
    ("white", pa.string()),
    ("black", pa.string()),
    ("white_title", pa.string()),
    ("black_title", pa.string()),
    ("result", pa.string()),           # "1-0", "0-1", "1/2-1/2", "*"
    ("white_elo", pa.int32()),
    ("black_elo", pa.int32()),
    ("white_rating_diff", pa.int32()),
    ("black_rating_diff", pa.int32()),
    ("tc_base", pa.int32()),           # seconds; null for correspondence ("-")
    ("tc_inc", pa.int32()),
    ("termination", pa.string()),      # Normal, Time forfeit, Abandoned, Rules infraction, Unterminated
])


def to_int(v: str | None) -> int | None:
    if v is None or v in ("?", "", "-"):
        return None
    try:
        return int(v)
    except ValueError:
        return None


def to_ts(date: str | None, time: str | None) -> datetime | None:
    if not date or not time or "?" in date or "?" in time:
        return None
    try:
        return datetime(int(date[0:4]), int(date[5:7]), int(date[8:10]),
                        int(time[0:2]), int(time[3:5]), int(time[6:8]), tzinfo=timezone.utc)
    except ValueError:
        return None


def to_row(h: dict[str, str]) -> dict | None:
    ts = to_ts(h.get("UTCDate"), h.get("UTCTime"))
    if ts is None or "White" not in h or "Black" not in h:
        return None
    base = inc = None
    tc = h.get("TimeControl", "-")
    if "+" in tc:
        base, inc = (to_int(x) for x in tc.split("+", 1))
    site = h.get("Site", "")
    return {
        "game_id": site.rsplit("/", 1)[-1] if site else None,
        "event": h.get("Event"),
        "ts": ts,
        "white": h["White"],
        "black": h["Black"],
        "white_title": h.get("WhiteTitle"),
        "black_title": h.get("BlackTitle"),
        "result": h.get("Result"),
        "white_elo": to_int(h.get("WhiteElo")),
        "black_elo": to_int(h.get("BlackElo")),
        "white_rating_diff": to_int(h.get("WhiteRatingDiff")),
        "black_rating_diff": to_int(h.get("BlackRatingDiff")),
        "tc_base": base,
        "tc_inc": inc,
        "termination": h.get("Termination"),
    }


def iter_headers(path: Path):
    """Yield one header dict per game. A new game starts at each [Event ...] line.

    Reads bytes and only decodes header lines; move text (most of the file) is skipped undecoded.
    """
    with path.open("rb") as fh:
        reader = zstandard.ZstdDecompressor().stream_reader(fh, read_size=1 << 20)
        lines = io.BufferedReader(reader, buffer_size=1 << 20)
        current: dict[str, str] = {}
        for raw in lines:
            if raw[:1] != b"[":
                continue  # move text, comments, blank lines
            m = HEADER.match(raw.decode("utf-8", "replace"))
            if not m:
                continue
            key, value = m.groups()
            if key == "Event" and current:
                yield current
                current = {}
            current[key] = value
        if current:
            yield current


def parse_file(src: Path, out_dir: Path, batch_size: int = 200_000) -> tuple[int, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / (src.name.replace(".pgn.zst", "") + ".parquet")
    tmp = dest.with_suffix(".parquet.part")
    kept = skipped = 0
    batch: list[dict] = []
    with pq.ParquetWriter(tmp, SCHEMA, compression="zstd") as writer:
        for h in iter_headers(src):
            row = to_row(h)
            if row is None:
                skipped += 1
                continue
            batch.append(row)
            if len(batch) >= batch_size:
                writer.write_table(pa.Table.from_pylist(batch, schema=SCHEMA))
                kept += len(batch)
                batch = []
        if batch:
            writer.write_table(pa.Table.from_pylist(batch, schema=SCHEMA))
            kept += len(batch)
    tmp.rename(dest)
    return kept, skipped


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+", type=Path)
    p.add_argument("--out", type=Path, default=Path("data/games"))
    p.add_argument("--force", action="store_true", help="re-parse months already done")
    args = p.parse_args()
    for src in sorted(args.files):
        dest = args.out / (src.name.replace(".pgn.zst", "") + ".parquet")
        if dest.exists() and not args.force:
            print(f"[skip] {dest.name}")
            continue
        kept, skipped = parse_file(src, args.out)
        print(f"[done] {src.name}: {kept:,} games kept, {skipped:,} skipped (missing date/players)")


if __name__ == "__main__":
    main()
