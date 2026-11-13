"""Download monthly Lichess rated-game dumps (.pgn.zst) and verify SHA-256.

Example:
    python src/download.py --variant chess960 --start 2025-04 --end 2026-09 --out data/raw
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

BASE = "https://database.lichess.org"


def month_range(start: str, end: str) -> list[str]:
    y, m = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    out = []
    while (y, m) <= (ey, em):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def file_name(variant: str, month: str) -> str:
    return f"lichess_db_{variant}_rated_{month}.pgn.zst"


def fetch_checksums(variant: str) -> dict[str, str]:
    url = f"{BASE}/{variant}/sha256sums.txt"
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            text = r.read().decode()
    except Exception as e:  # checksums are a nice-to-have
        print(f"[warn] could not fetch checksums ({e}); skipping verification")
        return {}
    sums = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            sums[parts[-1].lstrip("*")] = parts[0].lower()
    return sums


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as r, tmp.open("wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        for chunk in iter(lambda: r.read(1 << 20), b""):
            f.write(chunk)
            done += len(chunk)
            if total:
                sys.stdout.write(f"\r  {dest.name}: {done / total:6.1%} of {total / 1e6:,.0f} MB")
                sys.stdout.flush()
    print()
    tmp.rename(dest)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="chess960",
                   help="standard, chess960, atomic, antichess, crazyhouse, ...")
    p.add_argument("--start", required=True, help="first month, YYYY-MM")
    p.add_argument("--end", required=True, help="last month, YYYY-MM")
    p.add_argument("--out", default="data/raw")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sums = fetch_checksums(args.variant)

    for month in month_range(args.start, args.end):
        name = file_name(args.variant, month)
        dest = out / name
        expected = sums.get(name)
        if dest.exists() and (not expected or sha256_of(dest) == expected):
            print(f"[skip] {name} already downloaded")
            continue
        print(f"[get ] {name}")
        download(f"{BASE}/{args.variant}/{name}", dest)
        if expected:
            if sha256_of(dest) != expected:
                dest.unlink()
                raise SystemExit(f"[fail] checksum mismatch for {name}; file removed, rerun to retry")
            print("  checksum ok")


if __name__ == "__main__":
    main()
