#!/bin/zsh
# Downloads 18 months of Lichess rated Chess960 games (Apr 2025 - Sep 2026) into data/raw.
# Safe to re-run: finished files are kept, partial ones resume.
cd "$(dirname "$0")"
mkdir -p data/raw
LOG=data/raw/_download.log
echo "start $(date)" >> $LOG
for m in 2025-{04..12} 2026-{01..09}; do
  f=lichess_db_chess960_rated_$m.pgn.zst
  [ -f data/raw/$f ] && { echo "skip $f"; continue; }
  echo "=== $f"
  if curl -fL -C - --retry 3 -o data/raw/$f.part "https://database.lichess.org/chess960/$f"; then
    mv data/raw/$f.part data/raw/$f && echo "ok $f" >> $LOG
  else
    echo "FAIL $f" >> $LOG
  fi
done
echo "done $(date)" >> $LOG
echo
echo "All downloads finished. You can close this window."
