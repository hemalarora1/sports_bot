#!/usr/bin/env bash
# Live-dump every OptiTrack rigid-body key from Redis at 10 Hz.
# macOS-friendly replacement for `watch -n 0.1 redis-cli ...` (no GNU watch
# needed, no GNU date %N needed).
#
#   ./sports_bot/scripts/watch_ball.sh           # all rigid bodies
#   ./sports_bot/scripts/watch_ball.sh 8         # just ID 8

set -u

ID="${1:-}"
if [[ -n "$ID" ]]; then
    PATTERN="sai2::optitrack::rigid_body_pos::${ID}"
else
    PATTERN='sai2::optitrack::rigid_body_pos::*'
fi

while true; do
    clear
    # macOS-portable timestamp (BSD date has no %N; use python for ms).
    ts="$(python3 -c 'import time; t=time.time(); import datetime; print(datetime.datetime.fromtimestamp(t).strftime("%H:%M:%S.")+f"{int((t%1)*1000):03d}")')"
    printf '=== %s ===\n' "$ts"
    keys="$(redis-cli --scan --pattern "$PATTERN" | sort)"
    if [[ -z "$keys" ]]; then
        printf '(no keys matching %s — is the streamer running?)\n' "$PATTERN"
    else
        while IFS= read -r k; do
            printf '%-50s %s\n' "$k" "$(redis-cli get "$k")"
        done <<<"$keys"
    fi
    sleep 0.1
done
