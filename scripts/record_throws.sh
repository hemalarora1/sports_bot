#!/usr/bin/env bash
# Bring up the OptiTrack streamer (if not already running) and start the ball
# tracker recorder. Cleans up the streamer when the recorder exits.
#
# Assumes the 'opensai' conda env is already active.
#
#   ./sports_bot/scripts/record_throws.sh                          # defaults
#   RIGID_BODY_ID=1 ./sports_bot/scripts/record_throws.sh
#   ./sports_bot/scripts/record_throws.sh -o my_run.npz
#
# Env overrides:
#   MOTIVE_SERVER    Motive PC IP. Default 172.24.69.102 (SRC Kitchen).
#   RIGID_BODY_ID    Streaming ID for the PickleBall. Default 8.
#   MY_IP            Laptop IP to register with Motive's unicast list.
#                    Default: auto-detected from en0. Override when your
#                    en0 isn't the SRC-side interface or you want to pin
#                    to your assigned static SRC IP (e.g. 172.24.68.204).
#   STREAMER_MODE    NatNet transmission type: 'u' (unicast) or 'm' (multicast).
#                    Default 'm' — Motive is in Multicast as of 2026-05-19.
#                    Requires the laptop to be on SRC wifi with a static IP on
#                    that subnet (172.24.68.x for our setup) — multicast does
#                    not route across subnets. If you ever flip Motive back
#                    to Unicast, pass STREAMER_MODE=u; otherwise the command
#                    port handshakes but no data arrives.

set -euo pipefail

# Script lives at <OpenSai>/sports_bot/scripts/record_throws.sh.
# Don't use `git rev-parse` here: sports_bot is its own git repo nested inside
# OpenSai, so git would return the sports_bot dir, not OpenSai.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

MOTIVE_SERVER="${MOTIVE_SERVER:-172.24.69.102}"
RIGID_BODY_ID="${RIGID_BODY_ID:-8}"
STREAMER_MODE="${STREAMER_MODE:-m}"
KEY="sai2::optitrack::rigid_body_pos::${RIGID_BODY_ID}"

if [[ "$STREAMER_MODE" != "u" && "$STREAMER_MODE" != "m" ]]; then
    echo "[startup] STREAMER_MODE must be 'u' (unicast) or 'm' (multicast); got '$STREAMER_MODE'" >&2
    exit 1
fi

# -------- Redis -------------------------------------------------------------
if ! redis-cli ping >/dev/null 2>&1; then
    echo "[startup] redis not running; trying brew services start redis"
    brew services start redis >/dev/null 2>&1 || {
        echo "[startup] couldn't auto-start redis. run: redis-server &"
        exit 1
    }
    sleep 1
fi
echo "[startup] redis OK"

# -------- Reap stale streamers ---------------------------------------------
# A prior streamer can stick around in two ways: (a) the script that spawned
# it died ungracefully and the SIGTERM never fired, (b) the streamer is in a
# socket-retry loop and ignored a plain `kill`. Either way we end up with a
# zombie holding UDP port 1511, and the next streamer's bind() fails with
# "Address already in use." Always reap before bringing one up.
stale_pids="$(pgrep -f 'StreamDataSkeleton\.py' || true)"
if [[ -n "$stale_pids" ]]; then
    echo "[startup] reaping stale StreamDataSkeleton process(es): $stale_pids"
    # SIGTERM first, escalate to SIGKILL — Python in a retry loop ignores TERM.
    kill $stale_pids 2>/dev/null || true
    sleep 0.5
    still_alive="$(pgrep -f 'StreamDataSkeleton\.py' || true)"
    if [[ -n "$still_alive" ]]; then
        echo "[startup] SIGTERM didn't take, escalating to SIGKILL: $still_alive"
        kill -9 $still_alive 2>/dev/null || true
        sleep 0.3
    fi
fi

# Clear stale OptiTrack keys so an old, dead streamer's positions can't masquerade as live.
n_stale=$(redis-cli --scan --pattern 'sai2::optitrack::*' | wc -l | tr -d ' ')
if [[ "$n_stale" -gt 0 ]]; then
    redis-cli --scan --pattern 'sai2::optitrack::*' | xargs -I{} redis-cli del {} >/dev/null
    echo "[startup] cleared $n_stale stale optitrack key(s)"
fi

# -------- Streamer ----------------------------------------------------------
# A previously-running streamer would already be publishing fresh values to $KEY.
# We just deleted the key, so check whether anything writes to it within 1 s.
streamer_pid=""
sleep 1

cleanup() {
    if [[ -n "$streamer_pid" ]]; then
        echo "[cleanup] stopping streamer pid=$streamer_pid"
        kill "$streamer_pid" 2>/dev/null || true
        # Don't block forever on a stuck Python child — give it half a
        # second to exit, then SIGKILL.
        for _ in 1 2 3 4 5; do
            kill -0 "$streamer_pid" 2>/dev/null || break
            sleep 0.1
        done
        if kill -0 "$streamer_pid" 2>/dev/null; then
            echo "[cleanup] streamer pid=$streamer_pid didn't exit, SIGKILL"
            kill -9 "$streamer_pid" 2>/dev/null || true
        fi
        wait "$streamer_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# -------- Streamer ----------------------------------------------------------
if [[ -n "$(redis-cli get "$KEY" 2>/dev/null || true)" ]]; then
    echo "[startup] streamer already running (saw $KEY appear)"
else
    if [[ -z "${MY_IP:-}" ]]; then
        MY_IP="$(ifconfig en0 2>/dev/null | awk '/inet / {print $2}')"
    fi
    if [[ -z "${MY_IP:-}" ]]; then
        echo "[startup] couldn't read en0 IP and MY_IP not set. Connect to wifi" >&2
        echo "          (Stanford or SRC) first, or pass MY_IP=… explicitly." >&2
        exit 1
    fi
    case "$STREAMER_MODE" in
        u) mode_label="unicast"   ;;
        m) mode_label="multicast" ;;
    esac
    STREAMER_LOG="/tmp/optitrack_streamer_$(date +%Y%m%d_%H%M%S).log"
    echo "[startup] starting streamer: $MY_IP -> $MOTIVE_SERVER ($mode_label)"
    echo "[startup] log: $STREAMER_LOG"
    (
        cd "$REPO_ROOT/sports_bot/optitrack"
        PYTHONPATH=drivers/PythonClient \
            exec python -u StreamDataSkeleton.py "$MOTIVE_SERVER" "$MY_IP" "$STREAMER_MODE"
    ) >"$STREAMER_LOG" 2>&1 &
    streamer_pid=$!
    echo "[startup] streamer pid=$streamer_pid"

    # Wait up to 10 s for the first sample. The cleanup trap above will reap
    # the streamer on any exit path — including a failure here.
    for _ in $(seq 1 20); do
        if [[ -n "$(redis-cli get "$KEY" 2>/dev/null || true)" ]]; then
            echo "[startup] streamer publishing on $KEY"
            break
        fi
        # Bail early if the streamer died before publishing — e.g. socket
        # bind error. No point waiting the full 10 s.
        if ! kill -0 "$streamer_pid" 2>/dev/null; then
            echo "[startup] streamer pid=$streamer_pid exited before publishing." >&2
            echo "[startup] check $STREAMER_LOG" >&2
            exit 1
        fi
        sleep 0.5
    done
    if [[ -z "$(redis-cli get "$KEY" 2>/dev/null || true)" ]]; then
        echo "[startup] streamer did not publish $KEY within 10 s. Check $STREAMER_LOG"
        exit 1
    fi
fi

# -------- Recorder ----------------------------------------------------------
echo "[startup] starting recorder (Ctrl-C to stop and save)"
exec python -m sports_bot.state_machine.ball_tracker_test record \
    --ball-source optitrack \
    --optitrack-rigid-body-id "$RIGID_BODY_ID" \
    "$@"
