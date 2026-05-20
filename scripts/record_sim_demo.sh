#!/usr/bin/env bash
# Record the pickleball simulation running through one or more full FSM cycles.
#
# Brings up the sim+controller+FSM via pickleball/run.sh, screen-records with
# ffmpeg/avfoundation, fires a few balls, then tears the stack down. Output
# lands at sports_bot/media/sim_demo.mp4 — drop it into the deck (slide 4).
#
# macOS Screen Recording permission is required for the terminal running this
# script. The first run will fail with a permission prompt; grant access in
# System Settings → Privacy & Security → Screen Recording, then re-run.
#
# Usage:
#   ./scripts/record_sim_demo.sh                       # 25 s, screen 0, 3 balls
#   SCREEN=2 DURATION=40 BALLS=4 ./scripts/record_sim_demo.sh
#
# Env overrides:
#   FFMPEG     ffmpeg binary (default: bundled robocasa_fresh)
#   SCREEN     avfoundation video device index (default: 1 = primary screen)
#   DURATION   seconds to record (default: 25)
#   BALLS      ball launches during the recording (default: 3)
#   FRAMERATE  recording fps (default: 30)
#   CRF        x264 quality, lower=better (default: 23)
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPORTS_BOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MEDIA_DIR="$SPORTS_BOT_DIR/media"
mkdir -p "$MEDIA_DIR"

RUN_SH="$SPORTS_BOT_DIR/pickleball/run.sh"
OUT="$MEDIA_DIR/sim_demo.mp4"

FFMPEG="${FFMPEG:-/opt/homebrew/Caskroom/miniconda/base/envs/robocasa_fresh/bin/ffmpeg}"
SCREEN="${SCREEN:-1}"
DURATION="${DURATION:-25}"
BALLS="${BALLS:-3}"
FRAMERATE="${FRAMERATE:-30}"
CRF="${CRF:-23}"

# --- ANSI -------------------------------------------------------------------
if [[ -t 1 ]]; then
    C_RESET=$'\033[0m'; C_BOLD=$'\033[1m'
    C_GREEN=$'\033[32m'; C_YELLOW=$'\033[33m'; C_RED=$'\033[31m'; C_DIM=$'\033[2m'
else
    C_RESET=""; C_BOLD=""; C_GREEN=""; C_YELLOW=""; C_RED=""; C_DIM=""
fi
log()  { echo "${C_DIM}[record]${C_RESET} $*"; }
warn() { echo "${C_YELLOW}[record]${C_RESET} $*" >&2; }
err()  { echo "${C_RED}[record]${C_RESET} $*" >&2; }

# --- preflight --------------------------------------------------------------
if [[ ! -x "$FFMPEG" ]]; then
    err "ffmpeg not found at $FFMPEG (set FFMPEG=...)"; exit 1
fi
if [[ ! -x "$RUN_SH" ]]; then
    err "run.sh not found / not executable at $RUN_SH"; exit 1
fi
if ! redis-cli ping >/dev/null 2>&1; then
    err "redis-server is not running. Start it (\`redis-server\`) and retry."; exit 1
fi

REC_PID=""
SIM_BROUGHT_UP=0

cleanup() {
    if [[ -n "$REC_PID" ]] && kill -0 "$REC_PID" 2>/dev/null; then
        log "stopping recorder (pid $REC_PID)..."
        kill -INT "$REC_PID" 2>/dev/null || true
        wait "$REC_PID" 2>/dev/null || true
    fi
    if (( SIM_BROUGHT_UP == 1 )); then
        log "stopping sim stack..."
        "$RUN_SH" stop || true
    fi
}
trap cleanup EXIT INT TERM

# --- bring up the sim -------------------------------------------------------
state="$(redis-cli get sports_bot::fsm::state 2>/dev/null || true)"
if [[ -z "$state" ]]; then
    log "bringing up sim+controller+FSM..."
    # run.sh start drops into an interactive REPL; we just want it backgrounded.
    # The status/start subcommands don't enter the REPL.
    "$RUN_SH" stop >/dev/null 2>&1 || true
    # Start each piece via the helper's individual non-interactive subcommand.
    # ./run.sh restart re-runs start_simviz/start_controller/start_fsm without
    # entering the REPL, which is exactly what we want.
    if ! "$RUN_SH" restart; then
        err "sim stack failed to come up — check /tmp/sports_bot/*.log"
        exit 1
    fi
    SIM_BROUGHT_UP=1
else
    log "sim already up (FSM state: $state) — won't tear it down on exit"
fi

# Wait for READY
log "waiting for FSM READY..."
for _ in $(seq 1 50); do
    state="$(redis-cli get sports_bot::fsm::state 2>/dev/null || true)"
    if [[ "$state" == "READY" ]]; then break; fi
    sleep 0.2
done
if [[ "$state" != "READY" ]]; then
    warn "FSM didn't reach READY (state=$state); continuing anyway"
fi
log "${C_GREEN}sim ready${C_RESET}"

# --- start recording --------------------------------------------------------
rm -f "$OUT"
log "recording $DURATION s of screen $SCREEN → $OUT"
"$FFMPEG" -y \
    -hide_banner -loglevel error -stats \
    -f avfoundation -framerate "$FRAMERATE" -capture_cursor 0 \
    -i "$SCREEN:" \
    -t "$DURATION" \
    -vcodec libx264 -preset veryfast -crf "$CRF" -pix_fmt yuv420p \
    "$OUT" &
REC_PID=$!
sleep 1.5  # let ffmpeg open the capture session

# --- launch balls on a schedule --------------------------------------------
SPACING=$(awk -v d="$DURATION" -v n="$BALLS" 'BEGIN { printf "%.2f", (d - 4) / n }')
log "launching $BALLS balls, spaced ${SPACING}s apart"
for i in $(seq 1 "$BALLS"); do
    log "ball $i/$BALLS"
    "$RUN_SH" ball >/dev/null 2>&1 || warn "ball launch $i returned non-zero"
    if (( i < BALLS )); then
        sleep "$SPACING"
    fi
done

# --- wait for recording to finish ------------------------------------------
wait "$REC_PID"
rc=$?
REC_PID=""

if (( rc != 0 )); then
    err "ffmpeg exited with code $rc — most likely Screen Recording permission."
    err "grant the terminal app access in System Settings → Privacy & Security → Screen Recording, then re-run."
    exit "$rc"
fi

log "${C_GREEN}done${C_RESET}  →  $OUT"
log "drop this into slide 4 (or re-run scripts/build_presentation.py to embed automatically)."
