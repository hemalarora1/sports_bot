#!/usr/bin/env bash
# sports_bot pickleball: one-shot launcher + interactive REPL.
#
# Usage:
#   ./run.sh start                        # bring up sim+controller+FSM, then drop into REPL
#   ./run.sh stop                         # kill everything
#   ./run.sh status                       # process + redis snapshot
#   ./run.sh ball [px py pz vx vy vz]     # launch a ball (defaults: 5,0,2 / -7,0,1.5)
#   ./run.sh repl                         # re-enter the REPL (sim already running)
#   ./run.sh help
#
# Environment overrides:
#   PICKLEBALL_PYTHON  Path to a python with redis+numpy installed.
#                      Default: /opt/homebrew/Caskroom/miniconda/base/envs/opensai/bin/python
#   PICKLEBALL_LOGS    Log + pid directory. Default: /tmp/sports_bot
#
# Intentionally lenient: a REPL needs to survive a typo without exiting,
# and `set -u` chokes on empty-array expansions (very common with optional args).
set -o pipefail

# ---------- paths -------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BIN_DIR="$REPO_ROOT/cs225a/bin/pickleball"
SIMVIZ_BIN="$BIN_DIR/simviz_pickleball"
CONTROLLER_BIN="$BIN_DIR/controller_pickleball"

PYTHON_BIN="${PICKLEBALL_PYTHON:-/opt/homebrew/Caskroom/miniconda/base/envs/opensai/bin/python}"

LOG_DIR="${PICKLEBALL_LOGS:-/tmp/sports_bot}"
PID_DIR="$LOG_DIR/pids"
mkdir -p "$LOG_DIR" "$PID_DIR"

SIMVIZ_PID_FILE="$PID_DIR/simviz.pid"
CONTROLLER_PID_FILE="$PID_DIR/controller.pid"
FSM_PID_FILE="$PID_DIR/fsm.pid"

# ---------- ANSI colors -------------------------------------------------------
if [[ -t 1 ]]; then
    C_RESET=$'\033[0m'; C_BOLD=$'\033[1m'
    C_GREEN=$'\033[32m'; C_YELLOW=$'\033[33m'; C_RED=$'\033[31m'
    C_CYAN=$'\033[36m'; C_DIM=$'\033[2m'
else
    C_RESET=""; C_BOLD=""; C_GREEN=""; C_YELLOW=""; C_RED=""; C_CYAN=""; C_DIM=""
fi
log()  { echo "${C_DIM}[run]${C_RESET} $*"; }
warn() { echo "${C_YELLOW}[run]${C_RESET} $*" >&2; }
err()  { echo "${C_RED}[run]${C_RESET} $*" >&2; }

# ---------- helpers -----------------------------------------------------------
is_alive() {
    local pid_file="$1"
    [[ -f "$pid_file" ]] || return 1
    local pid; pid="$(cat "$pid_file" 2>/dev/null)"
    [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

ensure_redis() {
    if ! redis-cli ping >/dev/null 2>&1; then
        err "redis-server is not running. Start it (e.g. \`redis-server\`) and retry."
        exit 1
    fi
}

ensure_built() {
    if [[ ! -x "$SIMVIZ_BIN" || ! -x "$CONTROLLER_BIN" ]]; then
        err "binaries missing under $BIN_DIR"
        err "build with: (cd cs225a/build && cmake .. && make -j simviz_pickleball controller_pickleball)"
        exit 1
    fi
    if [[ ! -x "$PYTHON_BIN" ]]; then
        # Fall back to whatever python is in PATH; warn so the user knows.
        local fallback; fallback="$(command -v python || true)"
        if [[ -z "$fallback" ]]; then
            err "no python found; install one or set PICKLEBALL_PYTHON"; exit 1
        fi
        warn "opensai python not found at $PYTHON_BIN; falling back to $fallback"
        warn "(set PICKLEBALL_PYTHON to silence — must have redis + numpy)"
        PYTHON_BIN="$fallback"
    fi
}

# Block until a redis key is non-empty, or until timeout (seconds).
wait_for_redis_key() {
    local key="$1" timeout="${2:-10}" elapsed=0
    while :; do
        local v; v="$(redis-cli get "$key" 2>/dev/null || true)"
        if [[ -n "$v" && "$v" != $'\r' ]]; then return 0; fi
        sleep 0.2
        elapsed=$((elapsed + 1))
        if (( elapsed > timeout * 5 )); then
            err "timed out waiting for redis key: $key"
            return 1
        fi
    done
}

# ---------- start / stop ------------------------------------------------------
start_simviz() {
    if is_alive "$SIMVIZ_PID_FILE"; then
        log "simviz already running (pid $(cat "$SIMVIZ_PID_FILE"))"
        return 0
    fi
    log "starting simviz (log: $LOG_DIR/simviz.log)..."
    cd "$REPO_ROOT"
    "$SIMVIZ_BIN" >"$LOG_DIR/simviz.log" 2>&1 &
    echo $! > "$SIMVIZ_PID_FILE"
    if ! wait_for_redis_key "sai::sim::mmp_panda::sensors::q" 10; then
        err "simviz didn't publish joint state — check $LOG_DIR/simviz.log"
        return 1
    fi
    log "${C_GREEN}simviz up${C_RESET} (pid $(cat "$SIMVIZ_PID_FILE"))"
}

start_controller() {
    if is_alive "$CONTROLLER_PID_FILE"; then
        log "controller already running"
        return 0
    fi
    log "starting controller (log: $LOG_DIR/controller.log)..."
    "$CONTROLLER_BIN" >"$LOG_DIR/controller.log" 2>&1 &
    echo $! > "$CONTROLLER_PID_FILE"
    if ! wait_for_redis_key "sports_bot::state::racket::current_position" 10; then
        err "controller didn't publish racket state — check $LOG_DIR/controller.log"
        return 1
    fi
    log "${C_GREEN}controller up${C_RESET} (pid $(cat "$CONTROLLER_PID_FILE"))"
}

start_fsm() {
    if is_alive "$FSM_PID_FILE"; then
        log "fsm already running"
        return 0
    fi
    log "starting FSM (log: $LOG_DIR/fsm.log)..."
    cd "$REPO_ROOT"
    "$PYTHON_BIN" -u -m sports_bot.state_machine.pickleball_fsm \
        --robot-backend cs225a --ball-source optitrack \
        >"$LOG_DIR/fsm.log" 2>&1 &
    echo $! > "$FSM_PID_FILE"
    sleep 2
    local state; state="$(redis-cli get sports_bot::fsm::state)"
    if [[ -z "$state" ]]; then
        err "FSM didn't reach a state — tail $LOG_DIR/fsm.log"
        return 1
    fi
    log "${C_GREEN}fsm up${C_RESET} (pid $(cat "$FSM_PID_FILE"), state: ${C_BOLD}$state${C_RESET})"
}

stop_all() {
    local stopped=0
    for pf in "$FSM_PID_FILE" "$CONTROLLER_PID_FILE" "$SIMVIZ_PID_FILE"; do
        [[ -f "$pf" ]] || continue
        local pid; pid="$(cat "$pf")"
        local label; label="$(basename "$pf" .pid)"
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null && log "stopped $label (pid $pid)"
            stopped=$((stopped + 1))
        fi
        rm -f "$pf"
    done
    # Also catch any stragglers that lost their pid file.
    pkill -f simviz_pickleball 2>/dev/null && stopped=$((stopped + 1)) || true
    pkill -f controller_pickleball 2>/dev/null && stopped=$((stopped + 1)) || true
    pkill -f "sports_bot.state_machine.pickleball_fsm" 2>/dev/null && stopped=$((stopped + 1)) || true
    if (( stopped == 0 )); then log "nothing to stop"; fi
}

# ---------- status / state ----------------------------------------------------
show_status() {
    echo "${C_BOLD}Processes${C_RESET}"
    for label in simviz controller fsm; do
        local pf="$PID_DIR/$label.pid"
        if is_alive "$pf"; then
            printf "  %-12s ${C_GREEN}running${C_RESET}  pid=%s\n" "$label" "$(cat "$pf")"
        else
            printf "  %-12s ${C_RED}down${C_RESET}\n" "$label"
        fi
    done
    echo
    echo "${C_BOLD}Redis state${C_RESET}"
    printf "  fsm::state                  %s\n" "$(redis-cli get sports_bot::fsm::state || echo '<none>')"
    printf "  ball world position         %s\n" "$(redis-cli get sai2::optitrack::rigid_body_pos::1 || echo '<none>')"
    printf "  ball linear velocity        %s\n" "$(redis-cli get sports_bot::sim::ball::linear_velocity || echo '<none>')"
    printf "  racket current position     %s\n" "$(redis-cli get sports_bot::state::racket::current_position || echo '<none>')"
    printf "  racket goal position        %s\n" "$(redis-cli get sports_bot::cmd::racket::goal_position || echo '<none>')"
    printf "  base current pose           %s\n" "$(redis-cli get sports_bot::state::base::current_pose || echo '<none>')"
}

# ---------- ball launching ----------------------------------------------------
# Defaults reach the strike plane with z>0 and no floor bounce — clean swing test.
DEFAULT_POS=(5.0 0.0 2.0)
DEFAULT_VEL=(-7.0 0.0 1.5)

launch_ball() {
    local args=("$@")
    local pos vel
    if (( ${#args[@]} == 0 )); then
        pos=("${DEFAULT_POS[@]}"); vel=("${DEFAULT_VEL[@]}")
    elif (( ${#args[@]} == 6 )); then
        pos=("${args[0]}" "${args[1]}" "${args[2]}")
        vel=("${args[3]}" "${args[4]}" "${args[5]}")
    else
        err "ball expects 0 or 6 numbers (px py pz vx vy vz). Got: ${args[*]}"
        return 1
    fi
    cd "$REPO_ROOT"
    "$PYTHON_BIN" -m sports_bot.pickleball.launch_ball \
        --pos "${pos[@]}" --vel "${vel[@]}"
}

# ---------- log helpers -------------------------------------------------------
tail_log() {
    local target="${1:-simviz}"
    local file="$LOG_DIR/$target.log"
    if [[ ! -f "$file" ]]; then
        err "no log at $file"; return 1
    fi
    tail -n 60 "$file"
}

# ---------- watch (live FSM transitions) --------------------------------------
watch_fsm() {
    local seconds="${1:-10}"
    log "watching FSM state for ${seconds}s (Ctrl-C to abort)..."
    local end=$(( $(date +%s) + seconds ))
    local prev=""
    while (( $(date +%s) < end )); do
        local s; s="$(redis-cli get sports_bot::fsm::state)"
        if [[ "$s" != "$prev" ]]; then
            local bp rp
            bp="$(redis-cli get sai2::optitrack::rigid_body_pos::1)"
            rp="$(redis-cli get sports_bot::state::racket::current_position)"
            printf "${C_CYAN}%s${C_RESET}  state=${C_BOLD}%-10s${C_RESET}  ball=%s  racket=%s\n" \
                "$(date +%H:%M:%S)" "$s" "$bp" "$rp"
            prev="$s"
        fi
        sleep 0.05
    done
}

# ---------- REPL --------------------------------------------------------------
repl_help() {
    cat <<EOF
${C_BOLD}REPL commands${C_RESET}
  ball [px py pz vx vy vz]    Launch a ball (default: 5 0 2 / -7 0 1.5)
  shoot                       Alias for default-ball launch + watch FSM transitions for 5s
  state                       Show processes + redis snapshot
  watch [seconds]             Print every FSM state change (default 10s)
  tail simviz|controller|fsm  Show last 60 lines of a log
  restart                     Stop and restart the full stack
  stop                        Kill all processes (REPL stays open)
  start                       (Re)start any processes that are down
  quit                        Exit REPL (processes keep running; \`./run.sh stop\` to kill)
  help
EOF
}

repl() {
    echo
    echo "${C_BOLD}========= sports_bot REPL =========${C_RESET}"
    repl_help
    echo
    while :; do
        # Prompt shows current FSM state for situational awareness.
        local state; state="$(redis-cli get sports_bot::fsm::state 2>/dev/null || echo '?')"
        local prompt; prompt="${C_CYAN}sports_bot${C_RESET} [${C_BOLD}${state}${C_RESET}]> "
        local line
        if ! read -r -p "$prompt" line; then echo; break; fi
        # split into cmd + args
        local -a tokens; tokens=($line)
        local cmd="${tokens[0]:-}"
        local args=("${tokens[@]:1}")
        case "$cmd" in
            ""|help|h|?)        repl_help ;;
            ball|b)             launch_ball "${args[@]}" ;;
            shoot)              launch_ball; watch_fsm 5 ;;
            state|s|status)     show_status ;;
            watch|w)            watch_fsm "${args[0]:-10}" ;;
            tail|t)             tail_log "${args[0]:-simviz}" ;;
            restart)            stop_all; sleep 1; start_simviz && start_controller && start_fsm ;;
            stop)               stop_all ;;
            start)              start_simviz && start_controller && start_fsm ;;
            quit|exit|q)        break ;;
            *)                  err "unknown command: $cmd (try 'help')" ;;
        esac
    done
    log "left REPL — processes still running. Run \`$0 stop\` to kill them."
}

# ---------- top-level dispatch ------------------------------------------------
usage() {
    cat <<EOF
${C_BOLD}sports_bot pickleball launcher${C_RESET}

  $(basename "$0") start                          Bring up sim+controller+FSM, enter REPL
  $(basename "$0") stop                           Kill everything
  $(basename "$0") restart                        Stop + start
  $(basename "$0") status                         Process + redis snapshot
  $(basename "$0") ball [px py pz vx vy vz]       One-shot ball launch
  $(basename "$0") watch [seconds]                Print FSM state transitions
  $(basename "$0") tail simviz|controller|fsm     Last 60 lines of a log
  $(basename "$0") repl                           Re-enter the REPL (assumes already started)
  $(basename "$0") help

Logs:    $LOG_DIR
Python:  $PYTHON_BIN
EOF
}

cmd="${1:-help}"
shift || true

case "$cmd" in
    start)
        ensure_redis; ensure_built
        start_simviz
        start_controller
        start_fsm
        echo
        show_status
        repl
        ;;
    stop)     stop_all ;;
    restart)
        ensure_redis; ensure_built
        stop_all; sleep 1
        start_simviz; start_controller; start_fsm
        ;;
    status)   show_status ;;
    ball)     ensure_built; launch_ball "$@" ;;
    watch)    watch_fsm "${1:-10}" ;;
    tail)     tail_log "${1:-simviz}" ;;
    repl)     ensure_built; repl ;;
    help|--help|-h) usage ;;
    *)        err "unknown command: $cmd"; usage; exit 1 ;;
esac
