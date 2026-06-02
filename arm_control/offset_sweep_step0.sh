#!/usr/bin/env bash
set -euo pipefail

# Sweep compliantFrame z-offsets for the bare OpenSai Cartesian controller test.
#
# Important: OpenSai reads arm_test_picklebot.xml at controller startup, so this
# script pauses after each XML edit. Relaunch OpenSai with arm_test_picklebot.xml,
# then press Enter here to run the Step 0 sweep for that offset.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
XML_PATH="$ROOT_DIR/config_folder/xml_config_files/arm_test_picklebot.xml"
TEST_SCRIPT="$ROOT_DIR/sports_bot/arm_control/step0_controller_test.py"
LOG_DIR="$ROOT_DIR/log_files/step0_offset_sweep/$(date +%Y%m%d_%H%M%S)"
SUMMARY_PATH="$LOG_DIR/summary.txt"

OFFSETS=(0 0.1224 0.20 0.25 0.30 0.35)
PYTHON_BIN="${PYTHON_BIN:-python}"
POS_TOL_MM="${POS_TOL_MM:-8}"
ANG_TOL_DEG="${ANG_TOL_DEG:-3}"
TIMEOUT_S="${TIMEOUT_S:-12}"
EXTRA_ARGS=()

usage() {
  cat <<USAGE
Usage: $0 [options] [offset ...]

Options:
  --pos-tol-mm N     Position tolerance passed to step0_controller_test.py (default: $POS_TOL_MM)
  --ang-tol-deg N    Angle tolerance passed to step0_controller_test.py (default: $ANG_TOL_DEG)
  --timeout S        Timeout passed to step0_controller_test.py (default: $TIMEOUT_S)
  --no-pre-ready     Do not pass --pre-ready to the Step 0 test
  --                 Remaining args are passed through to step0_controller_test.py

Default offsets: ${OFFSETS[*]}
Examples:
  $0
  POS_TOL_MM=10 $0 0 0.1224 0.20 0.35
  $0 --pos-tol-mm 10 --timeout 15 0 0.1224 0.20 0.25 0.30 0.35
USAGE
}

USE_PRE_READY=1
CUSTOM_OFFSETS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h)
      usage
      exit 0
      ;;
    --pos-tol-mm)
      POS_TOL_MM="$2"
      shift 2
      ;;
    --ang-tol-deg)
      ANG_TOL_DEG="$2"
      shift 2
      ;;
    --timeout)
      TIMEOUT_S="$2"
      shift 2
      ;;
    --no-pre-ready)
      USE_PRE_READY=0
      shift
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    --*)
      EXTRA_ARGS+=("$1")
      shift
      ;;
    *)
      CUSTOM_OFFSETS+=("$1")
      shift
      ;;
  esac
done

if [[ ${#CUSTOM_OFFSETS[@]} -gt 0 ]]; then
  OFFSETS=("${CUSTOM_OFFSETS[@]}")
fi

mkdir -p "$LOG_DIR"

set_offset() {
  local offset="$1"
  "$PYTHON_BIN" - "$XML_PATH" "$offset" <<'PYSET'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
offset = sys.argv[2]
text = path.read_text()
pattern = re.compile(
    r'(<controller name="cartesian_controller">.*?<motionForceTask name="cartesian_task".*?<compliantFrame\s+xyz=")0 0 [^"]+("\s+rpy="0 0 0"\s*/>)',
    re.S,
)
new_text, count = pattern.subn(rf'\g<1>0 0 {offset}\2', text, count=1)
if count != 1:
    raise SystemExit('ERROR: could not find active cartesian_controller compliantFrame block')
path.write_text(new_text)
PYSET
}

run_test() {
  local offset="$1"
  local safe_offset="${offset//./p}"
  local log_path="$LOG_DIR/offset_${safe_offset}.log"
  local args=("$TEST_SCRIPT" --sweep --timeout "$TIMEOUT_S" --pos-tol-mm "$POS_TOL_MM" --ang-tol-deg "$ANG_TOL_DEG")
  if [[ "$USE_PRE_READY" == "1" ]]; then
    args+=(--pre-ready)
  fi
  args+=("${EXTRA_ARGS[@]}")

  echo "Running: $PYTHON_BIN ${args[*]}" | tee "$log_path"
  "$PYTHON_BIN" "${args[@]}" | tee -a "$log_path"

  {
    echo
    echo "===== offset $offset ====="
    grep -A 10 'SWEEP SUMMARY' "$log_path" || true
  } >> "$SUMMARY_PATH"
}

cat <<INFO
Offset sweep logs: $LOG_DIR
XML under test:    $XML_PATH
Step0 script:      $TEST_SCRIPT
Offsets:           ${OFFSETS[*]}
Tolerance:         ${POS_TOL_MM} mm, ${ANG_TOL_DEG} deg
Timeout:           ${TIMEOUT_S} s
INFO

for offset in "${OFFSETS[@]}"; do
  echo
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "Setting compliantFrame z-offset to $offset m"
  set_offset "$offset"
  grep -n -A 1 '<compliantFrame' "$XML_PATH" | head -n 6
  echo
  echo "Relaunch OpenSai with arm_test_picklebot.xml now so it loads offset $offset."
  read -r -p "Press Enter after OpenSai is relaunched and current_position is updating... "
  run_test "$offset"
done

echo
echo "Done. Summary: $SUMMARY_PATH"
cat "$SUMMARY_PATH"
