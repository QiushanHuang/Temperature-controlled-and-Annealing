#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${RUN_ROOT:-/Volumes/TRACER/heating_cooling}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
SUITE_DIR="${SUITE_DIR:-anneal_hot1.70_np4omp2_loops7_suite}"

MPI_RANKS="${MPI_RANKS:-4}"
OMP_THREADS="${OMP_THREADS:-2}"
MPIEXEC="${MPIEXEC:-mpiexec}"
LAMMPS_BIN="${LAMMPS_BIN:-/home/star/Research/software/lammps-22Jul2025/build/lmp}"
if [[ -z "${LAMMPS_ARGS+x}" ]]; then
  if (( OMP_THREADS > 1 )); then
    LAMMPS_ARGS="-sf omp -pk omp $OMP_THREADS"
  else
    LAMMPS_ARGS=""
  fi
fi

CPU_TOTAL="${CPU_TOTAL:-256}"
DRY_RUN_ONLY="${DRY_RUN_ONLY:-0}"
WAIT_FOR_FINISH="${WAIT_FOR_FINISH:-0}"
REPLACE_EXISTING="${REPLACE_EXISTING:-0}"
OVERWRITE_OUTPUTS="${OVERWRITE_OUTPUTS:-0}"

SESSION_PREFIX="${SESSION_PREFIX:-anneal_hc}"
LENGTHS="${LENGTHS:-L3 L7}"
SEEDS="${SEEDS:-111111 222222 333333 444444 555555 666666 777777}"
LOOPS="${LOOPS:-7}"
HOT_T="${HOT_T:-1.70}"

read -r -a LENGTH_ARGS <<< "$LENGTHS"
read -r -a SEED_ARGS <<< "$SEEDS"

if ! command -v tmux >/dev/null 2>&1; then
  echo "Cannot find tmux. Install tmux or load the server module first." >&2
  exit 2
fi

if (( MPI_RANKS <= 0 || OMP_THREADS <= 0 )); then
  echo "MPI_RANKS and OMP_THREADS must be positive." >&2
  exit 2
fi

if (( ${#SEED_ARGS[@]} != LOOPS )); then
  echo "SEEDS count (${#SEED_ARGS[@]}) must equal LOOPS (${LOOPS})." >&2
  exit 2
fi

if [[ -n "$OUTPUT_ROOT" && "$OUTPUT_ROOT" =~ [[:space:]] ]]; then
  echo "OUTPUT_ROOT contains whitespace: $OUTPUT_ROOT" >&2
  echo "LAMMPS input paths are intentionally kept whitespace-free." >&2
  echo "Create a no-space symlink, for example: ln -s '/media/star/My Passport2' /media/star/MyPassport2" >&2
  exit 2
fi

if [[ ! -x "$LAMMPS_BIN" ]]; then
  echo "LAMMPS_BIN is not executable: $LAMMPS_BIN" >&2
  exit 2
fi

LAMMPS_HELP="$("$LAMMPS_BIN" -h 2>&1 || true)"
REQUIRED_PACKAGES=(MOLECULE ASPHERE RIGID)
if (( OMP_THREADS > 1 )) || [[ " $LAMMPS_ARGS " == *" omp "* ]]; then
  REQUIRED_PACKAGES+=(OPENMP)
fi
MISSING_PACKAGES=()
for package in "${REQUIRED_PACKAGES[@]}"; do
  if ! grep -Eq "(^|[[:space:]])${package}($|[[:space:]])" <<< "$LAMMPS_HELP"; then
    MISSING_PACKAGES+=("$package")
  fi
done
if (( ${#MISSING_PACKAGES[@]} > 0 )); then
  echo "LAMMPS binary is missing required package(s): ${MISSING_PACKAGES[*]}" >&2
  echo "This restart/input needs MOLECULE for atom_style bond, ASPHERE for ellipsoids/Gay-Berne, RIGID for rigid/nvt/small, and OPENMP for -sf omp/-pk omp." >&2
  echo "Use or rebuild a LAMMPS binary with these packages enabled, then set LAMMPS_BIN to that executable." >&2
  exit 2
fi

SUITE_ABS="$RUN_ROOT/$SUITE_DIR"
MANIFEST="$SUITE_ABS/manifest.json"
TMUX_DIR="$SUITE_ABS/tmux"
RUNNER_DIR="$TMUX_DIR/runners"
LOG_DIR="$TMUX_DIR/logs"
STATUS_DIR="$TMUX_DIR/status"
mkdir -p "$RUNNER_DIR" "$LOG_DIR" "$STATUS_DIR"

CREATE_ARGS=(
  --root "$RUN_ROOT"
  --suite-dir "$SUITE_ABS"
  --lengths "${LENGTH_ARGS[@]}"
  --loops "$LOOPS"
  --seeds "${SEED_ARGS[@]}"
  --hot-T "$HOT_T"
  --mpi-ranks "$MPI_RANKS"
  --omp-threads "$OMP_THREADS"
  --mpiexec "$MPIEXEC"
  --lammps-bin "$LAMMPS_BIN"
)
if [[ -n "$LAMMPS_ARGS" ]]; then
  CREATE_ARGS+=(--lammps-args "$LAMMPS_ARGS")
fi

if [[ -n "$OUTPUT_ROOT" ]]; then
  CREATE_ARGS+=(--output-root "$OUTPUT_ROOT")
fi

if [[ "$DRY_RUN_ONLY" != "1" ]]; then
  CREATE_ARGS+=(--run-lammps)
fi

if [[ "$OVERWRITE_OUTPUTS" == "1" ]]; then
  CREATE_ARGS+=(--overwrite)
fi

echo "Code root    : $CODE_ROOT"
echo "Run root     : $RUN_ROOT"
echo "Output root  : ${OUTPUT_ROOT:-<inside each Tstar folder>}"
echo "Suite        : $SUITE_ABS"
echo "Lengths      : ${LENGTH_ARGS[*]}"
echo "Seeds        : ${SEED_ARGS[*]}"
echo "Loops/case   : $LOOPS"
echo "Hot T        : $HOT_T"
echo "Per case     : np=${MPI_RANKS}, omp=${OMP_THREADS}, CPUs=$(( MPI_RANKS * OMP_THREADS ))"
echo "MPI command  : ${MPIEXEC} -np ${MPI_RANKS} ${LAMMPS_BIN} ${LAMMPS_ARGS}"

PYTHONPATH="$CODE_ROOT:${PYTHONPATH:-}" python3 "$CODE_ROOT/scripts/create_heating_cooling_suite.py" "${CREATE_ARGS[@]}"

CASE_LIST="$TMUX_DIR/cases.tsv"
python3 - "$MANIFEST" <<'PY' > "$CASE_LIST"
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for case in manifest.get("cases", []):
    print(case["case_id"], case["params_path"], case["result_dir"], sep="\t")
PY

CASE_COUNT="$(wc -l < "$CASE_LIST" | tr -d ' ')"
CPUS_PER_CASE=$(( MPI_RANKS * OMP_THREADS ))
REQUESTED_CPUS=$(( CASE_COUNT * CPUS_PER_CASE ))

if (( CASE_COUNT <= 0 )); then
  echo "No cases found in manifest: $MANIFEST" >&2
  exit 2
fi

if (( REQUESTED_CPUS > CPU_TOTAL )); then
  echo "This tmux launcher starts one tmux session per L3/L7 Tstar case." >&2
  echo "Requested CPUs: ${REQUESTED_CPUS}; CPU_TOTAL=${CPU_TOTAL}." >&2
  echo "Reduce LENGTHS, or raise CPU_TOTAL if your allocation is larger." >&2
  exit 2
fi

echo "Cases        : $CASE_COUNT"
echo "tmux sessions: $CASE_COUNT"
echo "CPU request  : ${REQUESTED_CPUS} / ${CPU_TOTAL}"

if [[ "$DRY_RUN_ONLY" == "1" ]]; then
  echo "DRY_RUN_ONLY=1, stopping after params/manifest generation."
  exit 0
fi

SESSION_LIST="$TMUX_DIR/tmux_sessions.txt"
: > "$SESSION_LIST"

while IFS=$'\t' read -r case_id params_path result_dir; do
  session="${SESSION_PREFIX}_${case_id}"
  runner_script="$RUNNER_DIR/${case_id}.sh"
  tmux_log="$LOG_DIR/${case_id}.tmux.log"
  status_file="$STATUS_DIR/${case_id}.status"
  mkdir -p "$result_dir"

  if tmux has-session -t "$session" 2>/dev/null; then
    if [[ "$REPLACE_EXISTING" == "1" ]]; then
      tmux kill-session -t "$session"
    else
      echo "tmux session already exists: $session" >&2
      echo "Set REPLACE_EXISTING=1 to kill and relaunch it." >&2
      exit 2
    fi
  fi

  cat > "$runner_script" <<EOF
#!/usr/bin/env bash
set -u
cd $(printf '%q' "$RUN_ROOT")
export PYTHONPATH=$(printf '%q' "$CODE_ROOT"):\${PYTHONPATH:-}
echo "[START] $case_id \$(date)"
echo "[CASE] $case_id" > $(printf '%q' "$tmux_log")
echo "[RUN_ROOT] $RUN_ROOT" >> $(printf '%q' "$tmux_log")
echo "[RESULT] $result_dir" >> $(printf '%q' "$tmux_log")
echo "[PARAMS] $params_path" >> $(printf '%q' "$tmux_log")
echo "[COMMAND] OMP_NUM_THREADS=$OMP_THREADS $MPIEXEC -np $MPI_RANKS $LAMMPS_BIN $LAMMPS_ARGS" >> $(printf '%q' "$tmux_log")
set +e
python3 $(printf '%q' "$CODE_ROOT/run_anneal.py") $(printf '%q' "$params_path") >> $(printf '%q' "$tmux_log") 2>&1
rc=\$?
set -e
echo "\$rc" > $(printf '%q' "$status_file")
if [[ "\$rc" -eq 0 ]]; then
  echo "[DONE] $case_id \$(date)" >> $(printf '%q' "$tmux_log")
else
  echo "[FAILED] $case_id rc=\$rc \$(date)" >> $(printf '%q' "$tmux_log")
fi
exit "\$rc"
EOF
  chmod +x "$runner_script"

  rm -f "$status_file"
  tmux new-session -d -s "$session" "bash $(printf '%q' "$runner_script")"
  printf "%s\t%s\t%s\t%s\n" "$session" "$case_id" "$result_dir" "$tmux_log" >> "$SESSION_LIST"
  echo "Launched $session -> $result_dir"
done < "$CASE_LIST"

CHECK_SCRIPT="$TMUX_DIR/check_tmux_status.sh"
cat > "$CHECK_SCRIPT" <<EOF
#!/usr/bin/env bash
set -euo pipefail
echo "Active tmux sessions:"
tmux list-sessions -F '#S' 2>/dev/null | grep '^$(printf '%s' "$SESSION_PREFIX" | sed 's/[][\.^$*+?{}|()]/\\&/g')_' || true
echo
echo "Case status files:"
for status in $(printf '%q' "$STATUS_DIR")/*.status; do
  [[ -e "\$status" ]] || continue
  printf "%s " "\$(basename "\$status" .status)"
  cat "\$status"
done
EOF
chmod +x "$CHECK_SCRIPT"

echo
echo "tmux sessions written to: $SESSION_LIST"
echo "tmux logs are under     : $LOG_DIR"
if [[ -n "$OUTPUT_ROOT" ]]; then
  echo "case outputs are under  : $OUTPUT_ROOT/<L3|L7>/Tstar_x.xx/anneal_hot${HOT_T}_np${MPI_RANKS}omp${OMP_THREADS}_loops${LOOPS}"
else
  echo "case outputs are under  : each Tstar folder / anneal_hot${HOT_T}_np${MPI_RANKS}omp${OMP_THREADS}_loops${LOOPS}"
fi
echo "Check status with       : $CHECK_SCRIPT"
FIRST_SESSION="$(awk 'NR == 1 {print $1}' "$SESSION_LIST")"
echo "Attach example          : tmux attach -t $FIRST_SESSION"

if [[ "$WAIT_FOR_FINISH" == "1" ]]; then
  echo "WAIT_FOR_FINISH=1, waiting for all tmux sessions to exit..."
  while true; do
    active=0
    while IFS=$'\t' read -r session _case_id _result_dir _log_path; do
      if tmux has-session -t "$session" 2>/dev/null; then
        active=$((active + 1))
      fi
    done < "$SESSION_LIST"
    if (( active == 0 )); then
      break
    fi
    echo "Active sessions: $active"
    sleep 60
  done

  failures=0
  while IFS=$'\t' read -r _session case_id _result_dir _log_path; do
    status_file="$STATUS_DIR/${case_id}.status"
    if [[ ! -s "$status_file" ]] || [[ "$(cat "$status_file")" != "0" ]]; then
      echo "Failed or missing status: $case_id" >&2
      failures=$((failures + 1))
    fi
  done < "$SESSION_LIST"
  if (( failures > 0 )); then
    exit 1
  fi
fi
