#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="${WORKSPACE:-$PWD}"

MPI_RANKS="${MPI_RANKS:-4}"
OMP_THREADS="${OMP_THREADS:-9}"
MPIEXEC="${MPIEXEC:-mpiexec}"
LAMMPS_BIN="${LAMMPS_BIN:-/home/star/Research/software/lammps-22Jul2025/build/lmp}"
if [[ -z "${LAMMPS_ARGS+x}" ]]; then
  if (( OMP_THREADS > 1 )); then
    LAMMPS_ARGS="-sf omp -pk omp $OMP_THREADS"
  else
    LAMMPS_ARGS=""
  fi
fi

PARAMS="${PARAMS:-params.single_restart_loop.json}"
CPU_TOTAL="${CPU_TOTAL:-512}"
DRY_RUN_ONLY="${DRY_RUN_ONLY:-0}"
CLEAN_EXISTING="${CLEAN_EXISTING:-0}"
OVERWRITE_OUTPUTS="${OVERWRITE_OUTPUTS:-0}"
REPLACE_EXISTING="${REPLACE_EXISTING:-0}"
WAIT_FOR_FINISH="${WAIT_FOR_FINISH:-0}"
SESSION_PREFIX="${SESSION_PREFIX:-single_restart_np${MPI_RANKS}omp${OMP_THREADS}}"

TARGETS="${TARGETS:-}"
SEEDS="${SEEDS:-}"
LOOPS="${LOOPS:-}"
HOT_T="${HOT_T:-}"

if [[ "$WORKSPACE" =~ [[:space:]] ]]; then
  echo "WORKSPACE contains whitespace: $WORKSPACE" >&2
  echo "Run from a no-space symlink path such as \$HOME/MyPassport2/..." >&2
  exit 2
fi

if [[ "$LAMMPS_BIN" =~ [[:space:]] ]]; then
  echo "LAMMPS_BIN contains whitespace: $LAMMPS_BIN" >&2
  exit 2
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "Cannot find tmux. Install tmux or load the server module first." >&2
  exit 2
fi

if (( MPI_RANKS <= 0 || OMP_THREADS <= 0 )); then
  echo "MPI_RANKS and OMP_THREADS must be positive." >&2
  exit 2
fi

if [[ "$PARAMS" != /* ]]; then
  PARAMS="$WORKSPACE/$PARAMS"
fi

if [[ ! -f "$PARAMS" ]]; then
  echo "Cannot find params file: $PARAMS" >&2
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
  echo "This restart/input needs MOLECULE, ASPHERE, RIGID, and OPENMP when omp_threads > 1." >&2
  exit 2
fi

TMUX_DIR="${TMUX_DIR:-$WORKSPACE/.single_restart_tmux_np${MPI_RANKS}omp${OMP_THREADS}}"
RUNNER_DIR="$TMUX_DIR/runners"
LOG_DIR="$TMUX_DIR/logs"
STATUS_DIR="$TMUX_DIR/status"
mkdir -p "$RUNNER_DIR" "$LOG_DIR" "$STATUS_DIR"

EFFECTIVE_PARAMS="$TMUX_DIR/effective_params.json"
export MPI_RANKS OMP_THREADS MPIEXEC LAMMPS_BIN LAMMPS_ARGS TARGETS SEEDS LOOPS HOT_T OVERWRITE_OUTPUTS
python3 - "$PARAMS" "$EFFECTIVE_PARAMS" <<'PY'
import json
import os
import shlex
import sys
from pathlib import Path

source = Path(sys.argv[1])
dest = Path(sys.argv[2])
data = json.loads(source.read_text(encoding="utf-8"))
if not isinstance(data, dict):
    raise SystemExit(f"{source} must contain a JSON object")

data.setdefault("workspace", "cwd")
data.setdefault("input", {})
data.setdefault("output", {})
data.setdefault("simulation", {})
data.setdefault("run", {})

simulation = data["simulation"]
run = data["run"]
output = data["output"]

if os.environ.get("TARGETS"):
    simulation["target_T_list"] = [float(value) for value in shlex.split(os.environ["TARGETS"])]
if os.environ.get("SEEDS"):
    simulation["seeds"] = [int(value) for value in shlex.split(os.environ["SEEDS"])]
if os.environ.get("LOOPS"):
    simulation["loops"] = int(os.environ["LOOPS"])
elif "seeds" in simulation:
    simulation["loops"] = len(simulation["seeds"])
if os.environ.get("HOT_T"):
    simulation["hot_T"] = float(os.environ["HOT_T"])

run["run_lammps"] = False
run["mpi_ranks"] = int(os.environ["MPI_RANKS"])
run["omp_threads"] = int(os.environ["OMP_THREADS"])
run["mpiexec"] = os.environ["MPIEXEC"]
run["lammps_bin"] = os.environ["LAMMPS_BIN"]
run["lammps_args"] = shlex.split(os.environ.get("LAMMPS_ARGS", ""))
if os.environ.get("OVERWRITE_OUTPUTS") == "1":
    output["overwrite"] = True

dest.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

if [[ "$CLEAN_EXISTING" == "1" ]]; then
  CLEAN_LIST="$TMUX_DIR/cleanup_targets.txt"
  python3 - "$WORKSPACE" "$EFFECTIVE_PARAMS" <<'PY' > "$CLEAN_LIST"
import json
import sys
from pathlib import Path

workspace = Path(sys.argv[1])
params = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
template = params.get("output", {}).get("temperature_dir_template", "Tstar_{T:.2f}")
for target in params.get("simulation", {}).get("target_T_list", []):
    target_t = float(target)
    print(workspace / template.format(T=target_t, T_tag=f"{target_t:.2f}"))
print(workspace / "single_restart_suite_manifest.json")
PY
  while IFS= read -r target_path; do
    [[ -n "$target_path" ]] || continue
    if [[ "$target_path" == "$WORKSPACE/"* || "$target_path" == "$WORKSPACE/single_restart_suite_manifest.json" ]]; then
      rm -rf "$target_path"
    else
      echo "Refusing to clean path outside WORKSPACE: $target_path" >&2
      exit 2
    fi
  done < "$CLEAN_LIST"
  rm -rf "$RUNNER_DIR" "$LOG_DIR" "$STATUS_DIR"
  mkdir -p "$RUNNER_DIR" "$LOG_DIR" "$STATUS_DIR"
fi

echo "Code root    : $CODE_ROOT"
echo "Workspace    : $WORKSPACE"
echo "Params       : $PARAMS"
echo "Effective    : $EFFECTIVE_PARAMS"
echo "Launch mode  : parallel temperatures; sequential loops inside each temperature"
echo "Per temp      : np=${MPI_RANKS}, omp=${OMP_THREADS}, CPUs=$(( MPI_RANKS * OMP_THREADS ))"
echo "LAMMPS        : ${MPIEXEC} -np ${MPI_RANKS} ${LAMMPS_BIN} ${LAMMPS_ARGS}"
echo "Clean old     : $CLEAN_EXISTING"

(
  cd "$WORKSPACE"
  PYTHONPATH="$CODE_ROOT:${PYTHONPATH:-}" python3 "$CODE_ROOT/single_Restart_loop_version/run_single_restart_loops.py" "$EFFECTIVE_PARAMS"
)

MANIFEST="$WORKSPACE/single_restart_suite_manifest.json"
CASE_LIST="$TMUX_DIR/temperature_cases.tsv"
python3 - "$MANIFEST" <<'PY' > "$CASE_LIST"
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for case in manifest.get("cases", []):
    print(f'{float(case["target_T"]):.2f}', case["output_root"], case["run_script"], sep="\t")
PY

CASE_COUNT="$(wc -l < "$CASE_LIST" | tr -d ' ')"
CPUS_PER_TEMP=$(( MPI_RANKS * OMP_THREADS ))
REQUESTED_CPUS=$(( CASE_COUNT * CPUS_PER_TEMP ))

if (( CASE_COUNT <= 0 )); then
  echo "No temperature cases found in manifest: $MANIFEST" >&2
  exit 2
fi

if (( REQUESTED_CPUS > CPU_TOTAL )); then
  echo "This launcher starts one tmux session per temperature." >&2
  echo "Requested CPUs: ${REQUESTED_CPUS}; CPU_TOTAL=${CPU_TOTAL}." >&2
  echo "Reduce TARGETS, or raise CPU_TOTAL if your allocation is larger." >&2
  exit 2
fi

echo "Temperatures  : $CASE_COUNT"
echo "tmux sessions : $CASE_COUNT"
echo "CPU request  : ${REQUESTED_CPUS} / ${CPU_TOTAL}"

if [[ "$DRY_RUN_ONLY" == "1" ]]; then
  echo "DRY_RUN_ONLY=1, stopping after generation."
  exit 0
fi

SESSION_LIST="$TMUX_DIR/tmux_sessions.txt"
: > "$SESSION_LIST"

sanitize() {
  tr -c 'A-Za-z0-9_' '_' <<< "$1" | sed 's/_$//'
}

while IFS=$'\t' read -r target_tag output_root run_script; do
  case_id="Tstar_${target_tag}"
  session="$(sanitize "${SESSION_PREFIX}_${case_id}")"
  runner_script="$RUNNER_DIR/${case_id}.sh"
  tmux_log="$LOG_DIR/${case_id}.tmux.log"
  status_file="$STATUS_DIR/${case_id}.status"

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
cd $(printf '%q' "$output_root")
echo "[START] $case_id \$(date)" | tee $(printf '%q' "$tmux_log")
echo "[OUTPUT] $output_root" >> $(printf '%q' "$tmux_log")
echo "[RUN_SCRIPT] $run_script" >> $(printf '%q' "$tmux_log")
set +e
bash $(printf '%q' "$run_script") >> $(printf '%q' "$tmux_log") 2>&1
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
  printf "%s\t%s\t%s\t%s\n" "$session" "$case_id" "$output_root" "$tmux_log" >> "$SESSION_LIST"
  echo "Launched $session -> $output_root"
done < "$CASE_LIST"

CHECK_SCRIPT="$TMUX_DIR/check_tmux_status.sh"
cat > "$CHECK_SCRIPT" <<EOF
#!/usr/bin/env bash
set -euo pipefail
echo "Active tmux sessions:"
tmux list-sessions -F '#S' 2>/dev/null | grep '^$(printf '%s' "$SESSION_PREFIX" | sed 's/[][\.^$*+?{}|()]/\\&/g')_' || true
echo
echo "Temperature status files:"
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
echo "Check status with       : $CHECK_SCRIPT"
FIRST_SESSION="$(awk 'NR == 1 {print $1}' "$SESSION_LIST")"
echo "Attach example          : tmux attach -t $FIRST_SESSION"

if [[ "$WAIT_FOR_FINISH" == "1" ]]; then
  echo "WAIT_FOR_FINISH=1, waiting for all tmux sessions to exit..."
  while true; do
    active=0
    while IFS=$'\t' read -r session _case_id _output_root _log_path; do
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
  while IFS=$'\t' read -r _session case_id _output_root _log_path; do
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
