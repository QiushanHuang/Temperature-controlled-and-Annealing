#!/usr/bin/env python3
"""Run one manually placed restart through multiple target-temperature loop sets."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from anneal_restart_generator import (  # noqa: E402
    DEFAULT_COOL_DT,
    DEFAULT_COOL_STEPS_PER_DT,
    DEFAULT_DUMP_EVERY,
    DEFAULT_HOT_HOLD_STEPS,
    DEFAULT_RESTART_DUMP_MULTIPLE,
    DEFAULT_SAMPLE_EVERY,
    DEFAULT_SAMPLE_STEPS,
    DEFAULT_TARGET_HOLD_STEPS,
    DEFAULT_TCHAIN,
    DEFAULT_TDAMP,
    DEFAULT_TS,
    GeneratorConfig,
    finalize_config,
    write_project,
)


DEFAULT_PARAMS = Path(__file__).resolve().with_name("params.single_restart_loop.json")
DEFAULT_SEEDS = [111111, 222222, 333333, 444444, 555555, 666666, 777777]


@dataclass(frozen=True)
class SingleRunSettings:
    run_lammps: bool = False
    overwrite: bool = False


@dataclass(frozen=True)
class TemperatureCase:
    target_t: float
    output_root: Path
    run_script: Path
    loop_dirs: list[Path]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _absolute_no_resolve(path: str | Path) -> Path:
    expanded = Path(path).expanduser()
    if not expanded.is_absolute():
        expanded = Path.cwd() / expanded
    return Path(os.path.abspath(os.fspath(expanded)))


def _resolve_workspace(params_path: Path, data: dict[str, Any]) -> Path:
    workspace = data.get("workspace", "cwd")
    if workspace == "cwd":
        return Path.cwd().resolve()
    if workspace == "params":
        return params_path.parent.resolve()
    if isinstance(workspace, str):
        path = Path(workspace).expanduser()
        if not path.is_absolute():
            path = params_path.parent / path
        return path.resolve()
    raise ValueError("workspace must be 'cwd', 'params', or a path string")


def _discover_restart(workspace: Path, pattern: str) -> Path:
    matches = sorted(path for path in workspace.glob(pattern) if path.is_file() and not path.name.startswith("._"))
    if len(matches) == 1:
        return matches[0].resolve()
    if not matches:
        raise FileNotFoundError(
            f"No restart file matched {pattern!r} in {workspace}. "
            "Put exactly one Restart* file in this experiment folder or set input.restart_file."
        )
    joined = "\n".join(f"  {path.name}" for path in matches)
    raise ValueError(f"Multiple restart files matched {pattern!r} in {workspace}; set input.restart_file explicitly:\n{joined}")


def _resolve_restart(workspace: Path, input_cfg: dict[str, Any]) -> Path:
    restart_setting = input_cfg.get("restart_file", "auto")
    if restart_setting == "auto":
        return _discover_restart(workspace, str(input_cfg.get("restart_glob", "Restart*")))
    restart = Path(str(restart_setting)).expanduser()
    if not restart.is_absolute():
        restart = workspace / restart
    return restart.resolve()


def _target_list(simulation: dict[str, Any]) -> list[float]:
    raw = (
        simulation.get("target_T_list")
        or simulation.get("target_t_list")
        or simulation.get("target_temperatures")
        or simulation.get("targets")
    )
    if raw is None and ("target_T" in simulation or "target_t" in simulation):
        raw = [simulation.get("target_T", simulation.get("target_t"))]
    if not isinstance(raw, list) or not raw:
        raise ValueError("simulation.target_T_list must be a non-empty JSON list")
    targets = [float(value) for value in raw]
    if len(set(targets)) != len(targets):
        raise ValueError("simulation.target_T_list must not contain duplicates")
    return targets


def _steps_from_dump_count(
    simulation: dict[str, Any],
    *,
    dump_key: str,
    steps_key: str,
    default_steps: int,
    dump_every: int,
) -> int:
    if dump_key in simulation:
        return int(simulation[dump_key]) * int(dump_every)
    return int(simulation.get(steps_key, default_steps))


def _cool_steps_per_dt(simulation: dict[str, Any], dump_every: int) -> int:
    if "cool_dumps_per_dT" in simulation:
        return int(simulation["cool_dumps_per_dT"]) * int(dump_every)
    if "cool_dumps_per_dt" in simulation:
        return int(simulation["cool_dumps_per_dt"]) * int(dump_every)
    return int(simulation.get("cool_steps_per_dT", simulation.get("cool_steps_per_dt", DEFAULT_COOL_STEPS_PER_DT)))


def _json_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    raise ValueError(f"{name} must be a JSON boolean true/false")


def _lammps_args(raw_args: Any) -> list[str]:
    if raw_args in (None, ""):
        return []
    if isinstance(raw_args, str):
        return shlex.split(raw_args)
    if isinstance(raw_args, list):
        return [str(arg) for arg in raw_args]
    raise ValueError("run.lammps_args must be a string or a JSON list")


def _build_lammps_command(run: dict[str, Any]) -> str:
    if run.get("lammps_command"):
        return str(run["lammps_command"])
    mpi_ranks = int(run.get("mpi_ranks", 4))
    omp_threads = int(run.get("omp_threads", 2))
    if mpi_ranks <= 0:
        raise ValueError("run.mpi_ranks must be positive")
    if omp_threads <= 0:
        raise ValueError("run.omp_threads must be positive")

    parts: list[str] = []
    if omp_threads != 1:
        parts.append(f"OMP_NUM_THREADS={omp_threads}")
    if mpi_ranks != 1:
        parts.extend([str(run.get("mpiexec", "mpiexec")), "-np", str(mpi_ranks)])
    parts.append(str(run.get("lammps_bin", "/home/star/Research/software/lammps-22Jul2025/build/lmp")))
    parts.extend(_lammps_args(run.get("lammps_args", ["-sf", "omp", "-pk", "omp", str(omp_threads)] if omp_threads > 1 else [])))
    return " ".join(parts)


def _temperature_dir_name(template: str, target_t: float) -> str:
    return template.format(T=target_t, T_tag=f"{target_t:.2f}")


def _load_sections(params_path: Path) -> tuple[Path, Path, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    data = _read_json(params_path)
    workspace = _resolve_workspace(params_path, data)
    input_cfg = data.get("input", {})
    output_cfg = data.get("output", {})
    simulation = data.get("simulation", {})
    run = data.get("run", {})
    if not isinstance(input_cfg, dict) or not isinstance(output_cfg, dict) or not isinstance(simulation, dict) or not isinstance(run, dict):
        raise ValueError("params sections input/output/simulation/run must be JSON objects")
    restart = _resolve_restart(workspace, input_cfg)
    return workspace, restart, input_cfg, output_cfg, simulation, run


def generate_cases(params_path: str | Path = DEFAULT_PARAMS) -> tuple[list[TemperatureCase], SingleRunSettings, Path]:
    params_path = Path(params_path).expanduser().resolve()
    workspace, restart, input_cfg, output_cfg, simulation, run = _load_sections(params_path)
    targets = _target_list(simulation)
    dump_every = int(simulation.get("dump_every", DEFAULT_DUMP_EVERY))
    loops = int(simulation.get("loops", len(simulation.get("seeds", DEFAULT_SEEDS))))
    explicit_seeds = simulation.get("seeds", DEFAULT_SEEDS[:loops])
    explicit_seeds = [int(seed) for seed in explicit_seeds]
    template = str(output_cfg.get("temperature_dir_template", "Tstar_{T:.2f}"))

    settings = SingleRunSettings(
        run_lammps=_json_bool(run.get("run_lammps", False), name="run.run_lammps"),
        overwrite=_json_bool(output_cfg.get("overwrite", run.get("overwrite", False)), name="output.overwrite"),
    )

    cases: list[TemperatureCase] = []
    manifest_cases: list[dict[str, Any]] = []
    for target_t in targets:
        output_root = _absolute_no_resolve(workspace / _temperature_dir_name(template, target_t))
        cfg = finalize_config(
            GeneratorConfig(
                restart=restart,
                output_root=output_root,
                target_t=target_t,
                hot_t=float(simulation.get("hot_T", simulation.get("hot_t", 1.70))),
                loops=loops,
                explicit_seeds=explicit_seeds,
                seed_start=int(simulation.get("seed_start", 314159)),
                seed_step=int(simulation.get("seed_step", 7919)),
                ts=float(simulation.get("timestep", simulation.get("ts", DEFAULT_TS))),
                tdamp=float(simulation.get("Tdamp", simulation.get("tdamp", DEFAULT_TDAMP))),
                tchain=int(simulation.get("Tchain", simulation.get("tchain", DEFAULT_TCHAIN))),
                cool_dt=float(simulation.get("cool_dT", simulation.get("cool_dt", DEFAULT_COOL_DT))),
                cool_steps_per_dt=_cool_steps_per_dt(simulation, dump_every),
                cool_back_steps=int(simulation["cool_back_steps"]) if simulation.get("cool_back_steps") is not None else None,
                hot_hold_steps=_steps_from_dump_count(
                    simulation,
                    dump_key="hot_hold_dumps",
                    steps_key="hot_hold_steps",
                    default_steps=DEFAULT_HOT_HOLD_STEPS,
                    dump_every=dump_every,
                ),
                target_hold_steps=_steps_from_dump_count(
                    simulation,
                    dump_key="target_hold_dumps",
                    steps_key="target_hold_steps",
                    default_steps=DEFAULT_TARGET_HOLD_STEPS,
                    dump_every=dump_every,
                ),
                sample_steps=_steps_from_dump_count(
                    simulation,
                    dump_key="sample_dumps",
                    steps_key="sample_steps",
                    default_steps=DEFAULT_SAMPLE_STEPS,
                    dump_every=dump_every,
                ),
                sample_every=int(simulation.get("sample_every", DEFAULT_SAMPLE_EVERY)),
                dump_every=dump_every,
                restart_dump_multiple=int(
                    simulation.get("restart_interval_dump_frames", simulation.get("restart_dump_multiple", DEFAULT_RESTART_DUMP_MULTIPLE))
                ),
                head_id_override=str(input_cfg.get("head_id", "auto")),
                tail_id_override=str(input_cfg.get("tail_id", "auto")),
                lammps_command=_build_lammps_command(run),
            )
        )
        written = write_project(cfg, overwrite=settings.overwrite)
        case = TemperatureCase(
            target_t=target_t,
            output_root=written.output_root,
            run_script=written.run_script,
            loop_dirs=written.loop_dirs,
        )
        cases.append(case)
        manifest_cases.append(
            {
                "target_T": target_t,
                "output_root": str(case.output_root),
                "run_script": str(case.run_script),
                "loop_dirs": [str(loop_dir) for loop_dir in case.loop_dirs],
            }
        )

    manifest_path = workspace / "single_restart_suite_manifest.json"
    manifest = {
        "schema": "single-restart-loop-version-v1",
        "workspace": str(workspace),
        "restart": str(restart),
        "params_path": str(params_path),
        "target_T_list": targets,
        "hot_T": float(simulation.get("hot_T", simulation.get("hot_t", 1.70))),
        "loops": loops,
        "seeds": explicit_seeds,
        "cases": manifest_cases,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return cases, settings, manifest_path


def main(argv: Sequence[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    params_path = Path(args[0]).expanduser().resolve() if args else DEFAULT_PARAMS
    cases, settings, manifest_path = generate_cases(params_path)
    print(f"Wrote single-restart manifest: {manifest_path}")
    print(f"Temperature cases: {len(cases)}")
    for case in cases:
        print(f"  Tstar_{case.target_t:.2f}: {case.output_root}")
        print(f"    run_script: {case.run_script}")

    if settings.run_lammps:
        for case in cases:
            print(f"[RUN] Tstar_{case.target_t:.2f}")
            result = subprocess.run([str(case.run_script)], cwd=case.output_root, check=False)
            if result.returncode != 0:
                return int(result.returncode)
    else:
        print("Generation only. Set run.run_lammps=true to launch all generated run_all.sh scripts sequentially.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
