#!/usr/bin/env python3
"""Parameter-file entry point for restart annealing runs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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


DEFAULT_PARAMS = PROJECT_ROOT / "params.json"


@dataclass(frozen=True)
class RunSettings:
    run_lammps: bool = False
    overwrite: bool = False


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


def _resolve_project_path(project_root: Path, value: str | Path, *, resolve_symlinks: bool = True) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    if not resolve_symlinks:
        return _absolute_no_resolve(path)
    return path.resolve()


def _resolve_workspace(params_path: Path, data: dict[str, Any]) -> Path:
    workspace = data.get("workspace", "params")
    if workspace == "params":
        return params_path.parent
    if workspace == "cwd":
        return Path.cwd().resolve()
    if isinstance(workspace, str):
        return _resolve_project_path(params_path.parent, workspace)
    raise ValueError("workspace must be 'params', 'cwd', or a path string")


def _discover_restart(project_root: Path, pattern: str) -> Path:
    matches = sorted(path for path in project_root.glob(pattern) if path.is_file())
    if len(matches) == 1:
        return matches[0].resolve()
    if not matches:
        raise FileNotFoundError(
            f"No restart file matched {pattern!r} in {project_root}. "
            "Put one restart file in this folder or set input.restart_file in params.json."
        )
    joined = "\n".join(f"  {path.name}" for path in matches)
    raise ValueError(
        f"Multiple restart files matched {pattern!r} in {project_root}; set input.restart_file explicitly:\n{joined}"
    )


def _simulation_value(simulation: dict[str, Any], primary: str, fallback: str | None = None, default: Any = None) -> Any:
    if primary in simulation:
        return simulation[primary]
    if fallback is not None and fallback in simulation:
        return simulation[fallback]
    return default


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
    return int(
        _simulation_value(
            simulation,
            "cool_steps_per_dT",
            "cool_steps_per_dt",
            DEFAULT_COOL_STEPS_PER_DT,
        )
    )


def _json_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    raise ValueError(f"{name} must be a JSON boolean true/false")


def load_project_config(params_path: str | Path = DEFAULT_PARAMS) -> tuple[GeneratorConfig, RunSettings]:
    params_path = Path(params_path).expanduser().resolve()
    data = _read_json(params_path)
    project_root = _resolve_workspace(params_path, data)

    input_cfg = data.get("input", {})
    output_cfg = data.get("output", {})
    simulation = data.get("simulation", {})
    run = data.get("run", {})
    if not isinstance(input_cfg, dict) or not isinstance(output_cfg, dict) or not isinstance(simulation, dict) or not isinstance(run, dict):
        raise ValueError("params.json sections input/output/simulation/run must be JSON objects")

    restart_setting = input_cfg.get("restart_file", "auto")
    if restart_setting == "auto":
        restart = _discover_restart(project_root, str(input_cfg.get("restart_glob", "Restart*")))
    else:
        restart = _resolve_project_path(project_root, str(restart_setting))

    result_dir = run.get("result_dir")
    output_setting = result_dir if result_dir else output_cfg.get("output_root", "anneal_output")
    output_root = _resolve_project_path(project_root, str(output_setting), resolve_symlinks=False)

    loops = int(simulation["loops"])
    explicit_seeds = simulation.get("seeds")
    if explicit_seeds is not None:
        explicit_seeds = [int(seed) for seed in explicit_seeds]

    dump_every = int(simulation.get("dump_every", DEFAULT_DUMP_EVERY))

    cfg = finalize_config(
        GeneratorConfig(
            restart=restart,
            output_root=output_root,
            target_t=float(_simulation_value(simulation, "target_T", "target_t")),
            hot_t=float(_simulation_value(simulation, "hot_T", "hot_t")),
            loops=loops,
            explicit_seeds=explicit_seeds,
            seed_start=int(simulation.get("seed_start", 314159)),
            seed_step=int(simulation.get("seed_step", 7919)),
            ts=float(simulation.get("timestep", simulation.get("ts", DEFAULT_TS))),
            tdamp=float(simulation.get("Tdamp", simulation.get("tdamp", DEFAULT_TDAMP))),
            tchain=int(simulation.get("Tchain", simulation.get("tchain", DEFAULT_TCHAIN))),
            cool_dt=float(_simulation_value(simulation, "cool_dT", "cool_dt", DEFAULT_COOL_DT)),
            cool_steps_per_dt=_cool_steps_per_dt(simulation, dump_every),
            cool_back_steps=(
                int(simulation["cool_back_steps"]) if simulation.get("cool_back_steps") is not None else None
            ),
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
                simulation.get(
                    "restart_interval_dump_frames",
                    simulation.get("restart_dump_multiple", DEFAULT_RESTART_DUMP_MULTIPLE),
                )
            ),
            head_id_override=str(input_cfg.get("head_id", "auto")),
            tail_id_override=str(input_cfg.get("tail_id", "auto")),
            lammps_command=_build_lammps_command(run),
        )
    )
    run_settings = RunSettings(
        run_lammps=_json_bool(run.get("run_lammps", False), name="run.run_lammps"),
        overwrite=_json_bool(output_cfg.get("overwrite", run.get("overwrite", False)), name="output.overwrite"),
    )
    return cfg, run_settings


def _build_lammps_command(run: dict[str, Any]) -> str:
    if run.get("lammps_command"):
        return str(run["lammps_command"])

    lammps_bin = str(run.get("lammps_bin", "lmp"))
    mpi_ranks = int(run.get("mpi_ranks", 1))
    omp_threads = int(run.get("omp_threads", 1))
    if mpi_ranks <= 0:
        raise ValueError("run.mpi_ranks must be positive")
    if omp_threads <= 0:
        raise ValueError("run.omp_threads must be positive")

    parts: list[str] = []
    if omp_threads != 1:
        parts.append(f"OMP_NUM_THREADS={omp_threads}")
    if mpi_ranks != 1:
        parts.extend([str(run.get("mpiexec", "mpiexec")), "-np", str(mpi_ranks)])
    parts.append(lammps_bin)
    return " ".join(parts)


def _default_params_path() -> Path:
    cwd_params = Path.cwd() / "params.json"
    if cwd_params.is_file():
        return cwd_params
    return DEFAULT_PARAMS


def _print_config_summary(cfg: GeneratorConfig, run_settings: RunSettings) -> None:
    print("Resolved annealing parameters:")
    print(f"  restart={cfg.restart}")
    print(f"  output_root={cfg.output_root}")
    print(f"  target_T={cfg.target_t:g}")
    print(f"  hot_T={cfg.hot_t:g}")
    print(f"  loops={cfg.loops}")
    print(f"  seeds={cfg.explicit_seeds if cfg.explicit_seeds is not None else 'generated'}")
    print(f"  timestep={cfg.ts:g}")
    print(f"  Tdamp={cfg.tdamp:g}")
    print(f"  Tchain={cfg.tchain}")
    print(f"  hot_hold_steps={cfg.hot_hold_steps}")
    print(f"  cool_dT={cfg.cool_dt:g}")
    print(f"  cool_steps_per_dT={cfg.cool_steps_per_dt}")
    print(f"  cool_back_steps={cfg.cool_back_steps}")
    print(f"  target_hold_steps={cfg.target_hold_steps}")
    print(f"  sample_steps={cfg.sample_steps}")
    print(f"  sample_every={cfg.sample_every}")
    print(f"  dump_every={cfg.dump_every}")
    print(f"  restart_every={cfg.restart_every} ({cfg.restart_dump_multiple} dump intervals)")
    print(f"  lammps_command={cfg.lammps_command}")
    print(f"  run_lammps={run_settings.run_lammps}")


def main() -> int:
    params_paths = [Path(arg).expanduser().resolve() for arg in sys.argv[1:]] if len(sys.argv) > 1 else [_default_params_path()]
    for params_path in params_paths:
        print(f"Using params: {params_path}")
        cfg, run_settings = load_project_config(params_path)
        _print_config_summary(cfg, run_settings)
        written = write_project(cfg, overwrite=run_settings.overwrite)
        print(f"Wrote {len(written.loop_dirs)} loop directories under {written.output_root}")
        print(f"Run script: {written.run_script}")
        for loop_dir in written.loop_dirs:
            print(f"  {loop_dir}")

        if run_settings.run_lammps:
            print("Running LAMMPS via generated run_all.sh")
            result = subprocess.run([str(written.run_script)], cwd=written.output_root, check=False)
            if result.returncode != 0:
                return int(result.returncode)
        else:
            print("Generation only. Set run.run_lammps=true in params.json to launch LAMMPS automatically.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
