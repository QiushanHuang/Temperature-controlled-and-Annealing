#!/usr/bin/env python3
"""Create params files for every L3/L7 Tstar restart under heating_cooling."""

from __future__ import annotations

import argparse
import json
import stat
import sys
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Iterable, Sequence


CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path("/Volumes/TRACER/heating_cooling")
DEFAULT_LENGTHS = ("L3", "L7")
DEFAULT_SEEDS = (111111, 222222, 333333, 444444, 555555, 666666, 777777)
DEFAULT_HOT_T = 1.70
DEFAULT_DUMP_EVERY = 1000
DEFAULT_HOT_HOLD_DUMPS = 100
DEFAULT_COOL_DT = 0.1
DEFAULT_COOL_DUMPS_PER_DT = 100
DEFAULT_TARGET_HOLD_DUMPS = 200
DEFAULT_SAMPLE_DUMPS = 4000
DEFAULT_RESTART_INTERVAL_DUMPS = 500
DEFAULT_MPI_RANKS = 4
DEFAULT_OMP_THREADS = 2
DEFAULT_MPIEXEC = "mpiexec"
DEFAULT_LAMMPS_BIN = "/home/star/Research/software/lammps-22Jul2025/build/lmp"


@dataclass(frozen=True)
class SuiteCase:
    case_id: str
    length: str
    tstar: str
    workspace: Path
    restart: Path
    params_path: Path
    result_dir: Path


def _temperature_key(path: Path) -> Decimal:
    if not path.name.startswith("Tstar_"):
        raise ValueError(f"Tstar directory name must start with Tstar_: {path}")
    return Decimal(path.name.split("Tstar_", 1)[1])


def _restart_sort_key(path: Path) -> tuple[int, str]:
    suffix = path.name.rsplit(".", 1)[-1]
    try:
        return (int(suffix), path.name)
    except ValueError:
        return (-1, path.name)


def _find_restart(tstar_dir: Path) -> Path:
    candidates = sorted(
        (
            path
            for path in tstar_dir.glob("Restart*")
            if path.is_file() and not path.name.startswith("._")
        ),
        key=_restart_sort_key,
    )
    if not candidates:
        raise FileNotFoundError(f"No Restart* file found in {tstar_dir}")
    return candidates[-1].resolve()


def _case_id(length: str, tstar: str) -> str:
    return f"{length}_Tstar_{tstar}"


def _hot_tag(hot_t: float) -> str:
    return f"{float(hot_t):.2f}"


def _run_tag(mpi_ranks: int, omp_threads: int) -> str:
    return f"np{int(mpi_ranks)}omp{int(omp_threads)}"


def _params_payload(
    *,
    case: SuiteCase,
    loops: int,
    seeds: Sequence[int],
    hot_t: float,
    run_lammps: bool,
    mpi_ranks: int,
    omp_threads: int,
    mpiexec: str,
    lammps_bin: str,
    overwrite: bool,
) -> dict[str, object]:
    return {
        "workspace": str(case.workspace),
        "input": {
            "restart_file": case.restart.name,
            "restart_glob": "Restart*",
            "head_id": "auto",
            "tail_id": "auto",
        },
        "output": {
            "output_root": str(case.result_dir),
            "overwrite": overwrite,
        },
        "simulation": {
            "target_T": float(Decimal(case.tstar)),
            "hot_T": float(hot_t),
            "loops": int(loops),
            "seeds": [int(seed) for seed in seeds],
            "timestep": 0.001,
            "Tdamp": 0.1,
            "Tchain": 3,
            "dump_every": DEFAULT_DUMP_EVERY,
            "sample_every": DEFAULT_DUMP_EVERY,
            "hot_hold_dumps": DEFAULT_HOT_HOLD_DUMPS,
            "cool_dT": DEFAULT_COOL_DT,
            "cool_dumps_per_dT": DEFAULT_COOL_DUMPS_PER_DT,
            "cool_back_steps": None,
            "target_hold_dumps": DEFAULT_TARGET_HOLD_DUMPS,
            "sample_dumps": DEFAULT_SAMPLE_DUMPS,
            "restart_interval_dump_frames": DEFAULT_RESTART_INTERVAL_DUMPS,
        },
        "run": {
            "run_lammps": bool(run_lammps),
            "mpi_ranks": int(mpi_ranks),
            "omp_threads": int(omp_threads),
            "mpiexec": str(mpiexec),
            "lammps_bin": str(lammps_bin),
            "result_dir": str(case.result_dir),
        },
    }


def _write_run_all_script(suite_dir: Path, cases: Sequence[SuiteCase]) -> Path:
    script = suite_dir / "run_all_params.sh"
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"CODE_ROOT={str(CODE_ROOT)!r}",
        "",
    ]
    for case in cases:
        lines.extend(
            [
                f"echo '[CASE] {case.case_id}'",
                f"python3 \"$CODE_ROOT/run_anneal.py\" {str(case.params_path)!r}",
                "",
            ]
        )
    script.write_text("\n".join(lines), encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return script


def create_suite(
    *,
    root: str | Path = DEFAULT_ROOT,
    suite_dir: str | Path | None = None,
    output_root: str | Path | None = None,
    lengths: Iterable[str] = DEFAULT_LENGTHS,
    loops: int = 7,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    hot_t: float = DEFAULT_HOT_T,
    run_lammps: bool = False,
    mpi_ranks: int = DEFAULT_MPI_RANKS,
    omp_threads: int = DEFAULT_OMP_THREADS,
    mpiexec: str = DEFAULT_MPIEXEC,
    lammps_bin: str = DEFAULT_LAMMPS_BIN,
    overwrite: bool = False,
) -> Path:
    root = Path(root).expanduser().resolve()
    output_base = None
    if output_root is not None:
        output_base = Path(output_root).expanduser().resolve()
        if any(ch.isspace() for ch in str(output_base)):
            raise ValueError(
                f"output_root contains whitespace: {output_base}. "
                "Use a no-space mount path or symlink such as /media/star/MyPassport2."
            )
    if suite_dir is None:
        suite_path = root / f"anneal_hot{_hot_tag(hot_t)}_{_run_tag(mpi_ranks, omp_threads)}_loops{loops}_suite"
    else:
        suite_path = Path(suite_dir).expanduser()
        if not suite_path.is_absolute():
            suite_path = root / suite_path
        suite_path = suite_path.resolve()

    seeds = tuple(int(seed) for seed in seeds)
    if loops <= 0:
        raise ValueError("loops must be positive")
    if len(seeds) != loops:
        raise ValueError("number of seeds must equal loops")

    params_dir = suite_path / "params"
    params_dir.mkdir(parents=True, exist_ok=True)

    result_dir_name = f"anneal_hot{_hot_tag(hot_t)}_{_run_tag(mpi_ranks, omp_threads)}_loops{loops}"
    cases: list[SuiteCase] = []
    for length in lengths:
        length_dir = root / str(length)
        if not length_dir.is_dir():
            raise FileNotFoundError(f"Missing length directory: {length_dir}")
        tstar_dirs = sorted(
            (path for path in length_dir.iterdir() if path.is_dir() and path.name.startswith("Tstar_")),
            key=_temperature_key,
        )
        if not tstar_dirs:
            raise FileNotFoundError(f"No Tstar_* directories found under {length_dir}")
        for tstar_dir in tstar_dirs:
            tstar = tstar_dir.name.split("Tstar_", 1)[1]
            case_id = _case_id(str(length), tstar)
            result_base = output_base / str(length) / tstar_dir.name if output_base is not None else tstar_dir
            case = SuiteCase(
                case_id=case_id,
                length=str(length),
                tstar=tstar,
                workspace=tstar_dir.resolve(),
                restart=_find_restart(tstar_dir),
                params_path=(params_dir / f"{case_id}.json").resolve(),
                result_dir=(result_base / result_dir_name).resolve(),
            )
            payload = _params_payload(
                case=case,
                loops=loops,
                seeds=seeds,
                hot_t=hot_t,
                run_lammps=run_lammps,
                mpi_ranks=mpi_ranks,
                omp_threads=omp_threads,
                mpiexec=mpiexec,
                lammps_bin=lammps_bin,
                overwrite=overwrite,
            )
            case.params_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            cases.append(case)

    run_all_script = _write_run_all_script(suite_path, cases)
    manifest = {
        "schema": "temperature-controlled-anneal-heating-cooling-suite-v1",
        "suite_id": suite_path.name,
        "root": str(root),
        "output_root": str(output_base) if output_base is not None else None,
        "hot_T": float(hot_t),
        "loops": int(loops),
        "seeds": list(seeds),
        "protocol": {
            "dump_every": DEFAULT_DUMP_EVERY,
            "hot_hold_dumps": DEFAULT_HOT_HOLD_DUMPS,
            "cool_dT": DEFAULT_COOL_DT,
            "cool_dumps_per_dT": DEFAULT_COOL_DUMPS_PER_DT,
            "target_hold_dumps": DEFAULT_TARGET_HOLD_DUMPS,
            "sample_dumps": DEFAULT_SAMPLE_DUMPS,
            "restart_interval_dump_frames": DEFAULT_RESTART_INTERVAL_DUMPS,
        },
        "run": {
            "run_lammps": bool(run_lammps),
            "mpi_ranks": int(mpi_ranks),
            "omp_threads": int(omp_threads),
            "mpiexec": str(mpiexec),
            "lammps_bin": str(lammps_bin),
        },
        "run_all_script": str(run_all_script.resolve()),
        "cases": [
            {
                "case_id": case.case_id,
                "length": case.length,
                "tstar": case.tstar,
                "workspace": str(case.workspace),
                "restart": str(case.restart),
                "params_path": str(case.params_path),
                "result_dir": str(case.result_dir),
            }
            for case in cases
        ],
    }
    manifest_path = suite_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path.resolve()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create params files for all L3/L7 Tstar restart branches under /Volumes/TRACER/heating_cooling."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Root containing L3 and L7 directories.")
    parser.add_argument("--suite-dir", type=Path, default=None, help="Directory for manifest and generated params.")
    parser.add_argument("--output-root", type=Path, default=None, help="Optional no-space root for large LAMMPS outputs.")
    parser.add_argument("--lengths", nargs="+", default=list(DEFAULT_LENGTHS), help="Length directories to scan.")
    parser.add_argument("--loops", type=int, default=7, help="Independent loops per Tstar case.")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS), help="Velocity seeds, one per loop.")
    parser.add_argument("--hot-T", type=float, default=DEFAULT_HOT_T, dest="hot_t", help="High temperature for Gaussian velocity assignment.")
    parser.add_argument("--run-lammps", action="store_true", help="Set run.run_lammps=true in generated params.")
    parser.add_argument("--mpi-ranks", type=int, default=DEFAULT_MPI_RANKS, help="MPI ranks per case.")
    parser.add_argument("--omp-threads", type=int, default=DEFAULT_OMP_THREADS, help="OMP threads per MPI rank.")
    parser.add_argument("--mpiexec", default=DEFAULT_MPIEXEC, help="MPI launcher.")
    parser.add_argument("--lammps-bin", default=DEFAULT_LAMMPS_BIN, help="LAMMPS executable.")
    parser.add_argument("--overwrite", action="store_true", help="Set output.overwrite=true in generated params.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = create_suite(
        root=args.root,
        suite_dir=args.suite_dir,
        output_root=args.output_root,
        lengths=args.lengths,
        loops=args.loops,
        seeds=args.seeds,
        hot_t=args.hot_t,
        run_lammps=args.run_lammps,
        mpi_ranks=args.mpi_ranks,
        omp_threads=args.omp_threads,
        mpiexec=args.mpiexec,
        lammps_bin=args.lammps_bin,
        overwrite=args.overwrite,
    )
    data = json.loads(manifest.read_text(encoding="utf-8"))
    print(f"Wrote suite manifest: {manifest}")
    print(f"Cases: {len(data.get('cases', []))}")
    print(f"Run all params: {data['run_all_script']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
