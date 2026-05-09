#!/usr/bin/env python3
"""Generate independent annealing-branch LAMMPS inputs from one restart."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import shlex
import stat
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_TS = 0.001
DEFAULT_TDAMP = 0.1
DEFAULT_TCHAIN = 3
DEFAULT_COOL_DT = 0.1
DEFAULT_COOL_STEPS_PER_DT = 100000
DEFAULT_HOT_HOLD_STEPS = 100000
DEFAULT_TARGET_HOLD_STEPS = 200000
DEFAULT_SAMPLE_STEPS = 4000000
DEFAULT_SAMPLE_EVERY = 1000
DEFAULT_DUMP_EVERY = 1000
DEFAULT_RESTART_DUMP_MULTIPLE = 500
DEFAULT_SEED_START = 314159
DEFAULT_SEED_STEP = 7919


@dataclass(frozen=True)
class GeneratorConfig:
    restart: Path
    output_root: Path
    target_t: float
    hot_t: float
    loops: int
    explicit_seeds: list[int] | None = None
    seed_start: int = DEFAULT_SEED_START
    seed_step: int = DEFAULT_SEED_STEP
    ts: float = DEFAULT_TS
    tdamp: float = DEFAULT_TDAMP
    tchain: int = DEFAULT_TCHAIN
    cool_dt: float = DEFAULT_COOL_DT
    cool_steps_per_dt: int = DEFAULT_COOL_STEPS_PER_DT
    cool_back_steps: int | None = None
    hot_hold_steps: int = DEFAULT_HOT_HOLD_STEPS
    target_hold_steps: int = DEFAULT_TARGET_HOLD_STEPS
    sample_steps: int = DEFAULT_SAMPLE_STEPS
    sample_every: int = DEFAULT_SAMPLE_EVERY
    dump_every: int = DEFAULT_DUMP_EVERY
    restart_dump_multiple: int = DEFAULT_RESTART_DUMP_MULTIPLE
    restart_every: int | None = None
    head_id_override: str = "auto"
    tail_id_override: str = "auto"
    lammps_command: str = "lmp"
    dry_run: bool = False


@dataclass(frozen=True)
class WrittenProject:
    output_root: Path
    loop_dirs: list[Path]
    run_script: Path


def derive_cool_steps(target_t: float, hot_t: float, cool_dt: float, steps_per_dt: int) -> int:
    if not all(math.isfinite(value) for value in (target_t, hot_t, cool_dt)):
        raise ValueError("temperatures and cool_dt must be finite")
    if target_t <= 0 or hot_t <= 0:
        raise ValueError("target_t and hot_t must be positive")
    if cool_dt <= 0:
        raise ValueError("cool_dt must be positive")
    if steps_per_dt <= 0:
        raise ValueError("steps_per_dt must be positive")
    if hot_t <= target_t:
        raise ValueError("hot_t must be greater than target_t")
    scaled_steps = (hot_t - target_t) / cool_dt * steps_per_dt
    return max(1, int(round(scaled_steps)))


def build_seed_list(
    loops: int,
    explicit_seeds: Sequence[int] | None,
    seed_start: int,
    seed_step: int,
) -> list[int]:
    if loops <= 0:
        raise ValueError("loops must be positive")
    if explicit_seeds is not None:
        seeds = [int(seed) for seed in explicit_seeds]
        if len(seeds) != loops:
            raise ValueError("number of explicit seeds must equal loops")
    else:
        if seed_step <= 0:
            raise ValueError("seed_step must be positive")
        seeds = [int(seed_start + idx * seed_step) for idx in range(loops)]
    if any(seed <= 0 for seed in seeds):
        raise ValueError("all seeds must be positive")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")
    return seeds


def loop_dir_name(loop_index: int, seed: int) -> str:
    if loop_index <= 0:
        raise ValueError("loop_index must be positive")
    return f"loop{loop_index}_{seed}"


def _absolute_no_resolve(path: str | Path) -> Path:
    expanded = Path(path).expanduser()
    if not expanded.is_absolute():
        expanded = Path.cwd() / expanded
    return Path(os.path.abspath(os.fspath(expanded)))


def finalize_config(config: GeneratorConfig) -> GeneratorConfig:
    cool_back_steps = config.cool_back_steps
    if cool_back_steps is None:
        cool_back_steps = derive_cool_steps(
            target_t=config.target_t,
            hot_t=config.hot_t,
            cool_dt=config.cool_dt,
            steps_per_dt=config.cool_steps_per_dt,
        )

    restart_every = config.restart_every
    if restart_every is None:
        restart_every = config.dump_every * config.restart_dump_multiple

    finalized = GeneratorConfig(
        restart=Path(config.restart).expanduser().resolve(),
        output_root=_absolute_no_resolve(config.output_root),
        target_t=float(config.target_t),
        hot_t=float(config.hot_t),
        loops=int(config.loops),
        explicit_seeds=list(config.explicit_seeds) if config.explicit_seeds is not None else None,
        seed_start=int(config.seed_start),
        seed_step=int(config.seed_step),
        ts=float(config.ts),
        tdamp=float(config.tdamp),
        tchain=int(config.tchain),
        cool_dt=float(config.cool_dt),
        cool_steps_per_dt=int(config.cool_steps_per_dt),
        cool_back_steps=int(cool_back_steps),
        hot_hold_steps=int(config.hot_hold_steps),
        target_hold_steps=int(config.target_hold_steps),
        sample_steps=int(config.sample_steps),
        sample_every=int(config.sample_every),
        dump_every=int(config.dump_every),
        restart_dump_multiple=int(config.restart_dump_multiple),
        restart_every=int(restart_every),
        head_id_override=str(config.head_id_override),
        tail_id_override=str(config.tail_id_override),
        lammps_command=str(config.lammps_command),
        dry_run=bool(config.dry_run),
    )
    validate_config(finalized)
    return finalized


def validate_config(config: GeneratorConfig) -> None:
    if not config.restart.is_file():
        raise FileNotFoundError(f"restart file does not exist: {config.restart}")
    _reject_lammps_path_whitespace("restart", config.restart)
    _reject_lammps_path_whitespace("output_root", config.output_root)
    if not math.isfinite(config.target_t) or not math.isfinite(config.hot_t):
        raise ValueError("target_t and hot_t must be finite")
    if config.target_t <= 0 or config.hot_t <= 0:
        raise ValueError("target_t and hot_t must be positive")
    if config.hot_t <= config.target_t:
        raise ValueError("hot_t must be greater than target_t")
    if config.loops <= 0:
        raise ValueError("loops must be positive")
    positive_ints = {
        "tchain": config.tchain,
        "cool_steps_per_dt": config.cool_steps_per_dt,
        "cool_back_steps": config.cool_back_steps,
        "hot_hold_steps": config.hot_hold_steps,
        "target_hold_steps": config.target_hold_steps,
        "sample_steps": config.sample_steps,
        "sample_every": config.sample_every,
        "dump_every": config.dump_every,
        "restart_dump_multiple": config.restart_dump_multiple,
        "restart_every": config.restart_every,
    }
    for name, value in positive_ints.items():
        if value is None or int(value) <= 0:
            raise ValueError(f"{name} must be positive")
    for name, value in {"ts": config.ts, "tdamp": config.tdamp, "cool_dt": config.cool_dt}.items():
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive")
    if config.restart_every != config.dump_every * config.restart_dump_multiple:
        raise ValueError("restart_every must equal dump_every * restart_dump_multiple")
    _validate_atom_id_override("head_id_override", config.head_id_override)
    _validate_atom_id_override("tail_id_override", config.tail_id_override)
    build_seed_list(config.loops, config.explicit_seeds, config.seed_start, config.seed_step)


def _reject_lammps_path_whitespace(name: str, path: Path) -> None:
    if any(ch.isspace() for ch in str(path)):
        raise ValueError(
            f"{name} path contains whitespace, which this generator rejects because "
            f"LAMMPS command parsing can split unquoted paths: {path}"
        )


def _validate_atom_id_override(name: str, value: str) -> None:
    if value == "auto":
        return
    try:
        atom_id = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be 'auto' or a positive integer") from exc
    if atom_id <= 0 or str(atom_id) != value:
        raise ValueError(f"{name} must be 'auto' or a positive integer")


def _lammps_path(path: Path) -> str:
    return str(path).replace("\\", "\\\\")


def _phase_dump_block(dump_id: str, phase: str, filename_prefix: str) -> str:
    return f"""dump            {dump_id} all custom ${{dump_every}} ${{loop_dir}}/{phase}/{filename_prefix}.*.dump id type x y z xu yu zu vx vy vz fx fy fz c_orient[1] c_orient[2] c_orient[3] c_orient[4] c_shape[1] c_shape[2] c_shape[3] mass
dump_modify     {dump_id} colname c_orient[1] quatw colname c_orient[2] quati colname c_orient[3] quatj colname c_orient[4] quatk
dump_modify     {dump_id} colname c_shape[1] shapex colname c_shape[2] shapey colname c_shape[3] shapez"""


def _thermostat_block(start_t: str, stop_t: str, steps: str) -> str:
    return f"""fix             thermostat_lc rigid_lc rigid/nvt/small molecule temp {start_t} {stop_t} ${{Tdamp}}
fix             thermostat_sph sphere_thermo nvt temp {start_t} {stop_t} ${{Tdamp}} tchain ${{Tchain}}
fix_modify      thermostat_sph temp T_sph
run             {steps}
unfix           thermostat_lc
unfix           thermostat_sph"""


def _group_guard(group_name: str) -> str:
    return (
        f'if "$(count({group_name})) <= 0" then "print \'ERROR: {group_name} group is empty\'"\n'
        f'if "$(count({group_name})) <= 0" then "quit 2"'
    )


def _head_tail_setup(config: GeneratorConfig) -> str:
    if config.head_id_override == "auto":
        head_line = "variable        head_id equal c_min_atom_id"
    else:
        head_line = f"variable        head_id index {config.head_id_override}"

    if config.tail_id_override == "auto":
        tail_line = "variable        tail_id equal c_max_atom_id"
    else:
        tail_line = f"variable        tail_id index {config.tail_id_override}"

    return "\n".join(
        [
            head_line,
            tail_line,
            f'print           "ATOM-ID SETUP: head_id = ${{head_id}}, tail_id = ${{tail_id}} (overrides: head={config.head_id_override}, tail={config.tail_id_override})"',
        ]
    )


def render_lammps_input(config: GeneratorConfig, loop_index: int, seed: int) -> str:
    loop_name = loop_dir_name(loop_index, seed)
    loop_dir = config.output_root / loop_name
    final_t_tag = f"{config.target_t:.2f}"

    return f"""# Independent restart annealing branch generated by anneal_restart_generator.py
# Loop {loop_index}, seed {seed}
# Protocol: read one selected restart, assign Gaussian velocities at hot_T, relax, cool to target_T, hold, then sample.

variable        restart_file index {_lammps_path(config.restart)}
variable        output_root index {_lammps_path(config.output_root)}
variable        loop_dir index {_lammps_path(loop_dir)}
variable        loop_index index {loop_index}
variable        loop_seed index {seed}
variable        protocol_tag index independent_restart_anneal_branch

variable        target_T index {config.target_t:.8g}
variable        hot_T index {config.hot_t:.8g}
variable        final_T equal ${{target_T}}
variable        final_T_tag index {final_t_tag}
variable        ts index {config.ts:.8g}
variable        Tdamp index {config.tdamp:.8g}
variable        Tchain index {config.tchain}
variable        hot_hold_steps index {config.hot_hold_steps}
variable        cool_back_steps index {config.cool_back_steps}
variable        target_hold_steps index {config.target_hold_steps}
variable        sample_steps index {config.sample_steps}
variable        sample_every index {config.sample_every}
variable        dump_every index {config.dump_every}
variable        restart_dump_multiple index {config.restart_dump_multiple}
variable        restart_every index {config.restart_every}
variable        dry_run index {"yes" if config.dry_run else "no"}

variable        head_id_override index {config.head_id_override}
variable        tail_id_override index {config.tail_id_override}

log             ${{loop_dir}}/log.loop.lammps

read_restart    ${{restart_file}}
reset_timestep  0
timestep        ${{ts}}

neighbor        1.0 nsq
neigh_modify    delay 0 every 1 check yes
comm_modify     cutoff 35.0

variable        dA equal 0.00001
variable        k_bond_ss equal 250.0
variable        r0_ss equal 1.0
variable        r0_sa equal 0.5
variable        epsilon0 equal 1.0
variable        sig0 equal 1.0
variable        eps_a equal 1.0
variable        eps_b equal 1.0
variable        eps_c equal 0.2
variable        gb_gamma equal 1.0
variable        gb_upsilon equal 3.0
variable        gb_mu equal 1.0
variable        gb_rcut equal 5.0
variable        dB equal 1.0
variable        epsSS equal 1.0
variable        sigSS equal ${{dB}}
variable        epsSE equal ${{epsilon0}}
variable        sigSE equal ${{sig0}}
variable        rc_wca equal 1.122462048309373*v_sigSS
variable        sigAA equal ${{dA}}
variable        rcAA equal 1.122462048309373*v_sigAA
variable        sigSA equal 0.5*(v_sigSS+v_dA)
variable        rcSA equal 1.122462048309373*v_sigSA
variable        sigEA equal 0.5*(v_sig0+v_dA)
variable        rcEA equal 1.122462048309373*v_sigEA

bond_style      harmonic
bond_coeff      1 ${{k_bond_ss}} ${{r0_ss}}
bond_coeff      2 ${{k_bond_ss}} ${{r0_sa}}
bond_coeff      3 1.0 1.0
bond_coeff      4 1.0 1.0

special_bonds   lj 0.0 1.0 1.0
pair_style      hybrid gayberne ${{gb_gamma}} ${{gb_upsilon}} ${{gb_mu}} ${{gb_rcut}} lj/cut ${{rc_wca}}
pair_modify     shift yes
pair_coeff      1 1 gayberne ${{epsilon0}} ${{sig0}} ${{eps_a}} ${{eps_b}} ${{eps_c}} ${{eps_a}} ${{eps_b}} ${{eps_c}} ${{gb_rcut}}
pair_coeff      2 2 lj/cut ${{epsSS}} ${{sigSS}} ${{rc_wca}}
pair_coeff      3 3 lj/cut 0.000001 ${{sigAA}} ${{rcAA}}
pair_coeff      2 3 lj/cut 0.000001 ${{sigSA}} ${{rcSA}}
pair_coeff      1 3 lj/cut 0.000001 ${{sigEA}} ${{rcEA}}
pair_coeff      1 2 gayberne ${{epsSE}} ${{sigSE}} ${{eps_a}} ${{eps_b}} ${{eps_c}} 1.0 1.0 1.0 ${{gb_rcut}}

compute         atom_ids all property/atom id
compute         min_atom_id all reduce min c_atom_ids
compute         max_atom_id all reduce max c_atom_ids

run             0

{_head_tail_setup(config)}

group           ellipsoid empty
group           sphere empty
group           anchor empty
group           rigid_lc empty
group           head empty
group           tail empty
group           sphere_thermo empty
group           ellipsoid clear
group           sphere clear
group           anchor clear
group           rigid_lc clear
group           head clear
group           tail clear
group           sphere_thermo clear
group           ellipsoid type 1
group           sphere type 2
group           anchor type 3
group           rigid_lc type 1 3
group           head id ${{head_id}}
group           tail id ${{tail_id}}
group           sphere_thermo type 2

compute         T_sph sphere_thermo temp
compute_modify  T_sph extra/dof 0
compute         T_tail tail temp
compute_modify  T_tail extra/dof 0
compute         KE_ell ellipsoid ke
compute         ER_ell ellipsoid erotate/asphere
compute         orient all property/atom quatw quati quatj quatk
compute         shape all property/atom shapex shapey shapez
compute         pos all property/atom x y z

run             0

variable        L_contour equal count(sphere)*1.0+count(ellipsoid)*3.0
variable        dof_sph equal 3*count(sphere_thermo)
variable        dof_tail equal 3*count(tail)
variable        dof_ell equal 5*count(ellipsoid)
variable        dof_mix equal v_dof_sph+v_dof_ell
variable        T_ell equal 2.0*(c_KE_ell+c_ER_ell)/v_dof_ell
variable        T_mix equal (v_dof_sph*c_T_sph+v_dof_ell*v_T_ell)/(v_dof_mix)

{_group_guard("ellipsoid")}
{_group_guard("sphere_thermo")}
{_group_guard("rigid_lc")}
{_group_guard("head")}
{_group_guard("tail")}

# Gaussian velocity assignment creates independent branches by seed. The following
# high-temperature NVT hold is the thermalization stage for the rigid-body DOF.
velocity        all create ${{hot_T}} ${{loop_seed}} mom yes rot yes dist gaussian
velocity        all zero linear
velocity        all zero angular

variable        x_head equal xcm(head,x)
variable        x_tail equal xcm(tail,x)
variable        Lproj equal v_x_tail-v_x_head
variable        y_head equal xcm(head,y)
variable        y_tail equal xcm(tail,y)
variable        z_head equal xcm(head,z)
variable        z_tail equal xcm(tail,z)
variable        dx_ht equal v_x_tail-v_x_head
variable        dy_ht equal v_y_tail-v_y_head
variable        dz_ht equal v_z_tail-v_z_head
variable        Ree equal sqrt(v_dx_ht*v_dx_ht+v_dy_ht*v_dy_ht+v_dz_ht*v_dz_ht)
variable        contour_fraction equal v_Lproj/v_L_contour
variable        step_now equal step
variable        sim_time equal time

thermo          ${{sample_every}}
thermo_style    custom step temp v_T_mix c_T_sph v_T_ell c_T_tail v_dof_sph v_dof_ell v_dof_mix v_dof_tail v_Lproj v_Ree v_contour_fraction press pe ke etotal
thermo_modify   flush yes lost warn

fix             out all print ${{sample_every}} "${{step_now}} ${{sim_time}} ${{Lproj}} ${{Ree}} ${{contour_fraction}} ${{x_head}} ${{x_tail}} ${{y_head}} ${{y_tail}} ${{z_head}} ${{z_tail}}" file ${{loop_dir}}/loop_state.dat screen no title "# step time Lproj Ree contour_fraction x_head x_tail y_head y_tail z_head z_tail"

print           "ANNEAL SETUP: loop=${{loop_index}}, seed=${{loop_seed}}, restart=${{restart_file}}, hot_T=${{hot_T}}, target_T=${{target_T}}, hot_hold_steps=${{hot_hold_steps}}, cool_back_steps=${{cool_back_steps}}, target_hold_steps=${{target_hold_steps}}, sample_steps=${{sample_steps}}, dump_every=${{dump_every}}, restart_every=${{restart_every}}"

{_phase_dump_block("traj_hot_hold", "hot_hold", "traj.hot_hold")}
restart         ${{restart_every}} ${{loop_dir}}/hot_hold/Restart.hot_hold.*
{_thermostat_block("${hot_T}", "${hot_T}", "${hot_hold_steps}")}
write_restart   ${{loop_dir}}/hot_hold/After.hot_hold.bin
undump          traj_hot_hold

{_phase_dump_block("traj_cool", "cool_to_target", "traj.cool")}
restart         ${{restart_every}} ${{loop_dir}}/cool_to_target/Restart.cool.*
{_thermostat_block("${hot_T}", "${target_T}", "${cool_back_steps}")}
write_restart   ${{loop_dir}}/cool_to_target/After.cool_to_target.bin
undump          traj_cool

{_phase_dump_block("traj_target_hold", "target_hold", "traj.target_hold")}
restart         ${{restart_every}} ${{loop_dir}}/target_hold/Restart.target_hold.*
{_thermostat_block("${target_T}", "${target_T}", "${target_hold_steps}")}
write_restart   ${{loop_dir}}/target_hold/After.target_hold_T${{final_T_tag}}.bin
undump          traj_target_hold

{_phase_dump_block("traj_sample", "sampling", "traj.sample")}
restart         ${{restart_every}} ${{loop_dir}}/sampling/Restart.sample.*
{_thermostat_block("${target_T}", "${target_T}", "${sample_steps}")}
write_restart   ${{loop_dir}}/sampling/Final.sample_T${{final_T_tag}}.bin
undump          traj_sample

unfix           out
print           "ANNEAL COMPLETE: loop=${{loop_index}}, seed=${{loop_seed}}, final_restart=${{loop_dir}}/sampling/Final.sample_T${{final_T_tag}}.bin"
"""


def _manifest(config: GeneratorConfig, loop_index: int, seed: int) -> dict[str, object]:
    loop_name = loop_dir_name(loop_index, seed)
    data = asdict(config)
    data["restart"] = str(config.restart)
    data["output_root"] = str(config.output_root)
    data["loop_index"] = loop_index
    data["seed"] = seed
    data["loop_dir_name"] = loop_name
    data["loop_dir"] = str(config.output_root / loop_name)
    return data


def write_project(config: GeneratorConfig, overwrite: bool = False) -> WrittenProject:
    validate_config(config)
    seeds = build_seed_list(config.loops, config.explicit_seeds, config.seed_start, config.seed_step)
    loop_dirs = [config.output_root / loop_dir_name(loop_index, seed) for loop_index, seed in enumerate(seeds, start=1)]
    existing_loop_dirs = [loop_dir for loop_dir in loop_dirs if loop_dir.exists()]
    if existing_loop_dirs and not overwrite:
        joined = "\n".join(f"  {loop_dir}" for loop_dir in existing_loop_dirs)
        raise FileExistsError(f"loop directory already exists:\n{joined}")

    config.output_root.mkdir(parents=True, exist_ok=True)

    written_loop_dirs: list[Path] = []
    for loop_index, (seed, loop_dir) in enumerate(zip(seeds, loop_dirs), start=1):
        if loop_dir.exists() and overwrite:
            shutil.rmtree(loop_dir)
        loop_dir.mkdir(parents=True, exist_ok=True)
        for phase in ("hot_hold", "cool_to_target", "target_hold", "sampling"):
            (loop_dir / phase).mkdir(exist_ok=True)

        (loop_dir / "in.loop.lmp").write_text(
            render_lammps_input(config, loop_index=loop_index, seed=seed),
            encoding="utf-8",
        )
        (loop_dir / "command.txt").write_text(
            f"{config.lammps_command} -in in.loop.lmp\n",
            encoding="utf-8",
        )
        (loop_dir / "manifest.json").write_text(
            json.dumps(_manifest(config, loop_index=loop_index, seed=seed), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        written_loop_dirs.append(loop_dir)

    run_script = config.output_root / "run_all.sh"
    run_script.write_text(_render_run_script(config, written_loop_dirs), encoding="utf-8")
    run_script.chmod(run_script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return WrittenProject(output_root=config.output_root, loop_dirs=written_loop_dirs, run_script=run_script)


def _render_run_script(config: GeneratorConfig, loop_dirs: Iterable[Path]) -> str:
    cmd = shlex.split(config.lammps_command)
    if not cmd:
        cmd = ["lmp"]
    command_text = " ".join(shlex.quote(part) for part in cmd)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f"LAMMPS_CMD={shlex.quote(command_text)}",
        "",
    ]
    for loop_dir in loop_dirs:
        quoted_dir = shlex.quote(str(loop_dir))
        lines.extend(
            [
                f"echo '[RUN] {loop_dir.name}'",
                f"(cd {quoted_dir} && eval \"$LAMMPS_CMD\" -in in.loop.lmp 2>&1 | tee runner.log)",
                "",
            ]
        )
    return "\n".join(lines)


def _parse_seeds(raw: str | None) -> list[int] | None:
    if raw is None or raw.strip() == "":
        return None
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate independent LAMMPS annealing loops from one restart file.",
    )
    parser.add_argument("--restart", required=True, type=Path, help="Selected input restart file.")
    parser.add_argument("--output-root", required=True, type=Path, help="Directory to write loop folders into.")
    parser.add_argument("--target-T", required=True, type=float, dest="target_t", help="Target control temperature.")
    parser.add_argument("--hot-T", required=True, type=float, dest="hot_t", help="High annealing temperature.")
    parser.add_argument("--loops", required=True, type=int, help="Number of independent loops.")
    parser.add_argument("--seeds", default=None, help="Comma-separated explicit seed list. Length must match --loops.")
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START, help="First generated seed.")
    parser.add_argument("--seed-step", type=int, default=DEFAULT_SEED_STEP, help="Generated seed increment.")
    parser.add_argument("--ts", type=float, default=DEFAULT_TS, help="LAMMPS timestep.")
    parser.add_argument("--Tdamp", type=float, default=DEFAULT_TDAMP, dest="tdamp", help="NVT damping.")
    parser.add_argument("--Tchain", type=int, default=DEFAULT_TCHAIN, dest="tchain", help="Sphere thermostat chain length.")
    parser.add_argument("--cool-dT", type=float, default=DEFAULT_COOL_DT, dest="cool_dt", help="Temperature grid for default cool step derivation.")
    parser.add_argument("--cool-steps-per-dT", type=int, default=DEFAULT_COOL_STEPS_PER_DT, dest="cool_steps_per_dt", help="Steps per cool-dT interval.")
    parser.add_argument("--cool-back-steps", type=int, default=None, help="Override cooling ramp steps.")
    parser.add_argument("--hot-hold-steps", type=int, default=DEFAULT_HOT_HOLD_STEPS, help="High-temperature hold steps.")
    parser.add_argument("--target-hold-steps", type=int, default=DEFAULT_TARGET_HOLD_STEPS, help="Target-temperature equilibration hold steps before sampling.")
    parser.add_argument("--sample-steps", type=int, default=DEFAULT_SAMPLE_STEPS, help="Target-temperature sampling steps.")
    parser.add_argument("--sample-every", type=int, default=DEFAULT_SAMPLE_EVERY, help="Thermo and scalar print frequency.")
    parser.add_argument("--dump-every", type=int, default=DEFAULT_DUMP_EVERY, help="Dump frequency.")
    parser.add_argument("--restart-dump-multiple", type=int, default=DEFAULT_RESTART_DUMP_MULTIPLE, help="Write restart every dump_every times this multiplier.")
    parser.add_argument("--head-id", default="auto", dest="head_id_override", help="Head atom id override or auto.")
    parser.add_argument("--tail-id", default="auto", dest="tail_id_override", help="Tail atom id override or auto.")
    parser.add_argument("--lammps-command", default="lmp", help="Command used in run_all.sh, for example 'mpirun -np 8 lmp'.")
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into existing loop directories.")
    parser.add_argument("--dry-run", action="store_true", help="Write dry_run variable into LAMMPS input for record keeping.")
    return parser


def config_from_args(args: argparse.Namespace) -> GeneratorConfig:
    return finalize_config(
        GeneratorConfig(
            restart=args.restart,
            output_root=args.output_root,
            target_t=args.target_t,
            hot_t=args.hot_t,
            loops=args.loops,
            explicit_seeds=_parse_seeds(args.seeds),
            seed_start=args.seed_start,
            seed_step=args.seed_step,
            ts=args.ts,
            tdamp=args.tdamp,
            tchain=args.tchain,
            cool_dt=args.cool_dt,
            cool_steps_per_dt=args.cool_steps_per_dt,
            cool_back_steps=args.cool_back_steps,
            hot_hold_steps=args.hot_hold_steps,
            target_hold_steps=args.target_hold_steps,
            sample_steps=args.sample_steps,
            sample_every=args.sample_every,
            dump_every=args.dump_every,
            restart_dump_multiple=args.restart_dump_multiple,
            head_id_override=args.head_id_override,
            tail_id_override=args.tail_id_override,
            lammps_command=args.lammps_command,
            dry_run=args.dry_run,
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = config_from_args(args)
    written = write_project(config, overwrite=args.overwrite)
    print(f"Wrote {len(written.loop_dirs)} loop directories under {written.output_root}")
    print(f"Run script: {written.run_script}")
    for loop_dir in written.loop_dirs:
        print(f"  {loop_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
