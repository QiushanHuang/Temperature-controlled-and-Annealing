# single_Restart_loop_version

This workflow is for one manually selected restart in the experiment folder.

## What It Does

Run from the experiment folder that contains exactly one top-level `Restart*` file. The script uses that same restart for every target temperature, then creates:

```text
Tstar_1.04/
  loop1_111111/
  loop2_222222/
  ...
Tstar_1.12/
  loop1_111111/
  loop2_222222/
  ...
single_restart_suite_manifest.json
```

The per-loop LAMMPS protocol is the same as the existing annealing generator:

1. read the same selected restart
2. assign Gaussian velocities at `hot_T`
3. hold at `hot_T`
4. cool to the target temperature
5. hold at target
6. sample at target

## How To Use

Put the restart in the experiment folder:

```text
/path/to/my_experiment/Restart.manual
```

Edit `target_T_list` in:

```text
/path/to/code/single_Restart_loop_version/params.single_restart_loop.json
```

Generate the temperature folders:

```bash
cd /path/to/my_experiment
python3 /path/to/code/single_Restart_loop_version/run_single_restart_loops.py \
  /path/to/code/single_Restart_loop_version/params.single_restart_loop.json
```

To launch LAMMPS sequentially after generation, set:

```json
"run_lammps": true
```

For server work, keep `run_lammps` false and use the tmux launcher:

```bash
cd /path/to/my_experiment
PARAMS=params.single_restart_loop.json \
CPU_TOTAL=512 \
CLEAN_EXISTING=1 \
/path/to/code/single_Restart_loop_version/server_tmux_single_restart.sh
```

The tmux launcher starts one tmux session per target temperature, so different
`Tstar_x.xx` folders run at the same time. Inside each temperature, the generated
`run_all.sh` still runs `loop1_seed`, `loop2_seed`, ... sequentially.

Default server CPU settings are `np4omp9`:

```json
"mpi_ranks": 4,
"omp_threads": 9,
"lammps_args": ["-sf", "omp", "-pk", "omp", "9"]
```

Set `DRY_RUN_ONLY=1` to generate and inspect the plan without launching tmux.

`params.RE3000_np4omp9.json` is a ready-to-copy template for:

- restart file: `Restart.Cooling_heating_L7_from_RE3000`
- target temperatures: `1.20 1.10 1.00 0.90 0.80 0.70 0.60 0.50 0.40`
- CPU layout: `np4omp9`
