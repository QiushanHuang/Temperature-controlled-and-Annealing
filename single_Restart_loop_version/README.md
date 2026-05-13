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

For server work, keep `run_lammps` false first, inspect `single_restart_suite_manifest.json`, then run the generated `Tstar_x.xx/run_all.sh` scripts or use tmux manually per temperature.
