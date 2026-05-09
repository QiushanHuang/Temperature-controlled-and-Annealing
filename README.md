# Temperature-Controlled Annealing Restart Branches

This package generates independent LAMMPS annealing branches from the same selected restart file. The normal workflow is parameter-file driven; you do not need long `python --restart ... --hot-T ...` command lines.

## Single Case

Put one restart file in the case folder, edit `params.json`, then run:

```bash
cd /path/to/Tstar_1.40
python3 /Users/joshua/Desktop/MD/Temperature-controlled-and-Annealing/run_anneal.py \
  /Users/joshua/Desktop/MD/Temperature-controlled-and-Annealing/params.json
```

With `"workspace": "cwd"`, relative restart and output paths resolve against the directory you `cd` into. If `"restart_file": "auto"`, exactly one top-level `Restart*` file must be present.

## Current Protocol

Each loop starts from the same restart and uses its own seed:

1. `read_restart` from the selected restart.
2. Assign Gaussian velocities at `hot_T = 1.70`.
3. `hot_hold`: hold at `1.70` for `100` dumps = `100000` steps.
4. `cool_to_target`: cool linearly to `target_T` at `0.1` per `100` dumps.
5. `target_hold`: hold at `target_T` for `200` dumps = `200000` steps.
6. `sampling`: output `4000` sampling dumps = `4000000` steps.
7. Write sampling restart files every `500` dumps = `500000` steps.

Default cadence: `dump_every = 1000`, `sample_every = 1000`, `timestep = 0.001`, `Tdamp = 0.1`, `Tchain = 3`.

Phase folders and dump names are separated:

```text
loop1_111111/
  hot_hold/traj.hot_hold.*.dump
  cool_to_target/traj.cool.*.dump
  target_hold/traj.target_hold.*.dump
  sampling/traj.sample.*.dump
```

`dump_modify first yes` is not used, so a 100000-step phase at `dump_every=1000` produces 100 scheduled dump frames rather than an extra starting frame.

## Batch Suite For `/Volumes/TRACER/heating_cooling`

Generate params for every `L3/Tstar_*` and `L7/Tstar_*` folder:

```bash
python3 /Users/joshua/Desktop/MD/Temperature-controlled-and-Annealing/scripts/create_heating_cooling_suite.py \
  --root /Volumes/TRACER/heating_cooling \
  --suite-dir /Volumes/TRACER/heating_cooling/anneal_hot1.70_np4omp2_loops7_suite \
  --overwrite
```

To keep large dump/restart outputs on a separate disk, pass a no-space output root:

```bash
python3 /Users/joshua/Desktop/MD/Temperature-controlled-and-Annealing/scripts/create_heating_cooling_suite.py \
  --root /home/star/Research/QIUSHAN-HUANG/cooling-loop/heating_cooling \
  --suite-dir /home/star/Research/QIUSHAN-HUANG/cooling-loop/heating_cooling/anneal_hot1.70_np4omp2_loops7_suite \
  --output-root /media/star/MyPassport2/cooling-loop-output \
  --overwrite
```

If the disk is mounted as `/media/star/My Passport2`, create a no-space symlink first:

```bash
ln -s "/media/star/My Passport2" /media/star/MyPassport2
mkdir -p /media/star/MyPassport2/cooling-loop-output
```

This writes one params file per Tstar case under:

```text
/Volumes/TRACER/heating_cooling/anneal_hot1.70_np4omp2_loops7_suite/params/
```

Each generated params file has `loops = 7` and seeds:

```text
111111 222222 333333 444444 555555 666666 777777
```

Outputs are created inside each Tstar folder as:

```text
Tstar_x.xx/anneal_hot1.70_np4omp2_loops7/loopN_seed/
```

With `--output-root /media/star/MyPassport2/cooling-loop-output`, outputs are created as:

```text
/media/star/MyPassport2/cooling-loop-output/L3/Tstar_1.04/anneal_hot1.70_np4omp2_loops7/loopN_seed/
```

## Server tmux Launch

The tmux launcher follows the same pattern as `polymer-network-aEa-project`: it builds a manifest, creates `tmux/runners`, `tmux/logs`, `tmux/status`, checks CPU use, and launches one tmux session per L3/L7 Tstar case.

```bash
cd /Users/joshua/Desktop/MD/Temperature-controlled-and-Annealing
DRY_RUN_ONLY=1 ./scripts/server_tmux_heating_cooling.sh
```

Server example with the data in `/home/star/Research/QIUSHAN-HUANG/cooling-loop/heating_cooling` and outputs on the external disk symlink:

```bash
cd /home/star/Research/QIUSHAN-HUANG/cooling-loop/Temperature-controlled-and-Annealing

RUN_ROOT=/home/star/Research/QIUSHAN-HUANG/cooling-loop/heating_cooling \
OUTPUT_ROOT=/media/star/MyPassport2/cooling-loop-output \
CPU_TOTAL=512 \
DRY_RUN_ONLY=1 \
./scripts/server_tmux_heating_cooling.sh
```

After checking the manifest, launch real runs:

```bash
RUN_ROOT=/home/star/Research/QIUSHAN-HUANG/cooling-loop/heating_cooling \
OUTPUT_ROOT=/media/star/MyPassport2/cooling-loop-output \
CPU_TOTAL=512 \
OVERWRITE_OUTPUTS=1 \
./scripts/server_tmux_heating_cooling.sh
```

For the current `/Volumes/TRACER/heating_cooling` tree, L3+L7 contains 57 Tstar cases. At `np4omp2`, that is `57 * 4 * 2 = 456` CPUs if every case is launched at once. The script keeps a CPU guard, so leave `CPU_TOTAL` at your real allocation value.

Useful environment overrides:

```bash
MPI_RANKS=4
OMP_THREADS=2
MPIEXEC=mpiexec
LAMMPS_BIN=/home/star/Research/software/lammps-22Jul2025/build/lmp
CPU_TOTAL=512
LENGTHS="L3 L7"
SEEDS="111111 222222 333333 444444 555555 666666 777777"
```

The LAMMPS binary must include at least `MOLECULE`, `ASPHERE`, and `RIGID`. The tmux launcher checks this before launching cases.

## CPU Settings

The `run` section controls LAMMPS execution:

```json
"run": {
  "run_lammps": false,
  "mpi_ranks": 4,
  "omp_threads": 2,
  "mpiexec": "mpiexec",
  "lammps_bin": "/home/star/Research/software/lammps-22Jul2025/build/lmp",
  "result_dir": "anneal_hot1.70_np4omp2_loops7"
}
```

This generates:

```bash
OMP_NUM_THREADS=2 mpiexec -np 4 /home/star/Research/software/lammps-22Jul2025/build/lmp -in in.loop.lmp
```
