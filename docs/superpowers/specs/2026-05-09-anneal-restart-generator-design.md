# Anneal Restart Generator Design

## Goal

Build a reproducible LAMMPS input generator for independent annealing branches from one selected target-temperature restart file. Each loop starts from the same restart, uses a unique random seed, heats by reassigning Gaussian velocities at a specified high temperature, relaxes at that high temperature, cools back to the target temperature at a comparable rate, and then runs target-temperature production.

## Scientific Scope

The workflow tests whether a target-temperature structure can enter different Rg basins after a controlled thermal perturbation. It does not by itself prove an equilibrium target-temperature distribution; the target production window and independent loop statistics still determine the strength of that claim.

Each loop is an independent branch:

- Read the same user-selected restart file.
- Reset timestep to zero inside that loop.
- Assign velocities from a Gaussian distribution at `hot_T` using that loop's seed.
- Relax at `hot_T`.
- Cool from `hot_T` to `target_T`.
- Continue target-temperature production.

Loops are not chained. The final restart from `loop1` is not used as the input to `loop2`.

## Required LAMMPS Semantics

The generated inputs preserve the original force field and thermostat split:

- Type 1 atoms are ellipsoid particles.
- Type 2 atoms are sphere particles.
- Type 3 atoms are anchors.
- `rigid_lc` is type 1 plus type 3 and is integrated with `fix rigid/nvt/small molecule`.
- `sphere_thermo` is type 2 and is integrated with `fix nvt`.
- `fix_modify thermostat_sph temp T_sph` is retained.

The generated inputs do not repeat the original geometry rebuild by default. They do not change box dimensions, rotate the chain, recenter the structure, or reset image flags. This keeps the selected target-temperature restart's box and periodic image state intact.

The Gaussian velocity assignment happens before the `rigid/nvt/small` high-temperature integration fix is active. For rigid bodies, the exact initial reported temperature may not equal `hot_T` immediately; `hot_hold_steps` is therefore part of the thermalization protocol, not optional bookkeeping.

## Output Layout

For loop index `1` with seed `123456`, the default directory name is:

```text
loop1_123456
```

Each loop contains:

```text
loop1_123456/
  in.loop.lmp
  manifest.json
  log.loop.lammps
  loop_state.dat
  hot_relax/
  cool_to_target/
  target_production/
```

Dump output keeps the existing one-file-per-dump-frame pattern:

```text
traj.hot.*.dump
traj.cool.*.dump
traj.target.*.dump
```

Restart output is written every `restart_dump_multiple * dump_every` steps. The default multiplier is `500`, so with `dump_every = 1000`, restart output is every `500000` steps. Stage-boundary restart files are also written for reproducibility.

## Default Parameters

The user specifies `target_T` and `hot_T`; no high-temperature scan is generated automatically.

Default cooling rate is comparable to the original cooldown scripts:

```text
cool_steps = ceil((hot_T - target_T) / cool_dT) * cool_steps_per_dT
cool_dT = 0.02
cool_steps_per_dT = 100000
```

The generator allows overriding `cool_steps` directly.

Default production-oriented values:

- `timestep = 0.001`
- `Tdamp = 0.1`
- `Tchain = 3`
- `sample_every = 1000`
- `dump_every = 1000`
- `restart_dump_multiple = 500`
- `hot_hold_steps = 1000000`
- `target_hold_steps = 3900000`

## Reproducibility

Each loop records the seed in:

- The loop directory name.
- `manifest.json`.
- LAMMPS variables and printed setup lines inside `in.loop.lmp`.

The generator can use either an explicit seed list or a deterministic seed sequence from `seed_start` and `seed_step`.

LAMMPS command parsing can split unquoted paths containing spaces. This generator rejects restart and output-root paths containing whitespace instead of trying to quote every generated LAMMPS path.
