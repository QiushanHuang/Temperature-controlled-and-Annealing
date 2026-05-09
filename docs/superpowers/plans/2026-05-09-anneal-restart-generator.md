# Anneal Restart Generator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a tested Python generator that writes reproducible independent LAMMPS annealing loop inputs from one selected restart file.

**Architecture:** A single Python CLI owns parameter validation, cooling-step derivation, seed generation, LAMMPS template rendering, per-loop directory creation, manifests, and a `run_all.sh` helper. Tests exercise the pure functions and generated file content without running LAMMPS.

**Tech Stack:** Python standard library, `unittest`, generated LAMMPS input files.

---

### Task 1: Generator Behavior Tests

**Files:**
- Create: `/Users/joshua/Desktop/MD/Temperature-controlled-and-Annealing/tests/test_anneal_restart_generator.py`

- [ ] **Step 1: Write failing tests**

Create tests that import `anneal_restart_generator` and verify:

- default cooling steps follow `ceil((hot_T-target_T)/0.02) * 100000`;
- loop directory names use `loop{index}_{seed}`;
- explicit seeds override generated seeds;
- rendered LAMMPS does not include geometry rebuild commands;
- rendered LAMMPS includes the loop seed, Gaussian velocity creation, per-frame dump filenames, and restart cadence set to `dump_every * 500`.

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python3 -m unittest discover -s tests -v
```

Expected: import failure because `anneal_restart_generator.py` does not exist yet.

### Task 2: Python Generator

**Files:**
- Create: `/Users/joshua/Desktop/MD/Temperature-controlled-and-Annealing/anneal_restart_generator.py`

- [ ] **Step 1: Implement pure parameter helpers**

Implement `derive_cool_steps`, `build_seed_list`, `loop_dir_name`, and `validate_config`.

- [ ] **Step 2: Implement LAMMPS renderer**

Render a single loop input preserving the original force field, group definitions, computes, split thermostats, and dump fields while omitting geometry rebuild commands.

- [ ] **Step 3: Implement filesystem writer**

Create the output root, loop directories, phase subdirectories, `in.loop.lmp`, `manifest.json`, and `run_all.sh`.

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
python3 -m unittest discover -s tests -v
```

Expected: all tests pass.

### Task 3: Code Review

**Files:**
- Review: `/Users/joshua/Desktop/MD/Temperature-controlled-and-Annealing/anneal_restart_generator.py`
- Review: `/Users/joshua/Desktop/MD/Temperature-controlled-and-Annealing/tests/test_anneal_restart_generator.py`

- [ ] **Step 1: Dispatch code review subagent**

Ask a 5.5 xhigh review agent to inspect the generated LAMMPS semantics, output layout, validation, and tests.

- [ ] **Step 2: Apply necessary fixes**

Make only fixes that address concrete review findings.

- [ ] **Step 3: Re-run tests**

Run:

```bash
python3 -m unittest discover -s tests -v
```

Expected: all tests pass.

