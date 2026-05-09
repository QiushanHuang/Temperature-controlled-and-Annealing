import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import anneal_restart_generator as gen
import run_anneal
from scripts import create_heating_cooling_suite as suite


class AnnealRestartGeneratorTests(unittest.TestCase):
    def test_derive_cool_steps_uses_requested_cooling_rate(self):
        self.assertEqual(
            gen.derive_cool_steps(target_t=1.40, hot_t=1.52, cool_dt=0.02, steps_per_dt=100000),
            600000,
        )
        self.assertEqual(
            gen.derive_cool_steps(target_t=1.40, hot_t=1.41, cool_dt=0.02, steps_per_dt=100000),
            50000,
        )

    def test_build_seed_list_accepts_explicit_seeds(self):
        self.assertEqual(
            gen.build_seed_list(loops=3, explicit_seeds=[11, 22, 33], seed_start=1000, seed_step=7),
            [11, 22, 33],
        )

    def test_build_seed_list_generates_reproducible_sequence(self):
        self.assertEqual(
            gen.build_seed_list(loops=4, explicit_seeds=None, seed_start=1001, seed_step=17),
            [1001, 1018, 1035, 1052],
        )

    def test_finalize_config_rejects_nonfinite_or_nonpositive_temperatures(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            restart = root / "Restart.input"
            restart.write_bytes(b"restart")
            bad_configs = [
                gen.GeneratorConfig(restart=restart, output_root=root / "out1", target_t=0.0, hot_t=1.5, loops=1),
                gen.GeneratorConfig(restart=restart, output_root=root / "out2", target_t=1.0, hot_t=float("nan"), loops=1),
                gen.GeneratorConfig(restart=restart, output_root=root / "out3", target_t=float("inf"), hot_t=2.0, loops=1),
            ]

            for cfg in bad_configs:
                with self.subTest(cfg=cfg):
                    with self.assertRaises(ValueError):
                        gen.finalize_config(cfg)

    def test_finalize_config_rejects_paths_with_whitespace(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            restart_dir = root / "with space"
            restart_dir.mkdir()
            restart = restart_dir / "Restart.input"
            restart.write_bytes(b"restart")

            with self.assertRaisesRegex(ValueError, "whitespace"):
                gen.finalize_config(
                    gen.GeneratorConfig(
                        restart=restart,
                        output_root=root / "out",
                        target_t=1.40,
                        hot_t=1.52,
                        loops=1,
                    )
                )

    def test_numeric_head_tail_overrides_render_without_lammps_string_comparison(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            restart = root / "input.restart"
            restart.write_bytes(b"restart")
            cfg = gen.finalize_config(
                gen.GeneratorConfig(
                    restart=restart,
                    output_root=root / "out",
                    target_t=1.40,
                    hot_t=1.52,
                    loops=1,
                    explicit_seeds=[123456],
                    head_id_override="10",
                    tail_id_override="999",
                )
            )

            text = gen.render_lammps_input(cfg, loop_index=1, seed=123456)

        self.assertIn("variable        head_id index 10", text)
        self.assertIn("variable        tail_id index 999", text)
        self.assertNotIn("== auto", text)

    def test_invalid_head_tail_override_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            restart = root / "Restart.input"
            restart.write_bytes(b"restart")

            with self.assertRaisesRegex(ValueError, "head_id_override"):
                gen.finalize_config(
                    gen.GeneratorConfig(
                        restart=restart,
                        output_root=root / "out",
                        target_t=1.40,
                        hot_t=1.52,
                        loops=1,
                        head_id_override="first",
                    )
                )

    def test_loop_dir_name_places_seed_after_loop_index(self):
        self.assertEqual(gen.loop_dir_name(loop_index=1, seed=987654), "loop1_987654")

    def test_rendered_lammps_keeps_per_frame_dump_and_omits_geometry_rebuild(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            restart = root / "input.restart"
            restart.write_bytes(b"restart")
            cfg = gen.GeneratorConfig(
                restart=restart,
                output_root=root / "anneal_out",
                target_t=1.40,
                hot_t=1.52,
                loops=1,
                explicit_seeds=[123456],
            )
            cfg = gen.finalize_config(cfg)
            text = gen.render_lammps_input(cfg, loop_index=1, seed=123456)

        self.assertIn("variable        loop_seed index 123456", text)
        self.assertIn("velocity        all create ${hot_T} ${loop_seed} mom yes rot yes dist gaussian", text)
        self.assertIn("variable        restart_every index 500000", text)
        self.assertIn("restart         ${restart_every} ${loop_dir}/hot_hold/Restart.hot_hold.*", text)
        self.assertIn("dump            traj_hot_hold all custom ${dump_every} ${loop_dir}/hot_hold/traj.hot_hold.*.dump", text)
        self.assertIn("dump            traj_sample all custom ${dump_every} ${loop_dir}/sampling/traj.sample.*.dump", text)
        self.assertNotIn("dump_modify     traj_hot_hold first yes", text)
        self.assertIsNone(re.search(r'then "print [^"]+" then "quit 2"', text))
        self.assertIn("if \"$(count(ellipsoid)) <= 0\" then \"print 'ERROR: ellipsoid group is empty'\"", text)
        self.assertIn("if \"$(count(ellipsoid)) <= 0\" then \"quit 2\"", text)
        self.assertIn("if \"$(count(head)) <= 0\" then \"print 'ERROR: head group is empty'\"", text)
        self.assertIn("if \"$(count(tail)) <= 0\" then \"quit 2\"", text)
        self.assertNotIn("change_box", text)
        self.assertNotIn("displace_atoms", text)
        self.assertNotIn("set             group all image 0 0 0", text)

    def test_write_project_creates_loop_manifest_and_run_script(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            restart = root / "Restart.input"
            restart.write_bytes(b"restart")
            out = root / "out"
            cfg = gen.GeneratorConfig(
                restart=restart,
                output_root=out,
                target_t=1.40,
                hot_t=1.52,
                loops=2,
                explicit_seeds=[111111, 222222],
            )
            cfg = gen.finalize_config(cfg)

            written = gen.write_project(cfg)

            self.assertEqual(len(written.loop_dirs), 2)
            first_loop = out / "loop1_111111"
            second_loop = out / "loop2_222222"
            self.assertTrue((first_loop / "in.loop.lmp").is_file())
            self.assertTrue((first_loop / "hot_hold").is_dir())
            self.assertTrue((first_loop / "cool_to_target").is_dir())
            self.assertTrue((first_loop / "target_hold").is_dir())
            self.assertTrue((first_loop / "sampling").is_dir())
            self.assertTrue((second_loop / "in.loop.lmp").is_file())
            self.assertTrue((second_loop / "manifest.json").is_file())
            self.assertTrue((out / "run_all.sh").is_file())

            manifest = json.loads((first_loop / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["seed"], 111111)
            self.assertEqual(manifest["loop_dir_name"], "loop1_111111")
            self.assertEqual(manifest["restart_every"], 500000)
            self.assertEqual(manifest["cool_back_steps"], 120000)
            second_text = (second_loop / "in.loop.lmp").read_text(encoding="utf-8")
            self.assertIn("variable        loop_index index 2", second_text)
            self.assertIn("variable        loop_seed index 222222", second_text)
            run_script = (out / "run_all.sh").read_text(encoding="utf-8")
            self.assertIn("loop1_111111", run_script)
            self.assertIn("loop2_222222", run_script)

    def test_overwrite_removes_old_loop_outputs_before_rewriting(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            restart = root / "Restart.input"
            restart.write_bytes(b"restart")
            out = root / "out"
            cfg = gen.finalize_config(
                gen.GeneratorConfig(
                    restart=restart,
                    output_root=out,
                    target_t=1.40,
                    hot_t=1.52,
                    loops=1,
                    explicit_seeds=[111111],
                )
            )
            gen.write_project(cfg)
            stale_dump = out / "loop1_111111" / "sampling" / "traj.sample.1000.dump"
            stale_dump.write_text("old frame\n", encoding="utf-8")

            gen.write_project(cfg, overwrite=True)

            self.assertFalse(stale_dump.exists())
            self.assertTrue((out / "loop1_111111" / "in.loop.lmp").is_file())

    def test_load_params_resolves_restart_relative_to_params_file(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            restart = root / "Restart.cooldown.4000000"
            restart.write_bytes(b"restart")
            params = root / "params.json"
            params.write_text(
                json.dumps(
                    {
                        "input": {"restart_file": "Restart.cooldown.4000000"},
                        "output": {"output_root": "anneal_T1.40_hot1.52"},
                        "simulation": {
                            "target_T": 1.40,
                            "hot_T": 1.52,
                            "loops": 2,
                            "seeds": [111111, 222222],
                        },
                        "run": {"lammps_command": "lmp_serial", "run_lammps": False},
                    }
                ),
                encoding="utf-8",
            )

            cfg, run_cfg = run_anneal.load_project_config(params)

            self.assertEqual(cfg.restart, restart.resolve())
            self.assertEqual(cfg.output_root, (root / "anneal_T1.40_hot1.52").resolve())
            self.assertEqual(cfg.explicit_seeds, [111111, 222222])
            self.assertEqual(cfg.cool_back_steps, 120000)
            self.assertFalse(run_cfg.run_lammps)

    def test_load_params_auto_discovers_single_restart_in_project_folder(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            restart = root / "Restart.L7-Tstar_1.40.onlyE"
            restart.write_bytes(b"restart")
            params = root / "params.json"
            params.write_text(
                json.dumps(
                    {
                        "input": {"restart_file": "auto", "restart_glob": "Restart*"},
                        "output": {"output_root": "anneal_out"},
                        "simulation": {"target_T": 1.40, "hot_T": 1.42, "loops": 1},
                    }
                ),
                encoding="utf-8",
            )

            cfg, _ = run_anneal.load_project_config(params)

            self.assertEqual(cfg.restart, restart.resolve())

    def test_run_anneal_without_dash_args_uses_params_json(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            restart = root / "Restart.input"
            restart.write_bytes(b"restart")
            params = root / "params.json"
            params.write_text(
                json.dumps(
                    {
                        "input": {"restart_file": "Restart.input"},
                        "output": {"output_root": "anneal_out"},
                        "simulation": {
                            "target_T": 1.40,
                            "hot_T": 1.42,
                            "loops": 1,
                            "seeds": [123456],
                            "hot_hold_steps": 10,
                            "target_hold_steps": 20,
                        },
                        "run": {"run_lammps": False},
                    }
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [sys.executable, str(Path(run_anneal.__file__).resolve())],
                cwd=root,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((root / "anneal_out" / "loop1_123456" / "in.loop.lmp").is_file())
            self.assertIn("Wrote 1 loop directories", result.stdout)
            self.assertIn("target_T=1.4", result.stdout)
            self.assertIn("cool_back_steps=20000", result.stdout)

    def test_params_can_use_current_workdir_as_workspace(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config_dir = root / "configs"
            work_dir = root / "case_L7"
            config_dir.mkdir()
            work_dir.mkdir()
            restart = work_dir / "Restart.input"
            restart.write_bytes(b"restart")
            params = config_dir / "params_L7.json"
            params.write_text(
                json.dumps(
                    {
                        "workspace": "cwd",
                        "input": {"restart_file": "Restart.input"},
                        "output": {"output_root": "anneal_out"},
                        "simulation": {
                            "target_T": 1.40,
                            "hot_T": 1.42,
                            "loops": 1,
                            "seeds": [123456],
                        },
                        "run": {"run_lammps": False},
                    }
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [sys.executable, str(Path(run_anneal.__file__).resolve()), str(params)],
                cwd=work_dir,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((work_dir / "anneal_out" / "loop1_123456" / "in.loop.lmp").is_file())
            self.assertFalse((config_dir / "anneal_out").exists())

    def test_mpi_run_config_builds_lammps_command_and_result_dir(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            restart = root / "Restart.input"
            restart.write_bytes(b"restart")
            params = root / "params.json"
            params.write_text(
                json.dumps(
                    {
                        "input": {"restart_file": "Restart.input"},
                        "output": {"output_root": "anneal_inputs"},
                        "simulation": {
                            "target_T": 1.40,
                            "hot_T": 1.42,
                            "loops": 1,
                            "seeds": [123456],
                        },
                        "run": {
                            "run_lammps": False,
                            "mpi_ranks": 4,
                            "omp_threads": 2,
                            "mpiexec": "mpiexec",
                            "lammps_bin": "lmp_mpi",
                            "result_dir": "Result_MD_anneal",
                        },
                    }
                ),
                encoding="utf-8",
            )

            cfg, run_cfg = run_anneal.load_project_config(params)
            written = gen.write_project(cfg)

            self.assertEqual(cfg.output_root, (root / "Result_MD_anneal").resolve())
            self.assertEqual(cfg.lammps_command, "OMP_NUM_THREADS=2 mpiexec -np 4 lmp_mpi")
            run_script = (written.output_root / "run_all.sh").read_text(encoding="utf-8")
            self.assertIn("OMP_NUM_THREADS=2 mpiexec -np 4 lmp_mpi", run_script)
            self.assertFalse(run_cfg.run_lammps)

    def test_run_anneal_accepts_multiple_param_files_sequentially(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            restart = root / "Restart.input"
            restart.write_bytes(b"restart")
            first = root / "params_hot142.json"
            second = root / "params_hot152.json"
            common = {
                "workspace": "cwd",
                "input": {"restart_file": "Restart.input"},
                "simulation": {"target_T": 1.40, "loops": 1, "seeds": [123456]},
                "run": {"run_lammps": False},
            }
            first_data = dict(common)
            first_data["output"] = {"output_root": "anneal_hot142"}
            first_data["simulation"] = dict(common["simulation"], hot_T=1.42)
            second_data = dict(common)
            second_data["output"] = {"output_root": "anneal_hot152"}
            second_data["simulation"] = dict(common["simulation"], hot_T=1.52)
            first.write_text(json.dumps(first_data), encoding="utf-8")
            second.write_text(json.dumps(second_data), encoding="utf-8")

            result = subprocess.run(
                [sys.executable, str(Path(run_anneal.__file__).resolve()), str(first), str(second)],
                cwd=root,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((root / "anneal_hot142" / "loop1_123456" / "in.loop.lmp").is_file())
            self.assertTrue((root / "anneal_hot152" / "loop1_123456" / "in.loop.lmp").is_file())

    def test_dump_count_protocol_renders_separate_target_hold_and_sampling_stages(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            restart = root / "Restart.input"
            restart.write_bytes(b"restart")
            params = root / "params.json"
            params.write_text(
                json.dumps(
                    {
                        "input": {"restart_file": "Restart.input"},
                        "output": {"output_root": "anneal_out"},
                        "simulation": {
                            "target_T": 1.40,
                            "hot_T": 1.70,
                            "loops": 1,
                            "seeds": [123456],
                            "dump_every": 1000,
                            "hot_hold_dumps": 100,
                            "cool_dT": 0.1,
                            "cool_dumps_per_dT": 100,
                            "target_hold_dumps": 200,
                            "sample_dumps": 4000,
                            "restart_interval_dump_frames": 500,
                        },
                    }
                ),
                encoding="utf-8",
            )

            cfg, _ = run_anneal.load_project_config(params)
            text = gen.render_lammps_input(cfg, loop_index=1, seed=123456)

            self.assertEqual(cfg.hot_hold_steps, 100000)
            self.assertEqual(cfg.cool_steps_per_dt, 100000)
            self.assertEqual(cfg.cool_back_steps, 300000)
            self.assertEqual(cfg.target_hold_steps, 200000)
            self.assertEqual(cfg.sample_steps, 4000000)
            self.assertIn("variable        target_hold_steps index 200000", text)
            self.assertIn("variable        sample_steps index 4000000", text)
            self.assertIn("${loop_dir}/target_hold/traj.target_hold.*.dump", text)
            self.assertIn("${loop_dir}/sampling/traj.sample.*.dump", text)
            self.assertIn("write_restart   ${loop_dir}/sampling/Final.sample_T${final_T_tag}.bin", text)

    def test_dump_count_fields_are_converted_to_steps(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            restart = root / "Restart.input"
            restart.write_bytes(b"restart")
            params = root / "params.json"
            params.write_text(
                json.dumps(
                    {
                        "input": {"restart_file": "Restart.input"},
                        "output": {"output_root": "anneal_out"},
                        "simulation": {
                            "target_T": 1.45,
                            "hot_T": 1.65,
                            "loops": 1,
                            "seeds": [123456],
                            "dump_every": 250,
                            "hot_hold_dumps": 7,
                            "cool_dT": 0.05,
                            "cool_dumps_per_dT": 8,
                            "target_hold_dumps": 9,
                            "sample_dumps": 11,
                            "restart_interval_dump_frames": 13,
                        },
                    }
                ),
                encoding="utf-8",
            )

            cfg, _ = run_anneal.load_project_config(params)

            self.assertEqual(cfg.hot_hold_steps, 1750)
            self.assertEqual(cfg.cool_steps_per_dt, 2000)
            self.assertEqual(cfg.cool_back_steps, 8000)
            self.assertEqual(cfg.target_hold_steps, 2250)
            self.assertEqual(cfg.sample_steps, 2750)
            self.assertEqual(cfg.restart_every, 3250)

    def test_heating_cooling_suite_writes_params_for_l3_l7_tstar_dirs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for length, tstar, step in (
                ("L3", "1.40", "4000000"),
                ("L7", "1.36", "12000000"),
            ):
                tdir = root / length / f"Tstar_{tstar}"
                tdir.mkdir(parents=True)
                (tdir / f"Restart.cooldown.{step}").write_bytes(b"restart")

            manifest_path = suite.create_suite(
                root=root,
                suite_dir=root / "anneal_suite",
                lengths=("L3", "L7"),
                loops=7,
                seeds=(101, 102, 103, 104, 105, 106, 107),
                hot_t=1.7,
                run_lammps=False,
                mpi_ranks=4,
                overwrite=True,
            )

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["cases"]), 2)
            first_params = Path(manifest["cases"][0]["params_path"])
            data = json.loads(first_params.read_text(encoding="utf-8"))
            self.assertEqual(data["simulation"]["hot_T"], 1.7)
            self.assertEqual(data["simulation"]["loops"], 7)
            self.assertEqual(data["simulation"]["hot_hold_dumps"], 100)
            self.assertEqual(data["simulation"]["cool_dT"], 0.1)
            self.assertEqual(data["simulation"]["cool_dumps_per_dT"], 100)
            self.assertEqual(data["simulation"]["target_hold_dumps"], 200)
            self.assertEqual(data["simulation"]["sample_dumps"], 4000)
            self.assertEqual(data["run"]["mpi_ranks"], 4)
            self.assertEqual(data["run"]["omp_threads"], 2)
            self.assertEqual(data["run"]["lammps_bin"], "/home/star/Research/software/lammps-22Jul2025/build/lmp")
            self.assertTrue(data["run"]["result_dir"].endswith("np4omp2_loops7"))

    def test_run_anneal_launches_fake_lammps_for_multiple_loops(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            restart = root / "Restart.input"
            restart.write_bytes(b"restart")
            sequence = root / "sequence.txt"
            fake_lammps = root / "fake_lammps.sh"
            fake_lammps.write_text(
                "\n".join(
                    [
                        "#!/usr/bin/env bash",
                        "set -euo pipefail",
                        f"printf '%s\\n' \"$PWD\" >> {str(sequence)!r}",
                        "printf '%s\\n' \"$*\" > ran.marker",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            fake_lammps.chmod(0o755)
            params = root / "params.json"
            params.write_text(
                json.dumps(
                    {
                        "input": {"restart_file": "Restart.input"},
                        "output": {"output_root": "anneal_out"},
                        "simulation": {
                            "target_T": 1.40,
                            "hot_T": 1.70,
                            "loops": 3,
                            "seeds": [101, 202, 303],
                            "cool_back_steps": 1,
                            "hot_hold_steps": 1,
                            "target_hold_steps": 1,
                            "sample_steps": 1,
                        },
                        "run": {"run_lammps": True, "lammps_command": str(fake_lammps)},
                    }
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [sys.executable, str(Path(run_anneal.__file__).resolve())],
                cwd=root,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            expected_loops = ["loop1_101", "loop2_202", "loop3_303"]
            for loop_name in expected_loops:
                marker = root / "anneal_out" / loop_name / "ran.marker"
                self.assertTrue(marker.is_file(), loop_name)
                self.assertEqual(marker.read_text(encoding="utf-8").strip(), "-in in.loop.lmp")
            self.assertEqual(
                [Path(line).name for line in sequence.read_text(encoding="utf-8").splitlines()],
                expected_loops,
            )


if __name__ == "__main__":
    unittest.main()
