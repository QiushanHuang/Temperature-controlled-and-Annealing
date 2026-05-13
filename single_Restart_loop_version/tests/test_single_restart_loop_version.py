import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from single_Restart_loop_version import run_single_restart_loops as single


class SingleRestartLoopVersionTests(unittest.TestCase):
    def test_generates_temperature_folders_from_one_restart_in_cwd(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            restart = root / "Restart.manual"
            restart.write_bytes(b"restart")
            params = root / "params.single.json"
            params.write_text(
                json.dumps(
                    {
                        "workspace": "cwd",
                        "input": {"restart_file": "auto", "restart_glob": "Restart*"},
                        "output": {"temperature_dir_template": "Tstar_{T:.2f}", "overwrite": False},
                        "simulation": {
                            "target_T_list": [1.04, 1.12],
                            "hot_T": 1.70,
                            "loops": 2,
                            "seeds": [111111, 222222],
                            "dump_every": 1000,
                            "hot_hold_dumps": 100,
                            "cool_dT": 0.1,
                            "cool_dumps_per_dT": 100,
                            "target_hold_dumps": 200,
                            "sample_dumps": 4000,
                            "restart_interval_dump_frames": 500,
                        },
                        "run": {"run_lammps": False, "lammps_command": "lmp_serial"},
                    }
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [sys.executable, str(Path(single.__file__).resolve()), str(params)],
                cwd=root,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            for target in ("1.04", "1.12"):
                tdir = root / f"Tstar_{target}"
                self.assertTrue((tdir / "loop1_111111" / "in.loop.lmp").is_file())
                self.assertTrue((tdir / "loop2_222222" / "in.loop.lmp").is_file())
                self.assertTrue((tdir / "loop1_111111" / "sampling").is_dir())
                text = (tdir / "loop1_111111" / "in.loop.lmp").read_text(encoding="utf-8")
                self.assertIn(f"variable        target_T index {float(target):.8g}", text)
                self.assertIn(str(restart.resolve()), text)

            manifest = json.loads((root / "single_restart_suite_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual([case["target_T"] for case in manifest["cases"]], [1.04, 1.12])
            self.assertEqual(manifest["restart"], str(restart.resolve()))
            self.assertIn("Tstar_1.04", manifest["cases"][0]["output_root"])

    def test_rejects_multiple_restart_files_in_experiment_folder(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "Restart.a").write_bytes(b"restart")
            (root / "Restart.b").write_bytes(b"restart")
            params = root / "params.single.json"
            params.write_text(
                json.dumps(
                    {
                        "workspace": "cwd",
                        "input": {"restart_file": "auto", "restart_glob": "Restart*"},
                        "output": {"overwrite": False},
                        "simulation": {"target_T_list": [1.04], "hot_T": 1.70, "loops": 1, "seeds": [111111]},
                        "run": {"run_lammps": False},
                    }
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [sys.executable, str(Path(single.__file__).resolve()), str(params)],
                cwd=root,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Multiple restart files", result.stderr + result.stdout)

    def test_preserves_no_space_workspace_symlink_for_restart_path(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            real_dir = base / "My Passport2"
            real_dir.mkdir()
            symlink_dir = base / "MyPassport2"
            symlink_dir.symlink_to(real_dir, target_is_directory=True)
            restart = symlink_dir / "Restart.manual"
            restart.write_bytes(b"restart")
            params = symlink_dir / "params.single.json"
            params.write_text(
                json.dumps(
                    {
                        "workspace": "cwd",
                        "input": {"restart_file": "Restart.manual"},
                        "output": {"temperature_dir_template": "Tstar_{T:.2f}", "overwrite": False},
                        "simulation": {"target_T_list": [1.04], "hot_T": 1.70, "loops": 1, "seeds": [111111]},
                        "run": {"run_lammps": False, "lammps_command": "lmp_serial"},
                    }
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [sys.executable, str(Path(single.__file__).resolve()), str(params)],
                cwd=symlink_dir,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            text = (symlink_dir / "Tstar_1.04" / "loop1_111111" / "in.loop.lmp").read_text(encoding="utf-8")
            self.assertIn("MyPassport2", text)
            self.assertNotIn("My Passport2", text)

    def test_tmux_launcher_generates_parallel_temperature_sequential_loop_plan(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            fake_tmux = bin_dir / "tmux"
            fake_tmux.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            fake_tmux.chmod(0o755)
            fake_lmp = bin_dir / "lmp"
            fake_lmp.write_text(
                "#!/usr/bin/env bash\n"
                "if [[ \"${1:-}\" == \"-h\" ]]; then\n"
                "  echo 'Installed packages: MOLECULE ASPHERE RIGID OPENMP'\n"
                "  exit 0\n"
                "fi\n"
                "exit 0\n",
                encoding="utf-8",
            )
            fake_lmp.chmod(0o755)
            restart = root / "Restart.manual"
            restart.write_bytes(b"restart")
            params = root / "params.single.json"
            params.write_text(
                json.dumps(
                    {
                        "workspace": "cwd",
                        "input": {"restart_file": "Restart.manual"},
                        "output": {"temperature_dir_template": "Tstar_{T:.2f}", "overwrite": False},
                        "simulation": {"target_T_list": [1.20, 1.10], "hot_T": 1.70, "loops": 2, "seeds": [111111, 222222]},
                        "run": {"run_lammps": False, "mpi_ranks": 1, "omp_threads": 1, "lammps_bin": str(fake_lmp)},
                    }
                ),
                encoding="utf-8",
            )

            script = Path(single.__file__).resolve().with_name("server_tmux_single_restart.sh")
            env = os.environ.copy()
            env.update(
                {
                    "PATH": f"{bin_dir}{os.pathsep}{env.get('PATH', '')}",
                    "PARAMS": str(params),
                    "MPI_RANKS": "4",
                    "OMP_THREADS": "9",
                    "LAMMPS_BIN": str(fake_lmp),
                    "CPU_TOTAL": "512",
                    "DRY_RUN_ONLY": "1",
                    "CLEAN_EXISTING": "1",
                }
            )
            result = subprocess.run(
                ["bash", str(script)],
                cwd=root,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            self.assertIn("Launch mode  : parallel temperatures; sequential loops inside each temperature", result.stdout)
            self.assertIn("Per temp      : np=4, omp=9, CPUs=36", result.stdout)
            self.assertIn("CPU request  : 72 / 512", result.stdout)
            self.assertIn("DRY_RUN_ONLY=1", result.stdout)
            command = (root / "Tstar_1.20" / "loop1_111111" / "command.txt").read_text(encoding="utf-8")
            self.assertIn("OMP_NUM_THREADS=9 mpiexec -np 4", command)
            self.assertIn("-sf omp -pk omp 9", command)
            run_all = (root / "Tstar_1.20" / "run_all.sh").read_text(encoding="utf-8")
            self.assertLess(run_all.index("loop1_111111"), run_all.index("loop2_222222"))
            manifest = json.loads((root / "single_restart_suite_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual([case["target_T"] for case in manifest["cases"]], [1.2, 1.1])


if __name__ == "__main__":
    unittest.main()
