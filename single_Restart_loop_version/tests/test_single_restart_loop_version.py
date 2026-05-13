import json
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


if __name__ == "__main__":
    unittest.main()
