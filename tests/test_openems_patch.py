import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.solvers.openems_patch import PatchConfig, load_config, run_candidate
from src.verification.metrics import matching_metrics, refinement_metrics, solver_convergence


class ConfigTests(unittest.TestCase):
    def test_rejects_invalid_geometry_and_settings(self):
        for changes in (
            {"patch_x_mm": 60}, {"feed_x_mm": 16}, {"board_y_mm": 200},
            {"relative_permittivity": float("nan")}, {"threads": True},
            {"substrate_cells": 4.5}, {"end_criteria": 1},
            {"loss_tangent_at_reference": -0.1}, {"band_start_hz": 0.9e9},
            {"band_stop_hz": 2.3e9}, {"max_timesteps": 200001},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                PatchConfig(**changes)

    def test_requires_exact_config_fields(self):
        baseline = Path(__file__).resolve().parents[1] / "configs/openems_patch.json"
        self.assertEqual(load_config(baseline).patch_x_mm, 32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            data = json.loads(baseline.read_text())
            data["patch_width"] = data.pop("patch_x_mm")
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                load_config(path)


class MetricsTests(unittest.TestCase):
    frequency = np.array([2.3e9, 2.4e9, 2.45e9, 2.5e9, 2.6e9])

    def response(self, db):
        return {"frequency_hz": self.frequency, "s11": 10 ** (np.array(db) / 20)}

    def test_checks_worst_value_including_band_endpoints(self):
        good = self.response([-1, -10, -25, -10, -1])
        metrics = matching_metrics(**good, band_start_hz=2.4e9, band_stop_hz=2.5e9, limit_db=-10)
        self.assertTrue(metrics["sampled_band_matching_pass"])
        self.assertEqual(metrics["resonance_candidate_hz"], 2.45e9)
        bad = self.response([-1, -10, -25, -9.9, -1])
        self.assertFalse(matching_metrics(
            **bad, band_start_hz=2.4e9, band_stop_hz=2.5e9, limit_db=-10
        )["sampled_band_matching_pass"])

    def test_rejects_missing_band_edge_invalid_and_nonpassive_data(self):
        for response in (
            {"frequency_hz": self.frequency, "s11": np.array([1, 0.1, np.nan, 0.1, 1])},
            {"frequency_hz": self.frequency, "s11": np.array([1, 0.1, 0, 0.1, 1])},
            {"frequency_hz": self.frequency, "s11": np.array([1, 0.1, 2, 0.1, 1])},
            {"frequency_hz": self.frequency[::-1], "s11": np.ones(5)},
        ):
            with self.subTest(response=response), self.assertRaises(ValueError):
                matching_metrics(**response, band_start_hz=2.4e9, band_stop_hz=2.5e9, limit_db=-10)
        with self.assertRaises(ValueError):
            matching_metrics(
                **self.response([-1, -12, -25, -12, -1]),
                band_start_hz=2.41e9, band_stop_hz=2.5e9, limit_db=-10,
            )

    def test_refinement_measures_threshold_not_only_resonance(self):
        coarse = self.response([-1, -12, -25, -12, -1])
        stable = self.response([-1, -12.4, -25.4, -12.4, -1])
        unstable = self.response([-1, -12.6, -25.6, -12.6, -1])
        self.assertTrue(refinement_metrics(coarse, stable, 2.4e9, 2.5e9, -10)["mesh_stability_pass"])
        self.assertFalse(refinement_metrics(coarse, unstable, 2.4e9, 2.5e9, -10)["mesh_stability_pass"])
        self.assertFalse(refinement_metrics(coarse, stable, 2.4e9, 2.5e9, -30)["mesh_stability_pass"])
        mismatched = {**stable, "frequency_hz": self.frequency + 1}
        with self.assertRaises(ValueError):
            refinement_metrics(coarse, mismatched, 2.4e9, 2.5e9, -10)

    def test_tied_or_edge_minimum_is_not_a_resolved_resonance(self):
        for db in ([-1, -25, -25, -12, -1], [-30, -12, -20, -12, -1]):
            result = matching_metrics(
                **self.response(db), band_start_hz=2.4e9, band_stop_hz=2.5e9, limit_db=-10
            )
            self.assertIsNone(result["resonance_candidate_hz"])

    def test_convergence_rejects_caps_aborts_and_missing_completion(self):
        confirmed = (
            "Max. number of timesteps: 30000\n"
            "RunFDTD: end-criteria of -40dB reached after 1234 timesteps (-41.2dB)\n"
            "Time for 1234 iterations with 500 cells"
        )
        self.assertTrue(solver_convergence(
            confirmed, 30000, 1e-4
        )["energy_decay_reached"])
        for log in (
            "", "Time for 30000 iterations", "Time for 0 iterations",
            "Time for 1234 iterations with 500 cells",
            "Warning: Max. number of timesteps was reached\nTime for 2000 iterations",
            "simulation aborted\nTime for 2000 iterations",
            confirmed.replace("-40dB", "-20dB"),
            confirmed.replace("(-41.2dB)", "(-39.2dB)"),
        ):
            with self.subTest(log=log), self.assertRaises(ValueError):
                solver_convergence(log, 30000, 1e-4)


class JobTests(unittest.TestCase):
    def test_failures_are_retained_and_not_converted_to_success(self):
        for error, state in (
            (subprocess.CalledProcessError(1, "worker"), "failed"),
            (subprocess.TimeoutExpired("worker", 600), "timed_out"),
        ):
            with self.subTest(state=state), tempfile.TemporaryDirectory() as directory:
                run = Path(directory) / "candidate"
                with patch("src.solvers.openems_patch.subprocess.run", side_effect=error):
                    with self.assertRaises(type(error)):
                        run_candidate(PatchConfig(), run)
                manifest = json.loads((run / "manifest.json").read_text())
                self.assertEqual(manifest["status"], state)
                self.assertIn("error", manifest)
                self.assertTrue((run / "solver.log").exists())
                self.assertFalse((run / "matching.json").exists())

    def test_existing_run_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "sentinel").write_text("retained")
            with self.assertRaises(FileExistsError):
                run_candidate(PatchConfig(), run)
            self.assertEqual((run / "sentinel").read_text(), "retained")


if __name__ == "__main__":
    unittest.main()
