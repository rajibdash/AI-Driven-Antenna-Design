"""Bounded, headless full-wave study of a PEC patch with a lumped feed."""

import argparse
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

from src.verification.metrics import matching_metrics, refinement_metrics, solver_convergence


@dataclass(frozen=True)
class PatchConfig:
    patch_x_mm: float = 32.0
    patch_y_mm: float = 40.0
    feed_x_mm: float = -6.0
    board_x_mm: float = 60.0
    board_y_mm: float = 60.0
    substrate_thickness_mm: float = 1.524
    relative_permittivity: float = 3.38
    loss_tangent_at_reference: float = 0.001
    material_reference_hz: float = 2.45e9
    reference_impedance_ohm: float = 50.0
    band_start_hz: float = 2.4e9
    band_stop_hz: float = 2.5e9
    s11_limit_db: float = -10.0
    cells_per_wavelength: int = 20
    substrate_cells: int = 4
    max_timesteps: int = 30000
    end_criteria: float = 1e-6
    timeout_seconds: int = 600
    threads: int = 1

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{field.name} must be a finite number")
            if field.type is int and type(value) is not int:
                raise ValueError(f"{field.name} must be an integer")
            if field.name not in ("feed_x_mm", "s11_limit_db", "loss_tangent_at_reference") and value <= 0:
                raise ValueError(f"{field.name} must be positive")
        if not 0 <= self.loss_tangent_at_reference < 1 or not 0 < self.end_criteria < 1:
            raise ValueError("Loss tangent must be in [0,1); end criterion must be in (0,1)")
        if self.relative_permittivity < 1 or self.s11_limit_db >= 0:
            raise ValueError("Expected permittivity >= 1 and a negative S11 limit")
        if not (
            self.patch_x_mm < self.board_x_mm < 180
            and self.patch_y_mm < self.board_y_mm < 180
            and self.substrate_thickness_mm < 40
        ):
            raise ValueError("Patch must fit strictly inside the board and fixed air box")
        if not -self.patch_x_mm / 2 < self.feed_x_mm < self.patch_x_mm / 2:
            raise ValueError("Feed must lie strictly inside the patch")
        if not 1e9 <= self.band_start_hz < self.band_stop_hz <= 3e9:
            raise ValueError("Study band must be inside the fixed 1-3 GHz sweep")
        if not 1e9 <= self.material_reference_hz <= 3e9:
            raise ValueError("Material reference/far-field frequency must be inside the sweep")
        if not 10 <= self.cells_per_wavelength <= 60 or not 2 <= self.substrate_cells <= 20:
            raise ValueError("Mesh settings exceed the bounded study domain")
        if self.threads > 8 or self.timeout_seconds > 3600 or self.max_timesteps > 200000:
            raise ValueError("Resource settings exceed the study budget")


def load_config(path):
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError("Configuration must be a JSON object")
    names = {field.name for field in fields(PatchConfig)}
    if set(data) != names:
        raise ValueError(f"Configuration fields differ: missing={names - set(data)}, unknown={set(data) - names}")
    return PatchConfig(**data)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def artifact_hash(path):
    with path.open("rb") as artifact:
        return hashlib.file_digest(artifact, "sha256").hexdigest()


def simulate(config, run_dir):
    # Native bindings belong in a worker so its timeout also bounds native FDTD calls.
    import CSXCAD
    import openEMS
    from openEMS.physical_constants import C0, EPS0

    fdtd = openEMS.openEMS(NrTS=config.max_timesteps, EndCriteria=config.end_criteria)
    fdtd.SetGaussExcite(2e9, 1e9)
    fdtd.SetBoundaryCond(["MUR"] * 6)
    csx = CSXCAD.ContinuousStructure()
    fdtd.SetCSX(csx)
    grid = csx.GetGrid()
    grid.SetDeltaUnit(1e-3)
    resolution_mm = C0 / 3e9 / 1e-3 / config.cells_per_wavelength
    grid.AddLine("x", [-100, 100])
    grid.AddLine("y", [-100, 100])
    grid.AddLine("z", [-50, 100])
    thickness = config.substrate_thickness_mm

    patch = csx.AddMetal("patch")
    patch.AddBox(
        priority=10,
        start=[-config.patch_x_mm / 2, -config.patch_y_mm / 2, thickness],
        stop=[config.patch_x_mm / 2, config.patch_y_mm / 2, thickness],
    )
    fdtd.AddEdges2Grid(dirs="xy", properties=patch, metal_edge_res=resolution_mm / 2)
    conductivity = (
        config.loss_tangent_at_reference * 2 * np.pi * config.material_reference_hz
        * EPS0 * config.relative_permittivity
    )
    substrate = csx.AddMaterial("substrate", epsilon=config.relative_permittivity, kappa=conductivity)
    substrate.AddBox(
        priority=0,
        start=[-config.board_x_mm / 2, -config.board_y_mm / 2, 0],
        stop=[config.board_x_mm / 2, config.board_y_mm / 2, thickness],
    )
    ground = csx.AddMetal("ground")
    ground.AddBox(
        priority=10,
        start=[-config.board_x_mm / 2, -config.board_y_mm / 2, 0],
        stop=[config.board_x_mm / 2, config.board_y_mm / 2, 0],
    )
    fdtd.AddEdges2Grid(dirs="xy", properties=ground)
    grid.AddLine("z", np.linspace(0, thickness, config.substrate_cells + 1))
    port = fdtd.AddLumpedPort(
        1, config.reference_impedance_ohm,
        [config.feed_x_mm, 0, 0], [config.feed_x_mm, 0, thickness],
        "z", 1.0, priority=5, edges2grid="xy",
    )
    grid.SmoothMeshLines("all", resolution_mm, 1.4)
    far_field_box = fdtd.CreateNF2FFBox()
    fdtd.Write2XML(str(run_dir / "model.xml"))
    write_json(run_dir / "solver_setup.json", {
        "openEMS_version": openEMS.__version__, "CSXCAD_version": CSXCAD.__version__,
        "numpy_version": np.__version__, "python_version": platform.python_version(),
        "platform": platform.platform(), "length_unit_m": 1e-3,
        "coordinate_system": "right_handed_xyz", "boundary": ["MUR"] * 6,
        "air_box_mm": [[-100, 100], [-100, 100], [-50, 100]],
        "excitation_center_hz": 2e9, "excitation_corner_hz": 1e9,
        "mesh_max_cell_mm": resolution_mm, "mesh_growth_ratio": 1.4,
        "mesh_lines_mm": {axis: grid.GetLines(axis).tolist() for axis in "xyz"},
        "substrate_conductivity_s_per_m": conductivity,
        "conductor": "PEC", "feed": "z_directed_lumped_port_at_y_zero",
        "far_field_center_m": [0, 0, thickness / 2 * 1e-3],
        "far_field_angular_step_degrees": 10,
        "far_field_threading": "native_default",
        "exact_endcriteria": True,
    })
    result = fdtd.Run(
        str(run_dir), cleanup=False, numThreads=config.threads, exact_endcriteria=True
    )
    if result not in (None, 0):
        raise RuntimeError(f"openEMS setup failed with code {result}")
    frequency_hz = np.unique(np.concatenate((
        np.linspace(1e9, 3e9, 401),
        [config.band_start_hz, config.band_stop_hz, config.material_reference_hz],
        np.linspace(config.band_start_hz, config.band_stop_hz,
                    math.ceil((config.band_stop_hz - config.band_start_hz) / 1e6) + 1),
    )))
    port.CalcPort(str(run_dir), frequency_hz, ref_impedance=config.reference_impedance_ohm)
    with np.errstate(divide="raise", invalid="raise"):
        s11 = port.uf_ref / port.uf_inc
        impedance = port.uf_tot / port.if_tot
    if not np.all(np.isfinite(impedance)):
        raise ValueError("Non-finite input impedance")
    metrics = matching_metrics(
        frequency_hz, s11, config.band_start_hz, config.band_stop_hz, config.s11_limit_db
    )
    np.savez_compressed(
        run_dir / "response.npz", frequency_hz=frequency_hz, s11=s11,
        input_impedance_ohm=impedance, accepted_power_w=port.P_acc,
        incident_power_w=port.P_inc,
    )
    np.savetxt(
        run_dir / "s11.csv",
        np.column_stack((frequency_hz, s11.real, s11.imag, 20 * np.log10(np.abs(s11)),
                         impedance.real, impedance.imag)),
        delimiter=",", header="frequency_hz,s11_real,s11_imag,s11_db,zin_real_ohm,zin_imag_ohm",
        comments="",
    )
    # NF2FF phase centers use meters, unlike the geometry's millimeter coordinates.
    far_field = far_field_box.CalcNF2FF(
        str(run_dir), config.material_reference_hz,
        np.arange(0, 181, 10), np.arange(0, 360, 10),
        center=[0, 0, thickness / 2 * 1e-3],
    )
    arrays = {
        "theta_rad": far_field.theta, "phi_rad": far_field.phi,
        "frequency_hz": far_field.freq, "radius_m": far_field.r,
        "e_theta": far_field.E_theta[0], "e_phi": far_field.E_phi[0],
        "radiated_power_w": far_field.Prad, "max_directivity_linear": far_field.Dmax,
    }
    if not all(np.all(np.isfinite(value)) for value in arrays.values()):
        raise ValueError("Non-finite far-field export")
    if np.any(far_field.Prad <= 0) or np.any(far_field.Dmax <= 0):
        raise ValueError("Far-field power and directivity must be positive")
    np.savez_compressed(run_dir / "far_field.npz", **arrays)
    write_json(run_dir / "matching.json", metrics)


def run_candidate(config, run_dir):
    run_dir.mkdir()
    write_json(run_dir / "config.json", asdict(config))
    manifest = {
        "status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "adapter_sha256": artifact_hash(Path(__file__)),
        "verification_sha256": artifact_hash(
            Path(__file__).resolve().parents[1] / "verification/metrics.py"
        ),
    }
    write_json(run_dir / "manifest.json", manifest)
    started = time.monotonic()
    try:
        with (run_dir / "solver.log").open("w") as log:
            subprocess.run(
                [sys.executable, "-m", "src.solvers.openems_patch",
                 str(run_dir / "config.json"), "--worker", "--output", str(run_dir)],
                cwd=Path(__file__).resolve().parents[2],
                stdout=log, stderr=subprocess.STDOUT, check=True,
                timeout=config.timeout_seconds,
            )
        manifest["convergence"] = solver_convergence(
            (run_dir / "solver.log").read_text(), config.max_timesteps, config.end_criteria
        )
        manifest["matching"] = json.loads((run_dir / "matching.json").read_text())
        manifest["status"] = "succeeded"
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError, OSError) as error:
        manifest["status"] = "timed_out" if isinstance(error, subprocess.TimeoutExpired) else "failed"
        manifest["error"] = str(error)
        raise
    finally:
        manifest["elapsed_seconds"] = time.monotonic() - started
        manifest["artifacts_sha256"] = {
            file.name: artifact_hash(file)
            for file in run_dir.iterdir()
            if file.is_file() and file.name != "manifest.json"
        }
        write_json(run_dir / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="New directory; never overwritten")
    parser.add_argument("--patch-x-mm", type=float)
    parser.add_argument("--feed-x-mm", type=float)
    parser.add_argument("--verify", action="store_true", help="Three meshes plus a clean finest-mesh rerun")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        overrides = {
            key: value for key, value in {
                "patch_x_mm": args.patch_x_mm, "feed_x_mm": args.feed_x_mm
            }.items() if value is not None
        }
        config = replace(config, **overrides)
        output = args.output.resolve()
        if args.worker:
            simulate(config, output)
            return
        configs = [config]
        if args.verify:
            configs += [
                replace(config, cells_per_wavelength=config.cells_per_wavelength + step * 5,
                        substrate_cells=config.substrate_cells + step)
                for step in (1, 2)
            ]
            configs.append(configs[-1])
        output.mkdir(parents=True, exist_ok=False)
        results = [run_candidate(candidate, output / f"mesh-{index}")
                   for index, candidate in enumerate(configs)]
        report = {
            "status": "succeeded", "runs": [f"mesh-{index}" for index in range(len(results))],
            "sampled_band_matching_pass": results[-1]["matching"]["sampled_band_matching_pass"],
            "verification": "not_requested", "hardware_validated": False,
        }
        if args.verify:
            comparisons = []
            for index in range(3):
                with np.load(output / f"mesh-{index}/response.npz") as coarse, \
                        np.load(output / f"mesh-{index + 1}/response.npz") as fine:
                    comparisons.append(refinement_metrics(
                        coarse, fine, config.band_start_hz, config.band_stop_hz, config.s11_limit_db
                    ))
            report["comparisons"] = comparisons
            report["verification"] = "passed" if all(
                item["mesh_stability_pass"] for item in comparisons
            ) else "failed_or_unresolved"
        write_json(output / "study.json", report)
        print(json.dumps(report, indent=2))
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        parser.exit(1, f"Study failed: {error}\nInspect the retained run manifest and solver.log.\n")


if __name__ == "__main__":
    main()
