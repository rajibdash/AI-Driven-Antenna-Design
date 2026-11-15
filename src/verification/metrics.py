"""Numerical checks for the patch study; not hardware acceptance."""

import re

import numpy as np


def matching_metrics(frequency_hz, s11, band_start_hz, band_stop_hz, limit_db):
    frequency_hz = np.asarray(frequency_hz)
    s11 = np.asarray(s11)
    if (
        frequency_hz.ndim != 1
        or s11.shape != frequency_hz.shape
        or len(frequency_hz) < 2
        or not np.all(np.isfinite(frequency_hz))
        or not np.all(np.isfinite(s11))
        or not np.all(np.diff(frequency_hz) > 0)
    ):
        raise ValueError("Expected aligned, finite responses on an increasing frequency grid")
    if not frequency_hz[0] <= band_start_hz < band_stop_hz <= frequency_hz[-1]:
        raise ValueError("Response does not cover the required band")
    if not np.any(frequency_hz == band_start_hz) or not np.any(frequency_hz == band_stop_hz):
        raise ValueError("Both band endpoints must be sampled explicitly")
    magnitude = np.abs(s11)
    if np.any(magnitude > 1.001):
        raise ValueError("Passive one-port response exceeds unity beyond numerical tolerance")
    if np.any(magnitude == 0):
        raise ValueError("Zero reflection produces non-finite dB; refine/inspect the response")
    s11_db = 20 * np.log10(magnitude)
    band = (frequency_hz >= band_start_hz) & (frequency_hz <= band_stop_hz)
    worst = float(np.max(s11_db[band]))
    minimum = int(np.argmin(s11_db))
    # An interior, unique minimum below the threshold is only a resonance candidate.
    resonance = (
        float(frequency_hz[minimum])
        if 0 < minimum < len(s11_db) - 1
        and np.count_nonzero(s11_db == s11_db[minimum]) == 1
        and s11_db[minimum] < limit_db
        else None
    )
    return {
        "worst_band_s11_db": worst,
        "sampled_band_matching_pass": worst <= limit_db,
        "resonance_candidate_hz": resonance,
        "minimum_s11_db": float(s11_db[minimum]),
    }


def solver_convergence(log, max_timesteps, end_criteria):
    if "Max. number of timesteps was reached" in log or "simulation aborted" in log:
        raise ValueError("Solver stopped without reaching its energy-decay criterion")
    match = re.search(r"Time for\s+(\d+)\s+iterations", log)
    if match is None:
        raise ValueError("Solver log is missing its completion record")
    timesteps = int(match.group(1))
    if not 0 < timesteps < max_timesteps:
        raise ValueError("Solver reached its timestep cap; convergence is not established")
    decay = re.search(
        r"end-criteria of -([0-9.]+)dB reached after (\d+) timesteps \(-([0-9.]+)dB\)", log
    )
    expected_db = -10 * np.log10(end_criteria)
    if (
        decay is None
        or int(decay.group(2)) != timesteps
        or abs(float(decay.group(1)) - expected_db) > 0.05
        or float(decay.group(3)) + 0.05 < expected_db
    ):
        raise ValueError("Solver log does not confirm the configured energy-decay criterion")
    return {
        "completed_timesteps": timesteps, "energy_decay_reached": True,
        "final_energy_decay_db": -float(decay.group(3)),
    }


def refinement_metrics(coarse, fine, band_start_hz, band_stop_hz, limit_db):
    if not np.array_equal(coarse["frequency_hz"], fine["frequency_hz"]):
        raise ValueError("Mesh comparison requires identical frequency grids")
    coarse_metrics = matching_metrics(
        coarse["frequency_hz"], coarse["s11"], band_start_hz, band_stop_hz, limit_db
    )
    fine_metrics = matching_metrics(
        fine["frequency_hz"], fine["s11"], band_start_hz, band_stop_hz, limit_db
    )
    band = (coarse["frequency_hz"] >= band_start_hz) & (coarse["frequency_hz"] <= band_stop_hz)
    delta = float(np.max(np.abs(
        20 * np.log10(np.abs(coarse["s11"][band]))
        - 20 * np.log10(np.abs(fine["s11"][band]))
    )))
    first = coarse_metrics["resonance_candidate_hz"]
    second = fine_metrics["resonance_candidate_hz"]
    shift = abs(second - first) / first if first is not None and second is not None else None
    return {
        "max_band_s11_change_db": delta,
        "resonance_shift_fraction": shift,
        "mesh_stability_pass": shift is not None and shift <= 0.01 and delta <= 0.5,
    }
