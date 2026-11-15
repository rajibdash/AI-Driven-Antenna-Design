# Understanding `openems_patch.json`

This file configures one bounded openEMS simulation of a rectangular microstrip patch antenna. All lengths are in millimeters, frequencies are in hertz, and the solver's reference impedance is in ohms. The antenna and board are centered on the x/y origin; the patch lies above the substrate, and the feed is on the x-axis (`y = 0`).

The file is strict JSON. JSON does not support comments, and the loader rejects unknown keys, so keep explanations here rather than adding comment lines or extra properties to the configuration.

## Antenna geometry and material

| Setting | Meaning |
| --- | --- |
| `patch_x_mm` | Patch width along the x-axis. The patch extends from `-patch_x_mm / 2` to `+patch_x_mm / 2`. |
| `patch_y_mm` | Patch length along the y-axis. The patch extends from `-patch_y_mm / 2` to `+patch_y_mm / 2`. |
| `feed_x_mm` | Feed position along x; the lumped feed is at this x coordinate and at `y = 0`. It must be inside the patch. |
| `board_x_mm` | Board width along x. The board also defines the finite ground-plane size. |
| `board_y_mm` | Board length along y. The board also defines the finite ground-plane size. |
| `substrate_thickness_mm` | Dielectric layer thickness in z, between the ground plane and patch. |
| `relative_permittivity` | Relative dielectric constant (epsilon-r) of the substrate. |
| `loss_tangent_at_reference` | Substrate loss tangent used to calculate conductivity at `material_reference_hz`. The solver uses that calculated conductivity as a constant across the simulated frequency range; it does not model a constant loss tangent across the range. |
| `material_reference_hz` | Frequency used for the substrate conductivity calculation. It is also the frequency used for the far-field export. |

## Matching study

| Setting | Meaning |
| --- | --- |
| `reference_impedance_ohm` | Reference impedance in ohms for the lumped feed and S11 calculation; 50 ohms is the standard starting point. |
| `band_start_hz` | Inclusive start of the frequency band where input matching is evaluated. |
| `band_stop_hz` | Inclusive end of the frequency band where input matching is evaluated. |
| `s11_limit_db` | Maximum allowed worst-case S11 in that band. S11 is reported in dB; for example, `-10` means every sampled point in the band must be at or below -10 dB for `sampled_band_matching_pass` to be true. |

## Mesh, solver, and resource limits

| Setting | Meaning |
| --- | --- |
| `cells_per_wavelength` | Target number of free-space cells per wavelength at 3 GHz, used to set the maximum mesh-cell size in air. A larger value makes the mesh finer and usually increases runtime. |
| `substrate_cells` | Number of mesh divisions through the substrate thickness. |
| `max_timesteps` | Upper limit on FDTD time steps for a simulation. Reaching the cap is not treated as successful convergence. |
| `end_criteria` | Normalized field-energy threshold at which openEMS may stop the time-domain simulation. A smaller threshold asks the fields to decay further and can take longer. |
| `timeout_seconds` | Wall-clock time limit, in seconds, for each solver run. |
| `threads` | Number of CPU threads requested for each solver run. |

## Example usage

Run the baseline configuration from the repository root:

```bash
python -m src.solvers.openems_patch configs/openems_patch.json --output runs/baseline
```

The matching result is a numerical simulation result, not a guarantee of hardware performance. The substrate model is simplified, the patch and ground are modeled as perfect conductors, and the feed is an idealized lumped port. See the project README for installation, output, and mesh-verification details.
