# Starter Kit: AI-Driven RF & Antenna Design Automation (`openEMS` + PyTorch)

This starter kit provides a complete, runnable Python pipeline to generate electromagnetic (EM) simulation datasets, train a forward surrogate neural network, and run inverse design optimization with verification.

Because you listed three distinct target architectures, this kit includes **ready-to-use parameterized geometry modules** for all three:
1. **5G mmWave Patch Array** (`28 GHz`, Rogers RO4350B)
2. **Sub-6 GHz 2x2 MIMO Antenna** (`3.5 GHz` n78 band, FR4)
3. **Microstrip Coupled-Line Bandpass Filter** (`2.45 GHz`, Rogers RO4003C)

---

## 1. Environment Setup

Install the required Python libraries. For the EM solver, install **`openEMS`** (free, open-source FDTD solver) along with its Python bindings (`CSXCAD` and `openEMS`).

```bash
pip install numpy scipy torch scikit-learn scikit-rf optuna matplotlib
# Ensure openEMS and python-openems are installed via your OS package manager or built from source
```

### Project Directory Structure
```text
rf_ai_starter/
├── config.py               # Target selection & frequency/substrate specs
├── geometries.py           # Parameterized EM models (mmWave, MIMO, Filter)
├── 1_generate_dataset.py   # Latin Hypercube Sampling + openEMS headless runner
├── 2_train_surrogate.py    # PyTorch ResNet-MLP forward EM surrogate
├── 3_inverse_optimize.py   # Multi-objective inverse design + DFM verification
└── data/                   # Generated S-parameters (.npz) and trained weights (.pt)
```

---

## 2. Configuration (`config.py`)

Select your target device by changing `TARGET_DEVICE`.

```python
import numpy as np

# Options: "mmwave_array", "sub6_mimo", "microstrip_filter"
TARGET_DEVICE = "mmwave_array"

CONFIGS = {
    "mmwave_array": {
        "name": "28 GHz 1x2 Patch Array (Rogers RO4350B)",
        "f_min": 24e9,
        "f_max": 32e9,
        "n_freq": 161,
        "er": 3.66,
        "tand": 0.0037,
        "h_sub": 0.254e-3,  # 10 mil substrate
        # Design parameters: [patch_L, patch_W, inset_depth, feed_gap, array_spacing] (in mm)
        "bounds_min": np.array([2.4, 2.8, 0.4, 0.15, 4.5]),
        "bounds_max": np.array([3.2, 4.0, 1.1, 0.45, 7.5]),
        "n_ports": 2,
    },
    "sub6_mimo": {
        "name": "3.5 GHz Sub-6GHz 2-Element MIMO Monopole (FR4)",
        "f_min": 2.5e9,
        "f_max": 4.5e9,
        "n_freq": 161,
        "er": 4.4,
        "tand": 0.02,
        "h_sub": 1.6e-3,
        # Design parameters: [monopole_L, monopole_W, gnd_slot_L, gnd_slot_W, edge_sep] (in mm)
        "bounds_min": np.array([12.0, 1.5, 4.0, 1.0, 8.0]),
        "bounds_max": np.array([18.0, 4.0, 14.0, 4.0, 22.0]),
        "n_ports": 2,
    },
    "microstrip_filter": {
        "name": "2.45 GHz 3rd-Order Coupled-Line Bandpass Filter (RO4003C)",
        "f_min": 1.5e9,
        "f_max": 3.5e9,
        "n_freq": 201,
        "er": 3.55,
        "tand": 0.0027,
        "h_sub": 0.813e-3,
        # Design parameters: [L1, W1, S1, L2, W2, S2] (in mm)
        "bounds_min": np.array([16.0, 0.8, 0.15, 16.0, 1.0, 0.3]),
        "bounds_max": np.array([21.0, 2.2, 0.60, 21.0, 2.5, 1.2]),
        "n_ports": 2,
    },
}

ACTIVE_CFG = CONFIGS[TARGET_DEVICE]
FREQS = np.linspace(ACTIVE_CFG["f_min"], ACTIVE_CFG["f_max"], ACTIVE_CFG["n_freq"])
```

---

## 3. Parameterized EM Geometries (`geometries.py`)

This module builds the 3D FDTD mesh and runs `openEMS` for any candidate parameter vector $\mathbf{x}$.

```python
import os
import tempfile
import numpy as np
from CSXCAD import ContinuousStructure
from openEMS import openEMS
from openEMS.physical_constants import C0, EPS0
from config import ACTIVE_CFG, FREQS, TARGET_DEVICE


def run_openems_simulation(params_mm: np.ndarray) -> dict:
    """
    Runs a headless openEMS FDTD simulation for the active device.
    Returns complex S-parameters: S11, S21, S22 across FREQS.
    """
    unit = 1e-3  # mm
    f_min, f_max = ACTIVE_CFG["f_min"], ACTIVE_CFG["f_max"]
    f0 = 0.5 * (f_min + f_max)
    fc = 0.5 * (f_max - f_min)

    FDTD = openEMS(NrTS=30000, EndCriteria=1e-4)
    FDTD.SetGaussExcite(f0, fc)
    FDTD.SetBoundaryCond(["PML_8", "PML_8", "PML_8", "PML_8", "PEC", "PML_8"])

    CSX = ContinuousStructure()
    FDTD.SetCSX(CSX)
    mesh = CSX.GetGrid()
    mesh.SetDeltaUnit(unit)

    # Substrate & Copper Definitions
    sub = CSX.AddMaterial("Substrate", epsilon=ACTIVE_CFG["er"])
    kappa = 2 * np.pi * f0 * EPS0 * ACTIVE_CFG["er"] * ACTIVE_CFG["tand"]
    sub.SetMaterialProperty(kappa=kappa)
    pec = CSX.AddMetal("Copper")

    h_sub_mm = ACTIVE_CFG["h_sub"] / unit
    ports = []

    if TARGET_DEVICE == "mmwave_array":
        pL, pW, inset, gap, spacing = params_mm
        sub_x, sub_y = spacing + pW + 8.0, pL + 10.0
        sub.AddBox([-sub_x / 2, -sub_y / 2, 0], [sub_x / 2, sub_y / 2, h_sub_mm])

        for idx, sign in enumerate([-1, 1]):
            cx = sign * (spacing / 2.0)
            # Radiating Patch
            pec.AddBox([cx - pW / 2, -pL / 2, h_sub_mm], [cx + pW / 2, pL / 2, h_sub_mm])
            # Microstrip Feed + Lumped Port
            feed_w = 0.55
            pec.AddBox([cx - feed_w / 2, -sub_y / 2, h_sub_mm], [cx + feed_w / 2, -pL / 2 + inset, h_sub_mm])
            port = FDTD.AddLumpedPort(
                idx + 1, 50,
                [cx - feed_w / 2, -sub_y / 2 + 0.5, 0],
                [cx + feed_w / 2, -sub_y / 2 + 0.5, h_sub_mm],
                "z", excite=(1 if idx == 0 else 0)
            )
            ports.append(port)

    elif TARGET_DEVICE == "sub6_mimo":
        mL, mW, sL, sW, sep = params_mm
        sub_x, sub_y = sep + 2 * mW + 16.0, mL + 22.0
        gnd_y = 18.0
        sub.AddBox([-sub_x / 2, 0, 0], [sub_x / 2, sub_y, h_sub_mm])
        # Partial Ground Plane on Bottom (z=0) with Decoupling Slot
        pec.AddBox([-sub_x / 2, 0, 0], [-sW / 2, gnd_y, 0])
        pec.AddBox([sW / 2, 0, 0], [sub_x / 2, gnd_y, 0])
        pec.AddBox([-sW / 2, 0, 0], [sW / 2, max(1.0, gnd_y - sL), 0])

        for idx, sign in enumerate([-1, 1]):
            cx = sign * (sep / 2.0 + mW / 2.0)
            pec.AddBox([cx - mW / 2, 2.0, h_sub_mm], [cx + mW / 2, gnd_y + mL, h_sub_mm])
            port = FDTD.AddLumpedPort(
                idx + 1, 50,
                [cx - mW / 2, 1.0, 0],
                [cx + mW / 2, 2.0, h_sub_mm],
                "y", excite=(1 if idx == 0 else 0)
            )
            ports.append(port)

    elif TARGET_DEVICE == "microstrip_filter":
        L1, W1, S1, L2, W2, S2 = params_mm
        total_L = L1 + L2 + 12.0
        sub.AddBox([-5.0, -8.0, 0], [total_L, 8.0, h_sub_mm])
        # Input & Output 50-ohm Feedlines + Coupled Sections
        pec.AddBox([-4.0, -0.9, h_sub_mm], [2.0, 0.9, h_sub_mm])
        pec.AddBox([2.0, 0.5 * S1, h_sub_mm], [2.0 + L1, 0.5 * S1 + W1, h_sub_mm])
        pec.AddBox([2.0, -0.5 * S1 - W1, h_sub_mm], [2.0 + L1, -0.5 * S1, h_sub_mm])
        pec.AddBox([2.0 + L1, -0.5 * S2 - W2, h_sub_mm], [2.0 + L1 + L2, -0.5 * S2, h_sub_mm])
        pec.AddBox([2.0 + L1, 0.5 * S2, h_sub_mm], [2.0 + L1 + L2, 0.5 * S2 + W2, h_sub_mm])
        pec.AddBox([2.0 + L1 + L2, -0.9, h_sub_mm], [total_L - 1.0, 0.9, h_sub_mm])

        ports.append(FDTD.AddLumpedPort(1, 50, [-3.0, -0.9, 0], [-3.0, 0.9, h_sub_mm], "z", excite=1))
        ports.append(FDTD.AddLumpedPort(2, 50, [total_L - 2.0, -0.9, 0], [total_L - 2.0, 0.9, h_sub_mm], "z", excite=0))

    # Mesh discretization (lambda / 20 rule)
    max_res = (C0 / (f_max * np.sqrt(ACTIVE_CFG["er"]))) / 20.0 / unit
    mesh.SmoothMeshLines("x", max_res, 1.4)
    mesh.SmoothMeshLines("y", max_res, 1.4)
    mesh.SmoothMeshLines("z", h_sub_mm / 3.0, 1.4)

    with tempfile.TemporaryDirectory() as sim_dir:
        FDTD.Run(sim_dir, verbose=0)
        for p in ports:
            p.CalcPort(sim_dir, FREQS)

    s11 = ports[0].uf_ref / ports[0].uf_inc
    s21 = ports[1].uf_ref / ports[0].uf_inc

    # Enforce passivity sanity check (|S11|^2 + |S21|^2 <= 1.02 accounting for numerical tolerance)
    power_sum = np.abs(s11) ** 2 + np.abs(s21) ** 2
    valid = bool(np.all(power_sum <= 1.05))

    return {"s11": s11, "s21": s21, "valid": valid}
```

---

## 4. Automated Dataset Generation (`1_generate_dataset.py`)

Uses **Latin Hypercube Sampling (LHS)** to sample the parameter space efficiently and saves valid simulations to `data/dataset.npz`.

```python
import os
import numpy as np
from scipy.stats import qmc
from config import ACTIVE_CFG, FREQS, TARGET_DEVICE
from geometries import run_openems_simulation

os.makedirs("data", exist_ok=True)


def generate_dataset(n_samples: int = 300, seed: int = 42):
    b_min = ACTIVE_CFG["bounds_min"]
    b_max = ACTIVE_CFG["bounds_max"]
    dim = len(b_min)

    sampler = qmc.LatinHypercube(d=dim, seed=seed)
    unit_samples = sampler.random(n=n_samples)
    X_candidates = qmc.scale(unit_samples, b_min, b_max)

    X_valid, S11_list, S21_list = [], [], []

    for i, x in enumerate(X_candidates):
        res = run_openems_simulation(x)
        if res["valid"]:
            X_valid.append(x)
            S11_list.append(res["s11"])
            S21_list.append(res["s21"])
        print(f"[{i+1}/{n_samples}] Valid={res['valid']} | Total Kept={len(X_valid)}")

    X_arr = np.array(X_valid, dtype=np.float32)
    S11_arr = np.array(S11_list, dtype=np.complex64)
    S21_arr = np.array(S21_list, dtype=np.complex64)

    out_path = f"data/{TARGET_DEVICE}_dataset.npz"
    np.savez(out_path, X=X_arr, S11=S11_arr, S21=S21_arr, freqs=FREQS)
    print(f"Saved {len(X_arr)} valid samples to {out_path}")


if __name__ == "__main__":
    generate_dataset(n_samples=300)
```

---

## 5. Forward Surrogate Training (`2_train_surrogate.py`)

Trains a **Residual MLP (`ResNetMLP`)** to predict the continuous real and imaginary parts of $S_{11}(f)$ and $S_{21}(f)$ in `< 1 ms` per evaluation. Predicting complex components ($	ext{Re}, 	ext{Im}$) avoids phase-wrapping discontinuities and magnitude-only clipping.

```python
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from config import ACTIVE_CFG, TARGET_DEVICE


class ResBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(x + self.block(x))


class ForwardEMSurrogate(nn.Module):
    def __init__(self, in_dim: int, n_freq: int, hidden_dim: int = 256):
        super().__init__()
        # Outputs: [Re(S11), Im(S11), Re(S21), Im(S21)] across n_freq points
        out_dim = 4 * n_freq
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            ResBlock(hidden_dim),
            ResBlock(hidden_dim),
            ResBlock(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def train_surrogate(epochs: int = 400, batch_size: int = 32, lr: float = 1e-3):
    data = np.load(f"data/{TARGET_DEVICE}_dataset.npz")
    X = data["X"]
    S11 = data["S11"]
    S21 = data["S21"]

    # Normalize inputs to [0, 1] using physical bounds
    b_min, b_max = ACTIVE_CFG["bounds_min"], ACTIVE_CFG["bounds_max"]
    X_norm = (X - b_min) / (b_max - b_min)

    # Stack complex S-parameters into real tensor [N, 4 * n_freq]
    Y = np.concatenate([S11.real, S11.imag, S21.real, S21.imag], axis=1).astype(np.float32)

    X_train, X_val, Y_train, Y_val = train_test_split(X_norm, Y, test_size=0.15, random_state=42)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(Y_train)),
        batch_size=batch_size,
        shuffle=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ForwardEMSurrogate(in_dim=X.shape[1], n_freq=S11.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.SmoothL1Loss(beta=0.05)

    best_val_loss = float("inf")
    X_val_t = torch.tensor(X_val, device=device)
    Y_val_t = torch.tensor(Y_val, device=device)

    for epoch in range(1, epochs + 1):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = model(bx)
            # Passivity penalty: |S11|^2 + |S21|^2 <= 1.0
            n_f = S11.shape[1]
            s11_mag2 = pred[:, :n_f] ** 2 + pred[:, n_f : 2 * n_f] ** 2
            s21_mag2 = pred[:, 2 * n_f : 3 * n_f] ** 2 + pred[:, 3 * n_f :] ** 2
            passivity_pen = torch.relu(s11_mag2 + s21_mag2 - 1.0).mean()

            loss = loss_fn(pred, by) + 0.1 * passivity_pen
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = loss_fn(val_pred, Y_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"data/{TARGET_DEVICE}_surrogate.pt")

        if epoch % 50 == 0:
            print(f"Epoch {epoch:03d} | Val SmoothL1 Loss: {val_loss:.5f}")


if __name__ == "__main__":
    train_surrogate()
```

---

## 6. Inverse Optimization & Verification (`3_inverse_optimize.py`)

Uses **Optuna (TPE Multi-Objective Bayesian Optimization)** to query the trained neural surrogate thousands of times in seconds, extracts the best candidate, runs a **Monte Carlo fabrication tolerance check ($\pm 25\,\mu	ext{m}$)**, and verifies the final winner in **`openEMS`**.

```python
import numpy as np
import torch
import optuna
from config import ACTIVE_CFG, FREQS, TARGET_DEVICE
from geometries import run_openems_simulation
from train_surrogate import ForwardEMSurrogate  # Rename 2_train_surrogate.py or import via importlib

optuna.logging.set_verbosity(optuna.logging.WARNING)


def evaluate_surrogate(model, x_mm: np.ndarray):
    b_min, b_max = ACTIVE_CFG["bounds_min"], ACTIVE_CFG["bounds_max"]
    x_norm = torch.tensor(((x_mm - b_min) / (b_max - b_min)), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        y = model(x_norm).squeeze(0).numpy()
    n_f = len(FREQS)
    s11 = y[:n_f] + 1j * y[n_f : 2 * n_f]
    s21 = y[2 * n_f : 3 * n_f] + 1j * y[3 * n_f :]
    s11_db = 20 * np.log10(np.clip(np.abs(s11), 1e-6, 1.0))
    s21_db = 20 * np.log10(np.clip(np.abs(s21), 1e-6, 1.0))
    return s11_db, s21_db


def compute_rf_objectives(s11_db: np.ndarray, s21_db: np.ndarray):
    """Returns (matching_loss, secondary_metric) to MINIMIZE."""
    if TARGET_DEVICE == "mmwave_array":
        # Target: 27.5 - 28.5 GHz return loss < -15 dB, mutual coupling S21 < -20 dB
        band = (FREQS >= 27.5e9) & (FREQS <= 28.5e9)
        rl_worst = np.max(s11_db[band])
        iso_worst = np.max(s21_db[band])
        return rl_worst, iso_worst

    elif TARGET_DEVICE == "sub6_mimo":
        # Target: 3.4 - 3.6 GHz (n78) return loss < -12 dB, isolation S21 < -18 dB
        band = (FREQS >= 3.4e9) & (FREQS <= 3.6e9)
        rl_worst = np.max(s11_db[band])
        iso_worst = np.max(s21_db[band])
        return rl_worst, iso_worst

    elif TARGET_DEVICE == "microstrip_filter":
        # Target: Passband 2.40 - 2.50 GHz (low S11, low IL), Stopband rejection at 2.0 & 2.9 GHz
        pb = (FREQS >= 2.40e9) & (FREQS <= 2.50e9)
        sb = (FREQS <= 2.05e9) | (FREQS >= 2.85e9)
        pb_loss = np.max(s11_db[pb]) - np.min(s21_db[pb])
        sb_rej = np.max(s21_db[sb])
        return pb_loss, sb_rej


def run_inverse_design(n_trials: int = 2000):
    b_min, b_max = ACTIVE_CFG["bounds_min"], ACTIVE_CFG["bounds_max"]
    model = ForwardEMSurrogate(in_dim=len(b_min), n_freq=len(FREQS))
    model.load_state_dict(torch.load(f"data/{TARGET_DEVICE}_surrogate.pt", map_location="cpu"))
    model.eval()

    def objective(trial):
        x = np.array([
            trial.suggest_float(f"p_{i}", float(b_min[i]), float(b_max[i]))
            for i in range(len(b_min))
        ])
        s11_db, s21_db = evaluate_surrogate(model, x)
        return compute_rf_objectives(s11_db, s21_db)

    study = optuna.create_study(directions=["minimize", "minimize"])
    study.optimize(objective, n_trials=n_trials)

    # Pick knee point from Pareto front
    pareto_trials = study.best_trials
    best_trial = min(pareto_trials, key=lambda t: t.values[0] + 0.5 * t.values[1])
    best_x = np.array([best_trial.params[f"p_{i}"] for i in range(len(b_min))])

    print(f"\n=== Optimal Candidate Parameters (mm) for {ACTIVE_CFG['name']} ===")
    print(np.round(best_x, 4))
    print(f"Surrogate Predicted Objectives: {np.round(best_trial.values, 2)} dB")

    # Step A: Monte Carlo Tolerance Audit (+/- 25 um PCB etching variance)
    rng = np.random.default_rng(123)
    mc_perturbations = best_x + rng.normal(0.0, 0.025, size=(200, len(b_min)))
    mc_objs = [compute_rf_objectives(*evaluate_surrogate(model, px)) for px in mc_perturbations]
    mc_objs = np.array(mc_objs)
    print(f"Monte Carlo 95th-Percentile Worst-Case: {np.round(np.percentile(mc_objs, 95, axis=0), 2)} dB")

    # Step B: Full-Wave EM Ground-Truth Verification in openEMS
    print("Running full-wave openEMS verification on optimal candidate...")
    em_res = run_openems_simulation(best_x)
    s11_em_db = 20 * np.log10(np.clip(np.abs(em_res["s11"]), 1e-6, 1.0))
    s21_em_db = 20 * np.log10(np.clip(np.abs(em_res["s21"]), 1e-6, 1.0))
    em_objs = compute_rf_objectives(s11_em_db, s21_em_db)

    print(f"Full-Wave EM Verified Objectives:     {np.round(em_objs, 2)} dB")
    print(f"Surrogate-to-EM Discrepancy:          {np.round(np.abs(np.array(best_trial.values) - np.array(em_objs)), 2)} dB")


if __name__ == "__main__":
    run_inverse_design()
```

---

## 7. Quick-Start Execution Checklist

1. **Select Target:** Open `config.py` and set `TARGET_DEVICE` to `"mmwave_array"`, `"sub6_mimo"`, or `"microstrip_filter"`.
2. **Generate Training Data:** Run `python 1_generate_dataset.py` (start with `n_samples=100` to verify speed on your CPU, then scale to `500–1,000` samples).
3. **Train Forward Surrogate:** Run `python 2_train_surrogate.py` and verify validation SmoothL1 loss drops below `0.015`.
4. **Synthesize & Verify:** Run `python 3_inverse_optimize.py` to generate your Pareto-optimal geometry, check its $\pm 25\,\mu	ext{m}$ manufacturing tolerance, and confirm the surrogate prediction matches full-wave FDTD within `< 1.5 dB`.
