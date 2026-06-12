# Physics-Informed LSTM for Instantaneous Modal Parameter Identification of Base-Isolated Structures

**Paper:** *A Physics-Informed LSTM-Based System Identification Technique for Instantaneous Modal Parameters of Base-Isolated Structures under Seismic Excitation*

**Authors:** Juan Orozco, Francisco Hernandez, Rodrigo Astroza

**Journal:** Earthquake Engineering & Structural Dynamics (EESD)

---

## Repository Contents

```
├── models.py                        # LSTM model architectures
├── utils.py                         # Signal processing and loss utilities
├── process_window.py                # Window-by-window training step (TF compiled)
├── SDOF_aceleracion_promedio_tf.py  # Newmark time integration (TF)
│
├── 01_CNN_feature_extraction.ipynb  # Generate CNN feature vectors from seismic data
├── 02_train.ipynb                   # Train physics-informed LSTM (main notebook)
├── 03_inference.ipynb               # Load pre-trained weights and run inference
├── 04_plots_paper.ipynb             # Reproduce all figures in the paper
│
├── data/
│   ├── BI-BNCS_ICA100/  ← Ica station, 2007 Pisco (Peru) earthquake, 100% amplitude
│   ├── BI-BNCS_ICA50/   ← Ica station, 2007 Pisco (Peru) earthquake, 50% amplitude
│   ├── BI-BNCS_ICA140/  ← Ica station, 2007 Pisco (Peru) earthquake, 140% amplitude
│   ├── BI-BNCS_CNP100/  ← Canoga Park station, 1994 Northridge (CA) earthquake
│   ├── BI-BNCS_SP100/   ← San Pedro station, 2010 Maule (Chile) earthquake
│   └── BI-BNCS_LAC100/  ← LA City Terrace station, 1994 Northridge (CA) earthquake
│
└── results/
    └── BI-BNCS_ICA100_acc500_disp100_NM2_NDOF6_AV_Hiddim64/
        ├── best_model_mode1_*.weights.h5   ← Pre-trained weights (Mode 1)
        ├── best_model_mode2_*.weights.h5   ← Pre-trained weights (Mode 2)
        ├── best_model_contrib_*.weights.h5 ← Pre-trained weights (contribution net)
        ├── results_*.npz                   ← Saved predictions and true values
        └── losses_*.npz                    ← Training loss history
```

Each `data/` subfolder contains:
- `a1.txt` – `a6.txt`: floor absolute accelerations (m/s², sampled at 200 Hz)
- `d1.txt` – `d6.txt`: floor displacements (m)
- `u.txt`: ground acceleration input (m/s²)
- `du.txt`: ground displacement (m)
- `lstm_features_*V2.npy`: pre-extracted CNN feature vector (see Step 3)

---

## EESD Data and Code Availability

This repository meets **Tier 1 (Minimum Mandatory Standard)**:
- Test/validation data: included in `data/` (all six seismic cases)
- Pre-trained weights for the main case: included in `results/`
- Reviewer-ready runnable implementation: follow Steps 1–4 below

---

## Requirements

- Python 3.10–3.11
- TensorFlow 2.13+
- CUDA-capable GPU recommended (CPU training is slow for 7000 epochs)

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## Step-by-Step Instructions to Reproduce Key Results

### Step 1 — Clone the repository

```bash
git clone https://github.com/<your-username>/physics-informed-lstm-base-isolation.git
cd physics-informed-lstm-base-isolation
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — (Optional) Regenerate CNN feature vectors

The `lstm_features_*V2.npy` files are already included in `data/`. Skip this step unless you want to regenerate them.

Open `01_CNN_feature_extraction.ipynb` and set `SEISMIC_CASE` to the desired case, then run all cells. The output `.npy` file will be saved directly into the corresponding `data/` folder.

> **Important:** Run Jupyter from the **repository root**, not from inside `data/`.

### Step 4 — Run inference with pre-trained weights (fastest path)

Open `03_inference.ipynb`. The default settings point to the pre-trained weights for the main case (`BI-BNCS_ICA100`, 2 modes, 6 DOF, hidden dim 64). Run all cells to reproduce the instantaneous frequency and damping predictions shown in the paper.

```
SEISMIC_CASE   = 'BI-BNCS_ICA100'
RESULTS_FOLDER = 'BI-BNCS_ICA100_acc500_disp100_NM2_NDOF6_AV_Hiddim64'
```

### Step 5 — Train from scratch

Open `02_train.ipynb`. The **USER SETTINGS** block at the top contains all parameters. Default settings reproduce the main paper case:

```python
seismic_input      = "BI-BNCS_ICA100"   # seismic case in data/
N_MODES            = 2                   # number of modes to identify
N_DOF              = 6                   # degrees of freedom
NUM_EPOCHS         = 7000
current_hidden_dim = 64
ACC_LOSS_WEIGHT    = 5
DISPL_LOSS_WEIGHT  = 1
```

Run all cells. Trained weights and loss files are saved to `results/<folder_name>/`.

To train a different case, change `seismic_input` to any folder name in `data/` (e.g., `"BI-BNCS_ICA50"`).

> **Note:** Training 7000 epochs takes approximately 2–4 hours on a modern GPU. Set `NUM_EPOCHS = 100` for a quick smoke test.

### Step 6 — Reproduce paper figures

Open `04_plots_paper.ipynb` and run all cells. Figures are generated from the `.npz` result files in `results/`.

---

## Model Architecture

The method uses two coupled networks:

- **`PhysicsInformedLSTM`** — one instance per mode; outputs instantaneous damping ratio ξ(t) and natural frequency ω(t) for each time window
- **`PhysicsInformedLSTM_Contributions`** — outputs modal contribution factors φ(t) for each DOF

Both are trained end-to-end via a physics-based loss that penalises discrepancies in reconstructed floor accelerations and displacements against measured responses. The SDOF Newmark integrator (`SDOF_aceleracion_promedio_tf.py`) is embedded inside the training loop and differentiable through `tf.function`.

---

## Citation

If you use this code, please cite:

> Orozco, J., Hernández, F., Astroza, R. (2025). *A Physics-Informed LSTM-Based System Identification Technique for Instantaneous Modal Parameters of Base-Isolated Structures under Seismic Excitation*. Earthquake Engineering & Structural Dynamics. (under review)

---

## License

MIT License — see `LICENSE` for details.
