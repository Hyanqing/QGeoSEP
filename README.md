# QGeoSEP++ User Guide

This project uses a three-dimensional molecular graph neural network for multi-property regression of energetic molecules. It combines geometry-enhanced representations, multi-head graph attention incorporating edge features, and a knowledge embedding module that incorporates expert knowledge such as oxygen balance. Dynamic loss weighting based on predictive uncertainty further supports joint learning across multiple properties.

This guide follows the scripts in this repository's `scripts/` directory. Configuration files are read from `./config/`, and checkpoints and runtime outputs are stored under `./outs/`.

## 1. Predicted Properties and Model Structure

The provided final checkpoint uses the following property order. Do not change the order or capitalization when using this checkpoint:

```text
dv fs is Isp dt d dp hof q
```

| Identifier | Property |
| --- | --- |
| `dv` | Detonation velocity |
| `fs` | Friction sensitivity |
| `is` | Impact sensitivity |
| `Isp` | Specific impulse |
| `dt` | Decomposition temperature |
| `d` | Density |
| `dp` | Detonation pressure |
| `hof` | Heat of formation |
| `q` | Heat of explosion |

## 2. Repository Structure

```text
.
|-- configs/
|   |-- config.json                  # Atom, bond, and pretraining task configuration
|   |-- ems_9_3_endpoint.json         # Property counts (num) and proportions (sample)
|   `-- ems_9_3_status.json           # Mean/std associated with the provided final checkpoint
|-- data/
|   `-- ems-9_3/
|       |-- d.csv                    # Training data; one CSV per property
|       |-- ...
|       `-- x.csv                    # Inference input: no header, SMILES in the first column
|-- datasets/                        # Molecular graph construction and data loading
|-- models/                          # Encoders and downstream prediction models
|-- scripts/
|   |-- finetune/
|   |   |-- sub_fintune.sh           # Batch training launcher with configurable hyperparameter combinations
|   |   `-- finetune-transformer_ems.sh
|   `-- inference/
|       `-- inference_transformer.sh
|-- outs/
|   |-- checkpoints/
|   |   |-- pretrain/zinc22-10-gin/checkpoint_10.pt
|   |   |-- ems_final/checkpoint_final.pt
|   |   `-- ...                      # Newly trained checkpoints and logs
|   `-- inference/                   # Inference CSV outputs
|-- result/                          # Existing per-property prediction results; not training inputs
|-- finetune_transformer.py
|-- inference_transformer.py
`-- README.md
```

The CSV files shown under `data/` are input files that users need to prepare.

## 3. Runtime Environment

Activate a Python environment with the model dependencies installed before running the scripts. The scripts use `python` from the current environment. They do not activate a fixed Conda environment, load cluster modules, or set GPU visibility indices.

The main dependencies include:

- PyTorch: `torch`.
- PyTorch Geometric: `torch_geometric`, plus `torch_scatter` and `torch_sparse`, which the code imports directly.
- Molecular and data processing: `rdkit`, `numpy`, and `pandas`.
- Training and metrics: `scikit-learn`, `tqdm`, and `tensorboard`.

Create and activate the environment:

```bash
conda env create -n qgem -f environment.yml
conda activate qgem
```

Check the main dependencies in the prepared environment:

```bash
python -c "import torch, torch_geometric, torch_scatter, torch_sparse, rdkit, numpy, pandas, sklearn, tqdm, tensorboard; print('Dependencies imported successfully'); print('CUDA available:', torch.cuda.is_available())"
```

## 4. Data and Configuration Preparation

### 4.1 Training Data

The default training directory is `./data/ems-9_3/`. Prepare one CSV file with a header for each property:

```text
d.csv  dp.csv  dt.csv  dv.csv  fs.csv  hof.csv  is.csv  Isp.csv  q.csv
```

Each CSV must contain at least a `smiles` column and a property column matching its filename. For example, the header of `d.csv` is:

```text
smiles,d
```

### 4.2 Inference Data

The default input is `./data/ems-9_3/x.csv`: **no header, with SMILES in the first column**. Example format:

```text
CC(=O)OC1=CC=CC=C1C(=O)O
CCOC(=O)c1ccccc1
```

### 4.3 Property Configuration Files

| File | Contents | Purpose |
| --- | --- | --- |
| `config/ems_9_3_endpoint.json` | `num`, `sample` | Counts and proportions read by the training entry point; currently, only the `--use-old` branch uses `sample` to calculate gamma |
| `config/ems_9_3_status.json` | `mean`, `std` | Inverse normalization of inference predictions |

## 5. Training

### 5.1 Default Training

Run the following command from the project root:

```bash
bash scripts/finetune/sub_fintune.sh
```

For CPU debugging, use:

```bash
DEVICE=cpu NUM_WORKERS=0 EPOCHS=1 \
bash scripts/finetune/finetune-transformer_ems.sh
```

### 5.2 Training Outputs

The default output path is generated from the hyperparameters. For example:

```text
outs/checkpoints/ems-9_3_transformer_3/100M_128_120/
`-- gin-L3-HS512-D0.1-ELR1e-3-LR1e-4-2017-v3/
    |-- checkpoint_training_detail.pt
    |-- checkpoint_1.pt
    |-- ...
    |-- checkpoint_120.pt
    |-- log.txt
    `-- events.out.tfevents...
```

```bash
tensorboard --logdir ./outs/checkpoints
```

## 6. Inference and Application

### 6.1 Using the Provided Final Checkpoint

After preparing the input file, run:

```bash
bash scripts/inference/inference_transformer.sh
```

The default checkpoint, input, and output paths are:

```text
Checkpoint: ./outs/checkpoints/ems_final/checkpoint_final.pt
Input:      ./data/ems-9_3/x.csv
Output:     ./outs/inference/x.csv
```

### 6.2 Custom Input, Output, and Device

```bash
DEVICE=cpu NUM_WORKERS=0 \
SMILES_FILE=./data/ems-9_3/x.csv \
OUT_FILE=./outs/inference/predictions.csv \
bash scripts/inference/inference_transformer.sh 2017
```
