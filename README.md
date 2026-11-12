# RADVL

**Relation-Augmented Dual-View Learning with Discrepancy Guidance for circRNA–drug sensitivity association prediction**

RADVL integrates molecular similarity and observed associations to prioritize potential circRNA–drug sensitivity associations. This repository provides the model implementation, input preparation and evaluation utilities, a reference configuration, and component checks.

This is a **model-code distribution**. Datasets, baseline implementations, experimental results, prediction records, figures, and pretrained model weights are not included.

## Method overview

RADVL learns two complementary representation views and coordinates them through discrepancy-guided fusion:

- **Similarity view:** nonlinear graph learning and hypergraph propagation capture local dependencies and higher-order structural information.
- **Association view:** a collaborative variational autoencoder learns circRNA and drug representations from training associations. Covariance and variance constraints control redundancy while preserving representation diversity.
- **Discrepancy-guided fusion:** cross-view context and representation discrepancies regulate adaptive fusion and alignment.
- **Relation-augmented hypergraph (RAH):** interaction and similarity-neighborhood hyperedges provide an additional association-based prediction score. This score is separate from the two learned representation views.

Association-derived inputs are constructed from the fitting associations of each fold. Held-out positive associations are excluded from those inputs.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── prepare_inputs.py          # Validate matrices and prepare evaluation partitions
├── train.py                   # Launch training and held-out evaluation
├── configs/
│   └── reference.json         # Reference model and optimization settings
├── data/
│   └── README.md              # Input specifications and data-source attribution
├── docs/
│   ├── CODE_STRUCTURE.md      # Implementation organization
│   └── dependency_graph.json
├── radvl/
│   ├── __init__.py            # Public RADVL interface
│   ├── model.py              # Public model construction
│   ├── train_entry.py        # Training entry point
│   ├── model_*.py            # Encoders, fusion and prediction components
│   ├── training_*.py         # Training and evaluation procedures
│   ├── objectives*.py        # Prediction and representation objectives
│   ├── data_*.py             # Data loading and partition handling
│   ├── features*.py          # Feature construction
│   ├── hypergraph*.py        # Hypergraph construction and propagation
│   └── recorder_*.py         # Runtime recording
├── tests/                    # Component checks
└── source_manifest.csv       # Python source paths, sizes and SHA-256 checksums
```

The public model class is `RADVL`. See [Code structure](docs/CODE_STRUCTURE.md) for the organization of the internal modules.

## Environment and installation

The recorded reference environment uses the following versions:

| Dependency | Version |
| --- | --- |
| Python | 3.8.20 |
| PyTorch | 1.13.0+cu117 |
| CUDA version associated with the PyTorch build | 11.7 |
| NumPy | 1.23.5 |
| pandas | 2.0.3 |
| SciPy | 1.10.1 |
| scikit-learn | 1.3.2 |
| PyTorch Geometric | 2.6.1 |
| torch-sparse | 0.6.16+pt113cu117 |

Run the following commands from the repository root. This example uses Conda and the CUDA 11.7 builds of PyTorch and its compiled graph dependencies:

```bash
conda create -n radvl python=3.8 -y
conda activate radvl

python -m pip install "torch==1.13.0+cu117" --extra-index-url https://download.pytorch.org/whl/cu117
python -m pip install --only-binary=:all: "torch-scatter==2.1.0+pt113cu117" "torch-sparse==0.6.16+pt113cu117" --find-links https://data.pyg.org/whl/torch-1.13.0+cu117.html
python -m pip install -r requirements.txt
```

The compiled extensions must match the Python, PyTorch, CUDA and operating-system versions. The `torch-scatter` and `torch-sparse` installations are shown explicitly because `requirements.txt` alone does not install this complete compiled dependency stack. Consult the [official PyTorch previous-version instructions](https://pytorch.org/get-started/previous-versions/) and the [PyG wheel index for PyTorch 1.13.0 and CUDA 11.7](https://data.pyg.org/whl/torch-1.13.0%2Bcu117.html) when configuring another platform.

Check the imports after installation:

```bash
python -c "import torch, torch_sparse; from radvl import RADVL; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

The training code selects CUDA when available and otherwise selects CPU. The recorded environment uses the CUDA stack above; equivalent full-training behavior in other environments has not been established by the component checks.

## Input data

Place the following processed CSV matrices in `data/`, or provide another directory with `--data-dir`:

| File | Shape | Meaning |
| --- | --- | --- |
| `association.csv` | N × M | Binary circRNA-side source-label–drug association matrix |
| `gene_seq_sim.csv` | N × N | Static host-gene sequence similarity matrix |
| `drug_str_sim.csv` | M × M | Static drug structural similarity matrix |

N is the number of circRNA-side source labels, and M is the number of drugs. The reference processed benchmark contains 271 circRNA-side labels, 218 drugs and 4,134 positive associations.

All three CSV files must include a header row and an identifier column. The first column is read as the row index. For example, the association matrix has the following layout:

```csv
source_label,drug_A,drug_B,drug_C
label_A,1,0,0
label_B,0,1,0
```

These identifiers illustrate the file format; they are not benchmark identifiers. Supply complete matrices for your own data.

The input validator requires:

1. Association values are finite and binary, with `1` denoting a recorded positive association and `0` denoting an unlabeled pair.
2. Both axes of `gene_seq_sim.csv` exactly match the association-row identifiers and their order.
3. Both axes of `drug_str_sim.csv` exactly match the association-column identifiers and their order.
4. All similarity entries are finite, and the matrix dimensions agree.

Zero entries are not experimentally verified negative associations. They form the unlabeled pool used by the sampling protocol.

CircRNA-side benchmark labels are CircRiC source labels annotated by host genes. A host-gene label does not uniquely resolve a circular transcript or its back-splice junction (BSJ). Accordingly, `gene_seq_sim.csv` describes host-gene sequence similarity. See [Input data and source attribution](data/README.md) for the biological resolution and upstream sources.

## Quick start

After preparing the three input matrices, validate them and generate the evaluation partitions:

```bash
python prepare_inputs.py --split-seed 1234
```

Then run RADVL:

```bash
python train.py --config configs/reference.json --split-seed 1234
```

Preparation creates the protocol under `protocols/edge_val125_seed1234/`. The training launcher requires its `manifest.json` and writes new runtime outputs under `outputs/` by default.

For data stored elsewhere, pass the same input directory and split seed to both commands:

```bash
python prepare_inputs.py --data-dir "D:/RADVL_inputs" --split-seed 1234
python train.py --data-dir "D:/RADVL_inputs" --config configs/reference.json --split-seed 1234 --output "D:/RADVL_runs"
```

Replace the example paths with your own directories. To inspect the available launcher arguments:

```bash
python prepare_inputs.py --help
python train.py --help
```

### Repeated partitions

Each split seed defines one five-fold partition. To run five repeated partitions with seeds 1234–1238 in PowerShell:

```powershell
foreach ($splitSeed in 1234..1238) {
    python prepare_inputs.py --split-seed $splitSeed
    if ($LASTEXITCODE -ne 0) { throw "Input preparation failed for split $splitSeed" }
    python train.py --split-seed $splitSeed
    if ($LASTEXITCODE -ne 0) { throw "Training failed for split $splitSeed" }
}
```

The corresponding Bash commands are:

```bash
for split_seed in 1234 1235 1236 1237 1238; do
    python prepare_inputs.py --split-seed "$split_seed" || break
    python train.py --split-seed "$split_seed" || break
done
```

Five repetitions of five-fold evaluation produce 25 fold evaluations. The partition seed supplied by `--split-seed` is distinct from the model-training seed stored as `seed` in the configuration.

### Evaluation protocol

The supplied preparation script implements the `edge_val125` protocol:

- Recorded positive associations are divided into five outer folds.
- Each outer test fold contains approximately one fifth of the positives. One eighth of the remaining positives is reserved for validation, giving approximate fitting/validation/test proportions of 70%/10%/20%.
- Each evaluated subset uses a 1:1 ratio of positive and sampled unlabeled pairs.
- Association-derived inputs are prepared from the fitting associations within each fold.
- Model selection uses validation AUC. Test metrics are computed after restoring the selected model state.

The preparation wrapper generates this edge-partition protocol. Additional utilities inside the package do not imply that the wrapper automatically runs cold-start, case-study or external-benchmark experiments.

## Reference configuration

Model and optimization settings are read from [configs/reference.json](configs/reference.json). Copy this file to a separate configuration file when changing a setting, and pass that file through `--config`.

| Configuration key | Reference value | Description |
| --- | --- | --- |
| `seed` | 1234 | Model-training seed |
| `validation` | 5 | Number of outer folds |
| `epoch` | 800 | Maximum training epochs |
| `min_epochs` | 100 | First epoch eligible for model selection |
| `patience` | 40 | Early-stopping patience |
| `learning_rate` | 0.002 | Optimizer learning rate |
| `weight_decay` | 0.0005 | Optimizer weight decay |
| `dropout` | 0.1 | Dropout probability |
| `hid_dim` | 128 | Hidden representation dimension |
| `latent_dim` | 123 | Variational latent dimension |
| `vae_hidden_dim` | 512 | Variational encoder hidden dimension |
| `hypergraph_width` | 256 | Hypergraph branch width |
| `gin_layers` / `hgcn_layers` | 2 / 2 | Graph and hypergraph depths |
| `neighbors` / `clusters` | 8 / 9 | Neighborhood and clustering settings |
| `prediction_head` | `residual_mlp` | Prediction-head option |
| `ema_decay` | 0.99 | Exponential-moving-average decay |
| `protocol_name` | `edge_val125` | Prepared evaluation protocol |

The configuration also specifies the representation objectives, discrepancy gates, fusion and RAH settings. Internal configuration names beginning with `hap` are retained for implementation compatibility; they configure the RAH score branch rather than an additional representation view.

Keep the input matrices unchanged after preparing a partition. The stored protocol records input identity and partition settings, and incompatible inputs are rejected. For another dataset, use a separate working copy or an explicitly separate protocol location through the underlying protocol utilities.

## Python interface

Construct the public model from the reference settings:

```python
import json
from radvl import RADVL

with open("configs/reference.json", encoding="utf-8") as handle:
    settings = json.load(handle)

model = RADVL.from_config(
    n_circ=271,
    n_drug=218,
    parameters=settings,
)
```

Set `n_circ` and `n_drug` to the dimensions of your input matrices. This example constructs an untrained model; it does not load a fitted state or generate predictions.

A complete forward computation requires fold-specific association, graph and hypergraph inputs, together with the bound RAH score matrices. The training pipeline constructs these inputs and manages fitting, selection and evaluation. Use `train.py` for the complete workflow.

## Runtime outputs and metrics

The launcher records the effective settings in `parameters_split<seed>.json` under the requested output directory and sends the corresponding run to its `split<seed>/` subdirectory. The runner manages the nested scenario/run recording directories.

Depending on the recording stage, generated files include:

| File | Content |
| --- | --- |
| `run_config.json` | Run settings |
| `fold_metrics.csv` | Held-out metrics for individual folds |
| `summary_metrics.csv` | Aggregated fold metrics |
| `training_history.csv` | Training history |
| `validation_history.csv` | Validation history |
| `early_stopping_summary.csv` | Model-selection and stopping information |
| `protocol_audit.csv` | Partition and input audit records |
| `true_<fold>.npy`, `pred_<fold>.npy` | Held-out labels and prediction scores |

`AUC` is the area under the receiver operating characteristic curve. `AUPR` is computed by trapezoidal integration of the precision–recall curve; it is not the average-precision score. Threshold-based metrics use a threshold of 0.5. The aggregate recorder uses population standard deviations; calculate sample standard deviations from fold records with `ddof=1` when that convention is required.

Checkpoint-file saving is **off by default**. To save weights from your own future run:

```bash
python train.py --split-seed 1234 --save-checkpoints on
```

The default workflow still keeps and restores the selected state in memory for held-out evaluation. No fitted weights are supplied with this repository. Generated outputs and checkpoints are local runtime artifacts and are excluded by `.gitignore`.

## Component checks

Run these checks from the repository root:

```bash
python -m tests.test_balanced_objective
python -m tests.test_gradient_diagnostics
python -m tests.test_model_options
python -m tests.test_residual_prediction
python -m tests.test_similarity_fusion
```

These checks exercise objectives, gradient diagnostics, model options, residual prediction and similarity fusion. They do not replace evaluation on the required data or reproduce the manuscript's performance tables by themselves.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| A required CSV file is missing | Supply all three matrices and use the same `--data-dir` for preparation and training. |
| An identifier-order or dimension check fails | Align the similarity axes to the association row/column order and preserve identifiers during CSV export. |
| The protocol manifest is missing | Run `prepare_inputs.py` with the same split seed before training. |
| Stored protocol settings or data do not match | Keep the original inputs for that protocol, or prepare a distinct protocol location for the new dataset. |
| `torch_sparse` or `torch_scatter` cannot be imported | Install the compiled wheels matching the active Python/PyTorch/CUDA environment. |
| CUDA is unavailable | Check the active PyTorch build and GPU driver. The runner otherwise selects CPU. |

## Attribution and citation

Input specifications are derived from the GATECDA data release and the CircRiC resource. See [data/README.md](data/README.md) for the corresponding publications and biological interpretation. Public-source data and third-party software remain subject to their original providers' terms.

If you use RADVL in academic work, cite the associated manuscript, **“Relation-Augmented Dual-View Learning with Discrepancy Guidance for Predicting Circular Ribonucleic Acid-Drug Sensitivity Associations.”** Use the final bibliographic details from the published article when available.

This model-code distribution covers the implementation and its usage. It does not constitute the complete collection of data, prediction records and biological evidence materials described in the manuscript's Supporting Information.
