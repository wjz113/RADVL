# RADVL code structure

All implementation files use functional snake_case names and reside in the `radvl` package. The public model interface is `radvl.model.RADVL`.

## Core model

- `model.py`: the public model class and configuration interface.
- `model_core.py`, `model_components.py`: encoder, graph, attention and variational components.
- `model_similarity.py`: similarity-view encoders and fusion components.
- `model_residual.py`: residual prediction head.
- `model_balanced.py`: deterministic aggregation and alignment handling.
- `model_options.py`, `model_variants.py`: supported component switches.
- `training_rah.py`: relation-augmented hypergraph score construction and integration.
- `training_discrepancy.py`: discrepancy-guided fusion and related objective configuration.

## Training and preparation

`train.py` invokes `python -m radvl.train_entry`. The entry combines the model settings, reference training procedure, validation-based checkpoint selection and evaluation. `prepare_inputs.py` validates externally supplied matrices and uses `radvl.data_protocol` to prepare the five-fold partition.

`features.py`, `feature_inputs.py` and `features_similarity.py` prepare the association-dependent features. The hypergraph modules provide incidence construction and propagation operators. The `objectives` modules implement the prediction and regularization terms.

## Recording and configuration

The `recorder` modules record outputs produced by future local runs; no experimental output is bundled. The `settings` modules supply internal argument defaults, and `configs/reference.json` holds the public reference configuration. Internal setting keys are retained to preserve numerical compatibility.

Source hashes are listed in `source_manifest.csv`. `dependency_graph.json` describes the renamed package imports. The external renaming audit records the previous filenames for traceability without keeping duplicate legacy code in this package.

This organization changes module and class names, import paths and documentation. It does not change the model equations or the reference configuration values.
