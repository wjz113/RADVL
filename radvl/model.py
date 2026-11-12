"""Public RADVL model interface.

The implementation composes similarity and association representations,
discrepancy-guided fusion, and a separate relation-augmented prediction score.
The supporting modules retain the original numerical operations.
"""
from types import SimpleNamespace

from . import training_relation_gating as relation_gating
from . import model_options


def configure_model(parameters):
    """Apply model settings without changing the caller's configuration."""
    from . import (
        training_discrepancy,
        training_variance,
        training_readout,
        training_token_attention,
        training_context,
        training_masking,
        training_relations,
    )
    modules = (
        training_discrepancy,
        training_variance,
        training_readout,
        training_token_attention,
        training_context,
        training_masking,
        training_relations,
        relation_gating,
        model_options,
    )
    for module in modules:
        module.configure({key: parameters[key] for key in module.DEFAULTS if key in parameters})


class RADVL(model_options.RADVL):
    """Relation-Augmented Dual-View Learning with Discrepancy Guidance.

    Parameters are the circRNA-side node count, drug count, and an argument
    namespace containing the architecture settings. The forward interface is
    shared with the training entry and returns predictions and loss components.
    """

    @classmethod
    def from_config(cls, n_circ, n_drug, parameters):
        """Construct RADVL from a configuration dictionary."""
        configure_model(parameters)
        values = dict(parameters)
        values.setdefault('pair_chunk_rows', 32)
        return cls(n_circ, n_drug, SimpleNamespace(**values))


# Training and the public interface instantiate the same final model class.
relation_gating._WEIGHTED.RADVL = RADVL
