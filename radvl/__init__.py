"""RADVL model, input preparation, and training utilities."""

__all__ = ['RADVL']


def __getattr__(name):
    if name == 'RADVL':
        from .model import RADVL
        return RADVL
    raise AttributeError(name)
