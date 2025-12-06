from . import datasets
from .datasets import *  # noqa: F403
from .machado2009 import (
    matrix_anomalous_trichromacy_Machado2009,
    matrix_cvd_Machado2009,
    msds_cmfs_anomalous_trichromacy_Machado2009,
)

__all__ = datasets.__all__
__all__ += [
    "matrix_anomalous_trichromacy_Machado2009",
    "matrix_cvd_Machado2009",
    "msds_cmfs_anomalous_trichromacy_Machado2009",
]
