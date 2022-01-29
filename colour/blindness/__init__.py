#!/usr/bin/env python

from .datasets import *  # noqa
from . import datasets
from .machado2009 import (
    msds_cmfs_anomalous_trichromacy_Machado2009,
    matrix_anomalous_trichromacy_Machado2009,
    matrix_cvd_Machado2009,
)

__all__ = []
__all__ += datasets.__all__
__all__ += [
    "msds_cmfs_anomalous_trichromacy_Machado2009",
    "matrix_anomalous_trichromacy_Machado2009",
    "matrix_cvd_Machado2009",
]
