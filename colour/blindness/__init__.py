#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

import sys

from colour.utilities.deprecation import ModuleAPI, build_API_changes
from colour.utilities.documentation import is_documentation_building

from .datasets import *  # noqa
from . import datasets
from .machado2009 import (msds_cmfs_anomalous_trichromacy_Machado2009,
                          matrix_anomalous_trichromacy_Machado2009,
                          matrix_cvd_Machado2009)

__all__ = []
__all__ += datasets.__all__
__all__ += [
    'msds_cmfs_anomalous_trichromacy_Machado2009',
    'matrix_anomalous_trichromacy_Machado2009', 'matrix_cvd_Machado2009'
]


# ----------------------------------------------------------------------------#
# ---                API Changes and Deprecation Management                ---#
# ----------------------------------------------------------------------------#
class blindness(ModuleAPI):
    def __getattr__(self, attribute):
        return super(blindness, self).__getattr__(attribute)


# v0.3.16
API_CHANGES = {
    'ObjectRenamed': [
        [
            'colour.blindness.anomalous_trichromacy_cmfs_Machado2009',
            'colour.blindness.msds_cmfs_anomalous_trichromacy_Machado2009',
        ],
        [
            'colour.blindness.anomalous_trichromacy_matrix_Machado2009',
            'colour.blindness.matrix_anomalous_trichromacy_Machado2009',
        ],
        [
            'colour.blindness.cvd_matrix_Machado2009',
            'colour.blindness.matrix_cvd_Machado2009',
        ],
    ]
}
"""
Defines *colour.blindness* sub-package API changes.

API_CHANGES : dict
"""

if not is_documentation_building():
    sys.modules['colour.blindness'] = blindness(
        sys.modules['colour.blindness'], build_API_changes(API_CHANGES))

    del ModuleAPI, is_documentation_building, build_API_changes, sys
