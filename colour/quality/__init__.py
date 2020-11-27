# -*- coding: utf-8 -*-

from __future__ import absolute_import

import sys

from colour.utilities.deprecation import ModuleAPI, build_API_changes
from colour.utilities.documentation import is_documentation_building

from .datasets import *  # noqa
from . import datasets
from .cfi2017 import (ColourRendering_Specification_CIE2017,
                      colour_fidelity_index_CIE2017)
from .cri import ColourRendering_Specification_CRI, colour_rendering_index
from .cqs import (COLOUR_QUALITY_SCALE_METHODS,
                  ColourRendering_Specification_CQS, colour_quality_scale)
from .ssi import spectral_similarity_index
from .tm3018 import (ColourQuality_Specification_ANSIIESTM3018,
                     colour_fidelity_index_ANSIIESTM3018)
from colour.utilities import CaseInsensitiveMapping

__all__ = []
__all__ += datasets.__all__
__all__ += [
    'ColourRendering_Specification_CIE2017', 'colour_fidelity_index_CIE2017'
]
__all__ += [
    'ColourQuality_Specification_ANSIIESTM3018',
    'colour_fidelity_index_ANSIIESTM3018'
]
__all__ += ['ColourRendering_Specification_CRI', 'colour_rendering_index']
__all__ += [
    'ColourRendering_Specification_CQS', 'COLOUR_QUALITY_SCALE_METHODS',
    'colour_quality_scale'
]
__all__ += ['spectral_similarity_index']

COLOUR_FIDELITY_INDEX_METHODS = CaseInsensitiveMapping({
    'CIE 2017': colour_fidelity_index_CIE2017,
    'ANSI/IES TM-30-18': colour_fidelity_index_ANSIIESTM3018,
})
COLOUR_FIDELITY_INDEX_METHODS.__doc__ = """
Supported  *Colour Fidelity Index* (CFI) computation methods.

References
----------
:cite:`CIETC1-902017`, :cite:`ANSI2018`

COLOUR_FIDELITY_INDEX_METHODS : tuple
    **{'CIE 2017', 'ANSI/IES TM-30-18'}**
"""


def colour_fidelity_index(sd_test, additional_data=False, method='CIE 2017'):
    """
    Returns the *Colour Fidelity Index* (CFI) :math:`R_f` of given
    spectral distribution using given method.

    Parameters
    ----------
    sd_test : SpectralDistribution
        Test spectral distribution.
    additional_data : bool, optional
        Whether to output additional data.
    method : unicode, optional
        **{'CIE 2017', 'ANSI/IES TM-30-18'}**,
        Computation method.

    Returns
    -------
    numeric or ColourRendering_Specification_CIE2017 or \
ColourQuality_Specification_ANSIIESTM3018
        *Colour Fidelity Index* (CFI) :math:`R_f`.

    References
    ----------
    :cite:`CIETC1-902017`, :cite:`ANSI2018`

    Examples
    --------
    >>> from colour.colorimetry import SDS_ILLUMINANTS
    >>> sd = SDS_ILLUMINANTS['FL2']
    >>> colour_fidelity_index(sd)  # doctest: +ELLIPSIS
    70.1208254...
    """

    function = COLOUR_FIDELITY_INDEX_METHODS[method]

    return function(sd_test, additional_data)


__all__ += ['COLOUR_FIDELITY_INDEX_METHODS', 'colour_fidelity_index']


# ----------------------------------------------------------------------------#
# ---                API Changes and Deprecation Management                ---#
# ----------------------------------------------------------------------------#
class quality(ModuleAPI):
    def __getattr__(self, attribute):
        return super(quality, self).__getattr__(attribute)


# v0.3.16
API_CHANGES = {
    'ObjectRenamed': [
        [
            'colour.quality.CQS_Specification',
            'colour.quality.ColourRendering_Specification_CQS',
        ],
        [
            'colour.quality.CRI_Specification',
            'colour.quality.ColourRendering_Specification_CRI',
        ],
        [
            'colour.quality.TCS_SDS',
            'colour.quality.SDS_TCS',
        ],
        [
            'colour.quality.VS_SDS',
            'colour.quality.SDS_VS',
        ],
    ]
}
"""
Defines *colour.quality* sub-package API changes.

API_CHANGES : dict
"""

if not is_documentation_building():
    sys.modules['colour.quality'] = quality(sys.modules['colour.quality'],
                                            build_API_changes(API_CHANGES))

    del ModuleAPI, is_documentation_building, build_API_changes, sys
