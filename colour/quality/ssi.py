# -*- coding: utf-8 -*-
"""
Academy Spectral Similarity Index (SSI)
=======================================

Defines the *Academy Spectral Similarity Index* (SSI) computation objects:

-   :func:`colour.colour_quality_scale`

See Also
--------
`Academy Spectral Similarity Index Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/quality/ssi.ipynb>`_

References
----------
-   :cite:`TheAcademyofMotionPictureArtsandSciences2019` : The Academy of
    Motion Picture Arts and Sciences. (2019). Academy Spectral Similarity
    Index (SSI): Overview.
"""

from __future__ import division, unicode_literals

import numpy as np
from scipy.ndimage.filters import convolve1d

from colour.algebra import LinearInterpolator
from colour.colorimetry import SpectralShape

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['SSI_SPECTRAL_SHAPE', 'spectral_similarity_index']

SSI_SPECTRAL_SHAPE = SpectralShape(375, 675, 1)
"""
*Academy Spectral Similarity Index* (SSI) spectral shape.

SSI_SPECTRAL_SHAPE : SpectralShape
"""

_SSI_LARGE_SPECTRAL_SHAPE = SpectralShape(380, 670, 10)

_INTEGRATION_MATRIX = None


def spectral_similarity_index(sd_test, sd_reference):
    """
    Returns the *Academy Spectral Similarity Index* (SSI) of given test
    spectral distribution with given reference spectral distribution.

    Parameters
    ----------
    sd_test : SpectralDistribution
        Test spectral distribution.
    sd_reference : SpectralDistribution
        Reference spectral distribution.

    Returns
    -------
    numeric
        *Academy Spectral Similarity Index* (SSI).

    References
    ----------
    :cite:`TheAcademyofMotionPictureArtsandSciences2019`

    Examples
    --------
    >>> from colour import ILLUMINANTS_SDS
    >>> sd_test = ILLUMINANTS_SDS['C']
    >>> sd_reference = ILLUMINANTS_SDS['D65']
    >>> spectral_similarity_index(sd_test, sd_reference)
    94.0
    """

    global _INTEGRATION_MATRIX

    if _INTEGRATION_MATRIX is None:
        _INTEGRATION_MATRIX = np.zeros([
            len(_SSI_LARGE_SPECTRAL_SHAPE.range()),
            len(SSI_SPECTRAL_SHAPE.range())
        ])

        weights = np.array([0.5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.5])

        for i in range(_INTEGRATION_MATRIX.shape[0]):
            _INTEGRATION_MATRIX[i, (10 * i):(10 * i + 11)] = weights

    settings = {
        'interpolator': LinearInterpolator,
        'extrapolator_args': {
            'left': 0,
            'right': 0
        }
    }

    sd_test = sd_test.copy().align(SSI_SPECTRAL_SHAPE, **settings)
    sd_reference = sd_reference.copy().align(SSI_SPECTRAL_SHAPE, **settings)

    test_i = np.dot(_INTEGRATION_MATRIX, sd_test.values)
    reference_i = np.dot(_INTEGRATION_MATRIX, sd_reference.values)

    test_i /= np.sum(test_i)
    reference_i /= np.sum(reference_i)

    d_i = test_i - reference_i
    dr_i = d_i / (reference_i + np.mean(reference_i))
    wdr_i = dr_i * [
        12 / 45, 22 / 45, 32 / 45, 40 / 45, 44 / 45, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 11 / 15, 3 / 15
    ]
    c_wdr_i = convolve1d(np.hstack([0, wdr_i, 0]), [0.22, 0.56, 0.22])
    m_v = np.sum(c_wdr_i ** 2)

    SSI = np.around(100 - 32 * np.sqrt(m_v))

    return SSI
