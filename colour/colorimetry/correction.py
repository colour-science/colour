# -*- coding: utf-8 -*-
"""
Spectral Bandpass Dependence Correction
=======================================

Defines objects to perform spectral bandpass dependence correction.

The following correction methods are available:

-   :func:`colour.colorimetry.bandpass_correction_Stearns1988`:
    *Stearns and Stearns (1988)* spectral bandpass dependence correction
    method.
-   :attr:`colour.BANDPASS_CORRECTION_METHODS`: Supported spectral bandpass
    dependence correction methods.
-   :func:`colour.bandpass_correction`: Spectral bandpass dependence
    correction using given method.

See Also
--------
`Spectral Bandpass Dependence Correction Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/colorimetry/correction.ipynb>`_

References
----------
-   :cite:`Stearns1988a` : Stearns, E. I., & Stearns, R. E. (1988). An example
    of a method for correcting radiance data for Bandpass error. Color Research
    & Application, 13(4), 257-259. doi:10.1002/col.5080130410
-   :cite:`Westland2012f` : Westland, S., Ripamonti, C., & Cheung, V. (2012).
    Correction for Spectral Bandpass. In Computational Colour Science Using
    MATLAB (2nd ed., p. 38). ISBN:978-0-470-66569-5
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.utilities import CaseInsensitiveMapping

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'bandpass_correction_Stearns1988', 'BANDPASS_CORRECTION_METHODS',
    'bandpass_correction'
]

ALPHA_STEARNS = 0.083


def bandpass_correction_Stearns1988(sd):
    """
    Implements spectral bandpass dependence correction on given spectral
    distribution using *Stearns and Stearns (1988)* method.

    Parameters
    ----------
    sd : SpectralDistribution
        Spectral distribution.

    Returns
    -------
    SpectralDistribution
        Spectral bandpass dependence corrected spectral distribution.

    References
    ----------
    :cite:`Stearns1988a`, :cite:`Westland2012f`

    Examples
    --------
    >>> from colour import SpectralDistribution
    >>> from colour.utilities import numpy_print_options
    >>> data = {
    ...     500: 0.0651,
    ...     520: 0.0705,
    ...     540: 0.0772,
    ...     560: 0.0870,
    ...     580: 0.1128,
    ...     600: 0.1360
    ... }
    >>> with numpy_print_options(suppress=True):
    ...     bandpass_correction_Stearns1988(
    ...         SpectralDistribution(data))
    ... # doctest: +ELLIPSIS
    SpectralDistribution([[ 500.        ,    0.0646518...],
                          [ 520.        ,    0.0704293...],
                          [ 540.        ,    0.0769485...],
                          [ 560.        ,    0.0856928...],
                          [ 580.        ,    0.1129644...],
                          [ 600.        ,    0.1379256...]],
                         interpolator=SpragueInterpolator,
                         interpolator_args={},
                         extrapolator=Extrapolator,
                         extrapolator_args={...})
    """

    values = np.copy(sd.values)
    values[0] = (1 + ALPHA_STEARNS) * values[0] - ALPHA_STEARNS * values[1]
    values[-1] = (1 + ALPHA_STEARNS) * values[-1] - ALPHA_STEARNS * values[-2]
    for i in range(1, len(values) - 1):
        values[i] = (-ALPHA_STEARNS * values[i - 1] +
                     (1 + 2 * ALPHA_STEARNS) * values[i] -
                     ALPHA_STEARNS * values[i + 1])

    sd.values = values

    return sd


BANDPASS_CORRECTION_METHODS = CaseInsensitiveMapping({
    'Stearns 1988': bandpass_correction_Stearns1988
})
BANDPASS_CORRECTION_METHODS.__doc__ = """
Supported spectral bandpass dependence correction methods.

BANDPASS_CORRECTION_METHODS : CaseInsensitiveMapping
    **{'Stearns 1988', }**
"""


def bandpass_correction(sd, method='Stearns 1988'):
    """
    Implements spectral bandpass dependence correction on given spectral
    distribution using given method.

    Parameters
    ----------
    sd : SpectralDistribution
        Spectral distribution.
    method : unicode, optional
        ('Stearns 1988', )
        Correction method.

    Returns
    -------
    SpectralDistribution
        Spectral bandpass dependence corrected spectral distribution.
    """

    return BANDPASS_CORRECTION_METHODS.get(method)(sd)
