#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Academy Color Encoding System - Input Transform
===============================================

Defines the *Academy Color Encoding System* (ACES) *Input Transform* utilities:

-   :func:`spectral_to_aces_relative_exposure_values`

See Also
--------
`RGB Colourspaces Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
.. [1]  The Academy of Motion Picture Arts and Sciences, Science and
        Technology Council, & Academy Color Encoding System (ACES) Project
        Subcommittee. (n.d.). Academy Color Encoding System. Retrieved
        February 24, 2014, from
        http://www.oscars.org/science-technology/council/projects/aces.html
.. [2]  The Academy of Motion Picture Arts and Sciences, Science and
        Technology Council, & Academy Color Encoding System (ACES) Project
        Subcommittee. (2014). Technical Bulletin TB-2014-004 - Informative
        Notes on SMPTE ST 2065-1 – Academy Color Encoding Specification
        (ACES). Retrieved from
        https://github.com/ampas/aces-dev/tree/master/documents
.. [3]  The Academy of Motion Picture Arts and Sciences, Science and
        Technology Council, & Academy Color Encoding System (ACES) Project
        Subcommittee. (2014). Technical Bulletin TB-2014-012 - Academy Color
        Encoding System Version 1.0 Component Names. Retrieved from
        https://github.com/ampas/aces-dev/tree/master/documents
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.colorimetry import ILLUMINANTS_RELATIVE_SPDS
from colour.models.rgb import ACES_RICD

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['FLARE_PERCENTAGE',
           'S_FLARE_FACTOR',
           'spectral_to_aces_relative_exposure_values']

FLARE_PERCENTAGE = 0.00500
S_FLARE_FACTOR = 0.18000 / (0.18000 + FLARE_PERCENTAGE)


def spectral_to_aces_relative_exposure_values(
        spd,
        illuminant=ILLUMINANTS_RELATIVE_SPDS['D60']):
    """
    Converts given spectral power distribution to *ACES2065-1* colourspace
    relative exposure values.

    Parameters
    ----------
    spd : SpectralPowerDistribution
        Spectral power distribution.
    illuminant : SpectralPowerDistribution, optional
        *Illuminant* spectral power distribution.

    Returns
    -------
    ndarray, (3,)
        *ACES2065-1* colourspace relative exposure values array.

    Notes
    -----
    -   Output *ACES2065-1* colourspace relative exposure values array is in
        range [0, 1].

    See Also
    --------
    :func:`colour.colorimetry.tristimulus.spectral_to_XYZ`

    References
    ----------

    Examples
    --------
    >>> from colour import COLOURCHECKERS_SPDS
    >>> spd = COLOURCHECKERS_SPDS['ColorChecker N Ohta']['dark skin']
    >>> spectral_to_aces_relative_exposure_values(spd)  # doctest: +ELLIPSIS
    array([ 0.1187697...,  0.0870866...,  0.0589442...])
    """

    shape = ACES_RICD.shape
    if spd.shape != ACES_RICD.shape:
        spd = spd.clone().align(shape)

    if illuminant.shape != ACES_RICD.shape:
        illuminant = illuminant.clone().align(shape)

    spd = spd.values
    illuminant = illuminant.values

    r_bar, g_bar, b_bar = (ACES_RICD.r_bar.values,
                           ACES_RICD.g_bar.values,
                           ACES_RICD.b_bar.values)

    def k(x, y):
        """
        Computes the :math:`K_r`, :math:`K_g` or :math:`K_b` scale factors.
        """

        return 1 / np.sum(x * y)

    k_r = k(illuminant, r_bar)
    k_g = k(illuminant, g_bar)
    k_b = k(illuminant, b_bar)

    E_r = k_r * np.sum(illuminant * spd * r_bar)
    E_g = k_g * np.sum(illuminant * spd * g_bar)
    E_b = k_b * np.sum(illuminant * spd * b_bar)

    E_rgb = np.array([E_r, E_g, E_b])

    # Accounting for flare.
    E_rgb += FLARE_PERCENTAGE
    E_rgb *= S_FLARE_FACTOR

    return E_rgb
