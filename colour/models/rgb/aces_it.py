# -*- coding: utf-8 -*-
"""
Academy Color Encoding System - Input Transform
===============================================

Defines the *Academy Color Encoding System* (ACES) *Input Transform* utilities:

-   :func:`colour.sd_to_aces_relative_exposure_values`

See Also
--------
`RGB Colourspaces Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
-   :cite:`Forsythe2018` : Forsythe, A. (2018). Private Discussion with
    Mansencal, T.
-   :cite:`TheAcademyofMotionPictureArtsandSciences2014q` : The Academy of
    Motion Picture Arts and Sciences, Science and Technology Council, & Academy
    Color Encoding System (ACES) Project Subcommittee. (2014). Technical
    Bulletin TB-2014-004 - Informative Notes on SMPTE ST 2065-1 - Academy Color
    Encoding Specification (ACES). Retrieved from
    https://github.com/ampas/aces-dev/tree/master/documents
-   :cite:`TheAcademyofMotionPictureArtsandSciences2014r` : The Academy of
    Motion Picture Arts and Sciences, Science and Technology Council, & Academy
    Color Encoding System (ACES) Project Subcommittee. (2014). Technical
    Bulletin TB-2014-012 - Academy Color Encoding System Version 1.0 Component
    Names. Retrieved from
    https://github.com/ampas/aces-dev/tree/master/documents
-   :cite:`TheAcademyofMotionPictureArtsandSciencese` : The Academy of Motion
    Picture Arts and Sciences, Science and Technology Council, & Academy Color
    Encoding System (ACES) Project Subcommittee. (n.d.). Academy Color Encoding
    System. Retrieved February 24, 2014, from
    http://www.oscars.org/science-technology/council/projects/aces.html
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.colorimetry import ILLUMINANTS_SDS, sd_to_XYZ
from colour.models import XYZ_to_xy
from colour.models.rgb import (ACES_2065_1_COLOURSPACE, ACES_RICD, RGB_to_XYZ,
                               XYZ_to_RGB, normalised_primary_matrix)
from colour.utilities import from_range_1, tsplit

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'FLARE_PERCENTAGE', 'S_FLARE_FACTOR', 'sd_to_aces_relative_exposure_values'
]

FLARE_PERCENTAGE = 0.00500
S_FLARE_FACTOR = 0.18000 / (0.18000 + FLARE_PERCENTAGE)


def sd_to_aces_relative_exposure_values(
        sd,
        illuminant=ILLUMINANTS_SDS['D65'],
        apply_chromatic_adaptation=False,
        chromatic_adaptation_transform='CAT02'):
    """
    Converts given spectral distribution to *ACES2065-1* colourspace relative
    exposure values.

    Parameters
    ----------
    sd : SpectralDistribution
        Spectral distribution.
    illuminant : SpectralDistribution, optional
        *Illuminant* spectral distribution.
    apply_chromatic_adaptation : bool, optional
        Whether to apply chromatic adaptation using given transform.
    chromatic_adaptation_transform : unicode, optional
        **{'CAT02', 'XYZ Scaling', 'Von Kries', 'Bradford', 'Sharp',
        'Fairchild', 'CMCCAT97', 'CMCCAT2000', 'CAT02_BRILL_CAT', 'Bianco',
        'Bianco PC'}**,
        *Chromatic adaptation* transform.

    Returns
    -------
    ndarray, (3,)
        *ACES2065-1* colourspace relative exposure values array.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``XYZ``    | [0, 100]              | [0, 1]        |
    +------------+-----------------------+---------------+

    -   The chromatic adaptation method implemented here is a bit unusual
        as it involves building a new colourspace based on *ACES2065-1*
        colourspace primaries but using the whitepoint of the illuminant that
        the spectral distribution was measured under.

    References
    ----------
    :cite:`Forsythe2018`,
    :cite:`TheAcademyofMotionPictureArtsandSciences2014q`,
    :cite:`TheAcademyofMotionPictureArtsandSciences2014r`,
    :cite:`TheAcademyofMotionPictureArtsandSciencese`

    Examples
    --------
    >>> from colour import COLOURCHECKERS_SDS
    >>> sd = COLOURCHECKERS_SDS['ColorChecker N Ohta']['dark skin']
    >>> sd_to_aces_relative_exposure_values(sd)  # doctest: +ELLIPSIS
    array([ 0.1171785...,  0.0866347...,  0.0589707...])
    >>> sd_to_aces_relative_exposure_values(sd,
    ...     apply_chromatic_adaptation=True)  # doctest: +ELLIPSIS
    array([ 0.1180766...,  0.0869023...,  0.0589104...])
    """

    shape = ACES_RICD.shape
    if sd.shape != ACES_RICD.shape:
        sd = sd.copy().align(shape)

    if illuminant.shape != ACES_RICD.shape:
        illuminant = illuminant.copy().align(shape)

    s_v = sd.values
    i_v = illuminant.values

    r_bar, g_bar, b_bar = tsplit(ACES_RICD.values)

    def k(x, y):
        """
        Computes the :math:`K_r`, :math:`K_g` or :math:`K_b` scale factors.
        """

        return 1 / np.sum(x * y)

    k_r = k(i_v, r_bar)
    k_g = k(i_v, g_bar)
    k_b = k(i_v, b_bar)

    E_r = k_r * np.sum(i_v * s_v * r_bar)
    E_g = k_g * np.sum(i_v * s_v * g_bar)
    E_b = k_b * np.sum(i_v * s_v * b_bar)

    E_rgb = np.array([E_r, E_g, E_b])

    # Accounting for flare.
    E_rgb += FLARE_PERCENTAGE
    E_rgb *= S_FLARE_FACTOR

    if apply_chromatic_adaptation:
        xy = XYZ_to_xy(sd_to_XYZ(illuminant) / 100)
        NPM = normalised_primary_matrix(ACES_2065_1_COLOURSPACE.primaries, xy)
        XYZ = RGB_to_XYZ(E_rgb, xy, ACES_2065_1_COLOURSPACE.whitepoint, NPM,
                         chromatic_adaptation_transform)
        E_rgb = XYZ_to_RGB(XYZ, ACES_2065_1_COLOURSPACE.whitepoint,
                           ACES_2065_1_COLOURSPACE.whitepoint,
                           ACES_2065_1_COLOURSPACE.XYZ_to_RGB_matrix)

    return from_range_1(E_rgb)
