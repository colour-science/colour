# -*- coding: utf-8 -*-
"""
Hunter Rd,a,b Colour Scale
==========================

Defines the *Hunter Rd,a,b* colour scale transformations:

-   :func:`colour.XYZ_to_Hunter_Rdab`
-   :func:`colour.Hunter_Rdab_to_XYZ`

See Also
--------
`Hunter Rd,a,b Colour Scale Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/hunter_rdab.ipynb>`_

References
----------
-   :cite:`HunterLab2012a` : HunterLab. (2012). Hunter Rd,a,b Color Scale -
    History and Application. Retrieved from
    https://hunterlabdotcom.files.wordpress.com/2012/07/\
an-1016-hunter-rd-a-b-color-scale-update-12-07-03.pdf
"""

from __future__ import division, unicode_literals

from colour.colorimetry import HUNTERLAB_ILLUMINANTS
from colour.models import XYZ_to_K_ab_HunterLab1966
from colour.utilities import from_range_100, to_domain_100, tsplit, tstack

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2018 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['XYZ_to_Hunter_Rdab', 'Hunter_Rdab_to_XYZ']


def XYZ_to_Hunter_Rdab(XYZ,
                       XYZ_n=HUNTERLAB_ILLUMINANTS[
                           'CIE 1931 2 Degree Standard Observer']['D50'].XYZ_n,
                       K_ab=HUNTERLAB_ILLUMINANTS[
                           'CIE 1931 2 Degree Standard Observer']['D50'].K_ab):
    """
    Converts from *CIE XYZ* tristimulus values to *Hunter Rd,a,b* colour scale.

    Parameters
    ----------
    XYZ : array_like
        *CIE XYZ* tristimulus values.
    XYZ_n : array_like, optional
        Reference *illuminant* tristimulus values.
    K_ab : array_like, optional
        Reference *illuminant* chromaticity coefficients, if ``K_ab`` is set to
        *None* it will be computed using
        :func:`colour.XYZ_to_K_ab_HunterLab1966`.

    Returns
    -------
    ndarray
        *Hunter Rd,a,b* colour scale array.

    Notes
    -----
    -   Input *CIE XYZ* and reference *illuminant* tristimulus values are
        normalised to domain [0, 100].

    References
    ----------
    -   :cite:`HunterLab2012a`

    Examples
    --------
    >>> import numpy as np
    >>> XYZ = np.array([0.07049534, 0.10080000, 0.09558313]) * 100
    >>> D50 = HUNTERLAB_ILLUMINANTS[
    ...     'CIE 1931 2 Degree Standard Observer']['D50']
    >>> XYZ_to_Hunter_Rdab(XYZ, D50.XYZ_n, D50.K_ab)
    ... # doctest: +ELLIPSIS
    array([ 10.08      , -18.6765376...,  -3.4432992...])
    """

    X, Y, Z = tsplit(to_domain_100(XYZ))
    X_n, Y_n, Z_n = tsplit(to_domain_100(XYZ_n))
    K_a, K_b = (tsplit(XYZ_to_K_ab_HunterLab1966(XYZ_n))
                if K_ab is None else tsplit(K_ab))

    f = 0.51 * ((21 + 0.2 * Y) / (1 + 0.2 * Y))
    Y_Yn = Y / Y_n

    R_d = Y
    a_Rd = K_a * f * (X / X_n - Y_Yn)
    b_Rd = K_b * f * (Y_Yn - Z / Z_n)

    R_d_ab = tstack((R_d, a_Rd, b_Rd))

    return from_range_100(R_d_ab)


def Hunter_Rdab_to_XYZ(R_d_ab,
                       XYZ_n=HUNTERLAB_ILLUMINANTS[
                           'CIE 1931 2 Degree Standard Observer']['D50'].XYZ_n,
                       K_ab=HUNTERLAB_ILLUMINANTS[
                           'CIE 1931 2 Degree Standard Observer']['D50'].K_ab):
    """
    Converts from *Hunter Rd,a,b* colour scale to *CIE XYZ* tristimulus values.

    Parameters
    ----------
    R_d_ab : array_like
        *Hunter Rd,a,b* colour scale array.
    XYZ_n : array_like, optional
        Reference *illuminant* tristimulus values.
    K_ab : array_like, optional
        Reference *illuminant* chromaticity coefficients, if ``K_ab`` is set to
        *None* it will be computed using
        :func:`colour.XYZ_to_K_ab_HunterLab1966`.

    Returns
    -------
    ndarray
        *CIE XYZ* tristimulus values.

    Notes
    -----
    -   Input reference *illuminant* tristimulus values are normalised to
        domain [0, 100].
    -   Output *CIE XYZ* tristimulus values are normalised to range [0, 100].

    References
    ----------
    -   :cite:`HunterLab2012a`

    Examples
    --------
    >>> import numpy as np
    >>> R_d_ab = np.array([10.08000000, -18.67653764, -3.44329925])
    >>> D50 = HUNTERLAB_ILLUMINANTS[
    ...     'CIE 1931 2 Degree Standard Observer']['D50']
    >>> Hunter_Rdab_to_XYZ(R_d_ab, D50.XYZ_n, D50.K_ab)
    ... # doctest: +ELLIPSIS
    array([  7.049534...,  10.08    ...,   9.558313...])
    """

    R_d, a_Rd, b_Rd = tsplit(to_domain_100(R_d_ab))
    X_n, Y_n, Z_n = tsplit(to_domain_100(XYZ_n))
    K_a, K_b = (tsplit(XYZ_to_K_ab_HunterLab1966(XYZ_n))
                if K_ab is None else tsplit(K_ab))

    f = 0.51 * ((21 + 0.2 * R_d) / (1 + 0.2 * R_d))
    Rd_Yn = R_d / Y_n
    X = (a_Rd / (K_a * f) + Rd_Yn) * X_n
    Z = -(b_Rd / (K_b * f) - Rd_Yn) * Z_n

    XYZ = tstack((X, R_d, Z))

    return from_range_100(XYZ)
