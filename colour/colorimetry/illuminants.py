# -*- coding: utf-8 -*-
"""
Illuminants
===========

Defines the *CIE* illuminants computation related objects:

-   :func:`colour.sd_CIE_standard_illuminant_A`
-   :func:`colour.sd_CIE_illuminant_D_series`
-   :func:`colour.daylight_locus_function`

References
----------
-   :cite:`CIETC1-482004` : CIE TC 1-48. (2004). EXPLANATORY COMMENTS - 5. In
    CIE 015:2004 Colorimetry, 3rd Edition (pp. 68-68). ISBN:978-3-901906-33-6
-   :cite:`CIETC1-482004n` : CIE TC 1-48. (2004). 3.1 Recommendations
    concerning standard physical data of illuminants. In CIE 015:2004
    Colorimetry, 3rd Edition (pp. 12-13). ISBN:978-3-901906-33-6
-   :cite:`Wyszecki2000a` : Wyszecki, Günther, & Stiles, W. S. (2000).
    Equation I(1.2.1). In Color Science: Concepts and Methods, Quantitative
    Data and Formulae (p. 8). Wiley. ISBN:978-0-471-39918-6
-   :cite:`Wyszecki2000z` : Wyszecki, Günther, & Stiles, W. S. (2000). CIE
    Method of Calculating D-Illuminants. In Color Science: Concepts and
    Methods, Quantitative Data and Formulae (pp. 145-146). Wiley.
    ISBN:978-0-471-39918-6
"""

import numpy as np

from colour.algebra import LinearInterpolator
from colour.colorimetry import (
    SPECTRAL_SHAPE_DEFAULT,
    SDS_BASIS_FUNCTIONS_CIE_ILLUMINANT_D_SERIES,
    SpectralDistribution,
)
from colour.utilities import as_float_array, as_float, tsplit

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'sd_CIE_standard_illuminant_A',
    'sd_CIE_illuminant_D_series',
    'daylight_locus_function',
]


def sd_CIE_standard_illuminant_A(shape=SPECTRAL_SHAPE_DEFAULT):
    """
    *CIE Standard Illuminant A* is intended to represent typical, domestic,
    tungsten-filament lighting.

    Its spectral distribution is that of a Planckian radiator at a temperature
    of approximately 2856 K. *CIE Standard Illuminant A* should be used in all
    applications of colorimetry involving the use of incandescent lighting,
    unless there are specific reasons for using a different illuminant.

    Parameters
    ----------
    shape : SpectralShape, optional
        Spectral shape used to create the spectral distribution of the
        *CIE Standard Illuminant A*.

    Returns
    -------
    SpectralDistribution
        *CIE Standard Illuminant A*. spectral distribution.

    References
    ----------
    :cite:`CIETC1-482004n`

    Examples
    --------
    >>> from colour import SpectralShape
    >>> sd_CIE_standard_illuminant_A(SpectralShape(400, 700, 10))
    ... # doctest: +ELLIPSIS
    SpectralDistribution([[ 400.        ,   14.7080384...],
                          [ 410.        ,   17.6752521...],
                          [ 420.        ,   20.9949572...],
                          [ 430.        ,   24.6709226...],
                          [ 440.        ,   28.7027304...],
                          [ 450.        ,   33.0858929...],
                          [ 460.        ,   37.8120566...],
                          [ 470.        ,   42.8692762...],
                          [ 480.        ,   48.2423431...],
                          [ 490.        ,   53.9131532...],
                          [ 500.        ,   59.8610989...],
                          [ 510.        ,   66.0634727...],
                          [ 520.        ,   72.4958719...],
                          [ 530.        ,   79.1325945...],
                          [ 540.        ,   85.9470183...],
                          [ 550.        ,   92.9119589...],
                          [ 560.        ,  100.       ...],
                          [ 570.        ,  107.1837952...],
                          [ 580.        ,  114.4363383...],
                          [ 590.        ,  121.7312009...],
                          [ 600.        ,  129.0427389...],
                          [ 610.        ,  136.3462674...],
                          [ 620.        ,  143.6182057...],
                          [ 630.        ,  150.8361944...],
                          [ 640.        ,  157.9791857...],
                          [ 650.        ,  165.0275098...],
                          [ 660.        ,  171.9629200...],
                          [ 670.        ,  178.7686175...],
                          [ 680.        ,  185.4292591...],
                          [ 690.        ,  191.9309499...],
                          [ 700.        ,  198.2612232...]],
                         interpolator=SpragueInterpolator,
                         interpolator_kwargs={},
                         extrapolator=Extrapolator,
                         extrapolator_kwargs={...})
    """

    wavelengths = shape.range()
    values = (100 * (560 / wavelengths) ** 5 * (((np.exp(
        (1.435 * 10 ** 7) / (2848 * 560)) - 1) / (np.exp(
            (1.435 * 10 ** 7) / (2848 * wavelengths)) - 1))))

    return SpectralDistribution(
        values, wavelengths, name='CIE Standard Illuminant A')


def sd_CIE_illuminant_D_series(xy, M1_M2_rounding=True):
    """
    Returns the spectral distribution of given *CIE Illuminant D Series* using
    given *CIE xy* chromaticity coordinates.

    Parameters
    ----------
    xy : array_like
        *CIE xy* chromaticity coordinates.
    M1_M2_rounding : bool, optional
        Whether to round :math:`M1` and :math:`M2` variables to 3 decimal
        places in order to yield the internationally agreed values.

    Returns
    -------
    SpectralDistribution
        *CIE Illuminant D Series* spectral
        distribution.

    Notes
    -----
    -   The nominal *CIE xy* chromaticity coordinates which have been computed
        with :func:`colour.temperature.CCT_to_xy_CIE_D` must be given according
        to *CIE 015:2004* recommendation and thus multiplied by
        1.4388 / 1.4380.
    -   :math:`M1` and :math:`M2` variables are rounded to 3 decimal places
         according to *CIE 015:2004* recommendation.

    References
    ----------
    :cite:`CIETC1-482004`, :cite:`Wyszecki2000z`

    Examples
    --------
    >>> from colour.utilities import numpy_print_options
    >>> from colour.temperature import CCT_to_xy_CIE_D
    >>> CCT_D65 = 6500 * 1.4388 / 1.4380
    >>> xy = CCT_to_xy_CIE_D(CCT_D65)
    >>> with numpy_print_options(suppress=True):
    ...     sd_CIE_illuminant_D_series(xy)  # doctest: +ELLIPSIS
    SpectralDistribution([[ 300.     ,    0.0341...],
                          [ 305.     ,    1.6643...],
                          [ 310.     ,    3.2945...],
                          [ 315.     ,   11.7652...],
                          [ 320.     ,   20.236 ...],
                          [ 325.     ,   28.6447...],
                          [ 330.     ,   37.0535...],
                          [ 335.     ,   38.5011...],
                          [ 340.     ,   39.9488...],
                          [ 345.     ,   42.4302...],
                          [ 350.     ,   44.9117...],
                          [ 355.     ,   45.775 ...],
                          [ 360.     ,   46.6383...],
                          [ 365.     ,   49.3637...],
                          [ 370.     ,   52.0891...],
                          [ 375.     ,   51.0323...],
                          [ 380.     ,   49.9755...],
                          [ 385.     ,   52.3118...],
                          [ 390.     ,   54.6482...],
                          [ 395.     ,   68.7015...],
                          [ 400.     ,   82.7549...],
                          [ 405.     ,   87.1204...],
                          [ 410.     ,   91.486 ...],
                          [ 415.     ,   92.4589...],
                          [ 420.     ,   93.4318...],
                          [ 425.     ,   90.0570...],
                          [ 430.     ,   86.6823...],
                          [ 435.     ,   95.7736...],
                          [ 440.     ,  104.8649...],
                          [ 445.     ,  110.9362...],
                          [ 450.     ,  117.0076...],
                          [ 455.     ,  117.4099...],
                          [ 460.     ,  117.8122...],
                          [ 465.     ,  116.3365...],
                          [ 470.     ,  114.8609...],
                          [ 475.     ,  115.3919...],
                          [ 480.     ,  115.9229...],
                          [ 485.     ,  112.3668...],
                          [ 490.     ,  108.8107...],
                          [ 495.     ,  109.0826...],
                          [ 500.     ,  109.3545...],
                          [ 505.     ,  108.5781...],
                          [ 510.     ,  107.8017...],
                          [ 515.     ,  106.2957...],
                          [ 520.     ,  104.7898...],
                          [ 525.     ,  106.2396...],
                          [ 530.     ,  107.6895...],
                          [ 535.     ,  106.0475...],
                          [ 540.     ,  104.4055...],
                          [ 545.     ,  104.2258...],
                          [ 550.     ,  104.0462...],
                          [ 555.     ,  102.0231...],
                          [ 560.     ,  100.    ...],
                          [ 565.     ,   98.1671...],
                          [ 570.     ,   96.3342...],
                          [ 575.     ,   96.0611...],
                          [ 580.     ,   95.788 ...],
                          [ 585.     ,   92.2368...],
                          [ 590.     ,   88.6856...],
                          [ 595.     ,   89.3459...],
                          [ 600.     ,   90.0062...],
                          [ 605.     ,   89.8026...],
                          [ 610.     ,   89.5991...],
                          [ 615.     ,   88.6489...],
                          [ 620.     ,   87.6987...],
                          [ 625.     ,   85.4936...],
                          [ 630.     ,   83.2886...],
                          [ 635.     ,   83.4939...],
                          [ 640.     ,   83.6992...],
                          [ 645.     ,   81.863 ...],
                          [ 650.     ,   80.0268...],
                          [ 655.     ,   80.1207...],
                          [ 660.     ,   80.2146...],
                          [ 665.     ,   81.2462...],
                          [ 670.     ,   82.2778...],
                          [ 675.     ,   80.281 ...],
                          [ 680.     ,   78.2842...],
                          [ 685.     ,   74.0027...],
                          [ 690.     ,   69.7213...],
                          [ 695.     ,   70.6652...],
                          [ 700.     ,   71.6091...],
                          [ 705.     ,   72.9790...],
                          [ 710.     ,   74.349 ...],
                          [ 715.     ,   67.9765...],
                          [ 720.     ,   61.604 ...],
                          [ 725.     ,   65.7448...],
                          [ 730.     ,   69.8856...],
                          [ 735.     ,   72.4863...],
                          [ 740.     ,   75.087 ...],
                          [ 745.     ,   69.3398...],
                          [ 750.     ,   63.5927...],
                          [ 755.     ,   55.0054...],
                          [ 760.     ,   46.4182...],
                          [ 765.     ,   56.6118...],
                          [ 770.     ,   66.8054...],
                          [ 775.     ,   65.0941...],
                          [ 780.     ,   63.3828...],
                          [ 785.     ,   63.8434...],
                          [ 790.     ,   64.304 ...],
                          [ 795.     ,   61.8779...],
                          [ 800.     ,   59.4519...],
                          [ 805.     ,   55.7054...],
                          [ 810.     ,   51.959 ...],
                          [ 815.     ,   54.6998...],
                          [ 820.     ,   57.4406...],
                          [ 825.     ,   58.8765...],
                          [ 830.     ,   60.3125...]],
                         interpolator=LinearInterpolator,
                         interpolator_kwargs={},
                         extrapolator=Extrapolator,
                         extrapolator_kwargs={...})
    """

    x, y = tsplit(xy)

    M = 0.0241 + 0.2562 * x - 0.7341 * y
    M1 = (-1.3515 - 1.7703 * x + 5.9114 * y) / M
    M2 = (0.0300 - 31.4424 * x + 30.0717 * y) / M

    if M1_M2_rounding:
        M1 = np.around(M1, 3)
        M2 = np.around(M2, 3)

    S0 = SDS_BASIS_FUNCTIONS_CIE_ILLUMINANT_D_SERIES['S0']
    S1 = SDS_BASIS_FUNCTIONS_CIE_ILLUMINANT_D_SERIES['S1']
    S2 = SDS_BASIS_FUNCTIONS_CIE_ILLUMINANT_D_SERIES['S2']

    distribution = S0.values + M1 * S1.values + M2 * S2.values

    return SpectralDistribution(
        distribution,
        S0.wavelengths,
        name='CIE xy ({0}, {1}) - CIE Illuminant D Series'.format(*xy),
        interpolator=LinearInterpolator)


def daylight_locus_function(x_D):
    """
    Returns the daylight locus as *CIE xy* chromaticity coordinates.

    Parameters
    ----------
    x_D : numeric or array_like
        *x* chromaticity coordinates

    Returns
    -------
    numeric or array_like
        Daylight locus as *CIE xy* chromaticity coordinates.

    References
    ----------
    :cite:`Wyszecki2000a`

    Examples
    --------
    >>> daylight_locus_function(0.31270)  # doctest: +ELLIPSIS
    0.3291051...
    """

    x_D = as_float_array(x_D)

    y_D = -3.000 * x_D ** 2 + 2.870 * x_D - 0.275

    return as_float(y_D)
