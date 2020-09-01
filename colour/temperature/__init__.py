# -*- coding: utf-8 -*-

from __future__ import absolute_import

from colour.utilities import CaseInsensitiveMapping, filter_kwargs

from .cie_d import xy_to_CCT_CIE_D, CCT_to_xy_CIE_D
from .hernandez1999 import xy_to_CCT_Hernandez1999, CCT_to_xy_Hernandez1999
from .kang2002 import xy_to_CCT_Kang2002, CCT_to_xy_Kang2002
from .krystek1985 import uv_to_CCT_Krystek1985, CCT_to_uv_Krystek1985
from .mccamy1992 import xy_to_CCT_McCamy1992, CCT_to_xy_McCamy1992
from .ohno2013 import uv_to_CCT_Ohno2013, CCT_to_uv_Ohno2013
from .robertson1968 import uv_to_CCT_Robertson1968, CCT_to_uv_Robertson1968

__all__ = ['xy_to_CCT_CIE_D', 'CCT_to_xy_CIE_D']
__all__ += ['xy_to_CCT_Hernandez1999', 'CCT_to_xy_Hernandez1999']
__all__ += ['xy_to_CCT_Kang2002', 'CCT_to_xy_Kang2002']
__all__ += ['uv_to_CCT_Krystek1985', 'CCT_to_uv_Krystek1985']
__all__ += ['xy_to_CCT_McCamy1992', 'CCT_to_xy_McCamy1992']
__all__ += ['uv_to_CCT_Ohno2013', 'CCT_to_uv_Ohno2013']
__all__ += ['uv_to_CCT_Robertson1968', 'CCT_to_uv_Robertson1968']

UV_TO_CCT_METHODS = CaseInsensitiveMapping({
    'Krystek 1985': uv_to_CCT_Krystek1985,
    'Ohno 2013': uv_to_CCT_Ohno2013,
    'Robertson 1968': uv_to_CCT_Robertson1968
})
UV_TO_CCT_METHODS.__doc__ = """
Supported *CIE UCS* colourspace *uv* chromaticity coordinates to correlated
colour temperature :math:`T_{cp}` computation methods.

References
----------
:cite:`AdobeSystems2013`, :cite:`AdobeSystems2013a`, :cite:`Ohno2014a`,
:cite:`Wyszecki2000y`

UV_TO_CCT_METHODS : CaseInsensitiveMapping
    **{'Ohno 2013', 'Krystek 1985, 'Robertson 1968'}**

Aliases:

-   'ohno2013': 'Ohno 2013'
-   'robertson1968': 'Robertson 1968'
"""
UV_TO_CCT_METHODS['ohno2013'] = UV_TO_CCT_METHODS['Ohno 2013']
UV_TO_CCT_METHODS['robertson1968'] = UV_TO_CCT_METHODS['Robertson 1968']


def uv_to_CCT(uv, method='Ohno 2013', **kwargs):
    """
    Returns the correlated colour temperature :math:`T_{cp}` and
    :math:`\\Delta_{uv}` from given *CIE UCS* colourspace *uv* chromaticity
    coordinates using given method.

    Parameters
    ----------
    uv : array_like
        *CIE UCS* colourspace *uv* chromaticity coordinates.
    method : unicode, optional
        **{'Ohno 2013', 'Krystek 1985, 'Robertson 1968'}**,
        Computation method.

    Other Parameters
    ----------------
    cmfs : XYZ_ColourMatchingFunctions, optional
        {:func:`colour.temperature.uv_to_CCT_Ohno2013`},
        Standard observer colour matching functions.
    start : numeric, optional
        {:func:`colour.temperature.uv_to_CCT_Ohno2013`},
        Temperature range start in kelvins.
    end : numeric, optional
        {:func:`colour.temperature.uv_to_CCT_Ohno2013`},
        Temperature range end in kelvins.
    count : int, optional
        {:func:`colour.temperature.uv_to_CCT_Ohno2013`},
        Temperatures count in the planckian tables.
    iterations : int, optional
        {:func:`colour.temperature.uv_to_CCT_Ohno2013`},
        Number of planckian tables to generate.
    optimisation_kwargs : dict_like, optional
        {:func:`colour.temperature.uv_to_CCT_Krystek1985`},
        Parameters for :func:`scipy.optimize.minimize` definition.

    Returns
    -------
    ndarray
        Correlated colour temperature :math:`T_{cp}`, :math:`\\Delta_{uv}`.

    References
    ----------
    :cite:`AdobeSystems2013`, :cite:`AdobeSystems2013a`, :cite:`Krystek1985b`,
    :cite:`Ohno2014a`, :cite:`Wyszecki2000y`

    Examples
    --------
    >>> import colour.ndarray as np
    >>> uv = np.array([0.1978, 0.3122])
    >>> # Doctests skipping for Python 2.x compatibility.
    >>> uv_to_CCT(uv)  # doctest: +SKIP
    array([  6.5074738...e+03,   3.2233460...e-03])
    """

    function = UV_TO_CCT_METHODS[method]

    return function(uv, **filter_kwargs(function, **kwargs))


CCT_TO_UV_METHODS = CaseInsensitiveMapping({
    'Krystek 1985': CCT_to_uv_Krystek1985,
    'Ohno 2013': CCT_to_uv_Ohno2013,
    'Robertson 1968': CCT_to_uv_Robertson1968
})
CCT_TO_UV_METHODS.__doc__ = """
Supported correlated colour temperature :math:`T_{cp}` to *CIE UCS* colourspace
*uv* chromaticity coordinates computation methods.

References
----------
-   :cite:`AdobeSystems2013` : Adobe Systems. (2013). Adobe DNG Software
    Development Kit (SDK) - 1.3.0.0 -
    dng_sdk_1_3/dng_sdk/source/dng_temperature.cpp::dng_temperature::\
Set_xy_coord. https://www.adobe.com/support/downloads/dng/dng_sdk.html
-   :cite:`AdobeSystems2013a` : Adobe Systems. (2013). Adobe DNG Software
    Development Kit (SDK) - 1.3.0.0 -
    dng_sdk_1_3/dng_sdk/source/dng_temperature.cpp::dng_temperature::xy_coord.
    https://www.adobe.com/support/downloads/dng/dng_sdk.html
-   :cite:`Hernandez-Andres1999a` : Hernández-Andrés, J., Lee, R. L., &
    Romero, J. (1999). Calculating correlated color temperatures across the
    entire gamut of daylight and skylight chromaticities. Applied Optics,
    38(27),
    5703. doi:10.1364/AO.38.005703
-   :cite:`Kang2002a` : Kang, B., Moon, O., Hong, C., Lee, H., Cho, B., & Kim,
    Y. (2002). Design of advanced color: Temperature control system for HDTV
    applications. Journal of the Korean Physical Society, 41(6), 865-871.
-   :cite:`Krystek1985b` : Krystek, M. (1985). An algorithm to calculate
    correlated colour temperature. Color Research & Application, 10(1), 38-40.
    doi:10.1002/col.5080100109
-   :cite:`Ohno2014a` : Ohno, Yoshiro. (2014). Practical Use and Calculation of
    CCT and Duv. LEUKOS, 10(1), 47-55. doi:10.1080/15502724.2014.839020
-   :cite:`Wikipedia2001` : Wikipedia. (2001). Approximation. Retrieved June
    28, 2014, from http://en.wikipedia.org/wiki/Color_temperature#Approximation
-   :cite:`Wikipedia2001a` : Wikipedia. (2001). Color temperature. Retrieved
    June 28, 2014, from http://en.wikipedia.org/wiki/Color_temperature
-   :cite:`Wyszecki2000y` : Wyszecki, Günther, & Stiles, W. S. (2000).
    DISTRIBUTION TEMPERATURE, COLOR TEMPERATURE, AND CORRELATED COLOR
    TEMPERATURE. In Color Science: Concepts and Methods, Quantitative Data and
    Formulae (pp. 224-229). Wiley. ISBN:978-0-471-39918-6
-   :cite:`Wyszecki2000z` : Wyszecki, Günther, & Stiles, W. S. (2000). CIE
    Method of Calculating D-Illuminants. In Color Science: Concepts and
    Methods, Quantitative Data and Formulae (pp. 145-146). Wiley.
    ISBN:978-0-471-39918-6
"""
CCT_TO_UV_METHODS['ohno2013'] = CCT_TO_UV_METHODS['Ohno 2013']
CCT_TO_UV_METHODS['robertson1968'] = CCT_TO_UV_METHODS['Robertson 1968']


def CCT_to_uv(CCT_D_uv, method='Ohno 2013', **kwargs):
    """
    Returns the *CIE UCS* colourspace *uv* chromaticity coordinates from given
    correlated colour temperature :math:`T_{cp}` using given method.

    Parameters
    ----------
    CCT_D_uv : ndarray
        Correlated colour temperature :math:`T_{cp}`, :math:`\\Delta_{uv}`.
    method : unicode, optional
        **{'Ohno 2013', 'Robertson 1968', 'Krystek 1985}**,
        Computation method.

    Other Parameters
    ----------------
    cmfs : XYZ_ColourMatchingFunctions, optional
        {:func:`colour.temperature.CCT_to_uv_Ohno2013`},
        Standard observer colour matching functions.

    Returns
    -------
    ndarray
        *CIE UCS* colourspace *uv* chromaticity coordinates.

    References
    ----------
    :cite:`AdobeSystems2013`, :cite:`AdobeSystems2013a`, :cite:`Krystek1985b`,
    :cite:`Ohno2014a`, :cite:`Wyszecki2000y`

    Examples
    --------
    >>> import colour.ndarray as np
    >>> CCT_D_uv = np.array([6507.47380460, 0.00322335])
    >>> CCT_to_uv(CCT_D_uv)  # doctest: +ELLIPSIS
    array([ 0.1977999...,  0.3121999...])
    """

    function = CCT_TO_UV_METHODS[method]

    return function(CCT_D_uv, **filter_kwargs(function, **kwargs))


__all__ += ['UV_TO_CCT_METHODS', 'uv_to_CCT']
__all__ += ['CCT_TO_UV_METHODS', 'CCT_to_uv']

XY_TO_CCT_METHODS = CaseInsensitiveMapping({
    'CIE Illuminant D Series': xy_to_CCT_CIE_D,
    'Hernandez 1999': xy_to_CCT_Hernandez1999,
    'Kang 2002': xy_to_CCT_Kang2002,
    'McCamy 1992': xy_to_CCT_McCamy1992,
})
XY_TO_CCT_METHODS.__doc__ = """
Supported *CIE xy* chromaticity coordinates to correlated colour temperature
:math:`T_{cp}` computation methods.

References
----------
:cite:`Hernandez-Andres1999a`, :cite:`Kang2002a`, :cite:`Wikipedia2001`,
:cite:`Wikipedia2001a`,
:cite:`Wyszecki2000z`

XY_TO_CCT_METHODS : CaseInsensitiveMapping
    **{'McCamy 1992', 'CIE Illuminant D Series, 'Kang 2002',
    'Hernandez 1999'}**

Aliases:

-   'daylight': 'CIE Illuminant D Series'
-   'kang2002': 'Kang 2002'
-   'mccamy1992': 'McCamy 1992'
-   'hernandez1999': 'Hernandez 1999'
"""
XY_TO_CCT_METHODS['daylight'] = XY_TO_CCT_METHODS['CIE Illuminant D Series']
XY_TO_CCT_METHODS['kang2002'] = XY_TO_CCT_METHODS['Kang 2002']
XY_TO_CCT_METHODS['mccamy1992'] = XY_TO_CCT_METHODS['McCamy 1992']
XY_TO_CCT_METHODS['hernandez1999'] = XY_TO_CCT_METHODS['Hernandez 1999']


def xy_to_CCT(xy, method='CIE Illuminant D Series'):
    """
    Returns the correlated colour temperature :math:`T_{cp}` from given
    *CIE xy* chromaticity coordinates using given method.

    Parameters
    ----------
    xy : array_like
        *CIE xy* chromaticity coordinates.
    method : unicode, optional
        **{'CIE Illuminant D Series', 'Kang 2002', 'Hernandez 1999',
        'McCamy 1992'}**,
        Computation method.

    Other Parameters
    ----------------
    optimisation_kwargs : dict_like, optional
        {:func:`colour.temperature.xy_to_CCT_CIE_D`,
        :func:`colour.temperature.xy_to_CCT_Kang2002`},
        Parameters for :func:`scipy.optimize.minimize` definition.

    Returns
    -------
    numeric or ndarray
        Correlated colour temperature :math:`T_{cp}`.

    References
    ----------
    :cite:`Hernandez-Andres1999a`, :cite:`Kang2002a`, :cite:`Wikipedia2001`,
    :cite:`Wikipedia2001a`, :cite:`Wyszecki2000z`

    Examples
    --------
    >>> import colour.ndarray as np
    >>> xy_to_CCT(np.array([0.31270, 0.32900]))  # doctest: +ELLIPSIS
    6508.1175148...
    >>> xy_to_CCT(np.array([0.31270, 0.32900]), 'Hernandez 1999')
    ... # doctest: +ELLIPSIS
    6500.7420431...
    """

    return XY_TO_CCT_METHODS.get(method)(xy)


CCT_TO_XY_METHODS = CaseInsensitiveMapping({
    'CIE Illuminant D Series': CCT_to_xy_CIE_D,
    'Hernandez 1999': CCT_to_xy_Hernandez1999,
    'Kang 2002': CCT_to_xy_Kang2002,
    'McCamy 1992': CCT_to_xy_McCamy1992,
})
CCT_TO_XY_METHODS.__doc__ = """
Supported correlated colour temperature :math:`T_{cp}` to *CIE xy* chromaticity
coordinates computation methods.

References
----------
:cite:`Hernandez-Andres1999a`, :cite:`Kang2002a`, :cite:`Wikipedia2001`,
:cite:`Wikipedia2001a`, :cite:`Wyszecki2000z`

CCT_TO_XY_METHODS : CaseInsensitiveMapping
    **{'Kang 2002', 'CIE Illuminant D Series', 'Hernandez 1999',
    'McCamy 1992'}**

Aliases:

-   'daylight': 'CIE Illuminant D Series'
-   'kang2002': 'Kang 2002'
-   'mccamy1992': 'McCamy 1992'
-   'hernandez1999': 'Hernandez 1999'
"""
CCT_TO_XY_METHODS['daylight'] = CCT_TO_XY_METHODS['CIE Illuminant D Series']
CCT_TO_XY_METHODS['kang2002'] = CCT_TO_XY_METHODS['Kang 2002']
CCT_TO_XY_METHODS['mccamy1992'] = CCT_TO_XY_METHODS['McCamy 1992']
CCT_TO_XY_METHODS['hernandez1999'] = CCT_TO_XY_METHODS['Hernandez 1999']


def CCT_to_xy(CCT, method='CIE Illuminant D Series'):
    """
    Returns the *CIE xy* chromaticity coordinates from given correlated colour
    temperature :math:`T_{cp}` using given method.

    Parameters
    ----------
    CCT : numeric or array_like
        Correlated colour temperature :math:`T_{cp}`.
    method : unicode, optional
        **{'CIE Illuminant D Series', 'Hernandez 1999', 'Kang 2002',
        'McCamy 1992'}**,
        Computation method.

    Other Parameters
    ----------------
    optimisation_kwargs : dict_like, optional
        {:func:`colour.temperature.CCT_to_xy_Hernandez1999`,
        :func:`colour.temperature.CCT_to_xy_McCamy1992`},
        Parameters for :func:`scipy.optimize.minimize` definition.

    Returns
    -------
    ndarray
        *CIE xy* chromaticity coordinates.

    References
    ----------
    :cite:`Hernandez-Andres1999a`, :cite:`Kang2002a`, :cite:`Wikipedia2001`,
    :cite:`Wikipedia2001a`, :cite:`Wyszecki2000z`

    Examples
    --------
    >>> CCT_to_xy(6504.38938305)  # doctest: +ELLIPSIS
    array([ 0.3127077...,  0.3291128...])
    >>> CCT_to_xy(6504.38938305, 'Kang 2002')
    ... # doctest: +ELLIPSIS
    array([ 0.313426 ...,  0.3235959...])
    """

    return CCT_TO_XY_METHODS.get(method)(CCT)


__all__ += ['XY_TO_CCT_METHODS', 'xy_to_CCT']
__all__ += ['CCT_TO_XY_METHODS', 'CCT_to_xy']
