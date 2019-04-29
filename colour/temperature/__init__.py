# -*- coding: utf-8 -*-

from __future__ import absolute_import

from colour.utilities import CaseInsensitiveMapping, filter_kwargs

from .cie_d import CCT_to_xy_CIE_D
from .hernandez1999 import xy_to_CCT_Hernandez1999
from .kang2002 import CCT_to_xy_Kang2002
from .krystek1985 import uv_to_CCT_Krystek1985, CCT_to_uv_Krystek1985
from .mccamy1992 import xy_to_CCT_McCamy1992
from .ohno2013 import uv_to_CCT_Ohno2013, CCT_to_uv_Ohno2013
from .robertson1968 import uv_to_CCT_Robertson1968, CCT_to_uv_Robertson1968

__all__ = ['CCT_to_xy_CIE_D']
__all__ += ['xy_to_CCT_Hernandez1999']
__all__ += ['CCT_to_xy_Kang2002']
__all__ += ['uv_to_CCT_Krystek1985', 'CCT_to_uv_Krystek1985']
__all__ += ['xy_to_CCT_McCamy1992']
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
    optimisation_parameters : dict_like, optional
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
    >>> import numpy as np
    >>> from colour import STANDARD_OBSERVERS_CMFS
    >>> cmfs = STANDARD_OBSERVERS_CMFS['CIE 1931 2 Degree Standard Observer']
    >>> uv = np.array([0.1978, 0.3122])
    >>> uv_to_CCT(uv, cmfs=cmfs)  # doctest: +ELLIPSIS
    array([  6.5074738...e+03,   3.2233461...e-03])
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
:cite:`AdobeSystems2013`, :cite:`AdobeSystems2013a`, :cite:`Krystek1985b`,
:cite:`Ohno2014a`, :cite:`Wyszecki2000y`

CCT_TO_UV_METHODS : CaseInsensitiveMapping
    **{'Ohno 2013', 'Krystek 1985, 'Robertson 1968'}**

Aliases:

-   'ohno2013': 'Ohno 2013'
-   'robertson1968': 'Robertson 1968'
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
    >>> import numpy as np
    >>> from colour import STANDARD_OBSERVERS_CMFS
    >>> cmfs = STANDARD_OBSERVERS_CMFS['CIE 1931 2 Degree Standard Observer']
    >>> CCT_D_uv = np.array([6507.47380460, 0.00322335])
    >>> CCT_to_uv(CCT_D_uv, cmfs=cmfs)  # doctest: +ELLIPSIS
    array([ 0.1977999...,  0.3121999...])
    """

    function = CCT_TO_UV_METHODS[method]

    return function(CCT_D_uv, **filter_kwargs(function, **kwargs))


__all__ += ['UV_TO_CCT_METHODS', 'uv_to_CCT']
__all__ += ['CCT_TO_UV_METHODS', 'CCT_to_uv']

XY_TO_CCT_METHODS = CaseInsensitiveMapping({
    'McCamy 1992': xy_to_CCT_McCamy1992,
    'Hernandez 1999': xy_to_CCT_Hernandez1999
})
XY_TO_CCT_METHODS.__doc__ = """
Supported *CIE XYZ* tristimulus values *CIE xy* chromaticity coordinates to
correlated colour temperature :math:`T_{cp}` computation methods.

References
----------
:cite:`Hernandez-Andres1999a`, :cite:`Wikipedia2001`, :cite:`Wikipedia2001a`

XY_TO_CCT_METHODS : CaseInsensitiveMapping
    **{'McCamy 1992', 'Hernandez 1999'}**

Aliases:

-   'mccamy1992': 'McCamy 1992'
-   'hernandez1999': 'Hernandez 1999'
"""
XY_TO_CCT_METHODS['mccamy1992'] = XY_TO_CCT_METHODS['McCamy 1992']
XY_TO_CCT_METHODS['hernandez1999'] = XY_TO_CCT_METHODS['Hernandez 1999']


def xy_to_CCT(xy, method='McCamy 1992'):
    """
    Returns the correlated colour temperature :math:`T_{cp}` from given
    *CIE XYZ* tristimulus values *CIE xy* chromaticity coordinates using given
    method.

    Parameters
    ----------
    xy : array_like
        *CIE xy* chromaticity coordinates.
    method : unicode, optional
        **{'McCamy 1992', 'Hernandez 1999'}**,
        Computation method.

    Returns
    -------
    numeric or ndarray
        Correlated colour temperature :math:`T_{cp}`.

    References
    ----------
    :cite:`Hernandez-Andres1999a`, :cite:`Wikipedia2001`,
    :cite:`Wikipedia2001a`

    Examples
    --------
    >>> import numpy as np
    >>> xy_to_CCT(np.array([0.31270, 0.32900]))  # doctest: +ELLIPSIS
    6505.0805913...
    >>> xy_to_CCT(np.array([0.31270, 0.32900]), 'Hernandez 1999')
    ... # doctest: +ELLIPSIS
    6500.7420431...
    """

    return XY_TO_CCT_METHODS.get(method)(xy)


CCT_TO_XY_METHODS = CaseInsensitiveMapping({
    'Kang 2002': CCT_to_xy_Kang2002,
    'CIE Illuminant D Series': CCT_to_xy_CIE_D
})
CCT_TO_XY_METHODS.__doc__ = """
Supported correlated colour temperature :math:`T_{cp}` to *CIE XYZ* tristimulus
values *CIE xy* chromaticity coordinates computation methods.

References
----------
:cite:`Kang2002a`, :cite:`Wikipedia2001a`, :cite:`Wyszecki2000z`

CCT_TO_XY_METHODS : CaseInsensitiveMapping
    **{'Kang 2002', 'CIE Illuminant D Series'}**

Aliases:

-   'kang2002': 'Kang 2002'
-   'cie_d': 'Hernandez 1999'
"""
CCT_TO_XY_METHODS['kang2002'] = CCT_TO_XY_METHODS['Kang 2002']
CCT_TO_XY_METHODS['cie_d'] = CCT_TO_XY_METHODS['CIE Illuminant D Series']


def CCT_to_xy(CCT, method='Kang 2002'):
    """
    Returns the *CIE XYZ* tristimulus values *CIE xy* chromaticity coordinates
    from given correlated colour temperature :math:`T_{cp}` using given method.

    Parameters
    ----------
    CCT : numeric or array_like
        Correlated colour temperature :math:`T_{cp}`.
    method : unicode, optional
        **{'Kang 2002', 'CIE Illuminant D Series'}**,
        Computation method.

    Returns
    -------
    ndarray
        *CIE xy* chromaticity coordinates.

    References
    ----------
    :cite:`Kang2002a`, :cite:`Wikipedia2001a`, :cite:`Wyszecki2000z`

    Examples
    --------
    >>> CCT_to_xy(6504.38938305)  # doctest: +ELLIPSIS
    array([ 0.313426 ...,  0.3235959...])
    >>> CCT_to_xy(6504.38938305, 'CIE Illuminant D Series')
    ... # doctest: +ELLIPSIS
    array([ 0.3127077...,  0.3291128...])
    """

    return CCT_TO_XY_METHODS.get(method)(CCT)


__all__ += ['XY_TO_CCT_METHODS', 'xy_to_CCT']
__all__ += ['CCT_TO_XY_METHODS', 'CCT_to_xy']
