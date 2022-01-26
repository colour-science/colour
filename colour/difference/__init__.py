# -*- coding: utf-8 -*-
"""
References
----------
-   :cite:`ASTMInternational2007` : ASTM International. (2007). ASTM D2244-07 -
    Standard Practice for Calculation of Color Tolerances and Color Differences
    from Instrumentally Measured Color Coordinates: Vol. i (pp. 1-10).
    doi:10.1520/D2244-16
-   :cite:`Li2017` : Li, C., Li, Z., Wang, Z., Xu, Y., Luo, M. R., Cui, G.,
    Melgosa, M., Brill, M. H., & Pointer, M. (2017). Comprehensive color
    solutions: CAM16, CAT16, and CAM16-UCS. Color Research & Application,
    42(6), 703-718. doi:10.1002/col.22131
-   :cite:`Lindbloom2003c` : Lindbloom, B. (2003). Delta E (CIE 1976).
    Retrieved February 24, 2014, from
    http://brucelindbloom.com/Eqn_DeltaE_CIE76.html
-   :cite:`Lindbloom2009e` : Lindbloom, B. (2009). Delta E (CIE 2000).
    Retrieved February 24, 2014, from
    http://brucelindbloom.com/Eqn_DeltaE_CIE2000.html
-   :cite:`Lindbloom2009f` : Lindbloom, B. (2009). Delta E (CMC). Retrieved
    February 24, 2014, from http://brucelindbloom.com/Eqn_DeltaE_CMC.html
-   :cite:`Lindbloom2011a` : Lindbloom, B. (2011). Delta E (CIE 1994).
    Retrieved February 24, 2014, from
    http://brucelindbloom.com/Eqn_DeltaE_CIE94.html
-   :cite:`Luo2006b` : Luo, M. Ronnier, Cui, G., & Li, C. (2006). Uniform
    colour spaces based on CIECAM02 colour appearance model. Color Research &
    Application, 31(4), 320-330. doi:10.1002/col.20227
-   :cite:`Melgosa2013b` : Melgosa, M. (2013). CIE / ISO new standard:
    CIEDE2000. http://www.color.org/events/colorimetry/\
Melgosa_CIEDE2000_Workshop-July4.pdf
-   :cite:`Wikipedia2008b` : Wikipedia. (2008). Color difference. Retrieved
    August 29, 2014, from http://en.wikipedia.org/wiki/Color_difference
"""

from __future__ import annotations

from colour.hints import Any, ArrayLike, FloatingOrNDArray, Literal, Union
from colour.utilities import (
    CaseInsensitiveMapping,
    filter_kwargs,
    validate_method,
)

from .cam02_ucs import delta_E_CAM02LCD, delta_E_CAM02SCD, delta_E_CAM02UCS
from .cam16_ucs import delta_E_CAM16LCD, delta_E_CAM16SCD, delta_E_CAM16UCS
from .delta_e import (
    JND_CIE1976,
    delta_E_CIE1976,
    delta_E_CIE1994,
    delta_E_CIE2000,
    delta_E_CMC,
)
from .din99 import delta_E_DIN99
from .huang2015 import power_function_Huang2015
from .stress import index_stress_Garcia2007, INDEX_STRESS_METHODS, index_stress

__all__ = [
    'delta_E_CAM02LCD',
    'delta_E_CAM02SCD',
    'delta_E_CAM02UCS',
]
__all__ += [
    'delta_E_CAM16LCD',
    'delta_E_CAM16SCD',
    'delta_E_CAM16UCS',
]
__all__ += [
    'JND_CIE1976',
    'delta_E_CIE1976',
    'delta_E_CIE1994',
    'delta_E_CIE2000',
    'delta_E_CMC',
]
__all__ += [
    'delta_E_DIN99',
]
__all__ += [
    'power_function_Huang2015',
]
__all__ += [
    'index_stress_Garcia2007',
    'INDEX_STRESS_METHODS',
    'index_stress',
]

DELTA_E_METHODS: CaseInsensitiveMapping = CaseInsensitiveMapping({
    'CIE 1976': delta_E_CIE1976,
    'CIE 1994': delta_E_CIE1994,
    'CIE 2000': delta_E_CIE2000,
    'CMC': delta_E_CMC,
    'CAM02-LCD': delta_E_CAM02LCD,
    'CAM02-SCD': delta_E_CAM02SCD,
    'CAM02-UCS': delta_E_CAM02UCS,
    'CAM16-LCD': delta_E_CAM16LCD,
    'CAM16-SCD': delta_E_CAM16SCD,
    'CAM16-UCS': delta_E_CAM16UCS,
    'DIN99': delta_E_DIN99,
})
DELTA_E_METHODS.__doc__ = """
Supported :math:`\\Delta E_{ab}` computation methods.

References
----------
:cite:`ASTMInternational2007`, :cite:`Li2017`, :cite:`Lindbloom2003c`,
:cite:`Lindbloom2011a`, :cite:`Lindbloom2009e`, :cite:`Lindbloom2009f`,
:cite:`Luo2006b`, :cite:`Melgosa2013b`, :cite:`Wikipedia2008b`

Aliases:

-   'cie1976': 'CIE 1976'
-   'cie1994': 'CIE 1994'
-   'cie2000': 'CIE 2000'
"""
DELTA_E_METHODS['cie1976'] = DELTA_E_METHODS['CIE 1976']
DELTA_E_METHODS['cie1994'] = DELTA_E_METHODS['CIE 1994']
DELTA_E_METHODS['cie2000'] = DELTA_E_METHODS['CIE 2000']


def delta_E(a: ArrayLike,
            b: ArrayLike,
            method: Union[
                Literal['CIE 1976', 'CIE 1994', 'CIE 2000', 'CMC', 'CAM02-LCD',
                        'CAM02-SCD', 'CAM02-UCS', 'CAM16-LCD', 'CAM16-SCD',
                        'CAM16-UCS', 'DIN99'], str] = 'CIE 2000',
            **kwargs: Any) -> FloatingOrNDArray:
    """
    Returns the difference :math:`\\Delta E_{ab}` between two given
    *CIE L\\*a\\*b\\** or :math:`J'a'b'` colourspace arrays using given method.

    Parameters
    ----------
    a
        *CIE L\\*a\\*b\\** or :math:`J'a'b'` colourspace array :math:`a`.
    b
        *CIE L\\*a\\*b\\** or :math:`J'a'b'` colourspace array :math:`b`.
    method
        Computation method.

    Other Parameters
    ----------------
    textiles
        {:func:`colour.difference.delta_E_CIE1994`,
        :func:`colour.difference.delta_E_CIE2000`,
        :func:`colour.difference.delta_E_DIN99`},
        Textiles application specific parametric factors
        :math:`k_L=2,\\ k_C=k_H=1,\\ k_1=0.048,\\ k_2=0.014,\\ k_E=2,\
\\ k_CH=0.5` weights are used instead of
        :math:`k_L=k_C=k_H=1,\\ k_1=0.045,\\ k_2=0.015,\\ k_E=k_CH=1.0`.
    l
        {:func:`colour.difference.delta_E_CIE2000`},
        Lightness weighting factor.
    c
        {:func:`colour.difference.delta_E_CIE2000`},
        Chroma weighting factor.

    Returns
    -------
    :class:`numpy.floating` or :class:`numpy.ndarray`
        Colour difference :math:`\\Delta E_{ab}`.

    References
    ----------
    :cite:`ASTMInternational2007`, :cite:`Li2017`, :cite:`Lindbloom2003c`,
    :cite:`Lindbloom2011a`, :cite:`Lindbloom2009e`, :cite:`Lindbloom2009f`,
    :cite:`Luo2006b`, :cite:`Melgosa2013b`, :cite:`Wikipedia2008b`

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([100.00000000, 21.57210357, 272.22819350])
    >>> b = np.array([100.00000000, 426.67945353, 72.39590835])
    >>> delta_E(a, b)  # doctest: +ELLIPSIS
    94.0356490...
    >>> delta_E(a, b, method='CIE 2000')  # doctest: +ELLIPSIS
    94.0356490...
    >>> delta_E(a, b, method='CIE 1976')  # doctest: +ELLIPSIS
    451.7133019...
    >>> delta_E(a, b, method='CIE 1994')  # doctest: +ELLIPSIS
    83.7792255...
    >>> delta_E(a, b, method='CIE 1994', textiles=False)
    ... # doctest: +ELLIPSIS
    83.7792255...
    >>> delta_E(a, b, method='DIN99')  # doctest: +ELLIPSIS
    66.1119282...
    >>> a = np.array([54.90433134, -0.08450395, -0.06854831])
    >>> b = np.array([54.90433134, -0.08442362, -0.06848314])
    >>> delta_E(a, b, method='CAM02-UCS')  # doctest: +ELLIPSIS
    0.0001034...
    >>> delta_E(a, b, method='CAM16-LCD')  # doctest: +ELLIPSIS
    0.0001034...
    """

    method = validate_method(method, DELTA_E_METHODS)

    function = DELTA_E_METHODS[method]

    return function(a, b, **filter_kwargs(function, **kwargs))


__all__ += [
    'DELTA_E_METHODS',
    'delta_E',
]
