"""
References
----------
-   :cite:`Abasi2020a` : Abasi, S., Amani Tehran, M., & Fairchild, M. D. (2020).
    Distance metrics for very large color differences. Color Research &
    Application, 45(2), 208-223. doi:10.1002/col.22451
-   :cite:`ASTMInternational2007` : ASTM International. (2007). ASTM D2244-07 -
    Standard Practice for Calculation of Color Tolerances and Color Differences
    from Instrumentally Measured Color Coordinates: Vol. i (pp. 1-10).
    doi:10.1520/D2244-16
-   :cite:`InternationalTelecommunicationUnion2019` : International
    Telecommunication Union. (2019). Recommendation ITU-R BT.2124-0 -
    Objective metric for the assessment of the potential visibility of colour
    differences in television (pp. 1-36). http://www.itu.int/dms_pubrec/itu-r/\
rec/bt/R-REC-BT.470-6-199811-S!!PDF-E.pdf
-   :cite:`Li2017` : Li, C., Li, Z., Wang, Z., Xu, Y., Luo, M. R., Cui, G.,
    Melgosa, M., Brill, M. H., & Pointer, M. (2017). Comprehensive color
    solutions: CAM16, CAT16, and CAM16-UCS. Color Research & Application,
    42(6), 703-718. doi:10.1002/col.22131
-   :cite:`Lindbloom2003c` : Lindbloom, B. (2003). Delta E (CIE 1976).
    Retrieved February 24, 2014, from
    http://brucelindbloom.com/Eqn_DeltaE_CIE76.html
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
-   :cite:ISO18314-4_2024 : International Organization for Standardization. (2024).
    ISO 18314-4:2024. Analytical colorimetry, Part 4: Metamerism index for pairs
    of samples for change of illuminant (2nd ed., 24 pp.).
    ISO/TC 256. https://www.iso.org/standard/85116.html
"""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import Any, ArrayLike, NDArrayFloat, LiteralDeltaEMethod

from colour.utilities import (
    CanonicalMapping,
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
    delta_E_HyAB,
    delta_E_HyCH,
    delta_E_ITP,
)
from .din99 import delta_E_DIN99
from .huang2015 import power_function_Huang2015
from .metamerism_index import Lab_to_metamerism_index, XYZ_to_metamerism_index
from .stress import INDEX_STRESS_METHODS, index_stress, index_stress_Garcia2007

__all__ = [
    "delta_E_CAM02LCD",
    "delta_E_CAM02SCD",
    "delta_E_CAM02UCS",
]
__all__ += [
    "delta_E_CAM16LCD",
    "delta_E_CAM16SCD",
    "delta_E_CAM16UCS",
]
__all__ += [
    "JND_CIE1976",
    "delta_E_CIE1976",
    "delta_E_CIE1994",
    "delta_E_CIE2000",
    "delta_E_CMC",
    "delta_E_HyAB",
    "delta_E_HyCH",
    "delta_E_ITP",
]
__all__ += [
    "delta_E_DIN99",
]
__all__ += [
    "power_function_Huang2015",
]
__all__ += [
    "INDEX_STRESS_METHODS",
    "index_stress",
    "index_stress_Garcia2007",
]
__all__ += [
    "Lab_to_metamerism_index",
    "XYZ_to_metamerism_index",
]

DELTA_E_METHODS: CanonicalMapping = CanonicalMapping(
    {
        "CIE 1976": delta_E_CIE1976,
        "CIE 1994": delta_E_CIE1994,
        "CIE 2000": delta_E_CIE2000,
        "CMC": delta_E_CMC,
        "ITP": delta_E_ITP,
        "CAM02-LCD": delta_E_CAM02LCD,
        "CAM02-SCD": delta_E_CAM02SCD,
        "CAM02-UCS": delta_E_CAM02UCS,
        "CAM16-LCD": delta_E_CAM16LCD,
        "CAM16-SCD": delta_E_CAM16SCD,
        "CAM16-UCS": delta_E_CAM16UCS,
        "DIN99": delta_E_DIN99,
        "HyAB": delta_E_HyAB,
        "HyCH": delta_E_HyCH,
    }
)
DELTA_E_METHODS.__doc__ = """
Supported :math:`\\Delta E_{ab}` colour difference computation methods.

References
----------
:cite:`ASTMInternational2007`, :cite:`Abasi2020a`, :cite:`Li2017`,
:cite:`Lindbloom2003c`, :cite:`Lindbloom2011a`, :cite:`Lindbloom2009f`,
:cite:`Luo2006b`, :cite:`Melgosa2013b`, :cite:`Wikipedia2008b`

Aliases:

-   'cie1976': 'CIE 1976'
-   'cie1994': 'CIE 1994'
-   'cie2000': 'CIE 2000'
"""
DELTA_E_METHODS["cie1976"] = DELTA_E_METHODS["CIE 1976"]
DELTA_E_METHODS["cie1994"] = DELTA_E_METHODS["CIE 1994"]
DELTA_E_METHODS["cie2000"] = DELTA_E_METHODS["CIE 2000"]


def delta_E(
    a: ArrayLike,
    b: ArrayLike,
    method: LiteralDeltaEMethod | str = "CIE 2000",
    **kwargs: Any,
) -> NDArrayFloat:
    """
    Compute the colour difference :math:`\\Delta E_{ab}` between two
    specified *CIE L\\*a\\*b\\**, :math:`IC_TC_P`, or :math:`J'a'b'`
    colourspace arrays.

    Parameters
    ----------
    a
        *CIE L\\*a\\*b\\**, :math:`IC_TC_P`, or :math:`J'a'b'` colourspace
        array :math:`a`.
    b
        *CIE L\\*a\\*b\\**, :math:`IC_TC_P`, or :math:`J'a'b'` colourspace
        array :math:`b`.
    method
        Computation method.

    Other Parameters
    ----------------
    c
        {:func:`colour.difference.delta_E_CMC`},
        *Chroma* weighting factor.
    l
        {:func:`colour.difference.delta_E_CMC`},
        *Lightness* weighting factor.
    textiles
        {:func:`colour.difference.delta_E_CIE1994`,
        :func:`colour.difference.delta_E_CIE2000`,
        :func:`colour.difference.delta_E_DIN99`},
        Textiles application specific parametric factors
        :math:`k_L=2,\\ k_C=k_H=1,\\ k_1=0.048,\\ k_2=0.014,\\ k_E=2,\\ k_{CH}=0.5`
        weights are used instead of
        :math:`k_L=k_C=k_H=1,\\ k_1=0.045,\\ k_2=0.015,\\ k_E=k_{CH}=1.0`.

    Returns
    -------
    :class:`numpy.ndarray`
        Colour difference :math:`\\Delta E_{ab}`.

    References
    ----------
    :cite:`ASTMInternational2007`,
    :cite:`InternationalTelecommunicationUnion2019`, :cite:`Li2017`,
    :cite:`Lindbloom2003c`, :cite:`Lindbloom2011a`, :cite:`Lindbloom2009f`,
    :cite:`Luo2006b`, :cite:`Melgosa2013b`, :cite:`Wikipedia2008b`

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([48.99183622, -0.10561667, 400.65619925])
    >>> b = np.array([50.65907324, -0.11671910, 402.82235718])
    >>> delta_E(a, b)  # doctest: +ELLIPSIS
    1.6709303...
    >>> delta_E(a, b, method="CIE 2000")  # doctest: +ELLIPSIS
    1.6709303...
    >>> delta_E(a, b, method="CIE 1976")  # doctest: +ELLIPSIS
    2.7335037...
    >>> delta_E(a, b, method="CIE 1994")  # doctest: +ELLIPSIS
    1.6711191...
    >>> delta_E(a, b, method="CIE 1994", textiles=True)
    ... # doctest: +ELLIPSIS
    0.8404677...
    >>> delta_E(a, b, method="DIN99")  # doctest: +ELLIPSIS
    1.5591089...
    >>> a = np.array([0.4885468072, -0.04739350675, 0.07475401302])
    >>> b = np.array([0.4899203231, -0.04567508203, 0.07361341775])
    >>> delta_E(a, b, method="ITP")  # doctest: +ELLIPSIS
    1.42657228...
    >>> a = np.array([54.90433134, -0.08450395, -0.06854831])
    >>> b = np.array([54.90433134, -0.08442362, -0.06848314])
    >>> delta_E(a, b, method="CAM02-UCS")  # doctest: +ELLIPSIS
    0.0001034...
    >>> delta_E(a, b, method="CAM16-LCD")  # doctest: +ELLIPSIS
    0.0001034...
    >>> a = np.array([39.91531343, 51.16658481, 146.12933781])
    >>> b = np.array([53.12207516, -39.92365056, 249.54831278])
    >>> delta_E(a, b, method="HyAB")  # doctest: +ELLIPSIS
    151.0215481...
    >>> a = np.array([39.91531343, 51.16658481, 146.12933781])
    >>> b = np.array([53.12207516, -39.92365056, 249.54831278])
    >>> delta_E(a, b, method="HyCH")  # doctest: +ELLIPSIS
    48.66427941...
    """

    method = validate_method(method, tuple(DELTA_E_METHODS))

    function = DELTA_E_METHODS[method]

    return function(a, b, **filter_kwargs(function, **kwargs))


__all__ += [
    "DELTA_E_METHODS",
    "delta_E",
]
