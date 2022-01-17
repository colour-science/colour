# -*- coding: utf-8 -*-
"""
Luminance :math:`Y`
===================

Defines the *luminance* :math:`Y` computation objects.

The following methods are available:

-   :func:`colour.colorimetry.luminance_Newhall1943`: *luminance* :math:`Y`
    computation of given *Munsell* value :math:`V` using
    *Newhall, Nickerson and Judd (1943)* method.
-   :func:`colour.colorimetry.luminance_ASTMD1535`: *luminance* :math:`Y`
    computation of given *Munsell* value :math:`V` using *ASTM D1535-08e1*
    method.
-   :func:`colour.colorimetry.luminance_CIE1976`: *luminance* :math:`Y`
    computation of given *Lightness* :math:`L^*` as per *CIE 1976*
    recommendation.
-   :func:`colour.colorimetry.luminance_Fairchild2010`: *luminance* :math:`Y`
    computation of given *Lightness* :math:`L_{hdr}` using
    *Fairchild and Wyble (2010)* method.
-   :func:`colour.colorimetry.luminance_Fairchild2011`: *luminance* :math:`Y`
    computation of given *Lightness* :math:`L_{hdr}` using
    *Fairchild and Chen (2011)* method.
-   :func:`colour.colorimetry.luminance_Abebe2017`: *Luminance* :math:`Y`
    computation of given *Lightness* :math:`L` using
    *Abebe, Pouli, Larabi and Reinhard (2017)* method.
-   :attr:`colour.LUMINANCE_METHODS`: Supported *luminance* :math:`Y`
    computation methods.
-   :func:`colour.luminance`: *Luminance* :math:`Y` computation of given
    *Lightness* :math:`L^*` or given *Munsell* value :math:`V` using given
    method.

References
----------
-   :cite:`Abebe2017a` : Abebe, M. A., Pouli, T., Larabi, M.-C., & Reinhard,
    E. (2017). Perceptual Lightness Modeling for High-Dynamic-Range Imaging.
    ACM Transactions on Applied Perception, 15(1), 1-19. doi:10.1145/3086577
-   :cite:`ASTMInternational2008a` : ASTM International. (2008). ASTM
    D1535-08e1 - Standard Practice for Specifying Color by the Munsell System.
    doi:10.1520/D1535-08E01
-   :cite:`CIETC1-482004m` : CIE TC 1-48. (2004). CIE 1976 uniform colour
    spaces. In CIE 015:2004 Colorimetry, 3rd Edition (p. 24).
    ISBN:978-3-901906-33-6
-   :cite:`Fairchild2010` : Fairchild, M. D., & Wyble, D. R. (2010). hdr-CIELAB
    and hdr-IPT: Simple Models for Describing the Color of High-Dynamic-Range
    and Wide-Color-Gamut Images. Proc. of Color and Imaging Conference,
    322-326. ISBN:978-1-62993-215-6
-   :cite:`Fairchild2011` : Fairchild, M. D., & Chen, P. (2011). Brightness,
    lightness, and specifying color in high-dynamic-range scenes and images. In
    S. P. Farnand & F. Gaykema (Eds.), Proc. SPIE 7867, Image Quality and
    System Performance VIII (p. 78670O). doi:10.1117/12.872075
-   :cite:`Newhall1943a` : Newhall, S. M., Nickerson, D., & Judd, D. B. (1943).
    Final Report of the OSA Subcommittee on the Spacing of the Munsell Colors.
    Journal of the Optical Society of America, 33(7), 385.
    doi:10.1364/JOSA.33.000385
-   :cite:`Wikipedia2001b` : Wikipedia. (2001). Luminance. Retrieved February
    10, 2018, from https://en.wikipedia.org/wiki/Luminance
-   :cite:`Wyszecki2000bd` : Wyszecki, Günther, & Stiles, W. S. (2000). CIE
    1976 (L*u*v*)-Space and Color-Difference Formula. In Color Science:
    Concepts and Methods, Quantitative Data and Formulae (p. 167). Wiley.
    ISBN:978-0-471-39918-6
"""

import numpy as np

from colour.algebra import spow
from colour.biochemistry import (
    substrate_concentration_MichaelisMenten_Michaelis1913,
    substrate_concentration_MichaelisMenten_Abebe2017,
)
from colour.utilities import (
    CaseInsensitiveMapping,
    as_float_array,
    as_float,
    filter_kwargs,
    from_range_1,
    from_range_100,
    get_domain_range_scale,
    to_domain_10,
    to_domain_100,
    validate_method,
)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'luminance_Newhall1943',
    'luminance_ASTMD1535',
    'intermediate_luminance_function_CIE1976',
    'luminance_CIE1976',
    'luminance_Fairchild2010',
    'luminance_Fairchild2011',
    'luminance_Abebe2017',
    'LUMINANCE_METHODS',
    'luminance',
]


def luminance_Newhall1943(V):
    """
    Returns the *luminance* :math:`R_Y` of given *Munsell* value :math:`V`
    using *Newhall et al. (1943)* method.

    Parameters
    ----------
    V : numeric or array_like
        *Munsell* value :math:`V`.

    Returns
    -------
    numeric or array_like
        *luminance* :math:`R_Y`.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``V``      | [0, 10]               | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``R_Y``    | [0, 100]              | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Newhall1943a`

    Examples
    --------
    >>> luminance_Newhall1943(4.08244375)  # doctest: +ELLIPSIS
    12.5500788...
    """

    V = to_domain_10(V)

    R_Y = (1.2219 * V - 0.23111 * (V * V) + 0.23951 * (V ** 3) -
           0.021009 * (V ** 4) + 0.0008404 * (V ** 5))

    return as_float(from_range_100(R_Y))


def luminance_ASTMD1535(V):
    """
    Returns the *luminance* :math:`Y` of given *Munsell* value :math:`V` using
    *ASTM D1535-08e1* method.

    Parameters
    ----------
    V : numeric or array_like
        *Munsell* value :math:`V`.

    Returns
    -------
    numeric or array_like
        *luminance* :math:`Y`.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``V``      | [0, 10]               | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``Y``      | [0, 100]              | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`ASTMInternational2008a`

    Examples
    --------
    >>> luminance_ASTMD1535(4.08244375)  # doctest: +ELLIPSIS
    12.2363426...
    """

    V = to_domain_10(V)

    Y = (1.1914 * V - 0.22533 * (V ** 2) + 0.23352 * (V ** 3) -
         0.020484 * (V ** 4) + 0.00081939 * (V ** 5))

    return as_float(from_range_100(Y))


def intermediate_luminance_function_CIE1976(f_Y_Y_n, Y_n=100):
    """
    Returns the *luminance* :math:`Y` in the *luminance* :math:`Y`
    computation for given intermediate value :math:`f(Y/Yn)` using given
    reference white *luminance* :math:`Y_n` as per *CIE 1976* recommendation.

    Parameters
    ----------
    f_Y_Y_n : numeric or array_like
        Intermediate value :math:`f(Y/Yn)`.
    Y_n : numeric or array_like
        White reference *luminance* :math:`Y_n`.

    Returns
    -------
    numeric or array_like
        *luminance* :math:`Y`.

    Notes
    -----

    +-------------+-----------------------+---------------+
    | **Domain**  | **Scale - Reference** | **Scale - 1** |
    +=============+=======================+===============+
    | ``f_Y_Y_n`` | [0, 1]                | [0, 1]        |
    +-------------+-----------------------+---------------+

    +-------------+-----------------------+---------------+
    | **Range**   | **Scale - Reference** | **Scale - 1** |
    +=============+=======================+===============+
    | ``Y``       | [0, 100]              | [0, 100]      |
    +-------------+-----------------------+---------------+

    References
    ----------
    :cite:`CIETC1-482004m`, :cite:`Wyszecki2000bd`

    Examples
    --------
    >>> intermediate_luminance_function_CIE1976(0.495929964178047)
    ... # doctest: +ELLIPSIS
    12.1972253...
    >>> intermediate_luminance_function_CIE1976(0.504482161449319, 95)
    ... # doctest: +ELLIPSIS
    12.1972253...
    """

    f_Y_Y_n = as_float_array(f_Y_Y_n)
    Y_n = as_float_array(Y_n)

    Y = np.where(
        f_Y_Y_n > 24 / 116,
        Y_n * f_Y_Y_n ** 3,
        Y_n * (f_Y_Y_n - 16 / 116) * (108 / 841),
    )

    return as_float(Y)


def luminance_CIE1976(L_star, Y_n=100):
    """
    Returns the *luminance* :math:`Y` of given *Lightness* :math:`L^*` with
    given reference white *luminance* :math:`Y_n`.

    Parameters
    ----------
    L_star : numeric or array_like
        *Lightness* :math:`L^*`
    Y_n : numeric or array_like
        White reference *luminance* :math:`Y_n`.

    Returns
    -------
    numeric or array_like
        *luminance* :math:`Y`.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``L_star`` | [0, 100]              | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``Y``      | [0, 100]              | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`CIETC1-482004m`, :cite:`Wyszecki2000bd`

    Examples
    --------
    >>> luminance_CIE1976(41.527875844653451)  # doctest: +ELLIPSIS
    12.1972253...
    >>> luminance_CIE1976(41.527875844653451, 95)  # doctest: +ELLIPSIS
    11.5873640...
    """

    L_star = to_domain_100(L_star)
    Y_n = as_float_array(Y_n)

    f_Y_Y_n = (L_star + 16) / 116

    Y = intermediate_luminance_function_CIE1976(f_Y_Y_n, Y_n)

    return as_float(from_range_100(Y))


def luminance_Fairchild2010(L_hdr, epsilon=1.836):
    """
    Computes *luminance* :math:`Y` of given *Lightness* :math:`L_{hdr}` using
    *Fairchild and Wyble (2010)* method according to *Michaelis-Menten*
    kinetics.

    Parameters
    ----------
    L_hdr : array_like
        *Lightness* :math:`L_{hdr}`.
    epsilon : numeric or array_like, optional
        :math:`\\epsilon` exponent.

    Returns
    -------
    array_like
        *luminance* :math:`Y`.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``L_hdr``  | [0, 100]              | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``Y``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Fairchild2010`

    Examples
    --------
    >>> luminance_Fairchild2010(31.996390226262736, 1.836)
    ... # doctest: +ELLIPSIS
    0.1219722...
    """

    L_hdr = to_domain_100(L_hdr)

    Y = np.exp(
        np.log(
            substrate_concentration_MichaelisMenten_Michaelis1913(
                L_hdr - 0.02, 100, 0.184 ** epsilon)) / epsilon)

    return as_float(from_range_1(Y))


def luminance_Fairchild2011(L_hdr, epsilon=0.474, method='hdr-CIELAB'):
    """
    Computes *luminance* :math:`Y` of given *Lightness* :math:`L_{hdr}` using
    *Fairchild and Chen (2011)* method according to *Michaelis-Menten*
    kinetics.

    Parameters
    ----------
    L_hdr : array_like
        *Lightness* :math:`L_{hdr}`.
    epsilon : numeric or array_like, optional
        :math:`\\epsilon` exponent.
    method : str, optional
        **{'hdr-CIELAB', 'hdr-IPT'}**,
        *Lightness* :math:`L_{hdr}` computation method.

    Returns
    -------
    array_like
        *luminance* :math:`Y`.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``L_hdr``  | [0, 100]              | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``Y``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Fairchild2011`

    Examples
    --------
    >>> luminance_Fairchild2011(51.852958445912506)  # doctest: +ELLIPSIS
    0.1219722...
    >>> luminance_Fairchild2011(51.643108411718522, method='hdr-IPT')
    ... # doctest: +ELLIPSIS
    0.1219722...
    """

    L_hdr = to_domain_100(L_hdr)
    method = validate_method(method, ['hdr-CIELAB', 'hdr-IPT'])

    if method == 'hdr-cielab':
        maximum_perception = 247
    else:
        maximum_perception = 246

    Y = np.exp(
        np.log(
            substrate_concentration_MichaelisMenten_Michaelis1913(
                L_hdr - 0.02, maximum_perception, 2 ** epsilon)) / epsilon)

    return as_float(from_range_1(Y))


def luminance_Abebe2017(L, Y_n=100, method='Michaelis-Menten'):
    """
    Computes *luminance* :math:`Y` of *Lightness* :math:`L` using
    *Abebe, Pouli, Larabi and Reinhard (2017)* method according to
    *Michaelis-Menten* kinetics or *Stevens's Power Law*.

    Parameters
    ----------
    L : array_like
        *Lightness* :math:`L`.
    Y_n : numeric or array_like, optional
        Adapting luminance :math:`Y_n` in :math:`cd/m^2`.
    method : str, optional
        **{'Michaelis-Menten', 'Stevens'}**,
        *Luminance* :math:`Y` computation method.

    Returns
    -------
    array_like
        *Luminance* :math:`Y` in :math:`cd/m^2`.

    Notes
    -----

    -   *Abebe, Pouli, Larabi and Reinhard (2017)* method uses absolute
        luminance levels, thus the domain and range values for the *Reference*
        and *1* scales are only indicative that the data is not affected by
        scale transformations.

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``L``      | ``UN``                | ``UN``        |
    +------------+-----------------------+---------------+
    | ``Y_n``    | ``UN``                | ``UN``        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``Y``      | ``UN``                | ``UN``        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Abebe2017a`

    Examples
    --------
    >>> luminance_Abebe2017(0.486955571109229)  # doctest: +ELLIPSIS
    12.1972253...
    >>> luminance_Abebe2017(0.474544792145434, method='Stevens')
    ... # doctest: +ELLIPSIS
    12.1972253...
    """

    L = as_float_array(L)
    Y_n = as_float_array(Y_n)
    method = validate_method(method, ['Michaelis-Menten', 'Stevens'])

    if method == 'stevens':
        Y = np.where(
            Y_n <= 100,
            spow((L + 0.226) / 1.226, 1 / 0.266),
            spow((L + 0.127) / 1.127, 1 / 0.230),
        )
    else:
        Y = np.where(
            Y_n <= 100,
            spow(
                substrate_concentration_MichaelisMenten_Abebe2017(
                    L, 1.448, 0.635, 0.813), 1 / 0.582),
            spow(
                substrate_concentration_MichaelisMenten_Abebe2017(
                    L, 1.680, 1.584, 0.096), 1 / 0.293),
        )
    Y = Y * Y_n

    return as_float(Y)


LUMINANCE_METHODS = CaseInsensitiveMapping({
    'Newhall 1943': luminance_Newhall1943,
    'ASTM D1535': luminance_ASTMD1535,
    'CIE 1976': luminance_CIE1976,
    'Fairchild 2010': luminance_Fairchild2010,
    'Fairchild 2011': luminance_Fairchild2011,
    'Abebe 2017': luminance_Abebe2017
})
LUMINANCE_METHODS.__doc__ = """
Supported *luminance* computation methods.

References
----------
:cite:`ASTMInternational2008a`, :cite:`CIETC1-482004m`, :cite:`Fairchild2010`,
:cite:`Fairchild2011`, :cite:`Newhall1943a`, :cite:`Wyszecki2000bd`

LUMINANCE_METHODS : CaseInsensitiveMapping
    **{'Newhall 1943', 'ASTM D1535', 'CIE 1976', 'Fairchild 2010',
    'Fairchild 2011', 'Abebe 2017'}**

Aliases:

-   'astm2008': 'ASTM D1535'
-   'cie1976': 'CIE 1976'
"""
LUMINANCE_METHODS['astm2008'] = LUMINANCE_METHODS['ASTM D1535']
LUMINANCE_METHODS['cie1976'] = LUMINANCE_METHODS['CIE 1976']


def luminance(LV, method='CIE 1976', **kwargs):
    """
    Returns the *luminance* :math:`Y` of given *Lightness* :math:`L^*` or given
    *Munsell* value :math:`V`.

    Parameters
    ----------
    LV : numeric or array_like
        *Lightness* :math:`L^*` or *Munsell* value :math:`V`.
    method : str, optional
        **{'CIE 1976', 'Newhall 1943', 'ASTM D1535', 'Fairchild 2010',
        'Fairchild 2011', 'Abebe 2017'}**,
        Computation method.

    Other Parameters
    ----------------
    Y_n : numeric or array_like, optional
        {:func:`colour.colorimetry.luminance_CIE1976`},
        White reference *luminance* :math:`Y_n`.
    epsilon : numeric or array_like, optional
        {:func:`colour.colorimetry.lightness_Fairchild2010`,
        :func:`colour.colorimetry.lightness_Fairchild2011`},
        :math:`\\epsilon` exponent.

    Returns
    -------
    numeric or array_like
        *luminance* :math:`Y`.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``LV``     | [0, 100]              | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``Y``      | [0, 100]              | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Abebe2017a`, :cite:`ASTMInternational2008a`, :cite:`CIETC1-482004m`,
    :cite:`Fairchild2010`, :cite:`Fairchild2011`, :cite:`Newhall1943a`,
    :cite:`Wikipedia2001b`, :cite:`Wyszecki2000bd`

    Examples
    --------
    >>> luminance(41.527875844653451)  # doctest: +ELLIPSIS
    12.1972253...
    >>> luminance(41.527875844653451, Y_n=100)  # doctest: +ELLIPSIS
    12.1972253...
    >>> luminance(42.51993072812094, Y_n=95)  # doctest: +ELLIPSIS
    12.1972253...
    >>> luminance(4.08244375 * 10, method='Newhall 1943')
    ... # doctest: +ELLIPSIS
    12.5500788...
    >>> luminance(4.08244375 * 10, method='ASTM D1535')
    ... # doctest: +ELLIPSIS
    12.2363426...
    >>> luminance(29.829510892279330, epsilon=0.710, method='Fairchild 2011')
    ... # doctest: +ELLIPSIS
    12.1972253...
    """

    LV = as_float_array(LV)
    method = validate_method(method, LUMINANCE_METHODS)

    function = LUMINANCE_METHODS[method]

    # NOTE: "Abebe et al. (2017)" uses absolute luminance levels and has
    # undefined domain-range scale, yet we modify its behaviour consistency
    # with the other methods.
    domain_range_reference = get_domain_range_scale() == 'reference'
    domain_range_1 = get_domain_range_scale() == '1'

    domain_1 = (luminance_Fairchild2010, luminance_Fairchild2011)
    domain_10 = (luminance_Newhall1943, luminance_ASTMD1535)
    domain_undefined = (luminance_Abebe2017, )

    if function in domain_10 and domain_range_reference:
        LV = LV / 10

    if function in domain_undefined and domain_range_1:
        LV = LV * 100

    Y_V = function(LV, **filter_kwargs(function, **kwargs))

    if function in domain_1 and domain_range_reference:
        Y_V *= 100

    if function in domain_undefined and domain_range_1:
        Y_V /= 100

    return Y_V
