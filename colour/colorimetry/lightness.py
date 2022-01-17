# -*- coding: utf-8 -*-
"""
Lightness :math:`L`
===================

Defines the *Lightness* :math:`L` computation objects.

The following methods are available:

-   :func:`colour.colorimetry.lightness_Glasser1958`: *Lightness* :math:`L`
    computation of given *luminance* :math:`Y` using
    *Glasser, Mckinney, Reilly and Schnelle (1958)* method.
-   :func:`colour.colorimetry.lightness_Wyszecki1963`: *Lightness* :math:`W`
    computation of given *luminance* :math:`Y` using *Wyszecki (1963)* method.
-   :func:`colour.colorimetry.lightness_CIE1976`: *Lightness* :math:`L^*`
    computation of given *luminance* :math:`Y` as per *CIE 1976*
    recommendation.
-   :func:`colour.colorimetry.lightness_Fairchild2010`: *Lightness*
    :math:`L_{hdr}` computation of given *luminance* :math:`Y` using
    *Fairchild and Wyble (2010)* method.
-   :func:`colour.colorimetry.lightness_Fairchild2011`: *Lightness*
    :math:`L_{hdr}` computation of given *luminance* :math:`Y` using
    *Fairchild and Chen (2011)* method.
-   :func:`colour.colorimetry.lightness_Abebe2017`: *Lightness* :math:`L`
    computation of given *luminance* :math:`Y` using
    *Abebe, Pouli, Larabi and Reinhard (2017)* method.
-   :attr:`colour.LIGHTNESS_METHODS`: Supported *Lightness* :math:`L`
    computation methods.
-   :func:`colour.lightness`: *Lightness* :math:`L` computation of given
    *luminance* :math:`Y` using given method.

References
----------
-   :cite:`Abebe2017a` : Abebe, M. A., Pouli, T., Larabi, M.-C., & Reinhard,
    E. (2017). Perceptual Lightness Modeling for High-Dynamic-Range Imaging.
    ACM Transactions on Applied Perception, 15(1), 1-19. doi:10.1145/3086577
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
-   :cite:`Glasser1958a` : Glasser, L. G., McKinney, A. H., Reilly, C. D., &
    Schnelle, P. D. (1958). Cube-Root Color Coordinate System. Journal of the
    Optical Society of America, 48(10), 736. doi:10.1364/JOSA.48.000736
-   :cite:`Wikipedia2007c` : Nayatani, Y., Sobagaki, H., & Yano, K. H. T.
    (1995). Lightness dependency of chroma scales of a nonlinear
    color-appearance model and its latest formulation. Color Research &
    Application, 20(3), 156-167. doi:10.1002/col.5080200305
-   :cite:`Wyszecki1963b` : Wyszecki, Günter. (1963). Proposal for a New
    Color-Difference Formula. Journal of the Optical Society of America,
    53(11), 1318. doi:10.1364/JOSA.53.001318
-   :cite:`Wyszecki2000bd` : Wyszecki, Günther, & Stiles, W. S. (2000). CIE
    1976 (L*u*v*)-Space and Color-Difference Formula. In Color Science:
    Concepts and Methods, Quantitative Data and Formulae (p. 167). Wiley.
    ISBN:978-0-471-39918-6
"""

import numpy as np

from colour.algebra import spow
from colour.biochemistry import (
    reaction_rate_MichaelisMenten_Michaelis1913,
    reaction_rate_MichaelisMenten_Abebe2017,
)
from colour.utilities import (
    CaseInsensitiveMapping,
    as_float_array,
    as_float,
    filter_kwargs,
    from_range_100,
    get_domain_range_scale,
    to_domain_1,
    to_domain_100,
    usage_warning,
    validate_method,
)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'lightness_Glasser1958',
    'lightness_Wyszecki1963',
    'intermediate_lightness_function_CIE1976',
    'lightness_CIE1976',
    'lightness_Fairchild2010',
    'lightness_Fairchild2011',
    'lightness_Abebe2017',
    'LIGHTNESS_METHODS',
    'lightness',
]


def lightness_Glasser1958(Y):
    """
    Returns the *Lightness* :math:`L` of given *luminance* :math:`Y` using
    *Glasser et al. (1958)* method.

    Parameters
    ----------
    Y : numeric or array_like
        *luminance* :math:`Y`.

    Returns
    -------
    numeric or array_like
        *Lightness* :math:`L`.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``Y``      | [0, 100]              | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``L``      | [0, 100]              | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Glasser1958a`

    Examples
    --------
    >>> lightness_Glasser1958(12.19722535)  # doctest: +ELLIPSIS
    39.8351264...
    """

    Y = to_domain_100(Y)

    L = 25.29 * spow(Y, 1 / 3) - 18.38

    return as_float(from_range_100(L))


def lightness_Wyszecki1963(Y):
    """
    Returns the *Lightness* :math:`W` of given *luminance* :math:`Y` using
    *Wyszecki (1963)* method.


    Parameters
    ----------
    Y : numeric or array_like
        *luminance* :math:`Y`.

    Returns
    -------
    numeric or array_like
        *Lightness* :math:`W`.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``Y``      | [0, 100]              | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``W``      | [0, 100]              | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Wyszecki1963b`

    Examples
    --------
    >>> lightness_Wyszecki1963(12.19722535)  # doctest: +ELLIPSIS
    40.5475745...
    """

    Y = to_domain_100(Y)

    if np.any(Y < 1) or np.any(Y > 98):
        usage_warning('"W*" Lightness computation is only applicable for '
                      '1% < "Y" < 98%, unpredictable results may occur!')

    W = 25 * spow(Y, 1 / 3) - 17

    return as_float(from_range_100(W))


def intermediate_lightness_function_CIE1976(Y, Y_n=100):
    """
    Returns the intermediate value :math:`f(Y/Yn)` in the *Lightness*
    :math:`L^*` computation for given *luminance* :math:`Y` using given
    reference white *luminance* :math:`Y_n` as per *CIE 1976* recommendation.

    Parameters
    ----------
    Y : numeric or array_like
        *luminance* :math:`Y`.
    Y_n : numeric or array_like, optional
        White reference *luminance* :math:`Y_n`.

    Returns
    -------
    numeric or array_like
        Intermediate value :math:`f(Y/Yn)`.

    Notes
    -----

    +-------------+-----------------------+---------------+
    | **Domain**  | **Scale - Reference** | **Scale - 1** |
    +=============+=======================+===============+
    | ``Y``       | [0, 100]              | [0, 100]      |
    +-------------+-----------------------+---------------+

    +-------------+-----------------------+---------------+
    | **Range**   | **Scale - Reference** | **Scale - 1** |
    +=============+=======================+===============+
    | ``f_Y_Y_n`` | [0, 1]                | [0, 1]        |
    +-------------+-----------------------+---------------+

    References
    ----------
    :cite:`CIETC1-482004m`, :cite:`Wyszecki2000bd`

    Examples
    --------
    >>> intermediate_lightness_function_CIE1976(12.19722535)
    ... # doctest: +ELLIPSIS
    0.4959299...
    >>> intermediate_lightness_function_CIE1976(12.19722535, 95)
    ... # doctest: +ELLIPSIS
    0.5044821...
    """

    Y = as_float_array(Y)
    Y_n = as_float_array(Y_n)

    Y_Y_n = Y / Y_n

    f_Y_Y_n = np.where(
        Y_Y_n > (24 / 116) ** 3,
        spow(Y_Y_n, 1 / 3),
        (841 / 108) * Y_Y_n + 16 / 116,
    )

    return as_float(f_Y_Y_n)


def lightness_CIE1976(Y, Y_n=100):
    """
    Returns the *Lightness* :math:`L^*` of given *luminance* :math:`Y` using
    given reference white *luminance* :math:`Y_n` as per *CIE 1976*
    recommendation.

    Parameters
    ----------
    Y : numeric or array_like
        *luminance* :math:`Y`.
    Y_n : numeric or array_like, optional
        White reference *luminance* :math:`Y_n`.

    Returns
    -------
    numeric or array_like
        *Lightness* :math:`L^*`.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``Y``      | [0, 100]              | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``L_star`` | [0, 100]              | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`CIETC1-482004m`, :cite:`Wyszecki2000bd`

    Examples
    --------
    >>> lightness_CIE1976(12.19722535)  # doctest: +ELLIPSIS
    41.5278758...
    """

    Y = to_domain_100(Y)
    Y_n = as_float_array(Y_n)

    L_star = 116 * intermediate_lightness_function_CIE1976(Y, Y_n) - 16

    return as_float(from_range_100(L_star))


def lightness_Fairchild2010(Y, epsilon=1.836):
    """
    Computes *Lightness* :math:`L_{hdr}` of given *luminance* :math:`Y` using
    *Fairchild and Wyble (2010)* method according to *Michaelis-Menten*
    kinetics.

    Parameters
    ----------
    Y : array_like
        *luminance* :math:`Y`.
    epsilon : numeric or array_like, optional
        :math:`\\epsilon` exponent.

    Returns
    -------
    array_like
        *Lightness* :math:`L_{hdr}`.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``Y``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``L_hdr``  | [0, 100]              | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Fairchild2010`

    Examples
    --------
    >>> lightness_Fairchild2010(12.19722535 / 100)  # doctest: +ELLIPSIS
    31.9963902...
    """

    Y = to_domain_1(Y)

    maximum_perception = 100

    L_hdr = reaction_rate_MichaelisMenten_Michaelis1913(
        spow(Y, epsilon), maximum_perception, 0.184 ** epsilon) + 0.02

    return as_float(from_range_100(L_hdr))


def lightness_Fairchild2011(Y, epsilon=0.474, method='hdr-CIELAB'):
    """
    Computes *Lightness* :math:`L_{hdr}` of given *luminance* :math:`Y` using
    *Fairchild and Chen (2011)* method according to *Michaelis-Menten*
    kinetics.

    Parameters
    ----------
    Y : array_like
        *luminance* :math:`Y`.
    epsilon : numeric or array_like, optional
        :math:`\\epsilon` exponent.
    method : str, optional
        **{'hdr-CIELAB', 'hdr-IPT'}**,
        *Lightness* :math:`L_{hdr}` computation method.

    Returns
    -------
    array_like
        *Lightness* :math:`L_{hdr}`.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``Y``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``L_hdr``  | [0, 100]              | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Fairchild2011`

    Examples
    --------
    >>> lightness_Fairchild2011(12.19722535 / 100)  # doctest: +ELLIPSIS
    51.8529584...
    >>> lightness_Fairchild2011(12.19722535 / 100, method='hdr-IPT')
    ... # doctest: +ELLIPSIS
    51.6431084...
    """

    Y = to_domain_1(Y)
    method = validate_method(method, ['hdr-CIELAB', 'hdr-IPT'])

    if method == 'hdr-cielab':
        maximum_perception = 247
    else:
        maximum_perception = 246

    L_hdr = reaction_rate_MichaelisMenten_Michaelis1913(
        spow(Y, epsilon), maximum_perception, 2 ** epsilon) + 0.02

    return as_float(from_range_100(L_hdr))


def lightness_Abebe2017(Y, Y_n=100, method='Michaelis-Menten'):
    """
    Computes *Lightness* :math:`L` of given *luminance* :math:`Y` using
    *Abebe, Pouli, Larabi and Reinhard (2017)* method according to
    *Michaelis-Menten* kinetics or *Stevens's Power Law*.

    Parameters
    ----------
    Y : array_like
        *luminance* :math:`Y` in :math:`cd/m^2`.
    Y_n : numeric or array_like, optional
        Adapting luminance :math:`Y_n` in :math:`cd/m^2`.
    method : str, optional
        **{'Michaelis-Menten', 'Stevens'}**,
        *Lightness* :math:`L` computation method.

    Returns
    -------
    array_like
        *Lightness* :math:`L`.

    Notes
    -----

    -   *Abebe, Pouli, Larabi and Reinhard (2017)* method uses absolute
        luminance levels, thus the domain and range values for the *Reference*
        and *1* scales are only indicative that the data is not affected by
        scale transformations.

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``Y``      | ``UN``                | ``UN``        |
    +------------+-----------------------+---------------+
    | ``Y_n``    | ``UN``                | ``UN``        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``L``      | ``UN``                | ``UN``        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Abebe2017a`

    Examples
    --------
    >>> lightness_Abebe2017(12.19722535)  # doctest: +ELLIPSIS
    0.4869555...
    >>> lightness_Abebe2017(12.19722535, method='Stevens')
    ... # doctest: +ELLIPSIS
    0.4745447...
    """

    Y = as_float_array(Y)
    Y_n = as_float_array(Y_n)
    method = validate_method(method, ['Michaelis-Menten', 'Stevens'])

    Y_Y_n = Y / Y_n
    if method == 'stevens':
        L = np.where(
            Y_n <= 100,
            1.226 * spow(Y_Y_n, 0.266) - 0.226,
            1.127 * spow(Y_Y_n, 0.230) - 0.127,
        )
    else:
        L = np.where(
            Y_n <= 100,
            reaction_rate_MichaelisMenten_Abebe2017(
                spow(Y_Y_n, 0.582), 1.448, 0.635, 0.813),
            reaction_rate_MichaelisMenten_Abebe2017(
                spow(Y_Y_n, 0.293), 1.680, 1.584, 0.096),
        )

    return as_float(L)


LIGHTNESS_METHODS = CaseInsensitiveMapping({
    'Glasser 1958': lightness_Glasser1958,
    'Wyszecki 1963': lightness_Wyszecki1963,
    'CIE 1976': lightness_CIE1976,
    'Fairchild 2010': lightness_Fairchild2010,
    'Fairchild 2011': lightness_Fairchild2011,
    'Abebe 2017': lightness_Abebe2017
})
LIGHTNESS_METHODS.__doc__ = """
Supported *Lightness* computation methods.

References
----------
:cite:`CIETC1-482004m`, :cite:`Fairchild2010`, :cite:`Fairchild2011`,
:cite:`Glasser1958a`, :cite:`Wyszecki1963b`, :cite:`Wyszecki2000bd`

LIGHTNESS_METHODS : CaseInsensitiveMapping
    **{'Glasser 1958', 'Wyszecki 1963', 'CIE 1976', 'Fairchild 2010',
    'Fairchild 2011', 'Abebe 2017'}**

Aliases:

-   'Lstar1976': 'CIE 1976'
"""
LIGHTNESS_METHODS['Lstar1976'] = LIGHTNESS_METHODS['CIE 1976']


def lightness(Y, method='CIE 1976', **kwargs):
    """
    Returns the *Lightness* :math:`L` of given *luminance* :math:`Y` using
    given method.

    Parameters
    ----------
    Y : numeric or array_like
        *luminance* :math:`Y`.
    method : str, optional
        **{'CIE 1976', 'Glasser 1958', 'Wyszecki 1963', 'Fairchild 2010',
        'Fairchild 2011', 'Abebe 2017'}**,
        Computation method.

    Other Parameters
    ----------------
    Y_n : numeric or array_like, optional
        {:func:`colour.colorimetry.lightness_Abebe2017`,
        :func:`colour.colorimetry.lightness_CIE1976`},
        White reference *luminance* :math:`Y_n`.
    epsilon : numeric or array_like, optional
        {:func:`colour.colorimetry.lightness_Fairchild2010`,
        :func:`colour.colorimetry.lightness_Fairchild2011`},
        :math:`\\epsilon` exponent.

    Returns
    -------
    numeric or array_like
        *Lightness* :math:`L`.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``Y``      | [0, 100]              | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``L``      | [0, 100]              | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Abebe2017a`, :cite:`CIETC1-482004m`, :cite:`Fairchild2010`,
    :cite:`Fairchild2011`, :cite:`Glasser1958a`, :cite:`Wikipedia2007c`,
    :cite:`Wyszecki1963b`, :cite:`Wyszecki2000bd`

    Examples
    --------
    >>> lightness(12.19722535)  # doctest: +ELLIPSIS
    41.5278758...
    >>> lightness(12.19722535, Y_n=100)  # doctest: +ELLIPSIS
    41.5278758...
    >>> lightness(12.19722535, Y_n=95)  # doctest: +ELLIPSIS
    42.5199307...
    >>> lightness(12.19722535, method='Glasser 1958')  # doctest: +ELLIPSIS
    39.8351264...
    >>> lightness(12.19722535, method='Wyszecki 1963')  # doctest: +ELLIPSIS
    40.5475745...
    >>> lightness(12.19722535, epsilon=0.710, method='Fairchild 2011')
    ... # doctest: +ELLIPSIS
    29.8295108...
    >>> lightness(12.19722535, epsilon=0.710, method='Fairchild 2011')
    ... # doctest: +ELLIPSIS
    29.8295108...
    >>> lightness(12.19722535, method='Abebe 2017')
    ... # doctest: +ELLIPSIS
    48.6955571...
    """

    Y = as_float_array(Y)
    method = validate_method(method, LIGHTNESS_METHODS)

    function = LIGHTNESS_METHODS[method]

    # NOTE: "Abebe et al. (2017)" uses absolute luminance levels and has
    # undefined domain-range scale, yet we modify its behaviour consistency
    # with the other methods.
    domain_range_reference = get_domain_range_scale() == 'reference'
    domain_range_1 = get_domain_range_scale() == '1'
    domain_range_100 = get_domain_range_scale() == '100'

    domain_1 = (lightness_Fairchild2010, lightness_Fairchild2011)
    domain_undefined = (lightness_Abebe2017, )

    if function in domain_1 and domain_range_reference:
        Y = Y / 100

    if function in domain_undefined and domain_range_1:
        Y = Y * 100

    L = function(Y, **filter_kwargs(function, **kwargs))

    if function in domain_undefined and (domain_range_reference or
                                         domain_range_100):
        L *= 100

    return L
