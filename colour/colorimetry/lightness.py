# -*- coding: utf-8 -*-
"""
Lightness :math:`L`
===================

Defines *Lightness* :math:`L` computation objects.

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
-   :attr:`colour.LIGHTNESS_METHODS`: Supported *Lightness* :math:`L`
    computations methods.
-   :func:`colour.lightness`: *Lightness* :math:`L` computation of given
    *luminance* :math:`Y` using given method.

See Also
--------
`Lightness Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/colorimetry/lightness.ipynb>`_

References
----------
-   :cite:`CIETC1-482004m` : CIE TC 1-48. (2004). CIE 1976 uniform colour
    spaces. In CIE 015:2004 Colorimetry, 3rd Edition (p. 24).
    ISBN:978-3-901-90633-6
-   :cite:`Fairchild2010` : Fairchild, M. D., & Wyble, D. R. (2010).
    hdr-CIELAB and hdr-IPT: Simple Models for Describing the Color of
    High-Dynamic-Range and Wide-Color-Gamut Images. In Proc. of Color and
    Imaging Conference (pp. 322-326). ISBN:9781629932156
-   :cite:`Fairchild2011` : Fairchild, M. D., & Chen, P. (2011). Brightness,
    lightness, and specifying color in high-dynamic-range scenes and images.
    In S. P. Farnand & F. Gaykema (Eds.), Proc. SPIE 7867, Image Quality and
    System Performance VIII (p. 78670O). doi:10.1117/12.872075
-   :cite:`Glasser1958a` : Glasser, L. G., McKinney, A. H., Reilly, C. D., &
    Schnelle, P. D. (1958). Cube-Root Color Coordinate System. Journal of the
    Optical Society of America, 48(10), 736. doi:10.1364/JOSA.48.000736
-   :cite:`Wikipedia2007c` : Wikipedia. (2007). Lightness. Retrieved April
    13, 2014, from http://en.wikipedia.org/wiki/Lightness
-   :cite:`Wyszecki1963b` : Wyszecki, G. (1963). Proposal for a New
    Color-Difference Formula. Journal of the Optical Society of America,
    53(11), 1318. doi:10.1364/JOSA.53.001318
-   :cite:`Wyszecki2000bd` : Wyszecki, G., & Stiles, W. S. (2000). CIE 1976
    (L*u*v*)-Space and Color-Difference Formula. In Color Science: Concepts and
    Methods, Quantitative Data and Formulae (p. 167). Wiley.
    ISBN:978-0471399186
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.algebra import spow
from colour.biochemistry import reaction_rate_MichealisMenten
from colour.utilities import (CaseInsensitiveMapping, as_float_array, as_float,
                              filter_kwargs, from_range_100,
                              get_domain_range_scale, to_domain_1,
                              to_domain_100, usage_warning)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'lightness_Glasser1958', 'lightness_Wyszecki1963',
    'intermediate_lightness_function_CIE1976', 'lightness_CIE1976',
    'lightness_Fairchild2010', 'lightness_Fairchild2011', 'LIGHTNESS_METHODS',
    'lightness'
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

    return from_range_100(L)


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

    return from_range_100(W)


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

    f_Y_Y_n = as_float(
        np.where(
            Y_Y_n > (24 / 116) ** 3,
            spow(Y_Y_n, 1 / 3),
            (841 / 108) * Y_Y_n + 16 / 116,
        ))

    return f_Y_Y_n


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

    return from_range_100(L_star)


def lightness_Fairchild2010(Y, epsilon=1.836):
    """
    Computes *Lightness* :math:`L_{hdr}` of given *luminance* :math:`Y` using
    *Fairchild and Wyble (2010)* method according to *Michealis-Menten*
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

    Warning
    -------
    The input domain of that definition is non standard!

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

    L_hdr = reaction_rate_MichealisMenten(
        spow(Y, epsilon), maximum_perception, 0.184 ** epsilon) + 0.02

    return from_range_100(L_hdr)


def lightness_Fairchild2011(Y, epsilon=0.474, method='hdr-CIELAB'):
    """
    Computes *Lightness* :math:`L_{hdr}` of given *luminance* :math:`Y` using
    *Fairchild and Chen (2011)* method according to *Michealis-Menten*
    kinetics.

    Parameters
    ----------
    Y : array_like
        *luminance* :math:`Y`.
    epsilon : numeric or array_like, optional
        :math:`\\epsilon` exponent.
    method : unicode, optional
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

    if method.lower() == 'hdr-cielab':
        maximum_perception = 247
    else:
        maximum_perception = 246

    L_hdr = reaction_rate_MichealisMenten(
        spow(Y, epsilon), maximum_perception, 2 ** epsilon) + 0.02

    return from_range_100(L_hdr)


LIGHTNESS_METHODS = CaseInsensitiveMapping({
    'Glasser 1958': lightness_Glasser1958,
    'Wyszecki 1963': lightness_Wyszecki1963,
    'CIE 1976': lightness_CIE1976,
    'Fairchild 2010': lightness_Fairchild2010,
    'Fairchild 2011': lightness_Fairchild2011
})
LIGHTNESS_METHODS.__doc__ = """
Supported *Lightness* computations methods.

References
----------
:cite:`CIETC1-482004m`, :cite:`Fairchild2010`, :cite:`Fairchild2011`,
:cite:`Glasser1958a`, :cite:`Wyszecki1963b`, :cite:`Wyszecki2000bd`

LIGHTNESS_METHODS : CaseInsensitiveMapping
    **{'Glasser 1958', 'Wyszecki 1963', 'CIE 1976', 'Fairchild 2010',
    'Fairchild 2011'}**

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
    method : unicode, optional
        **{'CIE 1976', 'Glasser 1958', 'Wyszecki 1963', 'Fairchild 2010',
        'Fairchild 2011'}**,
        Computation method.

    Other Parameters
    ----------------
    Y_n : numeric or array_like, optional
        {:func:`colour.colorimetry.lightness_CIE1976`},
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
    :cite:`CIETC1-482004m`, :cite:`Fairchild2010`, :cite:`Fairchild2011`,
    :cite:`Glasser1958a`, :cite:`Wikipedia2007c`, :cite:`Wyszecki1963b`,
    :cite:`Wyszecki2000bd`

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
    """

    Y = as_float_array(Y)

    function = LIGHTNESS_METHODS[method]

    domain_range_reference = get_domain_range_scale() == 'reference'
    domain_1 = (lightness_Fairchild2010, lightness_Fairchild2011)

    if function in domain_1 and domain_range_reference:
        Y = Y / 100

    return function(Y, **filter_kwargs(function, **kwargs))
