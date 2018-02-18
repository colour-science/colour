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

See Also
--------
`Lightness Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/colorimetry/lightness.ipynb>`_

References
----------
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
-   :cite:`Lindbloom2003d` : Lindbloom, B. (2003). A Continuity Study of the
    CIE L* Function. Retrieved February 24, 2014, from
    http://brucelindbloom.com/LContinuity.html
-   :cite:`Wikipediabs` : Nayatani, Y., Sobagaki, H., & Yano, K. H. T. (1995).
    Lightness dependency of chroma scales of a nonlinear color-appearance model
    and its latest formulation. Color Research & Application, 20(3), 156-167.
    doi:10.1002/col.5080200305
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

from colour.biochemistry import reaction_rate_MichealisMenten
from colour.constants import CIE_E, CIE_K
from colour.utilities import (CaseInsensitiveMapping, as_numeric,
                              filter_kwargs, warning)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2018 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'lightness_Glasser1958', 'lightness_Wyszecki1963', 'lightness_CIE1976',
    'lightness_Fairchild2010', 'lightness_Fairchild2011', 'LIGHTNESS_METHODS',
    'lightness'
]


def lightness_Glasser1958(Y):
    """
    Returns the *Lightness* :math:`L` of given *luminance* :math:`Y` using
    *Glasser et alii (1958)* method.

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
    -   Input *luminance* :math:`Y` is in domain [0, 100].
    -   Output *Lightness* :math:`L` is in range [0, 100].

    References
    ----------
    -   :cite:`Glasser1958a`

    Examples
    --------
    >>> lightness_Glasser1958(10.08)  # doctest: +ELLIPSIS
    36.2505626...
    """

    Y = np.asarray(Y)

    L = 25.29 * (Y ** (1 / 3)) - 18.38

    return L


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
    -   Input *luminance* :math:`Y` is in domain [0, 100].
    -   Output *Lightness* :math:`W` is in range [0, 100].

    References
    ----------
    -   :cite:`Wyszecki1963b`

    Examples
    --------
    >>> lightness_Wyszecki1963(10.08)  # doctest: +ELLIPSIS
    37.0041149...
    """

    Y = np.asarray(Y)

    if np.any(Y < 1) or np.any(Y > 98):
        warning(('"W*" Lightness computation is only applicable for '
                 '1% < "Y" < 98%, unpredictable results may occur!'))

    W = 25 * (Y ** (1 / 3)) - 17

    return W


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
    -   Input *luminance* :math:`Y` and :math:`Y_n` are in domain [0, 100].
    -   Output *Lightness* :math:`L^*` is in range [0, 100].

    References
    ----------
    -   :cite:`Lindbloom2003d`
    -   :cite:`Wyszecki2000bd`

    Examples
    --------
    >>> lightness_CIE1976(10.08)  # doctest: +ELLIPSIS
    37.9856290...
    """

    Y = np.asarray(Y)
    Y_n = np.asarray(Y_n)

    Lstar = Y / Y_n

    Lstar = as_numeric(
        np.where(Lstar <= CIE_E, CIE_K * Lstar, 116 * Lstar ** (1 / 3) - 16))

    return Lstar


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
        :math:`\epsilon` exponent.

    Returns
    -------
    array_like
        *Lightness* :math:`L_{hdr}`.

    Warning
    -------
    The input domain of that definition is non standard!

    Notes
    -----
    -   Input *luminance* :math:`Y` is in domain [0, :math:`\infty`].

    References
    ----------
    -   :cite:`Fairchild2010`

    Examples
    --------
    >>> lightness_Fairchild2010(10.08 / 100)  # doctest: +ELLIPSIS
    24.9022902...
    """

    maximum_perception = 100

    Y = np.asarray(Y)

    L_hdr = reaction_rate_MichealisMenten(Y ** epsilon, maximum_perception,
                                          0.184 ** epsilon) + 0.02

    return L_hdr


def lightness_Fairchild2011(Y, epsilon=0.710, method='hdr-CIELAB'):
    """
    Computes *Lightness* :math:`L_{hdr}` of given *luminance* :math:`Y` using
    *Fairchild and Chen (2011)* method accordingly to *Michealis-Menten*
    kinetics.

    Parameters
    ----------
    Y : array_like
        *luminance* :math:`Y`.
    epsilon : numeric or array_like, optional
        :math:`\epsilon` exponent.
    method : unicode, optional
        **{'hdr-CIELAB', 'hdr-IPT'}**,
        *Lightness* :math:`L_{hdr}` computation method.

    Returns
    -------
    array_like
        *Lightness* :math:`L_{hdr}`.

    Warning
    -------
    The input domain of that definition is non standard!

    Notes
    -----
    -   Input *luminance* :math:`Y` is in domain [0, :math:`\infty`].

    References
    ----------
    -   :cite:`Fairchild2011`

    Examples
    --------
    >>> lightness_Fairchild2011(10.08 / 100)  # doctest: +ELLIPSIS
    26.45950981...
    >>> lightness_Fairchild2011(10.08 / 100, method='hdr-IPT')
    ... # doctest: +ELLIPSIS
    26.3524672...
    """

    Y = np.asarray(Y)

    if method.lower() == 'hdr-cielab':
        maximum_perception = 247
    else:
        maximum_perception = 246

    L_hdr = reaction_rate_MichealisMenten(Y ** epsilon, maximum_perception, 2
                                          ** epsilon) + 0.02

    return L_hdr


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
-   :cite:`Fairchild2010`
-   :cite:`Fairchild2011`
-   :cite:`Glasser1958a`
-   :cite:`Lindbloom2003d`
-   :cite:`Wyszecki1963b`
-   :cite:`Wyszecki2000bd`

LIGHTNESS_METHODS : CaseInsensitiveMapping
    **{'Glasser 1958', 'Wyszecki 1963', 'CIE 1976', 'Fairchild 2010',
    'Fairchild 2011'}**

Aliases:

-   'Lstar1976': 'CIE 1976'
"""
LIGHTNESS_METHODS['Lstar1976'] = LIGHTNESS_METHODS['CIE 1976']


def lightness(Y, method='CIE 1976', **kwargs):
    """
    Returns the *Lightness* :math:`L` using given method.

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
        :math:`\epsilon` exponent.

    Returns
    -------
    numeric or array_like
        *Lightness* :math:`L`.

    Notes
    -----
    -   Input *luminance* :math:`Y` and optional :math:`Y_n` are in domain
        [0, 100] or [0, :math:`\infty`].
    -   Output *Lightness* :math:`L` is in range [0, 100].

    References
    ----------
    -   :cite:`Fairchild2010`
    -   :cite:`Fairchild2011`
    -   :cite:`Glasser1958a`
    -   :cite:`Lindbloom2003d`
    -   :cite:`Wikipediabs`
    -   :cite:`Wyszecki1963b`
    -   :cite:`Wyszecki2000bd`

    Examples
    --------
    >>> lightness(10.08)  # doctest: +ELLIPSIS
    37.9856290...
    >>> lightness(10.08, Y_n=100)  # doctest: +ELLIPSIS
    37.9856290...
    >>> lightness(10.08, Y_n=95)  # doctest: +ELLIPSIS
    38.9165987...
    >>> lightness(10.08, method='Glasser 1958')  # doctest: +ELLIPSIS
    36.2505626...
    >>> lightness(10.08, method='Wyszecki 1963')  # doctest: +ELLIPSIS
    37.0041149...
    >>> lightness(10.08 / 100, epsilon=1.836, method='Fairchild 2010')
    ... # doctest: +ELLIPSIS
    24.9022902...
    """

    function = LIGHTNESS_METHODS[method]

    return function(Y, **filter_kwargs(function, **kwargs))
