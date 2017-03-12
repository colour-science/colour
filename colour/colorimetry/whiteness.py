#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Whiteness Index :math:`W`
=========================

Defines *whiteness* index :math:`W` computation objects:

-   :func:`whiteness_Berger1959`
-   :func:`whiteness_Taube1960`
-   :func:`whiteness_Stensby1968`
-   :func:`whiteness_ASTM313`
-   :func:`whiteness_Ganz1979`
-   :func:`whiteness_CIE2004`

See Also
--------
`Whiteness Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/colorimetry/whiteness.ipynb>`_

References
----------
.. [1]  Wikipedia. (n.d.). Whiteness. Retrieved September 17, 2014, from
        http://en.wikipedia.org/wiki/Whiteness
.. [2]  X-Rite, & Pantone. (2012). Color iQC and Color iMatch Color
        Calculations Guide. Retrieved from
        http://www.xrite.com/documents/literature/en/\
09_Color_Calculations_en.pdf
.. [3]  Wyszecki, G., & Stiles, W. S. (2000). Table I(6.5.3) Whiteness
        Formulae (Whiteness Measure Denoted by W). In Color Science: Concepts
        and Methods, Quantitative Data and Formulae (pp. 837–839). Wiley.
        ISBN:978-0471399186
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.utilities import CaseInsensitiveMapping, tsplit, tstack

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['whiteness_Berger1959',
           'whiteness_Taube1960',
           'whiteness_Stensby1968',
           'whiteness_ASTM313',
           'whiteness_Ganz1979',
           'whiteness_CIE2004',
           'WHITENESS_METHODS',
           'whiteness']


def whiteness_Berger1959(XYZ, XYZ_0):
    """
    Returns the *whiteness* index :math:`WI` of given sample *CIE XYZ*
    tristimulus values using *Berger (1959)* method. [2]_

    Parameters
    ----------
    XYZ : array_like
        *CIE XYZ* tristimulus values of sample.
    XYZ_0 : array_like
        *CIE XYZ* tristimulus values of reference white.

    Returns
    -------
    numeric or ndarray
        *Whiteness* :math:`WI`.

    Notes
    -----
    -   Input *CIE XYZ* and *CIE XYZ_0* tristimulus values are in domain
        [0, 100].
    -   *Whiteness* :math:`WI` values larger than 33.33 indicate a bluish
        white and values smaller than 33.33 indicate a yellowish white.

    Warning
    -------
    The input domain of that definition is non standard!

    Examples
    --------
    >>> XYZ = np.array([95.00000000, 100.00000000, 105.00000000])
    >>> XYZ_0 = np.array([94.80966767, 100.00000000, 107.30513595])
    >>> whiteness_Berger1959(XYZ, XYZ_0)  # doctest: +ELLIPSIS
    30.3638017...
    """

    X, Y, Z = tsplit(XYZ)
    X_0, _Y_0, Z_0 = tsplit(XYZ_0)

    WI = 0.333 * Y + 125 * (Z / Z_0) - 125 * (X / X_0)

    return WI


def whiteness_Taube1960(XYZ, XYZ_0):
    """
    Returns the *whiteness* index :math:`WI` of given sample *CIE XYZ*
    tristimulus values using *Taube (1960)* method. [2]_

    Parameters
    ----------
    XYZ : array_like
        *CIE XYZ* tristimulus values of sample.
    XYZ_0 : array_like
        *CIE XYZ* tristimulus values of reference white.

    Returns
    -------
    numeric or ndarray
        *Whiteness* :math:`WI`.

    Notes
    -----
    -   Input *CIE XYZ* and *CIE XYZ_0* tristimulus values are in domain
        [0, 100].
    -   *Whiteness* :math:`WI` values larger than 100 indicate a bluish
        white and values smaller than 100 indicate a yellowish white.

    Examples
    --------
    >>> XYZ = np.array([95.00000000, 100.00000000, 105.00000000])
    >>> XYZ_0 = np.array([94.80966767, 100.00000000, 107.30513595])
    >>> whiteness_Taube1960(XYZ, XYZ_0)  # doctest: +ELLIPSIS
    91.4071738...
    """

    _X, Y, Z = tsplit(XYZ)
    _X_0, _Y_0, Z_0 = tsplit(XYZ_0)

    WI = 400 * (Z / Z_0) - 3 * Y

    return WI


def whiteness_Stensby1968(Lab):
    """
    Returns the *whiteness* index :math:`WI` of given sample *CIE Lab*
    colourspace array using *Stensby (1968)* method. [2]_

    Parameters
    ----------
    Lab : array_like
        *CIE Lab* colourspace array of sample.

    Returns
    -------
    numeric or ndarray
        *Whiteness* :math:`WI`.

    Notes
    -----
    -   Input *CIE Lab* colourspace array is in domain [0, 100].
    -   *Whiteness* :math:`WI` values larger than 100 indicate a bluish
        white and values smaller than 100 indicate a yellowish white.

    Examples
    --------
    >>> Lab = np.array([100.00000000, -2.46875131, -16.72486654])
    >>> whiteness_Stensby1968(Lab)  # doctest: +ELLIPSIS
    142.7683456...
    """

    L, a, b = tsplit(Lab)

    WI = L - 3 * b + 3 * a

    return WI


def whiteness_ASTM313(XYZ):
    """
    Returns the *whiteness* index :math:`WI` of given sample *CIE XYZ*
    tristimulus values using *ASTM 313* method. [2]_

    Parameters
    ----------
    XYZ : array_like
        *CIE XYZ* tristimulus values of sample.

    Returns
    -------
    numeric or ndarray
        *Whiteness* :math:`WI`.

    Notes
    -----
    -   Input *CIE XYZ* tristimulus values are in domain [0, 100].

    Warning
    -------
    The input domain of that definition is non standard!

    Examples
    --------
    >>> XYZ = np.array([95.00000000, 100.00000000, 105.00000000])
    >>> whiteness_ASTM313(XYZ)  # doctest: +ELLIPSIS
    55.7400000...
    """

    _X, Y, Z = tsplit(XYZ)

    WI = 3.388 * Z - 3 * Y

    return WI


def whiteness_Ganz1979(xy, Y):
    """
    Returns the *whiteness* index :math:`W` and *tint* :math:`T` of given
    sample *xy* chromaticity coordinates using *Ganz and Griesser (1979)*
    method. [2]_

    Parameters
    ----------
    xy : array_like
        Chromaticity coordinates *xy* of sample.
    Y : numeric or array_like
        Tristimulus :math:`Y` value of sample.

    Returns
    -------
    ndarray
        *Whiteness* :math:`W` and *tint* :math:`T`.

    Notes
    -----
    -   Input tristimulus :math:`Y` value is in domain [0, 100].
    -   The formula coefficients are valid for
        *CIE Standard Illuminant D Series* *D65* and
        *CIE 1964 10 Degree Standard Observer*.
    -   Positive output *tint* :math:`T` values indicate a greener tint while
        negative values indicate a redder tint.
    -   Whiteness differences of less than 5 Ganz units appear to be
        indistinguishable to the human eye.
    -   Tint differences of less than 0.5 Ganz units appear to be
        indistinguishable to the human eye.

    Warning
    -------
    The input domain of that definition is non standard!

    Examples
    --------
    >>> xy = np.array([0.3167, 0.3334])
    >>> whiteness_Ganz1979(xy, 100)  # doctest: +ELLIPSIS
    array([ 85.6003766...,   0.6789003...])
    """

    x, y = tsplit(xy)
    Y = np.asarray(Y)

    W = Y - 1868.322 * x - 3695.690 * y + 1809.441
    T = -1001.223 * x + 748.366 * y + 68.261

    WT = tstack((W, T))

    return WT


def whiteness_CIE2004(xy,
                      Y,
                      xy_n,
                      observer='CIE 1931 2 Degree Standard Observer'):
    """
    Returns the *whiteness* :math:`W` or :math:`W_{10}` and *tint* :math:`T`
    or :math:`T_{10}` of given sample *xy* chromaticity coordinates using
    *CIE 2004* method.

    Parameters
    ----------
    xy : array_like
        Chromaticity coordinates *xy* of sample.
    Y : numeric or array_like
        Tristimulus :math:`Y` value of sample.
    xy_n : array_like
        Chromaticity coordinates *xy_n* of perfect diffuser.
    observer : unicode, optional
        **{'CIE 1931 2 Degree Standard Observer',
        'CIE 1964 10 Degree Standard Observer'}**,
        *CIE Standard Observer* used for computations, *tint* :math:`T` or
        :math:`T_{10}` value is dependent on viewing field angular subtense.

    Returns
    -------
    ndarray
        *Whiteness* :math:`W` or :math:`W_{10}` and *tint* :math:`T` or
        :math:`T_{10}` of given sample.

    Notes
    -----
    -   Input tristimulus :math:`Y` value is in domain [0, 100].
    -   This method may be used only for samples whose values of :math:`W` or
        :math:`W_{10}` lie within the following limits: greater than 40 and
        less than 5Y - 280, or 5Y10 - 280.
    -   This method may be used only for samples whose values of :math:`T` or
        :math:`T_{10}` lie within the following limits: greater than -4 and
        less than +2.
    -   Output *whiteness* :math:`W` or :math:`W_{10}` values larger than 100
        indicate a bluish white while values smaller than 100 indicate a
        yellowish white. [2]_
    -   Positive output *tint* :math:`T` or :math:`T_{10}` values indicate a
        greener tint while negative values indicate a redder tint.

    Warning
    -------
    The input domain of that definition is non standard!

    References
    ----------
    .. [4]  CIE TC 1-48. (2004). The evaluation of whiteness. In CIE 015:2004
            Colorimetry, 3rd Edition (p. 24). ISBN:978-3-901-90633-6

    Examples
    --------
    >>> xy = np.array([0.3167, 0.3334])
    >>> xy_n = np.array([0.3139, 0.3311])
    >>> whiteness_CIE2004(xy, 100, xy_n)  # doctest: +ELLIPSIS
    array([ 93.85...,  -1.305...])
    """

    x, y = tsplit(xy)
    Y = np.asarray(Y)
    x_n, y_n = tsplit(xy_n)

    W = Y + 800 * (x_n - x) + 1700 * (y_n - y)
    T = (1000 if '1931' in observer else 900) * (x_n - x) - 650 * (y_n - y)

    WT = tstack((W, T))

    return WT


WHITENESS_METHODS = CaseInsensitiveMapping(
    {'Berger 1959': whiteness_Berger1959,
     'Taube 1960': whiteness_Taube1960,
     'Stensby 1968': whiteness_Stensby1968,
     'ASTM 313': whiteness_ASTM313,
     'Ganz 1979': whiteness_Ganz1979,
     'CIE 2004': whiteness_CIE2004})
"""
Supported *whiteness* computations methods.

WHITENESS_METHODS : CaseInsensitiveMapping
    **{'CIE 2004', 'Berger 1959', 'Taube 1960', 'Stensby 1968', 'ASTM 313',
    'Ganz 1979', 'CIE 2004'}**

Aliases:

-   'cie2004': 'CIE 2004'
"""
WHITENESS_METHODS['cie2004'] = WHITENESS_METHODS['CIE 2004']


def whiteness(method='CIE 2004', **kwargs):
    """
    Returns the *whiteness* :math:`W` using given method.

    Parameters
    ----------
    method : unicode, optional
        **{'CIE 2004', 'Berger 1959', 'Taube 1960', 'Stensby 1968', 'ASTM 313',
        'Ganz 1979', 'CIE 2004'}**,
        Computation method.

    Other Parameters
    ----------------
    XYZ : array_like
        {:func:`whiteness_Berger1959`, :func:`whiteness_Taube1960`,
        :func:`whiteness_ASTM313`},
        *CIE XYZ* tristimulus values of sample.
    XYZ_0 : array_like
        {:func:`whiteness_Berger1959`, :func:`whiteness_Taube1960`},
        *CIE XYZ* tristimulus values of reference white.
    Lab : array_like
        {:func:`whiteness_Stensby1968`},
        *CIE Lab* colourspace array of sample.
    xy : array_like
        {:func:`whiteness_Ganz1979`, :func:`whiteness_CIE2004`},
        Chromaticity coordinates *xy* of sample.
    Y : numeric or array_like
        {:func:`whiteness_Ganz1979`, :func:`whiteness_CIE2004`},
        Tristimulus :math:`Y` value of sample.
    xy_n : array_like
        {:func:`whiteness_CIE2004`},
        Chromaticity coordinates *xy_n* of perfect diffuser.
    observer : unicode, optional
        {:func:`whiteness_CIE2004`},
        **{'CIE 1931 2 Degree Standard Observer',
        'CIE 1964 10 Degree Standard Observer'}**,
        *CIE Standard Observer* used for computations, *tint* :math:`T` or
        :math:`T_{10}` value is dependent on viewing field angular subtense.

    Returns
    -------
    numeric or ndarray
        *whiteness* :math:`W`.

    Examples
    --------
    >>> xy = np.array([0.3167, 0.3334])
    >>> Y = 100
    >>> xy_n = np.array([0.3139, 0.3311])
    >>> whiteness(xy=xy, Y=Y, xy_n=xy_n)  # doctest: +ELLIPSIS
    array([ 93.85...,  -1.305...])
    >>> XYZ = np.array([95.00000000, 100.00000000, 105.00000000])
    >>> XYZ_0 = np.array([94.80966767, 100.00000000, 107.30513595])
    >>> method = 'Taube 1960'
    >>> whiteness(XYZ=XYZ, XYZ_0=XYZ_0, method=method)  # doctest: +ELLIPSIS
    91.4071738...
    """

    return WHITENESS_METHODS.get(method)(**kwargs)
