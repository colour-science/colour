# -*- coding: utf-8 -*-
"""
Whiteness Index :math:`W`
=========================

Defines *whiteness* index :math:`W` computation objects:

-   :func:`colour.colorimetry.whiteness_Berger1959`: *Whiteness* index
    :math:`WI` computation of given sample *CIE XYZ* tristimulus values using
    *Berger (1959)* method.
-   :func:`colour.colorimetry.whiteness_Taube1960`: *Whiteness* index
    :math:`WI` computation of given sample *CIE XYZ* tristimulus values using
    *Taube (1960)* method.
-   :func:`colour.colorimetry.whiteness_Stensby1968`: *Whiteness* index
    :math:`WI` computation of given sample *CIE L\\*a\\*b\\** colourspace array
    using *Stensby (1968)* method.
-   :func:`colour.colorimetry.whiteness_ASTME313`: *Whiteness* index :math:`WI`
    of given sample *CIE XYZ* tristimulus values using *ASTM E313* method.
-   :func:`colour.colorimetry.whiteness_Ganz1979`: *Whiteness* index :math:`W`
    and *tint* :math:`T` computation of given sample *xy* chromaticity
    coordinates using *Ganz and Griesser (1979)* method.
-   :func:`colour.colorimetry.whiteness_CIE2004`: *Whiteness* :math:`W` or
    :math:`W_{10}` and *tint* :math:`T` or :math:`T_{10}` computation of given
    sample *xy* chromaticity coordinates using *CIE 2004* method.
-   :attr:`colour.WHITENESS_METHODS`: Supported *whiteness* computations
    methods.
-   :func:`colour.whiteness`: *Whiteness* :math:`W` computation using given
    method.

See Also
--------
`Whiteness Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/colorimetry/whiteness.ipynb>`_

References
----------
-   :cite:`CIETC1-482004k` : CIE TC 1-48. (2004). The evaluation of whiteness.
    In CIE 015:2004 Colorimetry, 3rd Edition (p. 24). ISBN:978-3-901-90633-6
-   :cite:`Wyszecki2000ba` : Wyszecki, G., & Stiles, W. S. (2000).
    Table I(6.5.3) Whiteness Formulae (Whiteness Measure Denoted by W). In
    Color Science: Concepts and Methods, Quantitative Data and Formulae
    (pp. 837-839). Wiley. ISBN:978-0471399186
-   :cite:`X-Rite2012a` : X-Rite, & Pantone. (2012). Color iQC and Color
    iMatch Color Calculations Guide. Retrieved from
    https://www.xrite.com/-/media/xrite/files/\
apps_engineering_techdocuments/c/09_color_calculations_en.pdf
-   :cite:`Wikipedia2004b` : Wikipedia. (2004). Whiteness. Retrieved September
    17, 2014, from http://en.wikipedia.org/wiki/Whiteness
"""

from __future__ import division, unicode_literals

from colour.utilities import (CaseInsensitiveMapping, filter_kwargs,
                              from_range_100, to_domain_100, tsplit, tstack)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'whiteness_Berger1959', 'whiteness_Taube1960', 'whiteness_Stensby1968',
    'whiteness_ASTME313', 'whiteness_Ganz1979', 'whiteness_CIE2004',
    'WHITENESS_METHODS', 'whiteness'
]


def whiteness_Berger1959(XYZ, XYZ_0):
    """
    Returns the *whiteness* index :math:`WI` of given sample *CIE XYZ*
    tristimulus values using *Berger (1959)* method.

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

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``XYZ``    | [0, 100]              | [0, 1]        |
    +------------+-----------------------+---------------+
    | ``XYZ_0``  | [0, 100]              | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``WI``     | [0, 100]              | [0, 1]        |
    +------------+-----------------------+---------------+

    -   *Whiteness* :math:`WI` values larger than 33.33 indicate a bluish
        white and values smaller than 33.33 indicate a yellowish white.

    References
    ----------
    :cite:`X-Rite2012a`

    Examples
    --------
    >>> import numpy as np
    >>> XYZ = np.array([95.00000000, 100.00000000, 105.00000000])
    >>> XYZ_0 = np.array([94.80966767, 100.00000000, 107.30513595])
    >>> whiteness_Berger1959(XYZ, XYZ_0)  # doctest: +ELLIPSIS
    30.3638017...
    """

    X, Y, Z = tsplit(to_domain_100(XYZ))
    X_0, _Y_0, Z_0 = tsplit(to_domain_100(XYZ_0))

    WI = 0.333 * Y + 125 * (Z / Z_0) - 125 * (X / X_0)

    return from_range_100(WI)


def whiteness_Taube1960(XYZ, XYZ_0):
    """
    Returns the *whiteness* index :math:`WI` of given sample *CIE XYZ*
    tristimulus values using *Taube (1960)* method.

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

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``XYZ``    | [0, 100]              | [0, 1]        |
    +------------+-----------------------+---------------+
    | ``XYZ_0``  | [0, 100]              | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``WI``     | [0, 100]              | [0, 1]        |
    +------------+-----------------------+---------------+

    -   *Whiteness* :math:`WI` values larger than 100 indicate a bluish
        white and values smaller than 100 indicate a yellowish white.

    References
    ----------
    :cite:`X-Rite2012a`

    Examples
    --------
    >>> import numpy as np
    >>> XYZ = np.array([95.00000000, 100.00000000, 105.00000000])
    >>> XYZ_0 = np.array([94.80966767, 100.00000000, 107.30513595])
    >>> whiteness_Taube1960(XYZ, XYZ_0)  # doctest: +ELLIPSIS
    91.4071738...
    """

    _X, Y, Z = tsplit(to_domain_100(XYZ))
    _X_0, _Y_0, Z_0 = tsplit(to_domain_100(XYZ_0))

    WI = 400 * (Z / Z_0) - 3 * Y

    return from_range_100(WI)


def whiteness_Stensby1968(Lab):
    """
    Returns the *whiteness* index :math:`WI` of given sample *CIE L\\*a\\*b\\**
    colourspace array using *Stensby (1968)* method.

    Parameters
    ----------
    Lab : array_like
        *CIE L\\*a\\*b\\** colourspace array of sample.

    Returns
    -------
    numeric or ndarray
        *Whiteness* :math:`WI`.

    Notes
    -----

    +------------+-----------------------+-----------------+
    | **Domain** | **Scale - Reference** | **Scale - 1**   |
    +============+=======================+=================+
    | ``Lab``    | ``L`` : [0, 100]      | ``L`` : [0, 1]  |
    |            |                       |                 |
    |            | ``a`` : [-100, 100]   | ``a`` : [-1, 1] |
    |            |                       |                 |
    |            | ``b`` : [-100, 100]   | ``b`` : [-1, 1] |
    +------------+-----------------------+-----------------+

    +------------+-----------------------+-----------------+
    | **Range**  | **Scale - Reference** | **Scale - 1**   |
    +============+=======================+=================+
    | ``WI``     | [0, 100]              | [0, 1]          |
    +------------+-----------------------+-----------------+

    -   *Whiteness* :math:`WI` values larger than 100 indicate a bluish
        white and values smaller than 100 indicate a yellowish white.

    References
    ----------
    :cite:`X-Rite2012a`

    Examples
    --------
    >>> import numpy as np
    >>> Lab = np.array([100.00000000, -2.46875131, -16.72486654])
    >>> whiteness_Stensby1968(Lab)  # doctest: +ELLIPSIS
    142.7683456...
    """

    L, a, b = tsplit(to_domain_100(Lab))

    WI = L - 3 * b + 3 * a

    return from_range_100(WI)


def whiteness_ASTME313(XYZ):
    """
    Returns the *whiteness* index :math:`WI` of given sample *CIE XYZ*
    tristimulus values using *ASTM E313* method.

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

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``XYZ``    | [0, 100]              | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``WI``     | [0, 100]              | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`X-Rite2012a`

    Examples
    --------
    >>> import numpy as np
    >>> XYZ = np.array([95.00000000, 100.00000000, 105.00000000])
    >>> whiteness_ASTME313(XYZ)  # doctest: +ELLIPSIS
    55.7400000...
    """

    _X, Y, Z = tsplit(to_domain_100(XYZ))

    WI = 3.388 * Z - 3 * Y

    return from_range_100(WI)


def whiteness_Ganz1979(xy, Y):
    """
    Returns the *whiteness* index :math:`W` and *tint* :math:`T` of given
    sample *xy* chromaticity coordinates using *Ganz and Griesser (1979)*
    method.

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

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``Y``      | [0, 100]              | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``WT``     | [0, 100]              | [0, 1]        |
    +------------+-----------------------+---------------+

    -   The formula coefficients are valid for
        *CIE Standard Illuminant D Series* *D65* and
        *CIE 1964 10 Degree Standard Observer*.
    -   Positive output *tint* :math:`T` values indicate a greener tint while
        negative values indicate a redder tint.
    -   Whiteness differences of less than 5 Ganz units appear to be
        indistinguishable to the human eye.
    -   Tint differences of less than 0.5 Ganz units appear to be
        indistinguishable to the human eye.

    References
    ----------
    :cite:`X-Rite2012a`

    Examples
    --------
    >>> import numpy as np
    >>> xy = np.array([0.3167, 0.3334])
    >>> whiteness_Ganz1979(xy, 100)  # doctest: +ELLIPSIS
    array([ 85.6003766...,   0.6789003...])
    """

    x, y = tsplit(xy)
    Y = to_domain_100(Y)

    W = Y - 1868.322 * x - 3695.690 * y + 1809.441
    T = -1001.223 * x + 748.366 * y + 68.261

    WT = tstack([W, T])

    return from_range_100(WT)


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

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``Y``      | [0, 100]              | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``WT``     | [0, 100]              | [0, 1]        |
    +------------+-----------------------+---------------+

    -   This method may be used only for samples whose values of :math:`W` or
        :math:`W_{10}` lie within the following limits: greater than 40 and
        less than 5Y - 280, or 5Y10 - 280.
    -   This method may be used only for samples whose values of :math:`T` or
        :math:`T_{10}` lie within the following limits: greater than -4 and
        less than +2.
    -   Output *whiteness* :math:`W` or :math:`W_{10}` values larger than 100
        indicate a bluish white while values smaller than 100 indicate a
        yellowish white.
    -   Positive output *tint* :math:`T` or :math:`T_{10}` values indicate a
        greener tint while negative values indicate a redder tint.

    References
    ----------
    :cite:`CIETC1-482004k`

    Examples
    --------
    >>> import numpy as np
    >>> xy = np.array([0.3167, 0.3334])
    >>> xy_n = np.array([0.3139, 0.3311])
    >>> whiteness_CIE2004(xy, 100, xy_n)  # doctest: +ELLIPSIS
    array([ 93.85...,  -1.305...])
    """

    x, y = tsplit(xy)
    Y = to_domain_100(Y)
    x_n, y_n = tsplit(xy_n)

    W = Y + 800 * (x_n - x) + 1700 * (y_n - y)
    T = (1000 if '1931' in observer else 900) * (x_n - x) - 650 * (y_n - y)

    WT = tstack([W, T])

    return from_range_100(WT)


WHITENESS_METHODS = CaseInsensitiveMapping({
    'Berger 1959': whiteness_Berger1959,
    'Taube 1960': whiteness_Taube1960,
    'Stensby 1968': whiteness_Stensby1968,
    'ASTM E313': whiteness_ASTME313,
    'Ganz 1979': whiteness_Ganz1979,
    'CIE 2004': whiteness_CIE2004
})
WHITENESS_METHODS.__doc__ = """
Supported *whiteness* computations methods.

References
----------
:cite:`CIETC1-482004k`, :cite:`X-Rite2012a`

WHITENESS_METHODS : CaseInsensitiveMapping
    **{'CIE 2004', 'Berger 1959', 'Taube 1960', 'Stensby 1968', 'ASTM E313',
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
        **{'CIE 2004', 'Berger 1959', 'Taube 1960', 'Stensby 1968',
        'ASTM E313', 'Ganz 1979'}**,
        Computation method.

    Other Parameters
    ----------------
    XYZ : array_like
        {:func:`colour.colorimetry.whiteness_Berger1959`,
        :func:`colour.colorimetry.whiteness_Taube1960`,
        :func:`colour.colorimetry.whiteness_ASTME313`},
        *CIE XYZ* tristimulus values of sample.
    XYZ_0 : array_like
        {:func:`colour.colorimetry.whiteness_Berger1959`,
        :func:`colour.colorimetry.whiteness_Taube1960`},
        *CIE XYZ* tristimulus values of reference white.
    Lab : array_like
        {:func:`colour.colorimetry.whiteness_Stensby1968`},
        *CIE L\\*a\\*b\\** colourspace array of sample.
    xy : array_like
        {:func:`colour.colorimetry.whiteness_Ganz1979`,
        :func:`colour.colorimetry.whiteness_CIE2004`},
        Chromaticity coordinates *xy* of sample.
    Y : numeric or array_like
        {:func:`colour.colorimetry.whiteness_Ganz1979`,
        :func:`colour.colorimetry.whiteness_CIE2004`},
        Tristimulus :math:`Y` value of sample.
    xy_n : array_like
        {:func:`colour.colorimetry.whiteness_CIE2004`},
        Chromaticity coordinates *xy_n* of perfect diffuser.
    observer : unicode, optional
        {:func:`colour.colorimetry.whiteness_CIE2004`},
        **{'CIE 1931 2 Degree Standard Observer',
        'CIE 1964 10 Degree Standard Observer'}**,
        *CIE Standard Observer* used for computations, *tint* :math:`T` or
        :math:`T_{10}` value is dependent on viewing field angular subtense.

    Returns
    -------
    numeric or ndarray
        *whiteness* :math:`W`.

    Notes
    -----

    +------------+-----------------------+-----------------+
    | **Domain** | **Scale - Reference** |   **Scale - 1** |
    +============+=======================+=================+
    | ``Lab``    | ``L`` : [0, 100]      | ``L`` : [0, 1]  |
    |            |                       |                 |
    |            | ``a`` : [-100, 100]   | ``a`` : [-1, 1] |
    |            |                       |                 |
    |            | ``b`` : [-100, 100]   | ``b`` : [-1, 1] |
    +------------+-----------------------+-----------------+
    | ``XYZ``    | [0, 100]              |   [0, 1]        |
    +------------+-----------------------+-----------------+
    | ``XYZ_0``  | [0, 100]              |   [0, 1]        |
    +------------+-----------------------+-----------------+
    | ``Y``      | [0, 100]              |   [0, 1]        |
    +------------+-----------------------+-----------------+

    +------------+-----------------------+-----------------+
    | **Range**  | **Scale - Reference** |   **Scale - 1** |
    +============+=======================+=================+
    | ``W``      | [0, 100]              |   [0, 1]        |
    +------------+-----------------------+-----------------+

    References
    ----------
    :cite:`CIETC1-482004k`, :cite:`Wyszecki2000ba`, :cite:`X-Rite2012a`,
    :cite:`Wikipedia2004b`

    Examples
    --------
    >>> import numpy as np
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

    function = WHITENESS_METHODS.get(method)

    return function(**filter_kwargs(function, **kwargs))
