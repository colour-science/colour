# -*- coding: utf-8 -*-
"""
Whiteness Index :math:`W`
=========================

Defines the *whiteness* index :math:`W` computation objects:

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
    and *tint* :math:`T` computation of given sample *CIE xy* chromaticity
    coordinates using *Ganz and Griesser (1979)* method.
-   :func:`colour.colorimetry.whiteness_CIE2004`: *Whiteness* :math:`W` or
    :math:`W_{10}` and *tint* :math:`T` or :math:`T_{10}` computation of given
    sample *CIE xy* chromaticity coordinates using *CIE 2004* method.
-   :attr:`colour.WHITENESS_METHODS`: Supported *whiteness* computations
    methods.
-   :func:`colour.whiteness`: *Whiteness* :math:`W` computation using given
    method.

References
----------
-   :cite:`CIETC1-482004k` : CIE TC 1-48. (2004). The evaluation of whiteness.
    In CIE 015:2004 Colorimetry, 3rd Edition (p. 24). ISBN:978-3-901906-33-6
-   :cite:`Wikipedia2004b` : Wikipedia. (2004). Whiteness. Retrieved September
    17, 2014, from http://en.wikipedia.org/wiki/Whiteness
-   :cite:`Wyszecki2000ba` : Wyszecki, Günther, & Stiles, W. S. (2000). Table
    I(6.5.3) Whiteness Formulae (Whiteness Measure Denoted by W). In Color
    Science: Concepts and Methods, Quantitative Data and Formulae (pp.
    837-839). Wiley. ISBN:978-0-471-39918-6
-   :cite:`X-Rite2012a` : X-Rite, & Pantone. (2012). Color iQC and Color iMatch
    Color Calculations Guide.
    https://www.xrite.com/-/media/xrite/files/apps_engineering_techdocuments/\
c/09_color_calculations_en.pdf
"""

from __future__ import annotations

from colour.hints import (
    Any,
    ArrayLike,
    FloatingOrNDArray,
    Literal,
    NDArray,
    Union,
)
from colour.utilities import (
    CaseInsensitiveMapping,
    as_float,
    as_float_array,
    get_domain_range_scale,
    filter_kwargs,
    from_range_100,
    to_domain_100,
    tsplit,
    tstack,
    validate_method,
)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'whiteness_Berger1959',
    'whiteness_Taube1960',
    'whiteness_Stensby1968',
    'whiteness_ASTME313',
    'whiteness_Ganz1979',
    'whiteness_CIE2004',
    'WHITENESS_METHODS',
    'whiteness',
]


def whiteness_Berger1959(XYZ: ArrayLike,
                         XYZ_0: ArrayLike) -> FloatingOrNDArray:
    """
    Returns the *whiteness* index :math:`WI` of given sample *CIE XYZ*
    tristimulus values using *Berger (1959)* method.

    Parameters
    ----------
    XYZ
        *CIE XYZ* tristimulus values of the sample.
    XYZ_0
        *CIE XYZ* tristimulus values of the reference white.

    Returns
    -------
    :class:`np.floating` or :class:`numpy.ndarray`
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

    return as_float(from_range_100(WI))


def whiteness_Taube1960(XYZ: ArrayLike, XYZ_0: ArrayLike) -> FloatingOrNDArray:
    """
    Returns the *whiteness* index :math:`WI` of given sample *CIE XYZ*
    tristimulus values using *Taube (1960)* method.

    Parameters
    ----------
    XYZ
        *CIE XYZ* tristimulus values of the sample.
    XYZ_0
        *CIE XYZ* tristimulus values of the reference white.

    Returns
    -------
    :class:`np.floating` or :class:`numpy.ndarray`
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

    return as_float(from_range_100(WI))


def whiteness_Stensby1968(Lab: ArrayLike) -> FloatingOrNDArray:
    """
    Returns the *whiteness* index :math:`WI` of given sample *CIE L\\*a\\*b\\**
    colourspace array using *Stensby (1968)* method.

    Parameters
    ----------
    Lab
        *CIE L\\*a\\*b\\** colourspace array of the sample.

    Returns
    -------
    :class:`np.floating` or :class:`numpy.ndarray`
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

    return as_float(from_range_100(WI))


def whiteness_ASTME313(XYZ: ArrayLike) -> FloatingOrNDArray:
    """
    Returns the *whiteness* index :math:`WI` of given sample *CIE XYZ*
    tristimulus values using *ASTM E313* method.

    Parameters
    ----------
    XYZ
        *CIE XYZ* tristimulus values of the sample.

    Returns
    -------
    :class:`np.floating` or :class:`numpy.ndarray`
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

    return as_float(from_range_100(WI))


def whiteness_Ganz1979(xy: ArrayLike, Y: FloatingOrNDArray) -> NDArray:
    """
    Returns the *whiteness* index :math:`W` and *tint* :math:`T` of given
    sample *CIE xy* chromaticity coordinates using *Ganz and Griesser (1979)*
    method.

    Parameters
    ----------
    xy
        Chromaticity coordinates *CIE xy* of the sample.
    Y
        Tristimulus :math:`Y` value of the sample.

    Returns
    -------
    :class:`numpy.ndarray`
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


def whiteness_CIE2004(
        xy: ArrayLike,
        Y: FloatingOrNDArray,
        xy_n: ArrayLike,
        observer: Literal['CIE 1931 2 Degree Standard Observer',
                          'CIE 1964 10 Degree Standard Observer'] = (
                              'CIE 1931 2 Degree Standard Observer')
) -> NDArray:
    """
    Returns the *whiteness* :math:`W` or :math:`W_{10}` and *tint* :math:`T`
    or :math:`T_{10}` of given sample *CIE xy* chromaticity coordinates using
    *CIE 2004* method.

    Parameters
    ----------
    xy
        Chromaticity coordinates *CIE xy* of the sample.
    Y
        Tristimulus :math:`Y` value of the sample.
    xy_n
        Chromaticity coordinates *xy_n* of a perfect diffuser.
    observer
        *CIE Standard Observer* used for computations, *tint* :math:`T` or
        :math:`T_{10}` value is dependent on viewing field angular subtense.

    Returns
    -------
    :class:`numpy.ndarray`
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


WHITENESS_METHODS: CaseInsensitiveMapping = CaseInsensitiveMapping({
    'Berger 1959': whiteness_Berger1959,
    'Taube 1960': whiteness_Taube1960,
    'Stensby 1968': whiteness_Stensby1968,
    'ASTM E313': whiteness_ASTME313,
    'Ganz 1979': whiteness_Ganz1979,
    'CIE 2004': whiteness_CIE2004
})
WHITENESS_METHODS.__doc__ = """
Supported *whiteness* computation methods.

References
----------
:cite:`CIETC1-482004k`, :cite:`X-Rite2012a`

Aliases:

-   'cie2004': 'CIE 2004'
"""
WHITENESS_METHODS['cie2004'] = WHITENESS_METHODS['CIE 2004']


def whiteness(XYZ: ArrayLike,
              XYZ_0: ArrayLike,
              method: Union[Literal['ASTM E313', 'CIE 2004', 'Berger 1959',
                                    'Ganz 1979', 'Stensby 1968', 'Taube 1960'],
                            str] = 'CIE 2004',
              **kwargs: Any) -> FloatingOrNDArray:
    """
    Returns the *whiteness* :math:`W` using given method.

    Parameters
    ----------
    XYZ
        *CIE XYZ* tristimulus values of the sample.
    XYZ_0
        *CIE XYZ* tristimulus values of the reference white.
    method
        Computation method.

    Other Parameters
    ----------------
    observer
        {:func:`colour.colorimetry.whiteness_CIE2004`},
        *CIE Standard Observer* used for computations, *tint* :math:`T` or
        :math:`T_{10}` value is dependent on viewing field angular subtense.

    Returns
    -------
    :class:`np.floating` or :class:`numpy.ndarray`
        *Whiteness* :math:`W`.

    Notes
    -----

    +------------+-----------------------+-----------------+
    | **Domain** | **Scale - Reference** |   **Scale - 1** |
    +============+=======================+=================+
    +------------+-----------------------+-----------------+
    | ``XYZ``    | [0, 100]              |   [0, 1]        |
    +------------+-----------------------+-----------------+
    | ``XYZ_0``  | [0, 100]              |   [0, 1]        |
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
    >>> from colour.models import xyY_to_XYZ
    >>> XYZ = xyY_to_XYZ(np.array([0.3167, 0.3334, 100]))
    >>> XYZ_0 = xyY_to_XYZ(np.array([0.3139, 0.3311, 100]))
    >>> whiteness(XYZ, XYZ_0)  # doctest: +ELLIPSIS
    array([ 93.85...,  -1.305...])
    >>> XYZ = np.array([95.00000000, 100.00000000, 105.00000000])
    >>> XYZ_0 = np.array([94.80966767, 100.00000000, 107.30513595])
    >>> whiteness(XYZ, XYZ_0, method='Taube 1960')  # doctest: +ELLIPSIS
    91.4071738...
    """

    XYZ = as_float_array(XYZ)
    XYZ_0 = as_float_array(XYZ_0)

    method = validate_method(method, WHITENESS_METHODS)

    kwargs.update({'XYZ': XYZ, 'XYZ_0': XYZ_0})

    function = WHITENESS_METHODS[method]

    if function is whiteness_Stensby1968:
        from colour.models import XYZ_to_Lab, XYZ_to_xy

        if get_domain_range_scale() == 'reference':
            XYZ = XYZ / 100
            XYZ_0 = XYZ_0 / 100

        kwargs.update({'Lab': XYZ_to_Lab(XYZ, XYZ_to_xy(XYZ_0))})
    elif function in (whiteness_Ganz1979, whiteness_CIE2004):
        from colour.models import XYZ_to_xy

        _X_0, Y_0, _Z_0 = tsplit(XYZ_0)
        kwargs.update({
            'xy': XYZ_to_xy(XYZ),
            'Y': Y_0,
            'xy_n': XYZ_to_xy(XYZ_0)
        })

    return function(**filter_kwargs(function, **kwargs))
