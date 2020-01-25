# -*- coding: utf-8 -*-
"""
Munsell Renotation System
=========================

Defines various objects for *Munsell Renotation System* computations:

-   :func:`colour.notation.munsell_value_Priest1920`: *Munsell* value :math:`V`
    computation of given *luminance* :math:`Y` using
    *Priest, Gibson and MacNicholas (1920)* method.
-   :func:`colour.notation.munsell_value_Munsell1933`: *Munsell* value
    :math:`V` computation of given *luminance* :math:`Y` using
    *Munsell, Sloan and Godlove (1933)* method.
-   :func:`colour.notation.munsell_value_Moon1943`: *Munsell* value :math:`V`
    computation of given *luminance* :math:`Y` using
    *Moon and Spencer (1943)* method.
-   :func:`colour.notation.munsell_value_Saunderson1944`: *Munsell* value
    :math:`V` computation of given *luminance* :math:`Y` using
    *Saunderson and Milner (1944)* method.
-   :func:`colour.notation.munsell_value_Ladd1955`: *Munsell* value :math:`V`
    computation of given *luminance* :math:`Y` using *Ladd and Pinney (1955)*
    method.
-   :func:`colour.notation.munsell_value_McCamy1987`: *Munsell* value :math:`V`
    computation of given *luminance* :math:`Y` using *McCamy (1987)* method.
-   :func:`colour.notation.munsell_value_ASTMD1535`: *Munsell* value
    :math:`V` computation of given *luminance* :math:`Y` using
    *ASTM D1535-08e1* method.
-   :attr:`colour.MUNSELL_VALUE_METHODS`: Supported *Munsell* value
    computation methods.
-   :func:`colour.munsell_value`: *Munsell* value :math:`V` computation of
    given *luminance* :math:`Y` using given method.
-   :func:`colour.munsell_colour_to_xyY`
-   :func:`colour.xyY_to_munsell_colour`

See Also
--------
`Munsell Renotation System Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/notation/munsell.ipynb>`_

Notes
-----
-   The Munsell Renotation data commonly available within the all.dat,
    experimental.dat and real.dat files features *CIE xyY* colourspace values
    that are scaled by a :math:`1 / 0.975 \\simeq 1.02568` factor. If you are
    performing conversions using *Munsell* *Colorlab* specification,
    e.g. *2.5R 9/2*, according to *ASTM D1535-08e1* method, you should not
    scale the output :math:`Y` Luminance. However, if you use directly the
    *CIE xyY* colourspace values from the Munsell Renotation data data, you
    should scale the :math:`Y` Luminance before conversions by a :math:`0.975`
    factor.

    *ASTM D1535-08e1* states that::

        The coefficients of this equation are obtained from the 1943 equation
        by multiplying each coefficient by 0.975, the reflectance factor of
        magnesium oxide with respect to the perfect reflecting diffuser, and
        rounding to ve digits of precision.

References
----------
-   :cite:`ASTMInternational1989a` : ASTM International. (1989). ASTM D1535-89
    - Standard Practice for Specifying Color by the Munsell System. Retrieved
    from http://www.astm.org/DATABASE.CART/HISTORICAL/D1535-89.htm
-   :cite:`Centore2012a` : Centore, P. (2012). An open-source inversion
    algorithm for the Munsell renotation. Color Research & Application, 37(6),
    455-464. doi:10.1002/col.20715
-   :cite:`Centore2014k` : Centore, P. (2014).
    MunsellAndKubelkaMunkToolboxApr2014 -
    MunsellRenotationRoutines/MunsellHueToASTMHue.m. Retrieved from
    https://github.com/colour-science/MunsellAndKubelkaMunkToolbox
-   :cite:`Centore2014l` : Centore, P. (2014).
    MunsellAndKubelkaMunkToolboxApr2014 -
    MunsellSystemRoutines/LinearVsRadialInterpOnRenotationOvoid.m.
    Retrieved from
    https://github.com/colour-science/MunsellAndKubelkaMunkToolbox
-   :cite:`Centore2014m` : Centore, P. (2014).
    MunsellAndKubelkaMunkToolboxApr2014 -
    MunsellRenotationRoutines/MunsellToxyY.m. Retrieved from
    https://github.com/colour-science/MunsellAndKubelkaMunkToolbox
-   :cite:`Centore2014n` : Centore, P. (2014).
    MunsellAndKubelkaMunkToolboxApr2014 -
    MunsellRenotationRoutines/FindHueOnRenotationOvoid.m. Retrieved from
    https://github.com/colour-science/MunsellAndKubelkaMunkToolbox
-   :cite:`Centore2014o` : Centore, P. (2014).
    MunsellAndKubelkaMunkToolboxApr2014 -
    MunsellSystemRoutines/BoundingRenotationHues.m. Retrieved from
    https://github.com/colour-science/MunsellAndKubelkaMunkToolbox
-   :cite:`Centore2014p` : Centore, P. (2014).
    MunsellAndKubelkaMunkToolboxApr2014 -
    MunsellRenotationRoutines/xyYtoMunsell.m. Retrieved from
    https://github.com/colour-science/MunsellAndKubelkaMunkToolbox
-   :cite:`Centore2014q` : Centore, P. (2014).
    MunsellAndKubelkaMunkToolboxApr2014 -
    MunsellRenotationRoutines/MunsellToxyForIntegerMunsellValue.m. Retrieved
    from https://github.com/colour-science/MunsellAndKubelkaMunkToolbox
-   :cite:`Centore2014r` : Centore, P. (2014).
    MunsellAndKubelkaMunkToolboxApr2014 -
    MunsellRenotationRoutines/MaxChromaForExtrapolatedRenotation.m. Retrieved
    from https://github.com/colour-science/MunsellAndKubelkaMunkToolbox
-   :cite:`Centore2014s` : Centore, P. (2014).
    MunsellAndKubelkaMunkToolboxApr2014 -
    MunsellRenotationRoutines/MunsellHueToChromDiagHueAngle.m. Retrieved from
    https://github.com/colour-science/MunsellAndKubelkaMunkToolbox
-   :cite:`Centore2014t` : Centore, P. (2014).
    MunsellAndKubelkaMunkToolboxApr2014 -
    MunsellRenotationRoutines/ChromDiagHueAngleToMunsellHue.m. Retrieved from
    https://github.com/colour-science/MunsellAndKubelkaMunkToolbox
-   :cite:`Centore2014u` : Centore, P. (2014).
    MunsellAndKubelkaMunkToolboxApr2014 -
    GeneralRoutines/CIELABtoApproxMunsellSpec.m. Retrieved from
    https://github.com/colour-science/MunsellAndKubelkaMunkToolbox
-   :cite:`Centorea` : Centore, P. (n.d.). The Munsell and Kubelka-Munk
    Toolbox. Retrieved January 23, 2018, from
    http://www.munsellcolourscienceforpainters.com/\
MunsellAndKubelkaMunkToolbox/MunsellAndKubelkaMunkToolbox.html
-   :cite:`Wikipedia2007c` : Wikipedia. (2007). Lightness. Retrieved April
    13, 2014, from http://en.wikipedia.org/wiki/Lightness
"""

from __future__ import division, unicode_literals

import numpy as np
import re
from collections import OrderedDict

from colour.algebra import (Extrapolator, LinearInterpolator,
                            cartesian_to_cylindrical, euclidean_distance,
                            polar_to_cartesian, spow)
from colour.colorimetry import ILLUMINANTS, luminance_ASTMD1535
from colour.constants import (DEFAULT_FLOAT_DTYPE, DEFAULT_INT_DTYPE,
                              INTEGER_THRESHOLD, FLOATING_POINT_NUMBER_PATTERN)
from colour.models import Lab_to_LCHab, XYZ_to_Lab, XYZ_to_xy, xyY_to_XYZ
from colour.volume import is_within_macadam_limits
from colour.notation import MUNSELL_COLOURS_ALL
from colour.utilities import (
    CaseInsensitiveMapping, Lookup, as_float_array, as_float, as_int,
    as_numeric, domain_range_scale, from_range_1, from_range_10,
    get_domain_range_scale, to_domain_1, to_domain_10, to_domain_100,
    is_integer, is_numeric, tsplit, usage_warning)

__author__ = 'Colour Developers, Paul Centore'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__copyright__ += ', '
__copyright__ += (
    'The Munsell and Kubelka-Munk Toolbox: Copyright  2010-2018 Paul Centore '
    '(Gales Ferry, CT 06335, USA); used by permission.')
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'MUNSELL_GRAY_PATTERN', 'MUNSELL_COLOUR_PATTERN', 'MUNSELL_GRAY_FORMAT',
    'MUNSELL_COLOUR_FORMAT', 'MUNSELL_GRAY_EXTENDED_FORMAT',
    'MUNSELL_COLOUR_EXTENDED_FORMAT', 'MUNSELL_HUE_LETTER_CODES',
    'MUNSELL_DEFAULT_ILLUMINANT',
    'MUNSELL_DEFAULT_ILLUMINANT_CHROMATICITY_COORDINATES',
    'munsell_value_Priest1920', 'munsell_value_Munsell1933',
    'munsell_value_Moon1943', 'munsell_value_Saunderson1944',
    'munsell_value_Ladd1955', 'munsell_value_McCamy1987',
    'munsell_value_ASTMD1535', 'MUNSELL_VALUE_METHODS', 'munsell_value',
    'munsell_specification_to_xyY', 'munsell_colour_to_xyY',
    'xyY_to_munsell_specification', 'xyY_to_munsell_colour',
    'parse_munsell_colour', 'is_grey_munsell_colour',
    'normalize_munsell_specification',
    'munsell_colour_to_munsell_specification',
    'munsell_specification_to_munsell_colour', 'xyY_from_renotation',
    'is_specification_in_renotation', 'bounding_hues_from_renotation',
    'hue_to_hue_angle', 'hue_angle_to_hue', 'hue_to_ASTM_hue',
    'interpolation_method_from_renotation_ovoid', 'xy_from_renotation_ovoid',
    'LCHab_to_munsell_specification', 'maximum_chroma_from_renotation',
    'munsell_specification_to_xy'
]

MUNSELL_GRAY_PATTERN = 'N(?P<value>{0})'.format(FLOATING_POINT_NUMBER_PATTERN)
MUNSELL_COLOUR_PATTERN = ('(?P<hue>{0})\\s*'
                          '(?P<letter>BG|GY|YR|RP|PB|B|G|Y|R|P)\\s*'
                          '(?P<value>{0})\\s*\\/\\s*(?P<chroma>[-+]?{0})'.
                          format(FLOATING_POINT_NUMBER_PATTERN))

MUNSELL_GRAY_FORMAT = 'N{0}'
MUNSELL_COLOUR_FORMAT = '{0} {1}/{2}'
MUNSELL_GRAY_EXTENDED_FORMAT = 'N{0:.{1}f}'
MUNSELL_COLOUR_EXTENDED_FORMAT = '{0:.{1}f}{2} {3:.{4}f}/{5:.{6}f}'

MUNSELL_HUE_LETTER_CODES = Lookup({
    'BG': 2,
    'GY': 4,
    'YR': 6,
    'RP': 8,
    'PB': 10,
    'B': 1,
    'G': 3,
    'Y': 5,
    'R': 7,
    'P': 9
})

MUNSELL_DEFAULT_ILLUMINANT = 'C'
MUNSELL_DEFAULT_ILLUMINANT_CHROMATICITY_COORDINATES = (ILLUMINANTS[
    'CIE 1931 2 Degree Standard Observer'][MUNSELL_DEFAULT_ILLUMINANT])

_MUNSELL_SPECIFICATIONS_CACHE = None
_MUNSELL_VALUE_ASTM_D1535_08_INTERPOLATOR_CACHE = None
_MUNSELL_MAXIMUM_CHROMAS_FROM_RENOTATION_CACHE = None


def _munsell_specifications():
    """
    Returns the *Munsell Renotation System* specifications and caches them if
    not existing.

    The *Munsell Renotation System* data is stored in
    :attr:`colour.notation.MUNSELL_COLOURS` attribute in a 2 columns form:

    (('2.5GY', 0.2, 2.0), (0.713, 1.414, 0.237)),
    (('5GY', 0.2, 2.0), (0.449, 1.145, 0.237)),
    (('7.5GY', 0.2, 2.0), (0.262, 0.837, 0.237)),
    ...,)

    The first column is converted from *Munsell* colour to specification using
    :func:`colour.notation.munsell.munsell_colour_to_munsell_specification`
    definition:

    ('2.5GY', 0.2, 2.0) ---> (2.5, 0.2, 2.0, 4)

    Returns
    -------
    ndarray
        *Munsell Renotation System* specifications.
    """

    global _MUNSELL_SPECIFICATIONS_CACHE

    if _MUNSELL_SPECIFICATIONS_CACHE is None:
        _MUNSELL_SPECIFICATIONS_CACHE = np.array([
            munsell_colour_to_munsell_specification(
                MUNSELL_COLOUR_FORMAT.format(*colour[0]))
            for colour in MUNSELL_COLOURS_ALL
        ])
    return _MUNSELL_SPECIFICATIONS_CACHE


def _munsell_value_ASTMD1535_interpolator():
    """
    Returns the *Munsell* value interpolator for *ASTM D1535-08e1* method and
    caches it if not existing.

    Returns
    -------
    Extrapolator
        *Munsell* value interpolator for *ASTM D1535-08e1* method.
    """

    global _MUNSELL_VALUE_ASTM_D1535_08_INTERPOLATOR_CACHE

    munsell_values = np.arange(0, 10, 0.001)
    if _MUNSELL_VALUE_ASTM_D1535_08_INTERPOLATOR_CACHE is None:
        _MUNSELL_VALUE_ASTM_D1535_08_INTERPOLATOR_CACHE = Extrapolator(
            LinearInterpolator(
                luminance_ASTMD1535(munsell_values), munsell_values))

    return _MUNSELL_VALUE_ASTM_D1535_08_INTERPOLATOR_CACHE


def _munsell_maximum_chromas_from_renotation():
    """
    Returns the maximum *Munsell* chromas from *Munsell Renotation System* data
    and caches them if not existing.

    Returns
    -------
    tuple
        Maximum *Munsell* chromas.
    """

    global _MUNSELL_MAXIMUM_CHROMAS_FROM_RENOTATION_CACHE

    if _MUNSELL_MAXIMUM_CHROMAS_FROM_RENOTATION_CACHE is None:
        chromas = OrderedDict()
        for munsell_colour in MUNSELL_COLOURS_ALL:
            hue, value, chroma, code = munsell_colour_to_munsell_specification(
                MUNSELL_COLOUR_FORMAT.format(*munsell_colour[0]))
            index = (hue, value, code)
            if index in chromas:
                chroma = max(chromas[index], chroma)

            chromas[index] = chroma

        _MUNSELL_MAXIMUM_CHROMAS_FROM_RENOTATION_CACHE = tuple(
            zip(chromas.keys(), chromas.values()))
    return _MUNSELL_MAXIMUM_CHROMAS_FROM_RENOTATION_CACHE


def munsell_value_Priest1920(Y):
    """
    Returns the *Munsell* value :math:`V` of given *luminance* :math:`Y` using
    *Priest et al. (1920)* method.

    Parameters
    ----------
    Y : numeric or array_like
        *luminance* :math:`Y`.

    Returns
    -------
    numeric or ndarray
        *Munsell* value :math:`V`.

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
    | ``V``      | [0, 10]               | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Wikipedia2007c`

    Examples
    --------
    >>> munsell_value_Priest1920(12.23634268)  # doctest: +ELLIPSIS
    3.4980484...
    """

    Y = to_domain_100(Y)

    V = 10 * np.sqrt(Y / 100)

    return from_range_10(V)


def munsell_value_Munsell1933(Y):
    """
    Returns the *Munsell* value :math:`V` of given *luminance* :math:`Y` using
    *Munsell et al. (1933)* method.

    Parameters
    ----------
    Y : numeric or array_like
        *luminance* :math:`Y`.

    Returns
    -------
    numeric or ndarray
        *Munsell* value :math:`V`.

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
    | ``V``      | [0, 10]               | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Wikipedia2007c`

    Examples
    --------
    >>> munsell_value_Munsell1933(12.23634268)  # doctest: +ELLIPSIS
    4.1627702...
    """

    Y = to_domain_100(Y)

    V = np.sqrt(1.4742 * Y - 0.004743 * (Y * Y))

    return from_range_10(V)


def munsell_value_Moon1943(Y):
    """
    Returns the *Munsell* value :math:`V` of given *luminance* :math:`Y` using
    *Moon and Spencer (1943)* method.


    Parameters
    ----------
    Y : numeric or array_like
        *luminance* :math:`Y`.

    Returns
    -------
    numeric or ndarray
        *Munsell* value :math:`V`.

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
    | ``V``      | [0, 10]               | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Wikipedia2007c`

    Examples
    --------
    >>> munsell_value_Moon1943(12.23634268)  # doctest: +ELLIPSIS
    4.0688120...
    """

    Y = to_domain_100(Y)

    V = 1.4 * spow(Y, 0.426)

    return from_range_10(V)


def munsell_value_Saunderson1944(Y):
    """
    Returns the *Munsell* value :math:`V` of given *luminance* :math:`Y` using
    *Saunderson and Milner (1944)* method.

    Parameters
    ----------
    Y : numeric
        *luminance* :math:`Y`.

    Returns
    -------
    numeric
        *Munsell* value :math:`V`.

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
    | ``V``      | [0, 10]               | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Wikipedia2007c`

    Examples
    --------
    >>> munsell_value_Saunderson1944(12.23634268)  # doctest: +ELLIPSIS
    4.0444736...
    """

    Y = to_domain_100(Y)

    V = 2.357 * spow(Y, 0.343) - 1.52

    return from_range_10(V)


def munsell_value_Ladd1955(Y):
    """
    Returns the *Munsell* value :math:`V` of given *luminance* :math:`Y` using
    *Ladd and Pinney (1955)* method.

    Parameters
    ----------
    Y : numeric or array_like
        *luminance* :math:`Y`.

    Returns
    -------
    numeric or ndarray
        *Munsell* value :math:`V`.

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
    | ``V``      | [0, 10]               | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Wikipedia2007c`

    Examples
    --------
    >>> munsell_value_Ladd1955(12.23634268)  # doctest: +ELLIPSIS
    4.0511633...
    """

    Y = to_domain_100(Y)

    V = 2.468 * spow(Y, 1 / 3) - 1.636

    return from_range_10(V)


def munsell_value_McCamy1987(Y):
    """
    Returns the *Munsell* value :math:`V` of given *luminance* :math:`Y` using
    *McCamy (1987)* method.

    Parameters
    ----------
    Y : numeric or array_like
        *luminance* :math:`Y`.

    Returns
    -------
    numeric or ndarray
        *Munsell* value :math:`V`.

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
    | ``V``      | [0, 10]               | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`ASTMInternational1989a`

    Examples
    --------
    >>> munsell_value_McCamy1987(12.23634268)  # doctest: +ELLIPSIS
    array(4.0814348...)
    """

    Y = to_domain_100(Y)

    V = np.where(
        Y <= 0.9,
        0.87445 * spow(Y, 0.9967),
        2.49268 * spow(Y, 1 / 3) - 1.5614 - (0.985 / ((
            (0.1073 * Y - 3.084) ** 2) + 7.54)) + (0.0133 / spow(Y, 2.3)) +
        0.0084 * np.sin(4.1 * spow(Y, 1 / 3) + 1) +
        (0.0221 / Y) * np.sin(0.39 * (Y - 2)) -
        (0.0037 / (0.44 * Y)) * np.sin(1.28 * (Y - 0.53)),
    )

    return from_range_10(V)


def munsell_value_ASTMD1535(Y):
    """
    Returns the *Munsell* value :math:`V` of given *luminance* :math:`Y` using
    an inverse lookup table from *ASTM D1535-08e1* method.

    Parameters
    ----------
    Y : numeric or array_like
        *luminance* :math:`Y`

    Returns
    -------
    numeric or ndarray
        *Munsell* value :math:`V`.

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
    | ``V``      | [0, 10]               | [0, 1]        |
    +------------+-----------------------+---------------+

    -   The *Munsell* value* computation with *ASTM D1535-08e1* method is only
        defined for domain [0, 100].

    References
    ----------
    :cite:`ASTMInternational1989a`

    Examples
    --------
    >>> munsell_value_ASTMD1535(12.23634268)  # doctest: +ELLIPSIS
    4.0824437...
    """

    Y = to_domain_100(Y)

    V = _munsell_value_ASTMD1535_interpolator()(Y)

    return from_range_10(V)


MUNSELL_VALUE_METHODS = CaseInsensitiveMapping({
    'Priest 1920': munsell_value_Priest1920,
    'Munsell 1933': munsell_value_Munsell1933,
    'Moon 1943': munsell_value_Moon1943,
    'Saunderson 1944': munsell_value_Saunderson1944,
    'Ladd 1955': munsell_value_Ladd1955,
    'McCamy 1987': munsell_value_McCamy1987,
    'ASTM D1535': munsell_value_ASTMD1535
})
MUNSELL_VALUE_METHODS.__doc__ = """
Supported *Munsell* value computation methods.

References
----------
:cite:`ASTMInternational1989a`, :cite:`Wikipedia2007c`

MUNSELL_VALUE_METHODS : CaseInsensitiveMapping
    **{'Priest 1920', 'Munsell 1933', 'Moon 1943', 'Saunderson 1944',
    'Ladd 1955', 'McCamy 1987', 'ASTM D1535'}**

Aliases:

-   'astm2008': 'ASTM D1535'
"""
MUNSELL_VALUE_METHODS['astm2008'] = (MUNSELL_VALUE_METHODS['ASTM D1535'])


def munsell_value(Y, method='ASTM D1535'):
    """
    Returns the *Munsell* value :math:`V` of given *luminance* :math:`Y` using
    given method.

    Parameters
    ----------
    Y : numeric or array_like
        *luminance* :math:`Y`.
    method : unicode, optional
        **{'ASTM D1535', 'Priest 1920', 'Munsell 1933', 'Moon 1943',
        'Saunderson 1944', 'Ladd 1955', 'McCamy 1987'}**,
        Computation method.

    Returns
    -------
    numeric or ndarray
        *Munsell* value :math:`V`.

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
    | ``V``      | [0, 10]               | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`ASTMInternational1989a`, :cite:`Wikipedia2007c`

    Examples
    --------
    >>> munsell_value(12.23634268)  # doctest: +ELLIPSIS
    4.0824437...
    >>> munsell_value(12.23634268, method='Priest 1920') # doctest: +ELLIPSIS
    3.4980484...
    >>> munsell_value(12.23634268, method='Munsell 1933') # doctest: +ELLIPSIS
    4.1627702...
    >>> munsell_value(12.23634268, method='Moon 1943') # doctest: +ELLIPSIS
    4.0688120...
    >>> munsell_value(12.23634268, method='Saunderson 1944')
    ... # doctest: +ELLIPSIS
    4.0444736...
    >>> munsell_value(12.23634268, method='Ladd 1955') # doctest: +ELLIPSIS
    4.0511633...
    >>> munsell_value(12.23634268, method='McCamy 1987') # doctest: +ELLIPSIS
    array(4.0814348...)
    """

    return MUNSELL_VALUE_METHODS.get(method)(Y)


def _domain_range_scale_factor():
    """
    Returns the domain-range scale factor for *Munsell Renotation System*.

    Returns
    -------
    ndarray
        Domain-range scale factor for *Munsell Renotation System*.
    """

    return np.array([10, 10, 50 if get_domain_range_scale() == '1' else 2, 10])


def _munsell_specification_to_xyY(specification):
    """
    Converts given *Munsell* *Colorlab* specification to *CIE xyY* colourspace.


    Parameters
    ----------
    specification : array_like
        *Munsell* *Colorlab* specification.

    Returns
    -------
    ndarray
        *CIE xyY* colourspace array.
    """

    specification = normalize_munsell_specification(specification)

    if is_grey_munsell_colour(specification):
        specification = as_float_array(to_domain_10(specification))
        hue, value, chroma, code = specification
    else:
        specification = to_domain_10(specification,
                                     _domain_range_scale_factor())
        hue, value, chroma, code = ([as_float(i) for i in specification[0:3]] +
                                    [DEFAULT_INT_DTYPE(specification[-1])])

        assert 0 <= hue <= 10, (
            '"{0}" specification hue must be normalised to domain '
            '[0, 10]!'.format(specification))
        assert 0 <= value <= 10, (
            '"{0}" specification value must be normalised to domain '
            '[0, 10]!'.format(specification))

    with domain_range_scale('ignore'):
        Y = luminance_ASTMD1535(value)

    if is_integer(value):
        value_minus = value_plus = round(value)
    else:
        value_minus = np.floor(value)
        value_plus = value_minus + 1

    specification_minus = (value_minus if is_grey_munsell_colour(specification)
                           else (hue, value_minus, chroma, code))
    x_minus, y_minus = munsell_specification_to_xy(specification_minus)

    plus_specification = (value_plus
                          if (is_grey_munsell_colour(specification) or
                              value_plus == 10) else (hue, value_plus, chroma,
                                                      code))
    x_plus, y_plus = munsell_specification_to_xy(plus_specification)

    if value_minus == value_plus:
        x = x_minus
        y = y_minus
    else:
        with domain_range_scale('ignore'):
            Y_minus = luminance_ASTMD1535(value_minus)
            Y_plus = luminance_ASTMD1535(value_plus)

        x = LinearInterpolator((Y_minus, Y_plus), (x_minus, x_plus))(Y)
        y = LinearInterpolator((Y_minus, Y_plus), (y_minus, y_plus))(Y)

    return np.array([x, y, from_range_1(Y / 100)])


def munsell_specification_to_xyY(specification):
    """
    Converts given *Munsell* *Colorlab* specification to *CIE xyY* colourspace.

    Parameters
    ----------
    specification : array_like
        *Munsell* *Colorlab* specification.

    Returns
    -------
    ndarray
        *CIE xyY* colourspace array.

    Notes
    -----

    +-------------------+-----------------------+---------------+
    | **Domain**        | **Scale - Reference** | **Scale - 1** |
    +===================+=======================+===============+
    | ``specification`` | ``hue``    : [0, 10]  | [0, 1]        |
    |                   |                       |               |
    |                   | ``value``  : [0, 10]  | [0, 1]        |
    |                   |                       |               |
    |                   | ``chroma`` : [0, 50]  | [0, 1]        |
    |                   |                       |               |
    |                   | ``code``   : [0, 10]  | [0, 1]        |
    +-------------------+-----------------------+---------------+

    +-------------------+-----------------------+---------------+
    | **Range**         | **Scale - Reference** | **Scale - 1** |
    +===================+=======================+===============+
    | ``xyY``           | [0, 1]                | [0, 1]        |
    +-------------------+-----------------------+---------------+

    References
    ----------
    :cite:`Centore2014m`

    Examples
    --------
    >>> munsell_specification_to_xyY(np.array([2.1, 8.0, 17.9, 4]))
    ... # doctest: +ELLIPSIS
    array([ 0.4400632...,  0.5522428...,  0.5761962...])
    >>> munsell_specification_to_xyY(np.array([np.nan, 8.9, np.nan, np.nan]))
    ... # doctest: +ELLIPSIS
    array([ 0.31006  ,  0.31616  ,  0.7461345...])
    """

    specification = as_float_array(specification)
    shape = list(specification.shape)

    xyY = [
        _munsell_specification_to_xyY(a)
        for a in specification.reshape([-1, 4])
    ]

    shape[-1] = 3

    return as_float_array(xyY).reshape(shape)


def munsell_colour_to_xyY(munsell_colour):
    """
    Converts given *Munsell* colour to *CIE xyY* colourspace.

    Parameters
    ----------
    munsell_colour : unicode or array_like
        *Munsell* colour.

    Returns
    -------
    ndarray
        *CIE xyY* colourspace array.

    Notes
    -----

    +-----------+-----------------------+---------------+
    | **Range** | **Scale - Reference** | **Scale - 1** |
    +===========+=======================+===============+
    | ``xyY``   | [0, 1]                | [0, 1]        |
    +-----------+-----------------------+---------------+

    References
    ----------
    :cite:`Centorea`, :cite:`Centore2012a`

    Examples
    --------
    >>> munsell_colour_to_xyY('4.2YR 8.1/5.3')  # doctest: +ELLIPSIS
    array([ 0.3873694...,  0.3575165...,  0.59362   ])
    >>> munsell_colour_to_xyY('N8.9')  # doctest: +ELLIPSIS
    array([ 0.31006  ,  0.31616  ,  0.7461345...])
    """

    munsell_colour = np.array(munsell_colour)
    shape = list(munsell_colour.shape)

    specification = np.array([
        munsell_colour_to_munsell_specification(a)
        for a in np.ravel(munsell_colour)
    ])

    return munsell_specification_to_xyY(
        from_range_10(
            specification.reshape(shape + [4]), _domain_range_scale_factor()))


def _xyY_to_munsell_specification(xyY):
    """
    Converts from *CIE xyY* colourspace to *Munsell* *Colorlab* specification.

    Parameters
    ----------
    xyY : array_like
        *CIE xyY* colourspace array.

    Returns
    -------
    ndarray
        *Munsell* *Colorlab* specification.

    Raises
    ------
    ValueError
        If the given *CIE xyY* colourspace array is not within MacAdam
        limits.
    RuntimeError
        If the maximum iterations count has been reached without converging to
        a result.
    """

    x, y, Y = tsplit(xyY)
    Y = to_domain_1(Y)

    if not is_within_macadam_limits(xyY, MUNSELL_DEFAULT_ILLUMINANT):
        usage_warning('"{0}" is not within "MacAdam" limits for illuminant '
                      '"{1}"!'.format(xyY, MUNSELL_DEFAULT_ILLUMINANT))

    with domain_range_scale('ignore'):
        value = munsell_value_ASTMD1535(Y * 100)

    if is_integer(value):
        value = round(value)

    with domain_range_scale('ignore'):
        x_center, y_center, Y_center = _munsell_specification_to_xyY(value)

    rho_input, phi_input, _z_input = cartesian_to_cylindrical(
        (x - x_center, y - y_center, Y_center))
    phi_input = np.degrees(phi_input)

    grey_threshold = 1e-7
    if rho_input < grey_threshold:
        return from_range_10(normalize_munsell_specification(value))

    X, Y, Z = xyY_to_XYZ([x, y, Y])
    xi, yi = MUNSELL_DEFAULT_ILLUMINANT_CHROMATICITY_COORDINATES
    Xr, Yr, Zr = xyY_to_XYZ([xi, yi, Y])

    XYZ = np.array([X, Y, Z])
    XYZr = np.array([(1 / Yr) * Xr, 1, (1 / Yr) * Zr])

    Lab = XYZ_to_Lab(XYZ, XYZ_to_xy(XYZr))
    LCHab = Lab_to_LCHab(Lab)
    hue_initial, _value_initial, chroma_initial, code_initial = (
        LCHab_to_munsell_specification(LCHab))
    specification_current = [
        hue_initial, value, (5 / 5.5) * chroma_initial, code_initial
    ]

    convergence_threshold = 1e-7
    iterations_maximum = 64
    iterations = 0

    while iterations <= iterations_maximum:
        iterations += 1

        hue_current, _value_current, chroma_current, code_current = (
            specification_current)
        hue_angle_current = hue_to_hue_angle(hue_current, code_current)

        chroma_maximum = maximum_chroma_from_renotation(
            hue_current, value, code_current)
        if chroma_current > chroma_maximum:
            chroma_current = specification_current[2] = chroma_maximum

        with domain_range_scale('ignore'):
            x_current, y_current, _Y_current = _munsell_specification_to_xyY(
                specification_current)

        rho_current, phi_current, _z_current = cartesian_to_cylindrical(
            (x_current - x_center, y_current - y_center, Y_center))
        phi_current = np.degrees(phi_current)
        phi_current_difference = (360 - phi_input + phi_current) % 360
        if phi_current_difference > 180:
            phi_current_difference -= 360

        phi_differences = [phi_current_difference]
        hue_angles = [hue_angle_current]
        hue_angles_differences = [0]

        iterations_maximum_inner = 16
        iterations_inner = 0
        extrapolate = False

        while np.sign(min(phi_differences)) == np.sign(
                max(phi_differences)) and extrapolate is False:
            iterations_inner += 1

            if iterations_inner > iterations_maximum_inner:
                # NOTE: This exception is likely never raised in practice:
                # 300K iterations with random numbers never reached this code
                # path, it is kept for consistency with the reference
                # implementation.
                raise RuntimeError(  # pragma: no cover
                    ('Maximum inner iterations count reached without '
                     'convergence!'))

            hue_angle_inner = ((hue_angle_current + iterations_inner *
                                (phi_input - phi_current)) % 360)
            hue_angle_difference_inner = (
                iterations_inner * (phi_input - phi_current) % 360)
            if hue_angle_difference_inner > 180:
                hue_angle_difference_inner -= 360

            hue_inner, code_inner = hue_angle_to_hue(hue_angle_inner)

            with domain_range_scale('ignore'):
                x_inner, y_inner, _Y_inner = _munsell_specification_to_xyY(
                    (hue_inner, value, chroma_current, code_inner))

            if len(phi_differences) >= 2:
                extrapolate = True

            if extrapolate is False:
                rho_inner, phi_inner, _z_inner = cartesian_to_cylindrical(
                    (x_inner - x_center, y_inner - y_center, Y_center))
                phi_inner = np.degrees(phi_inner)
                phi_inner_difference = ((360 - phi_input + phi_inner) % 360)
                if phi_inner_difference > 180:
                    phi_inner_difference -= 360

                phi_differences.append(phi_inner_difference)
                hue_angles.append(hue_angle_inner)
                hue_angles_differences.append(hue_angle_difference_inner)

        phi_differences = np.array(phi_differences)
        hue_angles_differences = np.array(hue_angles_differences)

        phi_differences_indexes = phi_differences.argsort()

        phi_differences = phi_differences[phi_differences_indexes]
        hue_angles_differences = (
            hue_angles_differences[phi_differences_indexes])

        hue_angle_difference_new = Extrapolator(
            LinearInterpolator(phi_differences,
                               hue_angles_differences))(0) % 360
        hue_angle_new = (hue_angle_current + hue_angle_difference_new) % 360

        hue_new, code_new = hue_angle_to_hue(hue_angle_new)
        specification_current = [hue_new, value, chroma_current, code_new]

        with domain_range_scale('ignore'):
            x_current, y_current, _Y_current = _munsell_specification_to_xyY(
                specification_current)

        chroma_scale = 50 if get_domain_range_scale() == '1' else 2

        difference = euclidean_distance((x, y), (x_current, y_current))
        if difference < convergence_threshold:
            return from_range_10(
                np.array(specification_current),
                np.array([10, 10, chroma_scale, 10]))

        # TODO: Consider refactoring implementation.
        hue_current, _value_current, chroma_current, code_current = (
            specification_current)
        chroma_maximum = maximum_chroma_from_renotation(
            hue_current, value, code_current)

        # NOTE: This condition is likely never "True" while producing a valid
        # "Munsell Specification" in practice: 100K iterations with random
        # numbers never reached this code path while producing a valid
        # "Munsell Specification".
        if chroma_current > chroma_maximum:
            chroma_current = specification_current[2] = chroma_maximum

        with domain_range_scale('ignore'):
            x_current, y_current, _Y_current = _munsell_specification_to_xyY(
                specification_current)

        rho_current, phi_current, _z_current = cartesian_to_cylindrical(
            (x_current - x_center, y_current - y_center, Y_center))

        rho_bounds = [rho_current]
        chroma_bounds = [chroma_current]

        iterations_maximum_inner = 16
        iterations_inner = 0
        while not (min(rho_bounds) < rho_input < max(rho_bounds)):
            iterations_inner += 1

            if iterations_inner > iterations_maximum_inner:
                raise RuntimeError(('Maximum inner iterations count reached '
                                    'without convergence!'))

            chroma_inner = (((rho_input / rho_current) ** iterations_inner) *
                            chroma_current)
            if chroma_inner > chroma_maximum:
                chroma_inner = specification_current[2] = chroma_maximum

            specification_inner = (hue_current, value, chroma_inner,
                                   code_current)

            with domain_range_scale('ignore'):
                x_inner, y_inner, _Y_inner = _munsell_specification_to_xyY(
                    specification_inner)

            rho_inner, phi_inner, _z_inner = cartesian_to_cylindrical(
                (x_inner - x_center, y_inner - y_center, Y_center))

            rho_bounds.append(rho_inner)
            chroma_bounds.append(chroma_inner)

        rho_bounds = np.array(rho_bounds)
        chroma_bounds = np.array(chroma_bounds)

        rhos_bounds_indexes = rho_bounds.argsort()

        rho_bounds = rho_bounds[rhos_bounds_indexes]
        chroma_bounds = chroma_bounds[rhos_bounds_indexes]
        chroma_new = LinearInterpolator(rho_bounds, chroma_bounds)(rho_input)

        specification_current = [hue_current, value, chroma_new, code_current]

        with domain_range_scale('ignore'):
            x_current, y_current, _Y_current = _munsell_specification_to_xyY(
                specification_current)

        difference = euclidean_distance((x, y), (x_current, y_current))
        if difference < convergence_threshold:
            return from_range_10(
                np.array(specification_current),
                np.array([10, 10, chroma_scale, 10]))

    # NOTE: This exception is likely never raised in practice: 300K iterations
    # with random numbers never reached this code path, it is kept for
    # consistency with the reference # implementation
    raise RuntimeError(  # pragma: no cover
        'Maximum outside iterations count reached without convergence!')


def xyY_to_munsell_specification(xyY):
    """
    Converts from *CIE xyY* colourspace to *Munsell* *Colorlab* specification.

    Parameters
    ----------
    xyY : array_lik
        *CIE xyY* colourspace array.

    Returns
    -------
    ndarray
        *Munsell* *Colorlab* specification.

    Raises
    ------
    ValueError
        If the given *CIE xyY* colourspace array is not within MacAdam
        limits.
    RuntimeError
        If the maximum iterations count has been reached without converging to
        a result.

    Notes
    -----

    +-------------------+-----------------------+---------------+
    | **Domain**        | **Scale - Reference** | **Scale - 1** |
    +===================+=======================+===============+
    | ``xyY``           | [0, 1]                | [0, 1]        |
    +-------------------+-----------------------+---------------+

    +-------------------+-----------------------+---------------+
    | **Range**         | **Scale - Reference** | **Scale - 1** |
    +===================+=======================+===============+
    | ``specification`` | ``hue``    : [0, 10]  | [0, 1]        |
    |                   |                       |               |
    |                   | ``value``  : [0, 10]  | [0, 1]        |
    |                   |                       |               |
    |                   | ``chroma`` : [0, 50]  | [0, 1]        |
    |                   |                       |               |
    |                   | ``code``   : [0, 10]  | [0, 1]        |
    +-------------------+-----------------------+---------------+

    References
    ----------
    :cite:`Centore2014p`

    Examples
    --------
    >>> xyY = np.array([0.38736945, 0.35751656, 0.59362000])
    >>> xyY_to_munsell_specification(xyY)  # doctest: +ELLIPSIS
    array([ 4.2000019...,  8.0999999...,  5.2999996...,  6.        ])
    """

    xyY = as_float_array(xyY)
    shape = list(xyY.shape)

    specification = [
        _xyY_to_munsell_specification(a) for a in xyY.reshape([-1, 3])
    ]

    shape[-1] = 4

    return as_float_array(specification).reshape(shape)


def xyY_to_munsell_colour(xyY,
                          hue_decimals=1,
                          value_decimals=1,
                          chroma_decimals=1):
    """
    Converts from *CIE xyY* colourspace to *Munsell* colour.

    Parameters
    ----------
    xyY : array_like, (3,)
        *CIE xyY* colourspace array.
    hue_decimals : int
        Hue formatting decimals.
    value_decimals : int
        Value formatting decimals.
    chroma_decimals : int
        Chroma formatting decimals.

    Returns
    -------
    unicode
        *Munsell* colour.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``xyY``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Centorea`, :cite:`Centore2012a`

    Examples
    --------
    >>> xyY = np.array([0.38736945, 0.35751656, 0.59362000])
    >>> # Doctests skip for Python 2.x compatibility.
    >>> xyY_to_munsell_colour(xyY)  # doctest: +SKIP
    '4.2YR 8.1/5.3'
    """

    specification = to_domain_10(
        xyY_to_munsell_specification(xyY), _domain_range_scale_factor())
    shape = list(specification.shape)

    munsell_colour = [
        munsell_specification_to_munsell_colour(
            a, hue_decimals, value_decimals, chroma_decimals)
        for a in specification.reshape([-1, 4])
    ]
    munsell_colour = np.array(munsell_colour).reshape(shape[:-1])
    munsell_colour = str(munsell_colour) if shape == [4] else munsell_colour

    return munsell_colour


def parse_munsell_colour(munsell_colour):
    """
    Parses given *Munsell* colour and returns an intermediate *Munsell*
    *Colorlab* specification.

    Parameters
    ----------
    munsell_colour : unicode
        *Munsell* colour.

    Returns
    -------
    ndarray
        Intermediate *Munsell* *Colorlab* specification.

    Raises
    ------
    ValueError
        If the given specification is not a valid *Munsell Renotation System*
        colour specification.

    Examples
    --------
    >>> parse_munsell_colour('N5.2')
    array([ nan,  5.2,  nan,  nan])
    >>> parse_munsell_colour('0YR 2.0/4.0')
    array([ 0.,  2.,  4.,  6.])
    """

    match = re.match(MUNSELL_GRAY_PATTERN, munsell_colour, flags=re.IGNORECASE)
    if match:
        return as_float_array([
            np.nan,
            DEFAULT_FLOAT_DTYPE(match.group('value')),
            np.nan,
            np.nan,
        ])

    match = re.match(
        MUNSELL_COLOUR_PATTERN, munsell_colour, flags=re.IGNORECASE)
    if match:
        return as_float_array([
            DEFAULT_FLOAT_DTYPE(match.group('hue')),
            DEFAULT_FLOAT_DTYPE(match.group('value')),
            DEFAULT_FLOAT_DTYPE(match.group('chroma')),
            MUNSELL_HUE_LETTER_CODES.get(match.group('letter').upper()),
        ])

    raise ValueError(
        ('"{0}" is not a valid "Munsell Renotation System" colour '
         'specification!').format(munsell_colour))


def is_grey_munsell_colour(specification):
    """
    Returns if given *Munsell* *Colorlab* specification is a grey colour.

    Parameters
    ----------
    specification : array_like
        *Munsell* *Colorlab* specification.

    Returns
    -------
    bool
        Is specification a grey colour.

    Examples
    --------
    >>> is_grey_munsell_colour(np.array([0.0, 2.0, 4.0, 6]))
    False
    >>> is_grey_munsell_colour(np.array([np.nan, 0.5, np.nan, np.nan]))
    True
    """

    specification = as_float_array(specification)

    specification = specification[~np.isnan(specification)]

    return is_numeric(as_numeric(specification))


def normalize_munsell_specification(specification):
    """
    Normalises given *Munsell* *Colorlab* specification.

    Parameters
    ----------
    specification : array_like
        *Munsell* *Colorlab* specification.

    Returns
    -------
    ndarray
        Normalised *Munsell* *Colorlab* specification.

    Examples
    --------
    >>> normalize_munsell_specification(
    ...     np.array([0.0, 2.0, 4.0, 6]))
    array([ 10.,   2.,   4.,   7.])
    >>> normalize_munsell_specification(
    ...     np.array([np.nan, 0.5, np.nan, np.nan]))
    array([ nan,  0.5,  nan,  nan])
    """

    if is_grey_munsell_colour(specification):
        specification = as_float_array(specification)

        value = as_numeric(specification[~np.isnan(specification)])

        return as_float_array([np.nan, value, np.nan, np.nan])
    else:
        hue, value, chroma, code = specification

        if hue == 0:
            # 0YR is equivalent to 10R.
            hue, code = 10, (code + 1) % 10

        if chroma == 0:
            return as_float_array([np.nan, value, np.nan, np.nan])
        else:
            return as_float_array([hue, value, chroma, code])


def munsell_colour_to_munsell_specification(munsell_colour):
    """
    Convenient definition to retrieve a normalised *Munsell* *Colorlab*
    specification from given *Munsell* colour.

    Parameters
    ----------
    munsell_colour : unicode
        *Munsell* colour.

    Returns
    -------
    array_like
        Normalised *Munsell* *Colorlab* specification.

    Examples
    --------
    >>> munsell_colour_to_munsell_specification('N5.2')
    array([ nan,  5.2,  nan,  nan])
    >>> munsell_colour_to_munsell_specification('0YR 2.0/4.0')
    array([ 10.,   2.,   4.,   7.])
    """

    return normalize_munsell_specification(
        parse_munsell_colour(munsell_colour))


def munsell_specification_to_munsell_colour(specification,
                                            hue_decimals=1,
                                            value_decimals=1,
                                            chroma_decimals=1):
    """
    Converts from *Munsell* *Colorlab* specification to given *Munsell* colour.

    Parameters
    ----------
    specification : array_like
        *Munsell* *Colorlab* specification.
    hue_decimals : int, optional
        Hue formatting decimals.
    value_decimals : int, optional
        Value formatting decimals.
    chroma_decimals : int, optional
        Chroma formatting decimals.

    Returns
    -------
    unicode
        *Munsell* colour.

    Examples
    --------
    >>> # Doctests skip for Python 2.x compatibility.
    >>> munsell_specification_to_munsell_colour(
    ...     np.array([np.nan, 5.2, np.nan, np.nan]))  # doctest: +SKIP
    'N5.2'
    >>> # Doctests skip for Python 2.x compatibility.
    >>> munsell_specification_to_munsell_colour(
    ...     np.array([10, 2.0, 4.0, 7]))  # doctest: +SKIP
    '10.0R 2.0/4.0'
    """

    if is_grey_munsell_colour(specification):
        hue, value, chroma, code = normalize_munsell_specification(
            specification)

        return MUNSELL_GRAY_EXTENDED_FORMAT.format(value, value_decimals)
    else:
        rounding_decimals = (hue_decimals, value_decimals, chroma_decimals, 1)
        hue, value, chroma, code = [
            round(component, rounding_decimals[i])
            for i, component in enumerate(specification)
        ]
        code_values = MUNSELL_HUE_LETTER_CODES.values()

        assert 0 <= hue <= 10, (
            '"{0}" specification hue must be normalised to domain '
            '[0, 10]!'.format(specification))
        assert 0 <= value <= 10, (
            '"{0}" specification value must be normalised to domain '
            '[0, 10]!'.format(specification))
        assert 2 <= chroma <= 50, (
            '"{0}" specification chroma must be normalised to domain '
            '[2, 50]!'.format(specification))
        assert code in code_values, (
            '"{0}" specification code must one of "{1}"!'.format(
                specification, code_values))

        if hue == 0:
            hue, code = 10, (code + 1) % 10

        if value == 0:
            return MUNSELL_GRAY_EXTENDED_FORMAT.format(value, value_decimals)
        else:
            hue_letter = MUNSELL_HUE_LETTER_CODES.first_key_from_value(code)

            return MUNSELL_COLOUR_EXTENDED_FORMAT.format(
                hue, hue_decimals, hue_letter, value, value_decimals, chroma,
                chroma_decimals)


def xyY_from_renotation(specification):
    """
    Returns given existing *Munsell* *Colorlab* specification *CIE xyY*
    colourspace vector from *Munsell Renotation System* data.

    Parameters
    ----------
    specification : array_like
        *Munsell* *Colorlab* specification.

    Returns
    -------
    ndarray
        *CIE xyY* colourspace vector.

    Raises
    ------
    ValueError
        If the given specification doesn't exist in *Munsell Renotation System*
        data.

    Examples
    --------
    >>> xyY_from_renotation(np.array([2.5, 0.2, 2.0, 4]))  # doctest: +ELLIPSIS
    array([ 0.71...,  1.41...,  0.23...])
    """

    specification = normalize_munsell_specification(specification)

    try:
        index = np.where(
            (_munsell_specifications() == specification).all(axis=-1))

        return MUNSELL_COLOURS_ALL[as_int(index[0])][1]
    except Exception:
        raise ValueError(
            ('"{0}" specification does not exists in '
             '"Munsell Renotation System" data!').format(specification))


def is_specification_in_renotation(specification):
    """
    Returns if given *Munsell* *Colorlab* specification is in
    *Munsell Renotation System* data.

    Parameters
    ----------
    specification : array_like
        *Munsell* *Colorlab* specification.

    Returns
    -------
    bool
        Is specification in *Munsell Renotation System* data.

    Examples
    --------
    >>> is_specification_in_renotation(np.array([2.5, 0.2, 2.0, 4]))
    True
    >>> is_specification_in_renotation(np.array([64, 0.2, 2.0, 4]))
    False
    """

    try:
        xyY_from_renotation(specification)

        return True
    except ValueError:
        return False


def bounding_hues_from_renotation(hue, code):
    """
    Returns for a given hue the two bounding hues from
    *Munsell Renotation System* data.

    Parameters
    ----------
    hue : numeric
        *Munsell* *Colorlab* specification hue.
    code : numeric
        *Munsell* *Colorlab* specification code.

    Returns
    -------
    ndarray
        Bounding hues.

    References
    ----------
    :cite:`Centore2014o`

    Examples
    --------
    >>> bounding_hues_from_renotation(3.2, 4)
    array([[ 2.5,  4. ],
           [ 5. ,  4. ]])

    # Coverage Doctests

    >>> bounding_hues_from_renotation(0.0, 1)
    array([[ 10.,   2.],
           [ 10.,   2.]])
    """

    if hue % 2.5 == 0:
        if hue == 0:
            hue_cw = 10
            code_cw = (code + 1) % 10
        else:
            hue_cw = hue
            code_cw = code
        hue_ccw = hue_cw
        code_ccw = code_cw
    else:
        hue_cw = 2.5 * np.floor(hue / 2.5)
        hue_ccw = (hue_cw + 2.5) % 10
        if hue_ccw == 0:
            hue_ccw = 10

        if hue_cw == 0:
            hue_cw = 10
            code_cw = (code + 1) % 10
            if code_cw == 0:
                code_cw = 10
        else:
            code_cw = code
        code_ccw = code

    return as_float_array([(hue_cw, code_cw), (hue_ccw, code_ccw)])


def hue_to_hue_angle(hue, code):
    """
    Converts from the *Munsell* *Colorlab* specification hue to hue angle in
    degrees.

    Parameters
    ----------
    hue : numeric
        *Munsell* *Colorlab* specification hue.
    code : numeric
        *Munsell* *Colorlab* specification code.

    Returns
    -------
    numeric
        Hue angle in degrees.

    References
    ----------
    :cite:`Centore2014s`

    Examples
    --------
    >>> hue_to_hue_angle(3.2, 4)
    65.5
    """

    single_hue = ((17 - code) % 10 + (hue / 10) - 0.5) % 10

    return LinearInterpolator(
        (0, 2, 3, 4, 5, 6, 8, 9, 10),
        (0, 45, 70, 135, 160, 225, 255, 315, 360))(single_hue)


def hue_angle_to_hue(hue_angle):
    """
    Converts from hue angle in degrees to the *Munsell* *Colorlab*
    specification hue.

    Parameters
    ----------
    hue_angle : numeric
        Hue angle in degrees.

    Returns
    -------
    ndarray
        (*Munsell* *Colorlab* specification hue, *Munsell* *Colorlab*
        specification code).

    References
    ----------
    :cite:`Centore2014t`

    Examples
    --------
    >>> hue_angle_to_hue(65.54)  # doctest: +ELLIPSIS
    array([ 3.216,  4.   ])
    """

    single_hue = LinearInterpolator((0, 45, 70, 135, 160, 225, 255, 315, 360),
                                    (0, 2, 3, 4, 5, 6, 8, 9, 10))(hue_angle)

    if single_hue <= 0.5:
        code = 7
    elif single_hue <= 1.5:
        code = 6
    elif single_hue <= 2.5:
        code = 5
    elif single_hue <= 3.5:
        code = 4
    elif single_hue <= 4.5:
        code = 3
    elif single_hue <= 5.5:
        code = 2
    elif single_hue <= 6.5:
        code = 1
    elif single_hue <= 7.5:
        code = 10
    elif single_hue <= 8.5:
        code = 9
    elif single_hue <= 9.5:
        code = 8
    else:
        code = 7

    hue = (10 * (single_hue % 1) + 5) % 10
    if hue == 0:
        hue = 10

    return as_float_array([hue, code])


def hue_to_ASTM_hue(hue, code):
    """
    Converts from the *Munsell* *Colorlab* specification hue to *ASTM* hue
    number.

    Parameters
    ----------
    hue : numeric
        *Munsell* *Colorlab* specification hue.
    code : numeric
        *Munsell* *Colorlab* specification code.

    Returns
    -------
    numeric
        *ASTM* hue number.

    References
    ----------
    :cite:`Centore2014k`

    Examples
    --------
    >>> hue_to_ASTM_hue(3.2, 4)  # doctest: +ELLIPSIS
    33.2...
    """

    ASTM_hue = 10 * ((7 - code) % 10) + hue

    return 100 if ASTM_hue == 0 else ASTM_hue


def interpolation_method_from_renotation_ovoid(specification):
    """
    Returns whether to use linear or radial interpolation when drawing ovoids
    through data points in the *Munsell Renotation System* data from given
    specification.

    Parameters
    ----------
    specification : array_like
        *Munsell* *Colorlab* specification.

    Returns
    -------
    unicode or None ('Linear', 'Radial', None)
        Interpolation method.

    References
    ----------
    :cite:`Centore2014l`

    Examples
    --------
    >>> # Doctests skip for Python 2.x compatibility.
    >>> interpolation_method_from_renotation_ovoid((2.5, 5.0, 12.0, 4))
    ... # doctest: +SKIP
    'Radial'
    """

    specification = normalize_munsell_specification(specification)

    interpolation_methods = {0: None, 1: 'Linear', 2: 'Radial'}
    interpolation_method = 0
    if is_grey_munsell_colour(specification):
        # No interpolation needed for grey colours.
        interpolation_method = 0
    else:
        hue, value, chroma, code = specification

        assert 0 <= value <= 10, (
            '"{0}" specification value must be normalised to domain '
            '[0, 10]!'.format(specification))
        assert is_integer(value), (
            '"{0}" specification value must be an integer!'.format(
                specification))

        value = round(value)

        # Ideal white, no interpolation needed.
        if value == 10:
            interpolation_method = 0

        assert 2 <= chroma <= 50, (
            '"{0}" specification chroma must be normalised to domain '
            '[2, 50]!'.format(specification))
        assert abs(
            2 * (chroma / 2 - round(chroma / 2))) <= INTEGER_THRESHOLD, ((
                '"{0}" specification chroma must be an integer and '
                'multiple of 2!').format(specification))

        chroma = 2 * round(chroma / 2)

        # Standard Munsell Renotation System hue, no interpolation needed.
        if hue % 2.5 == 0:
            interpolation_method = 0

        ASTM_hue = hue_to_ASTM_hue(hue, code)

        # NOTE: The first level "else" clauses are likely never reached in
        # practice, they are kept for consistency with the reference
        # implementation.
        if value == 1:
            if chroma == 2:
                if 15 < ASTM_hue < 30 or 60 < ASTM_hue < 85:
                    interpolation_method = 2
                else:
                    interpolation_method = 1
            elif chroma == 4:
                if 12.5 < ASTM_hue < 27.5 or 57.5 < ASTM_hue < 80:
                    interpolation_method = 2
                else:
                    interpolation_method = 1
            elif chroma == 6:
                if 55 < ASTM_hue < 80:
                    interpolation_method = 2
                else:
                    interpolation_method = 1
            elif chroma == 8:
                if 67.5 < ASTM_hue < 77.5:
                    interpolation_method = 2
                else:
                    interpolation_method = 1
            elif chroma >= 10:
                # NOTE: This condition is likely never "True" while producing a
                # valid "Munsell Specification" in practice: 1M iterations with
                # random numbers never reached this code path while producing a
                # valid "Munsell Specification".
                if 72.5 < ASTM_hue < 77.5:  # pragma: no cover
                    interpolation_method = 2
                else:
                    interpolation_method = 1
            else:  # pragma: no cover
                interpolation_method = 1
        elif value == 2:
            if chroma == 2:
                if 15 < ASTM_hue < 27.5 or 77.5 < ASTM_hue < 80:
                    interpolation_method = 2
                else:
                    interpolation_method = 1
            elif chroma == 4:
                if 12.5 < ASTM_hue < 30 or 62.5 < ASTM_hue < 80:
                    interpolation_method = 2
                else:
                    interpolation_method = 1
            elif chroma == 6:
                if 7.5 < ASTM_hue < 22.5 or 62.5 < ASTM_hue < 80:
                    interpolation_method = 2
                else:
                    interpolation_method = 1
            elif chroma == 8:
                if 7.5 < ASTM_hue < 15 or 60 < ASTM_hue < 80:
                    interpolation_method = 2
                else:
                    interpolation_method = 1
            elif chroma >= 10:
                if 65 < ASTM_hue < 77.5:
                    interpolation_method = 2
                else:
                    interpolation_method = 1
            else:  # pragma: no cover
                interpolation_method = 1
        elif value == 3:
            if chroma == 2:
                if 10 < ASTM_hue < 37.5 or 65 < ASTM_hue < 85:
                    interpolation_method = 2
                else:
                    interpolation_method = 1
            elif chroma == 4:
                if 5 < ASTM_hue < 37.5 or 55 < ASTM_hue < 72.5:
                    interpolation_method = 2
                else:
                    interpolation_method = 1
            elif chroma in (6, 8, 10):
                if 7.5 < ASTM_hue < 37.5 or 57.5 < ASTM_hue < 82.5:
                    interpolation_method = 2
                else:
                    interpolation_method = 1
            elif chroma >= 12:
                if 7.5 < ASTM_hue < 42.5 or 57.5 < ASTM_hue < 80:
                    interpolation_method = 2
                else:
                    interpolation_method = 1
            else:  # pragma: no cover
                interpolation_method = 1
        elif value == 4:
            if chroma in (2, 4):
                if 7.5 < ASTM_hue < 42.5 or 57.5 < ASTM_hue < 85:
                    interpolation_method = 2
                else:
                    interpolation_method = 1
            elif chroma in (6, 8):
                if 7.5 < ASTM_hue < 40 or 57.5 < ASTM_hue < 82.5:
                    interpolation_method = 2
                else:
                    interpolation_method = 1
            elif chroma >= 10:
                if 7.5 < ASTM_hue < 40 or 57.5 < ASTM_hue < 80:
                    interpolation_method = 2
                else:
                    interpolation_method = 1
            else:  # pragma: no cover
                interpolation_method = 1
        elif value == 5:
            if chroma == 2:
                if 5 < ASTM_hue < 37.5 or 55 < ASTM_hue < 85:
                    interpolation_method = 2
                else:
                    interpolation_method = 1
            elif chroma in (4, 6, 8):
                if 2.5 < ASTM_hue < 42.5 or 55 < ASTM_hue < 85:
                    interpolation_method = 2
                else:
                    interpolation_method = 1
            elif chroma >= 10:
                if 2.5 < ASTM_hue < 42.5 or 55 < ASTM_hue < 82.5:
                    interpolation_method = 2
                else:
                    interpolation_method = 1
            else:  # pragma: no cover
                interpolation_method = 1
        elif value == 6:
            if chroma in (2, 4):
                if 5 < ASTM_hue < 37.5 or 55 < ASTM_hue < 87.5:
                    interpolation_method = 2
                else:
                    interpolation_method = 1
            elif chroma == 6:
                if 5 < ASTM_hue < 42.5 or 57.5 < ASTM_hue < 87.5:
                    interpolation_method = 2
                else:
                    interpolation_method = 1
            elif chroma in (8, 10):
                if 5 < ASTM_hue < 42.5 or 60 < ASTM_hue < 85:
                    interpolation_method = 2
                else:
                    interpolation_method = 1
            elif chroma in (12, 14):
                if 5 < ASTM_hue < 42.5 or 60 < ASTM_hue < 82.5:
                    interpolation_method = 2
                else:
                    interpolation_method = 1
            elif chroma >= 16:
                if 5 < ASTM_hue < 42.5 or 60 < ASTM_hue < 80:
                    interpolation_method = 2
                else:
                    interpolation_method = 1
            else:  # pragma: no cover
                interpolation_method = 1
        elif value == 7:
            if chroma in (2, 4, 6):
                if 5 < ASTM_hue < 42.5 or 60 < ASTM_hue < 85:
                    interpolation_method = 2
                else:
                    interpolation_method = 1
            elif chroma == 8:
                if 5 < ASTM_hue < 42.5 or 60 < ASTM_hue < 82.5:
                    interpolation_method = 2
                else:
                    interpolation_method = 1
            elif chroma == 10:
                if (30 < ASTM_hue < 42.5 or 5 < ASTM_hue < 25 or
                        60 < ASTM_hue < 82.5):
                    interpolation_method = 2
                else:
                    interpolation_method = 1
            elif chroma == 12:
                if (30 < ASTM_hue < 42.5 or 7.5 < ASTM_hue < 27.5 or
                        80 < ASTM_hue < 82.5):
                    interpolation_method = 2
                else:
                    interpolation_method = 1
            elif chroma >= 14:
                if (32.5 < ASTM_hue < 40 or 7.5 < ASTM_hue < 15 or
                        80 < ASTM_hue < 82.5):
                    interpolation_method = 2
                else:
                    interpolation_method = 1
            else:  # pragma: no cover
                interpolation_method = 1
        elif value == 8:
            if chroma in (2, 4, 6, 8, 10, 12):
                if 5 < ASTM_hue < 40 or 60 < ASTM_hue < 85:
                    interpolation_method = 2
                else:
                    interpolation_method = 1
            elif chroma >= 14:
                if (32.5 < ASTM_hue < 40 or 5 < ASTM_hue < 15 or
                        60 < ASTM_hue < 85):
                    interpolation_method = 2
                else:
                    interpolation_method = 1
            else:  # pragma: no cover
                interpolation_method = 1
        elif value == 9:
            if chroma in (2, 4):
                if 5 < ASTM_hue < 40 or 55 < ASTM_hue < 80:
                    interpolation_method = 2
                else:
                    interpolation_method = 1
            elif chroma in (6, 8, 10, 12, 14):
                if 5 < ASTM_hue < 42.5:
                    interpolation_method = 2
                else:
                    interpolation_method = 1
            elif chroma >= 16:
                if 35 < ASTM_hue < 42.5:
                    interpolation_method = 2
                else:
                    interpolation_method = 1
            else:  # pragma: no cover
                interpolation_method = 1

    return interpolation_methods.get(interpolation_method)


def xy_from_renotation_ovoid(specification):
    """
    Converts given *Munsell* *Colorlab* specification to *CIE xy* chromaticity
    coordinates on *Munsell Renotation System* ovoid.
    The *CIE xy* point will be on the ovoid about the achromatic point,
    corresponding to the *Munsell* *Colorlab* specification
    value and chroma.

    Parameters
    ----------
    specification : array_like
        *Munsell* *Colorlab* specification.

    Returns
    -------
    ndarray
        *CIE xy* chromaticity coordinates.

    Raises
    ------
    ValueError
        If an invalid interpolation method is retrieved from internal
        computations.

    References
    ----------
    :cite:`Centore2014n`

    Examples
    --------
    >>> xy_from_renotation_ovoid((2.5, 5.0, 12.0, 4))
    ... # doctest: +ELLIPSIS
    array([ 0.4333...,  0.5602...])
    >>> xy_from_renotation_ovoid((np.nan, 8, np.nan, np.nan))
    ... # doctest: +ELLIPSIS
    array([ 0.31006...,  0.31616...])
    """

    specification = normalize_munsell_specification(specification)

    if is_grey_munsell_colour(specification):
        return MUNSELL_DEFAULT_ILLUMINANT_CHROMATICITY_COORDINATES
    else:
        hue, value, chroma, code = specification

        assert 1 <= value <= 9, (
            '"{0}" specification value must be normalised to domain '
            '[1, 9]!'.format(specification))
        assert is_integer(value), (
            '"{0}" specification value must be an integer!'.format(
                specification))

        value = round(value)

        assert 2 <= chroma <= 50, (
            '"{0}" specification chroma must be normalised to domain '
            '[2, 50]!'.format(specification))
        assert abs(
            2 * (chroma / 2 - round(chroma / 2))) <= INTEGER_THRESHOLD, ((
                '"{0}" specification chroma must be an integer and '
                'multiple of 2!').format(specification))

        chroma = 2 * round(chroma / 2)

        # Checking if renotation data is available without interpolation using
        # given threshold.
        threshold = 1e-7
        if (abs(hue) < threshold or abs(hue - 2.5) < threshold or
                abs(hue - 5) < threshold or abs(hue - 7.5) < threshold or
                abs(hue - 10) < threshold):
            hue = 2.5 * round(hue / 2.5)

            x, y, _Y = xyY_from_renotation((hue, value, chroma, code))

            return as_float_array([x, y])

        hue_cw, hue_ccw = bounding_hues_from_renotation(hue, code)
        hue_minus, code_minus = hue_cw
        hue_plus, code_plus = hue_ccw

        x_grey, y_grey = MUNSELL_DEFAULT_ILLUMINANT_CHROMATICITY_COORDINATES

        specification_minus = (hue_minus, value, chroma, code_minus)
        x_minus, y_minus, Y_minus = xyY_from_renotation(specification_minus)
        rho_minus, phi_minus, _z_minus = cartesian_to_cylindrical(
            (x_minus - x_grey, y_minus - y_grey, Y_minus))
        phi_minus = np.degrees(phi_minus)

        specification_plus = (hue_plus, value, chroma, code_plus)
        x_plus, y_plus, Y_plus = xyY_from_renotation(specification_plus)
        rho_plus, phi_plus, _z_plus = cartesian_to_cylindrical(
            (x_plus - x_grey, y_plus - y_grey, Y_plus))
        phi_plus = np.degrees(phi_plus)

        lower_hue_angle = hue_to_hue_angle(hue_minus, code_minus)
        hue_angle = hue_to_hue_angle(hue, code)
        upper_hue_angle = hue_to_hue_angle(hue_plus, code_plus)

        if phi_minus - phi_plus > 180:
            phi_plus += 360

        if lower_hue_angle == 0:
            lower_hue_angle = 360

        if lower_hue_angle > upper_hue_angle:
            if lower_hue_angle > hue_angle:
                lower_hue_angle -= 360
            else:
                lower_hue_angle -= 360
                hue_angle -= 360

        interpolation_method = interpolation_method_from_renotation_ovoid(
            specification).lower()

        assert interpolation_method is not None, (
            'Interpolation method must be one of : "{0}"'.format(', '.join(
                ['Linear', 'radial'])))

        if interpolation_method == 'linear':
            x = LinearInterpolator((lower_hue_angle, upper_hue_angle),
                                   (x_minus, x_plus))(hue_angle)
            y = LinearInterpolator((lower_hue_angle, upper_hue_angle),
                                   (y_minus, y_plus))(hue_angle)
        elif interpolation_method == 'radial':
            theta = LinearInterpolator((lower_hue_angle, upper_hue_angle),
                                       (phi_minus, phi_plus))(hue_angle)
            rho = LinearInterpolator((lower_hue_angle, upper_hue_angle),
                                     (rho_minus, rho_plus))(hue_angle)

            x, y = tsplit(
                polar_to_cartesian((rho, np.radians(theta))) +
                as_float_array((x_grey, y_grey)))

        return as_float_array([x, y])


def LCHab_to_munsell_specification(LCHab):
    """
    Converts from *CIE L\\*C\\*Hab* colourspace to approximate *Munsell*
    *Colorlab* specification.

    Parameters
    ----------
    LCHab : array_like, (3,)
        *CIE L\\*C\\*Hab* colourspace array.

    Returns
    -------
    ndarray
        *Munsell* *Colorlab* specification.

    References
    ----------
    :cite:`Centore2014u`

    Examples
    --------
    >>> LCHab = np.array([100, 17.50664796, 244.93046842])
    >>> LCHab_to_munsell_specification(LCHab)  # doctest: +ELLIPSIS
    array([  8.0362412...,  10.        ,   3.5013295...,   1.        ])
    """

    L, C, Hab = tsplit(LCHab)

    if Hab == 0:
        code = 8
    elif Hab <= 36:
        code = 7
    elif Hab <= 72:
        code = 6
    elif Hab <= 108:
        code = 5
    elif Hab <= 144:
        code = 4
    elif Hab <= 180:
        code = 3
    elif Hab <= 216:
        code = 2
    elif Hab <= 252:
        code = 1
    elif Hab <= 288:
        code = 10
    elif Hab <= 324:
        code = 9
    else:
        code = 8

    hue = LinearInterpolator((0, 36), (0, 10))(Hab % 36)
    if hue == 0:
        hue = 10

    value = L / 10
    chroma = C / 5

    return as_float_array([hue, value, chroma, code])


def maximum_chroma_from_renotation(hue, value, code):
    """
    Returns the maximum *Munsell* chroma from *Munsell Renotation System* data
    using given *Munsell* *Colorlab* specification hue, *Munsell* *Colorlab*
    specification value and *Munsell* *Colorlab* specification code.

    Parameters
    ----------
    hue : numeric
        *Munsell* *Colorlab* specification hue.
    value : numeric
        *Munsell* value code.
    code : numeric
        *Munsell* *Colorlab* specification code.

    Returns
    -------
    numeric
        Maximum chroma.

    References
    ----------
    :cite:`Centore2014r`

    Examples
    --------
    >>> maximum_chroma_from_renotation(2.5, 5, 5)
    14.0
    """

    # Ideal white, no chroma.
    if value >= 9.99:
        return 0

    assert 1 <= value <= 10, (
        '"{0}" value must be normalised to domain [1, 10]!'.format(value))

    if value % 1 == 0:
        value_minus = value
        value_plus = value
    else:
        value_minus = np.floor(value)
        value_plus = value_minus + 1

    hue_cw, hue_ccw = bounding_hues_from_renotation(hue, code)
    hue_cw, code_cw = hue_cw
    hue_ccw, code_ccw = hue_ccw

    maximum_chromas = _munsell_maximum_chromas_from_renotation()
    specification_for_indexes = [chroma[0] for chroma in maximum_chromas]

    ma_limit_mcw = maximum_chromas[specification_for_indexes.index(
        (hue_cw, value_minus, code_cw))][1]
    ma_limit_mccw = maximum_chromas[specification_for_indexes.index(
        (hue_ccw, value_minus, code_ccw))][1]

    if value_plus <= 9:
        ma_limit_pcw = maximum_chromas[specification_for_indexes.index(
            (hue_cw, value_plus, code_cw))][1]
        ma_limit_pccw = maximum_chromas[specification_for_indexes.index(
            (hue_ccw, value_plus, code_ccw))][1]
        max_chroma = min(ma_limit_mcw, ma_limit_mccw, ma_limit_pcw,
                         ma_limit_pccw)
    else:
        L = luminance_ASTMD1535(value)
        L9 = luminance_ASTMD1535(9)
        L10 = luminance_ASTMD1535(10)

        max_chroma = min(
            LinearInterpolator((L9, L10), (ma_limit_mcw, 0))(L),
            LinearInterpolator((L9, L10), (ma_limit_mccw, 0))(L))
    return max_chroma


def munsell_specification_to_xy(specification):
    """
    Converts given *Munsell* *Colorlab* specification to *CIE xy* chromaticity
    coordinates by interpolating over *Munsell Renotation System* data.

    Parameters
    ----------
    specification : array_like
        *Munsell* *Colorlab* specification.

    Returns
    -------
    ndarray
        *CIE xy* chromaticity coordinates.

    References
    ----------
    :cite:`Centore2014q`

    Examples
    --------
    >>> # Doctests ellipsis for Python 2.x compatibility.
    >>> munsell_specification_to_xy((2.1, 8.0, 17.9, 4))
    ... # doctest: +ELLIPSIS
    array([ 0.4400632...,  0.5522428...])
    >>> munsell_specification_to_xy((np.nan, 8, np.nan, np.nan))
    ... # doctest: +ELLIPSIS
    array([ 0.31006...,  0.31616...])
    """

    specification = normalize_munsell_specification(specification)

    if is_grey_munsell_colour(specification):
        return MUNSELL_DEFAULT_ILLUMINANT_CHROMATICITY_COORDINATES
    else:
        hue, value, chroma, code = specification

        assert 0 <= value <= 10, (
            '"{0}" specification value must be normalised to domain '
            '[0, 10]!'.format(specification))
        assert is_integer(value), (
            '"{0}" specification value must be an integer!'.format(
                specification))

        value = round(value)

        if chroma % 2 == 0:
            chroma_minus = chroma_plus = chroma
        else:
            chroma_minus = 2 * np.floor(chroma / 2)
            chroma_plus = chroma_minus + 2

        if chroma_minus == 0:
            # Smallest chroma ovoid collapses to illuminant chromaticity
            # coordinates.
            x_minus, y_minus = (
                MUNSELL_DEFAULT_ILLUMINANT_CHROMATICITY_COORDINATES)
        else:
            x_minus, y_minus = xy_from_renotation_ovoid((hue, value,
                                                         chroma_minus, code))

        x_plus, y_plus = xy_from_renotation_ovoid((hue, value, chroma_plus,
                                                   code))

        if chroma_minus == chroma_plus:
            x = x_minus
            y = y_minus
        else:
            x = LinearInterpolator((chroma_minus, chroma_plus),
                                   (x_minus, x_plus))(chroma)
            y = LinearInterpolator((chroma_minus, chroma_plus),
                                   (y_minus, y_plus))(chroma)

        return as_float_array([x, y])
