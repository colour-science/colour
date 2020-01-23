# -*- coding: utf-8 -*-
"""
Yellowness Index :math:`Y`
==========================

Defines *yellowness* index :math:`Y` computation objects:

-   :func:`colour.colorimetry.yellowness_ASTMD1925`: *Yellowness* index
    :math:`YI` computation of given sample *CIE XYZ* tristimulus values using
    *ASTM D1925* method.
-   :func:`colour.colorimetry.yellowness_ASTME313`: *Yellowness* index
    :math:`YI` computation of given sample *CIE XYZ* tristimulus values using
    *ASTM E313* method.
-   :attr:`colour.YELLOWNESS_METHODS`: Supported *yellowness* computations
    methods.
-   :func:`colour.whiteness`: *Yellowness* :math:`YI` computation using given
    method.

See Also
--------
`Yellowness Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/colorimetry/yellowness.ipynb>`_

References
----------
-   :cite:`X-Rite2012a` : X-Rite, & Pantone. (2012). Color iQC and Color
    iMatch Color Calculations Guide. Retrieved from
    https://www.xrite.com/-/media/xrite/files/\
apps_engineering_techdocuments/c/09_color_calculations_en.pdf
"""

from __future__ import division, unicode_literals

from colour.utilities import (CaseInsensitiveMapping, from_range_100,
                              to_domain_100, tsplit)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'yellowness_ASTMD1925', 'yellowness_ASTME313', 'YELLOWNESS_METHODS',
    'yellowness'
]


def yellowness_ASTMD1925(XYZ):
    """
    Returns the *yellowness* index :math:`YI` of given sample *CIE XYZ*
    tristimulus values using *ASTM D1925* method.

    ASTM D1925 has been specifically developed for the definition of the
    Yellowness of homogeneous, non-fluorescent, almost neutral-transparent,
    white-scattering or opaque plastics as they will be reviewed under daylight
    condition. It can be other materials as well, as long as they fit into this
    description.

    Parameters
    ----------
    XYZ : array_like
        *CIE XYZ* tristimulus values of sample.

    Returns
    -------
    numeric or ndarray
        *Whiteness* :math:`YI`.

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
    | ``YI``     | [0, 100]              | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`X-Rite2012a`

    Examples
    --------
    >>> import numpy as np
    >>> XYZ = np.array([95.00000000, 100.00000000, 105.00000000])
    >>> yellowness_ASTMD1925(XYZ)  # doctest: +ELLIPSIS
    10.2999999...
    """

    X, Y, Z = tsplit(to_domain_100(XYZ))

    YI = (100 * (1.28 * X - 1.06 * Z)) / Y

    return from_range_100(YI)


def yellowness_ASTME313(XYZ):
    """
    Returns the *yellowness* index :math:`YI` of given sample *CIE XYZ*
    tristimulus values using *ASTM E313* method.

    ASTM E313 has successfully been used for a variety of white or near white
    materials. This includes coatings, Plastics, Textiles.

    Parameters
    ----------
    XYZ : array_like
        *CIE XYZ* tristimulus values of sample.

    Returns
    -------
    numeric or ndarray
        *Whiteness* :math:`YI`.

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
    | ``YI``     | [0, 100]              | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`X-Rite2012a`

    Examples
    --------
    >>> import numpy as np
    >>> XYZ = np.array([95.00000000, 100.00000000, 105.00000000])
    >>> yellowness_ASTME313(XYZ)  # doctest: +ELLIPSIS
    11.0650000...
    """

    _X, Y, Z = tsplit(to_domain_100(XYZ))

    WI = 100 * (1 - (0.847 * Z) / Y)

    return from_range_100(WI)


YELLOWNESS_METHODS = CaseInsensitiveMapping({
    'ASTM D1925': yellowness_ASTMD1925,
    'ASTM E313': yellowness_ASTME313
})
YELLOWNESS_METHODS.__doc__ = """
Supported *yellowness* computation methods.

References
----------
:cite:`X-Rite2012a`

YELLOWNESS_METHODS : CaseInsensitiveMapping
    **{'ASTM E313', 'ASTM D1925'}**
"""


def yellowness(XYZ, method='ASTM E313'):
    """
    Returns the *yellowness* :math:`W` using given method.

    Parameters
    ----------
    XYZ : array_like
        *CIE XYZ* tristimulus values of sample.
    method : unicode, optional
        **{'ASTM E313', 'ASTM D1925'}**,
        Computation method.

    Returns
    -------
    numeric or ndarray
        *yellowness* :math:`Y`.

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
    | ``YI``     | [0, 100]              | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`X-Rite2012a`

    Examples
    --------
    >>> import numpy as np
    >>> XYZ = np.array([95.00000000, 100.00000000, 105.00000000])
    >>> yellowness(XYZ)  # doctest: +ELLIPSIS
    11.0650000...
    >>> method = 'ASTM D1925'
    >>> yellowness(XYZ, method=method)  # doctest: +ELLIPSIS
    10.2999999...
    """

    return YELLOWNESS_METHODS.get(method)(XYZ)
