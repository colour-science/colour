# -*- coding: utf-8 -*-
"""
DICOM - Grayscale Standard Display Function
===========================================

Defines the *DICOM - Grayscale Standard Display Function* electro-optical
transfer function (EOTF / EOCF) and its inverse:

-   :func:`colour.models.eotf_inverse_DICOMGSDF`
-   :func:`colour.models.eotf_DICOMGSDF`

The Grayscale Standard Display Function is defined for the Luminance Range
from :math:`0.05` to :math:`4000 cd/m^2`. The minimum Luminance corresponds
to the lowest practically useful Luminance of cathode-ray-tube (CRT) monitors
and the maximum exceeds the unattenuated Luminance of very bright light-boxes
used for interpreting X-Ray mammography. The Grayscale Standard Display
Function explicitly includes the effects of the diffused ambient Illuminance.

See Also
--------
`RGB Colourspaces Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
-   :cite:`NationalElectricalManufacturersAssociation2004b` : National
    Electrical Manufacturers Association. (2004). Digital Imaging and
    Communications in Medicine (DICOM) Part 14: Grayscale Standard Display
    Function. Retrieved from http://medical.nema.org/dicom/2004/04_14PU.PDF
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.utilities import (Structure, as_float, as_int, from_range_1,
                              to_domain_1)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['DICOMGSDF_CONSTANTS', 'eotf_inverse_DICOMGSDF', 'eotf_DICOMGSDF']

DICOMGSDF_CONSTANTS = Structure(
    a=-1.3011877,
    b=-2.5840191e-2,
    c=8.0242636e-2,
    d=-1.0320229e-1,
    e=1.3646699e-1,
    f=2.8745620e-2,
    g=-2.5468404e-2,
    h=-3.1978977e-3,
    k=1.2992634e-4,
    m=1.3635334e-3,
    A=71.498068,
    B=94.593053,
    C=41.912053,
    D=9.8247004,
    E=0.28175407,
    F=-1.1878455,
    G=-0.18014349,
    H=0.14710899,
    I=-0.017046845)  # noqa
"""
*DICOM Grayscale Standard Display Function* constants.

DICOMGSDF_CONSTANTS : Structure
"""


def eotf_inverse_DICOMGSDF(L, out_int=False):
    """
    Defines the *DICOM - Grayscale Standard Display Function* inverse
    electro-optical transfer function (EOTF / EOCF).

    Parameters
    ----------
    L : numeric or array_like
        *Luminance* :math:`L`.
    out_int : bool, optional
        Whether to return value as integer code value or float equivalent of a
        code value at a given bit depth.

    Returns
    -------
    numeric or ndarray
        Just-Noticeable Difference (JND) Index, :math:`j`.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``L``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``J``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`NationalElectricalManufacturersAssociation2004b`

    Examples
    --------
    >>> eotf_inverse_DICOMGSDF(130.0662)  # doctest: +ELLIPSIS
    0.5004862...
    >>> eotf_inverse_DICOMGSDF(130.0662, out_int=True)
    512
    """

    L = to_domain_1(L)

    L_lg = np.log10(L)

    A = DICOMGSDF_CONSTANTS.A
    B = DICOMGSDF_CONSTANTS.B
    C = DICOMGSDF_CONSTANTS.C
    D = DICOMGSDF_CONSTANTS.D
    E = DICOMGSDF_CONSTANTS.E
    F = DICOMGSDF_CONSTANTS.F
    G = DICOMGSDF_CONSTANTS.G
    H = DICOMGSDF_CONSTANTS.H
    I = DICOMGSDF_CONSTANTS.I  # noqa

    J = (A + B * L_lg + C * L_lg ** 2 + D * L_lg ** 3 + E * L_lg ** 4 +
         F * L_lg ** 5 + G * L_lg ** 6 + H * L_lg ** 7 + I * L_lg ** 8)

    if out_int:
        return as_int(np.round(J))
    else:
        return as_float(from_range_1(J / 1023))


def eotf_DICOMGSDF(J, in_int=False):
    """
    Defines the *DICOM - Grayscale Standard Display Function* electro-optical
    transfer function (EOTF / EOCF).

    Parameters
    ----------
    J : numeric or array_like
        Just-Noticeable Difference (JND) Index, :math:`j`.
    in_int : bool, optional
        Whether to treat the input value as integer code value or float
        equivalent of a code value at a given bit depth.

    Returns
    -------
    numeric or ndarray
        Corresponding *luminance* :math:`L`.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``J``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``L``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`NationalElectricalManufacturersAssociation2004b`

    Examples
    --------
    >>> eotf_DICOMGSDF(0.500486263438448)  # doctest: +ELLIPSIS
    130.0628647...
    >>> eotf_DICOMGSDF(512, in_int=True)  # doctest: +ELLIPSIS
    130.0652840...
    """

    J = to_domain_1(J)

    if not in_int:
        J = J * 1023

    a = DICOMGSDF_CONSTANTS.a
    b = DICOMGSDF_CONSTANTS.b
    c = DICOMGSDF_CONSTANTS.c
    d = DICOMGSDF_CONSTANTS.d
    e = DICOMGSDF_CONSTANTS.e
    f = DICOMGSDF_CONSTANTS.f
    g = DICOMGSDF_CONSTANTS.g
    h = DICOMGSDF_CONSTANTS.h
    k = DICOMGSDF_CONSTANTS.k
    m = DICOMGSDF_CONSTANTS.m

    J_ln = np.log(J)
    J_ln2 = J_ln ** 2
    J_ln3 = J_ln ** 3
    J_ln4 = J_ln ** 4
    J_ln5 = J_ln ** 5

    L = ((a + c * J_ln + e * J_ln2 + g * J_ln3 + m * J_ln4) /
         (1 + b * J_ln + d * J_ln2 + f * J_ln3 + h * J_ln4 + k * J_ln5))
    L = 10 ** L

    return as_float(from_range_1(L))
