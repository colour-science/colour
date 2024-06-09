"""
Recommendation ITU-T H.273 Transfer Characteristics
===================================================

Define the *Recommendation ITU-T H.273* transfer functions that do not belong
in another specification or standard, or have been modified for inclusion:

-   :func:`colour.models.oetf_H273_Log`
-   :func:`colour.models.oetf_inverse_H273_Log`
-   :func:`colour.models.oetf_H273_LogSqrt`
-   :func:`colour.models.oetf_inverse_H273_LogSqrt`
-   :func:`colour.models.oetf_H273_IEC61966_2`
-   :func:`colour.models.oetf_inverse_H273_IEC61966_2`
-   :func:`colour.models.eotf_H273_ST428_1`
-   :func:`colour.models.eotf_inverse_H273_ST428_1`

References
----------
-   :cite:`InternationalTelecommunicationUnion2021` : International
    Telecommunication Union. (2021). Recommendation ITU-T H.273 -
    Coding-independent code points for video signal type identification.
    https://www.itu.int/rec/T-REC-H.273-202107-I/en
-   :cite:`SocietyofMotionPictureandTelevisionEngineers2019` : Society of
    Motion Picture and Television Engineers. (2019). ST 428-1:2019 - D-Cinema
    Distribution Master — Image Characteristic. doi:10.5594/SMPTE.ST428-1.2019
"""

import numpy as np

from colour.algebra import spow
from colour.models.rgb.transfer_functions import (
    eotf_DCDM,
    eotf_inverse_DCDM,
    eotf_inverse_sRGB,
    eotf_sRGB,
)
from colour.utilities import (
    as_float,
    as_float_array,
    from_range_1,
    to_domain_1,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "oetf_H273_Log",
    "oetf_inverse_H273_Log",
    "oetf_H273_LogSqrt",
    "oetf_inverse_H273_LogSqrt",
    "oetf_H273_IEC61966_2",
    "oetf_inverse_H273_IEC61966_2",
    "eotf_inverse_H273_ST428_1",
    "eotf_H273_ST428_1",
]


def oetf_H273_Log(L_c):
    """
    Define *Recommendation ITU-T H.273* opto-electronic transfer function
    (OETF) for logarithmic encoding (100:1 range).

    Parameters
    ----------
    L_c
        Scene *Luminance* :math:`L_c`.

    Returns
    -------
    :class:`numpy.ndarray`
        Corresponding electrical signal :math:`V`.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``L_c``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``V``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`InternationalTelecommunicationUnion2021`

    Warnings
    --------
    -   The function is clamped to domain [0.01, np.inf].

    Examples
    --------
    >>> oetf_H273_Log(0.18)  # doctest: +ELLIPSIS
    0.6276362525516...
    >>> oetf_H273_Log(0.01)  # doctest: +ELLIPSIS
    0.0
    >>> oetf_H273_Log(0.001)  # doctest: +ELLIPSIS
    0.0
    >>> oetf_H273_Log(1.0)  # doctest: +ELLIPSIS
    1.0
    """

    L_c = to_domain_1(L_c)

    V = np.where(
        L_c >= 0.01,
        # L_c in [0.01, 1] range
        1 + np.log10(L_c) / 2,
        # L_c in [0, 0.01] range
        0,
    )

    return as_float(from_range_1(V))


def oetf_inverse_H273_Log(V):
    """
    Define *Recommendation ITU-T H.273* inverse-opto-electronic transfer
    function (OETF) for logarithmic encoding (100:1 range).

    Parameters
    ----------
    V
        Electrical signal :math:`V`.

    Returns
    -------
    :class:`numpy.ndarray`
        Corresponding scene *Luminance* :math:`L_c`.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``V``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``L_c``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`InternationalTelecommunicationUnion2021`

    Warnings
    --------
    -   The function is clamped to domain
        [:func:`colour.models.oetf_H273_Log` (0.01), np.inf].

    Examples
    --------
    >>> oetf_inverse_H273_Log(0.6276362525516)  # doctest: +ELLIPSIS
    0.17999999...
    >>> oetf_inverse_H273_Log(0.0)  # doctest: +ELLIPSIS
    0.01
    >>> oetf_inverse_H273_Log(1.0)  # doctest: +ELLIPSIS
    1.0
    """

    V = to_domain_1(V)

    L_c = np.where(
        oetf_H273_Log(0.01) <= V,
        # L_c in [0.01, 1] range
        spow(10, (V - 1) * 2),
        # L_c in [0, 0.01] range
        0,
    )

    return as_float(from_range_1(L_c))


def oetf_H273_LogSqrt(L_c):
    """
    Define *Recommendation ITU-T H.273* opto-electronic transfer function
    (OETF) for logarithmic encoding (100\\*Sqrt(10):1 range).

    Parameters
    ----------
    L_c
        Scene *Luminance* :math:`L_c`.

    Returns
    -------
    :class:`numpy.ndarray`
        Corresponding electrical signal :math:`V`.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``L_c``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``V``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`InternationalTelecommunicationUnion2021`

    Warnings
    --------
    -   The function is clamped to domain
        [:func:`colour.models.oetf_H273_LogSqrt` (sqrt(10) / 1000), np.inf].

    Examples
    --------
    >>> oetf_H273_LogSqrt(0.18)  # doctest: +ELLIPSIS
    0.702109002041...
    >>> oetf_H273_LogSqrt(0.003162277660168)  # doctest: +ELLIPSIS
    0.0
    >>> oetf_H273_LogSqrt(0.0001)  # doctest: +ELLIPSIS
    0.0
    >>> oetf_H273_LogSqrt(1.0)  # doctest: +ELLIPSIS
    1.0
    """

    L_c = to_domain_1(L_c)

    V = np.where(
        L_c >= np.sqrt(10) / 1000,
        # L_c in [sqrt(10)/1000, 1] range
        1 + np.log10(L_c) / 2.5,
        # L_c in [0, sqrt(10)/1000] range
        0,
    )

    return as_float(from_range_1(V))


def oetf_inverse_H273_LogSqrt(V):
    """
    Define *Recommendation ITU-T H.273* inverse-opto-electronic transfer
    function (OETF) for logarithmic encoding (100\\*Sqrt(10):1 range).

    Parameters
    ----------
    V
        Electrical signal :math:`V`.

    Returns
    -------
    :class:`numpy.ndarray`
        Corresponding scene *Luminance* :math:`L_c`.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``V``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``L_c``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`InternationalTelecommunicationUnion2021`

    Warnings
    --------
    -   The function is clamped to domain [sqrt(10) / 1000, np.inf].

    Examples
    --------
    >>> oetf_inverse_H273_LogSqrt(0.702109002041)  # doctest: +ELLIPSIS
    0.1799999999...
    >>> oetf_inverse_H273_LogSqrt(0.0)  # doctest: +ELLIPSIS
    0.00316227766...
    >>> oetf_inverse_H273_LogSqrt(1.0)  # doctest: +ELLIPSIS
    1.0
    """

    V = to_domain_1(V)

    L_c = np.where(
        oetf_H273_LogSqrt(np.sqrt(10) / 1000) <= V,
        # L_c in [sqrt(10)/1000, 1] range
        spow(10, (V - 1) * 2.5),
        # L_c in [0, sqrt(10)/1000] range
        0,
    )

    return as_float(from_range_1(L_c))


def oetf_H273_IEC61966_2(L_c):
    """
    Define *Recommendation ITU-T H.273* opto-electronic transfer function
    (OETF) for *IEC 61966-2* family of transfer functions (*2-1 sRGB*,
    *2-1 sYCC*, *2-4 xvYCC*).

    Parameters
    ----------
    L_c
        Scene *Luminance* :math:`L_c`.

    Returns
    -------
    :class:`numpy.ndarray`
        Corresponding electrical signal :math:`V`.

    Notes
    -----
    Usage in :cite:`InternationalTelecommunicationUnion2021` is as follows:

    -   For *IEC 61966-2-1 sRGB (MatrixCoefficients=0)*, the function is only
        defined for :math:`L_c` in [0-1] range.
    -   For *IEC 61966-2-1 sYCC (MatrixCoefficients=5)* and
        *IEC 61966-2-4 xvYCC*, the function is defined for any real-valued
        :math:`L_c`.

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``L_c``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``V``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    -   :cite:`InternationalTelecommunicationUnion2021`

    Examples
    --------
    >>> oetf_H273_IEC61966_2(0.18)  # doctest: +ELLIPSIS
    0.4613561295004...
    >>> oetf_H273_IEC61966_2(-0.18)  # doctest: +ELLIPSIS
    -0.4613561295004...
    """

    V = np.where(
        as_float_array(L_c) >= 0,
        eotf_inverse_sRGB(L_c),
        -eotf_inverse_sRGB(-L_c),
    )

    return as_float(V)


def oetf_inverse_H273_IEC61966_2(V):
    """
    Define *Recommendation ITU-T H.273* inverse opto-electronic transfer
    function (OETF) for *IEC 61966-2* family of transfer functions (*2-1 sRGB*,
    *2-1 sYCC*, *2-4 xvYCC*).

    Parameters
    ----------
    V
        Electrical signal :math:`V`.

    Returns
    -------
    :class:`numpy.ndarray`
        Corresponding scene luminance :math:`L_c`.

    Notes
    -----
    Usage in :cite:`InternationalTelecommunicationUnion2021` is as follows:

    -   For *IEC 61966-2-1 sRGB (MatrixCoefficients=0)*, the function is only
        defined for :math:`L_c` in [0-1] range.
    -   For *IEC 61966-2-1 sYCC (MatrixCoefficients=5)* and
        *IEC 61966-2-4 xvYCC*, the function is defined for any real-valued
        :math:`L_c`.

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``V``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``L_c``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    -   :cite:`InternationalTelecommunicationUnion2021`

    Examples
    --------
    >>> oetf_inverse_H273_IEC61966_2(0.461356129500)  # doctest: +ELLIPSIS
    0.1799999999...
    >>> oetf_inverse_H273_IEC61966_2(-0.461356129500)  # doctest: +ELLIPSIS
    -0.1799999999...
    """

    L_c = np.where(
        as_float_array(V) >= 0,
        eotf_sRGB(V),
        -eotf_sRGB(-V),
    )

    return as_float(L_c)


def eotf_inverse_H273_ST428_1(L_o):
    """
    Define *Recommendation ITU-T H.273* inverse electro-optical transfer
    function (EOTF) for *SMPTE ST 428-1 (2019)*.

    Parameters
    ----------
    L_o
        Output display *Luminance* :math:`L_o` of the image.

    Returns
    -------
    :class:`numpy.ndarray`
        Corresponding electrical signal :math:`V`.

    Notes
    -----
    -   The function given in :cite:`InternationalTelecommunicationUnion2021`
        multiplies :math:`L_o` by 48 contrary to what is given in
        :cite:`SocietyofMotionPictureandTelevisionEngineers2019` and
        :func:`colour.models.eotf_inverse_DCDM`.

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``L_o``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``V``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    -   :cite:`InternationalTelecommunicationUnion2021`,
        :cite:`SocietyofMotionPictureandTelevisionEngineers2019`

    Examples
    --------
    >>> eotf_inverse_H273_ST428_1(0.18)  # doctest: +ELLIPSIS
    0.5000483...
    """

    L_o = to_domain_1(L_o)

    return as_float(from_range_1(eotf_inverse_DCDM(L_o * 48)))


def eotf_H273_ST428_1(V):
    """
    Define the *SMPTE ST 428-1 (2019)* electro-optical transfer function (EOTF).

    Parameters
    ----------
    V
        Electrical signal :math:`V`.

    Returns
    -------
    :class:`numpy.ndarray`
        Corresponding output display *Luminance* :math:`L_o` of the image.

    Notes
    -----
    -   The function given in :cite:`InternationalTelecommunicationUnion2021`
        divides :math:`L_o` by 48 contrary to what is given in
        :cite:`SocietyofMotionPictureandTelevisionEngineers2019` and
        :func:`colour.models.eotf_DCDM`.

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``V``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``L_o``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    -   :cite:`InternationalTelecommunicationUnion2021`,
        :cite:`SocietyofMotionPictureandTelevisionEngineers2019`

    Examples
    --------
    >>> eotf_H273_ST428_1(0.5000483377172)  # doctest: +ELLIPSIS
    0.1799999...
    """

    V = to_domain_1(V)

    return as_float(from_range_1(eotf_DCDM(V) / 48))
