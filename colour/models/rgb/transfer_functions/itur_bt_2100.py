# -*- coding: utf-8 -*-
"""
ITU-R BT.2100
=============

Defines *ITU-R BT.2100* opto-electrical transfer functions (OETF / OECF),
opto-optical transfer functions (OOTF / OOCF) and electro-optical transfer
functions (EOTF / EOCF) and their reverse:

-   :func:`colour.models.oetf_BT2100_PQ`
-   :func:`colour.models.oetf_reverse_BT2100_PQ`
-   :func:`colour.models.eotf_BT2100_PQ`
-   :func:`colour.models.eotf_reverse_BT2100_PQ`
-   :func:`colour.models.ootf_BT2100_PQ`
-   :func:`colour.models.ootf_reverse_BT2100_PQ`
-   :func:`colour.models.oetf_BT2100_HLG`
-   :func:`colour.models.oetf_reverse_BT2100_HLG`
-   :func:`colour.models.eotf_BT2100_HLG`
-   :func:`colour.models.eotf_reverse_BT2100_HLG`
-   :func:`colour.models.ootf_BT2100_HLG`
-   :func:`colour.models.ootf_reverse_BT2100_HLG`

See Also
--------
`RGB Colourspaces Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
-   :cite:`Borer2017a` : Borer, T. (2017). Private Discussion with
    Mansencal, T. and Shaw, N.
-   :cite:`InternationalTelecommunicationUnion2016a` : International
    Telecommunication Union. (2016). Recommendation ITU-R BT.2100-1 - Image
    parameter values for high dynamic range television for use in production
    and international programme exchange. Retrieved from
    https://www.itu.int/dms_pubrec/itu-r/rec/bt/\
R-REC-BT.2100-1-201706-I!!PDF-E.pdf
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.models.rgb.transfer_functions import (
    eotf_BT1886, eotf_ST2084, eotf_reverse_BT1886, oetf_ARIBSTDB67, oetf_BT709,
    oetf_ST2084, oetf_reverse_ARIBSTDB67, oetf_reverse_BT709)
from colour.models.rgb.transfer_functions.arib_std_b67 import (
    ARIBSTDB67_CONSTANTS)
from colour.utilities import (Structure, as_float_array, as_float,
                              from_range_1, to_domain_1, tsplit, tstack,
                              usage_warning)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'oetf_BT2100_PQ', 'oetf_reverse_BT2100_PQ', 'eotf_BT2100_PQ',
    'eotf_reverse_BT2100_PQ', 'ootf_BT2100_PQ', 'ootf_reverse_BT2100_PQ',
    'BT2100_HLG_WEIGHTS', 'BT2100_HLG_CONSTANTS', 'gamma_function_BT2100_HLG',
    'oetf_BT2100_HLG', 'oetf_reverse_BT2100_HLG', 'eotf_BT2100_HLG',
    'eotf_reverse_BT2100_HLG', 'ootf_BT2100_HLG', 'ootf_reverse_BT2100_HLG'
]


def oetf_BT2100_PQ(E):
    """
    Defines *Recommendation ITU-R BT.2100* *Reference PQ* opto-electrical
    transfer function (OETF / OECF).

    The OETF maps relative scene linear light into the non-linear *PQ* signal
    value.

    Parameters
    ----------
    E : numeric or array_like
        :math:`E = {R_S, G_S, B_S; Y_S; or I_S}` is the signal determined by
        scene light and scaled by camera exposure.

    Returns
    -------
    numeric or ndarray
        :math:`E'` is the resulting non-linear signal (:math:`R'`, :math:`G'`,
        :math:`B'`).

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``E``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``E_p``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Borer2017a`, :cite:`InternationalTelecommunicationUnion2016a`

    Examples
    --------
    >>> oetf_BT2100_PQ(0.1)  # doctest: +ELLIPSIS
    0.7247698...
    """

    return oetf_ST2084(ootf_BT2100_PQ(E), 10000)


def oetf_reverse_BT2100_PQ(E_p):
    """
    Defines *Recommendation ITU-R BT.2100* *Reference PQ* reverse
    opto-electrical transfer function (OETF / OECF).

    Parameters
    ----------
    E_p : numeric or array_like
        :math:`E'` is the resulting non-linear signal (:math:`R'`, :math:`G'`,
        :math:`B'`).

    Returns
    -------
    numeric or ndarray
        :math:`E = {R_S, G_S, B_S; Y_S; or I_S}` is the signal determined by
        scene light and scaled by camera exposure.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``E_p``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``E``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Borer2017a`, :cite:`InternationalTelecommunicationUnion2016a`

    Examples
    --------
    >>> oetf_reverse_BT2100_PQ(0.724769816665726)  # doctest: +ELLIPSIS
    0.0999999...
    """

    return ootf_reverse_BT2100_PQ(eotf_ST2084(E_p, 10000))


def eotf_BT2100_PQ(E_p):
    """
    Defines *Recommendation ITU-R BT.2100* *Reference PQ* electro-optical
    transfer function (EOTF / EOCF).

    The EOTF maps the non-linear *PQ* signal into display light.

    Parameters
    ----------
    E_p : numeric or array_like
        :math:`E'` denotes a non-linear colour value :math:`{R', G', B'}` or
        :math:`{L', M', S'}` in *PQ* space [0, 1].

    Returns
    -------
    numeric or ndarray
        :math:`F_D` is the luminance of a displayed linear component
        :math:`{R_D, G_D, B_D}` or :math:`Y_D` or :math:`I_D`, in
        :math:`cd/m^2`.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``E_p``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``F_D``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Borer2017a`, :cite:`InternationalTelecommunicationUnion2016a`

    Examples
    --------
    >>> eotf_BT2100_PQ(0.724769816665726)  # doctest: +ELLIPSIS
    779.9883608...
    """

    return eotf_ST2084(E_p, 10000)


def eotf_reverse_BT2100_PQ(F_D):
    """
    Defines *Recommendation ITU-R BT.2100* *Reference PQ* reverse
    electro-optical transfer function (EOTF / EOCF).

    Parameters
    ----------
    F_D : numeric or array_like
        :math:`F_D` is the luminance of a displayed linear component
        :math:`{R_D, G_D, B_D}` or :math:`Y_D` or :math:`I_D`, in
        :math:`cd/m^2`.

    Returns
    -------
    numeric or ndarray
        :math:`E'` denotes a non-linear colour value :math:`{R', G', B'}` or
        :math:`{L', M', S'}` in *PQ* space [0, 1].

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``F_D``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``E_p``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Borer2017a`, :cite:`InternationalTelecommunicationUnion2016a`

    Examples
    --------
    >>> eotf_reverse_BT2100_PQ(779.988360834085370)  # doctest: +ELLIPSIS
    0.7247698...
    """

    return oetf_ST2084(F_D, 10000)


def ootf_BT2100_PQ(E):
    """
    Defines *Recommendation ITU-R BT.2100* *Reference PQ* opto-optical transfer
    function (OOTF / OOCF).

    The OOTF maps relative scene linear light to display linear light.

    Parameters
    ----------
    E : numeric or array_like
        :math:`E = {R_S, G_S, B_S; Y_S; or I_S}` is the signal determined by
        scene light and scaled by camera exposure.

    Returns
    -------
    numeric or ndarray
        :math:`F_D` is the luminance of a displayed linear component
        (:math:`R_D`, :math:`G_D`, :math:`B_D`; :math:`Y_D`; or :math:`I_D`).

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``E``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``F_D``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Borer2017a`, :cite:`InternationalTelecommunicationUnion2016a`

    Examples
    --------
    >>> ootf_BT2100_PQ(0.1)  # doctest: +ELLIPSIS
    779.9883608...
    """

    E = as_float_array(E)

    return 100 * eotf_BT1886(oetf_BT709(59.5208 * E))


def ootf_reverse_BT2100_PQ(F_D):
    """
    Defines *Recommendation ITU-R BT.2100* *Reference PQ* reverse opto-optical
    transfer function (OOTF / OOCF).

    Parameters
    ----------
    F_D : numeric or array_like
        :math:`F_D` is the luminance of a displayed linear component
        (:math:`R_D`, :math:`G_D`, :math:`B_D`; :math:`Y_D`; or :math:`I_D`).

    Returns
    -------
    numeric or ndarray
        :math:`E = {R_S, G_S, B_S; Y_S; or I_S}` is the signal determined by
        scene light and scaled by camera exposure.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``F_D``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``E``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Borer2017a`, :cite:`InternationalTelecommunicationUnion2016a`

    Examples
    --------
    >>> ootf_reverse_BT2100_PQ(779.988360834115840)  # doctest: +ELLIPSIS
    0.1000000...
    """

    F_D = as_float_array(F_D)

    return oetf_reverse_BT709(eotf_reverse_BT1886(F_D / 100)) / 59.5208


BT2100_HLG_WEIGHTS = np.array([0.2627, 0.6780, 0.0593])
"""
Luminance weights for *Recommendation ITU-R BT.2100* *Reference HLG*.

BT2100_HLG_WEIGHTS : ndarray
"""

BT2100_HLG_CONSTANTS = Structure(
    a=ARIBSTDB67_CONSTANTS.a,
    b=1 - 4 * ARIBSTDB67_CONSTANTS.a,
    c=0.5 - ARIBSTDB67_CONSTANTS.a * np.log(4 * ARIBSTDB67_CONSTANTS.a))
"""
*Recommendation ITU-R BT.2100* *Reference HLG* constants expressed in their
analytical form in contrast to the *ARIB STD-B67 (Hybrid Log-Gamma)* numerical
reference.

Those constants are not currently in use and are provided as a reference only.

References
----------
:cite:`InternationalTelecommunicationUnion2016a`

BT2100_HLG_CONSTANTS : Structure
"""


def gamma_function_BT2100_HLG(L_W=1000):
    """
    Returns the *Reference HLG* system gamma value for given display nominal
    peak luminance.

    Parameters
    ----------
    L_W : numeric, optional
        :math:`L_W` is nominal peak luminance of the display in :math:`cd/m^2`
        for achromatic pixels.

    Returns
    -------
    numeric
        *Reference HLG* system gamma value.

    Examples
    --------
    >>> gamma_function_BT2100_HLG()
    1.2
    >>> gamma_function_BT2100_HLG(2000)  # doctest: +ELLIPSIS
    1.3264325...
    >>> gamma_function_BT2100_HLG(4000)  # doctest: +ELLIPSIS
    1.4528651...
    """

    gamma = 1.2 + 0.42 * np.log10(L_W / 1000)

    return gamma


def oetf_BT2100_HLG(E):
    """
    Defines *Recommendation ITU-R BT.2100* *Reference HLG* opto-electrical
    transfer function (OETF / OECF).

    The OETF maps relative scene linear light into the non-linear *HLG* signal
    value.

    Parameters
    ----------
    E : numeric or array_like
        :math:`E` is the signal for each colour component
        :math:`{R_S, G_S, B_S}` proportional to scene linear light and scaled
        by camera exposure.

    Returns
    -------
    numeric or ndarray
        :math:`E'` is the resulting non-linear signal :math:`{R', G', B'}`.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``E``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``E_p``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Borer2017a`, :cite:`InternationalTelecommunicationUnion2016a`

    Examples
    --------
    >>> oetf_BT2100_HLG(0.18 / 12)  # doctest: +ELLIPSIS
    0.2121320...
    """

    return oetf_ARIBSTDB67(12 * E)


def oetf_reverse_BT2100_HLG(E):
    """
    Defines *Recommendation ITU-R BT.2100* *Reference HLG* reverse
    opto-electrical transfer function (OETF / OECF).

    Parameters
    ----------
    E_p : numeric or array_like
        :math:`E'` is the resulting non-linear signal :math:`{R', G', B'}`.

    Returns
    -------
    numeric or ndarray
        :math:`E` is the signal for each colour component
        :math:`{R_S, G_S, B_S}` proportional to scene linear light and scaled
        by camera exposure.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``E_p``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``E``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Borer2017a`, :cite:`InternationalTelecommunicationUnion2016a`

    Examples
    --------
    >>> oetf_reverse_BT2100_HLG(0.212132034355964)  # doctest: +ELLIPSIS
    0.0149999...
    """

    return oetf_reverse_ARIBSTDB67(E) / 12


def eotf_BT2100_HLG(E_p, L_B=0, L_W=1000, gamma=None):
    """
    Defines *Recommendation ITU-R BT.2100* *Reference HLG* electro-optical
    transfer function (EOTF / EOCF).

    The EOTF maps the non-linear *HLG* signal into display light.

    Parameters
    ----------
    E_p : numeric or array_like
        :math:`E'` denotes a non-linear colour value :math:`{R', G', B'}` or
        :math:`{L', M', S'}` in *HLG* space.
    L_B : numeric, optional
        :math:`L_B` is the display luminance for black in :math:`cd/m^2`.
    L_W : numeric, optional
        :math:`L_W` is nominal peak luminance of the display in :math:`cd/m^2`
        for achromatic pixels.
    gamma : numeric, optional
        System gamma value, 1.2 at the nominal display peak luminance of
        :math:`1000 cd/m^2`.

    Returns
    -------
    numeric or ndarray
        Luminance :math:`F_D` of a displayed linear component
        :math:`{R_D, G_D, B_D}` or :math:`Y_D` or :math:`I_D`, in
        :math:`cd/m^2`.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``E_p``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``F_D``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Borer2017a`, :cite:`InternationalTelecommunicationUnion2016a`

    Examples
    --------
    >>> eotf_BT2100_HLG(0.212132034355964)  # doctest: +ELLIPSIS
    6.4760398...
    """

    return ootf_BT2100_HLG(oetf_reverse_ARIBSTDB67(E_p) / 12, L_B, L_W, gamma)


def eotf_reverse_BT2100_HLG(F_D, L_B=0, L_W=1000, gamma=None):
    """
    Defines *Recommendation ITU-R BT.2100* *Reference HLG* reverse
    electro-optical transfer function (EOTF / EOCF).

    Parameters
    ----------
    F_D : numeric or array_like
        Luminance :math:`F_D` of a displayed linear component
        :math:`{R_D, G_D, B_D}` or :math:`Y_D` or :math:`I_D`, in
        :math:`cd/m^2`.
    L_B : numeric, optional
        :math:`L_B` is the display luminance for black in :math:`cd/m^2`.
    L_W : numeric, optional
        :math:`L_W` is nominal peak luminance of the display in :math:`cd/m^2`
        for achromatic pixels.
    gamma : numeric, optional
        System gamma value, 1.2 at the nominal display peak luminance of
        :math:`1000 cd/m^2`.

    Returns
    -------
    numeric or ndarray
        :math:`E'` denotes a non-linear colour value :math:`{R', G', B'}` or
        :math:`{L', M', S'}` in *HLG* space.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``F_D``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``E_p``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Borer2017a`, :cite:`InternationalTelecommunicationUnion2016a`

    Examples
    --------
    >>> eotf_reverse_BT2100_HLG(6.476039825649814)  # doctest: +ELLIPSIS
    0.2121320...
    """

    return oetf_ARIBSTDB67(ootf_reverse_BT2100_HLG(F_D, L_B, L_W, gamma) * 12)


def ootf_BT2100_HLG(E, L_B=0, L_W=1000, gamma=None):
    """
    Defines *Recommendation ITU-R BT.2100* *Reference HLG* opto-optical
    transfer function (OOTF / OOCF).

    The OOTF maps relative scene linear light to display linear light.

    Parameters
    ----------
    E : numeric or array_like
        :math:`E` is the signal for each colour component
        :math:`{R_S, G_S, B_S}` proportional to scene linear light and scaled
        by camera exposure.
    L_B : numeric, optional
        :math:`L_B` is the display luminance for black in :math:`cd/m^2`.
    L_W : numeric, optional
        :math:`L_W` is nominal peak luminance of the display in :math:`cd/m^2`
        for achromatic pixels.
    gamma : numeric, optional
        System gamma value, 1.2 at the nominal display peak luminance of
        :math:`1000 cd/m^2`.

    Returns
    -------
    numeric or ndarray
        :math:`F_D` is the luminance of a displayed linear component
        :math:`{R_D, G_D, or B_D}`, in :math:`cd/m^2`.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``E``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``F_D``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Borer2017a`, :cite:`InternationalTelecommunicationUnion2016a`

    Examples
    --------
    >>> ootf_BT2100_HLG(0.1)  # doctest: +ELLIPSIS
    63.0957344...
    """

    E = np.atleast_1d(to_domain_1(E))

    if E.shape[-1] != 3:
        usage_warning(
            '"Recommendation ITU-R BT.2100" "Reference HLG OOTF" uses '
            'RGB Luminance in computations and expects a vector input, thus '
            'the given input array will be stacked to compose a vector for '
            'internal computations but a single component will be output.')
        R_S = G_S = B_S = E
    else:
        R_S, G_S, B_S = tsplit(E)

    alpha = L_W - L_B
    beta = L_B

    Y_S = np.sum(BT2100_HLG_WEIGHTS * tstack([R_S, G_S, B_S]), axis=-1)

    if gamma is None:
        gamma = gamma_function_BT2100_HLG(L_W)

    R_D = alpha * R_S * np.abs(Y_S) ** (gamma - 1) + beta
    G_D = alpha * G_S * np.abs(Y_S) ** (gamma - 1) + beta
    B_D = alpha * B_S * np.abs(Y_S) ** (gamma - 1) + beta

    if E.shape[-1] != 3:
        return as_float(from_range_1(R_D))
    else:
        RGB_D = tstack([R_D, G_D, B_D])

        return from_range_1(RGB_D)


def ootf_reverse_BT2100_HLG(F_D, L_B=0, L_W=1000, gamma=None):
    """
    Defines *Recommendation ITU-R BT.2100* *Reference HLG* reverse opto-optical
    transfer function (OOTF / OOCF).

    Parameters
    ----------
    F_D : numeric or array_like
        :math:`F_D` is the luminance of a displayed linear component
        :math:`{R_D, G_D, or B_D}`, in :math:`cd/m^2`.
    L_B : numeric, optional
        :math:`L_B` is the display luminance for black in :math:`cd/m^2`.
    L_W : numeric, optional
        :math:`L_W` is nominal peak luminance of the display in :math:`cd/m^2`
        for achromatic pixels.
    gamma : numeric, optional
        System gamma value, 1.2 at the nominal display peak luminance of
        :math:`1000 cd/m^2`.

    Returns
    -------
    numeric or ndarray
        :math:`E` is the signal for each colour component
        :math:`{R_S, G_S, B_S}` proportional to scene linear light and scaled
        by camera exposure.

    Notes
    -----

    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``F_D``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``E``      | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Borer2017a`, :cite:`InternationalTelecommunicationUnion2016a`

    Examples
    --------
    >>> ootf_reverse_BT2100_HLG(63.095734448019336)  # doctest: +ELLIPSIS
    0.1000000...
    """

    F_D = np.atleast_1d(to_domain_1(F_D))

    if F_D.shape[-1] != 3:
        usage_warning(
            '"Recommendation ITU-R BT.2100" "Reference HLG OOTF" uses '
            'RGB Luminance in computations and expects a vector input, thus '
            'the given input array will be stacked to compose a vector for '
            'internal computations but a single component will be output.')
        R_D = G_D = B_D = F_D
    else:
        R_D, G_D, B_D = tsplit(F_D)

    Y_D = np.sum(BT2100_HLG_WEIGHTS * tstack([R_D, G_D, B_D]), axis=-1)

    alpha = L_W - L_B
    beta = L_B

    if gamma is None:
        gamma = gamma_function_BT2100_HLG(L_W)

    R_S = np.where(
        Y_D == beta,
        0.0,
        (np.abs((Y_D - beta) / alpha) **
         ((1 - gamma) / gamma)) * (R_D - beta) / alpha,
    )
    G_S = np.where(
        Y_D == beta,
        0.0,
        (np.abs((Y_D - beta) / alpha) **
         ((1 - gamma) / gamma)) * (G_D - beta) / alpha,
    )
    B_S = np.where(
        Y_D == beta,
        0.0,
        (np.abs((Y_D - beta) / alpha) **
         ((1 - gamma) / gamma)) * (B_D - beta) / alpha,
    )

    if F_D.shape[-1] != 3:
        return as_float(from_range_1(R_S))
    else:
        RGB_S = tstack([R_S, G_S, B_S])

        return from_range_1(RGB_S)
