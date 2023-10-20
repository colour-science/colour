"""
Recommendation ITU-R BT.2100
============================

Defines the *Recommendation ITU-R BT.2100* opto-electrical transfer functions
(OETF), opto-optical transfer functions (OOTF / OOCF) and electro-optical
transfer functions (EOTF) and their inverse:

-   :func:`colour.models.oetf_BT2100_PQ`
-   :func:`colour.models.oetf_inverse_BT2100_PQ`
-   :func:`colour.models.eotf_BT2100_PQ`
-   :func:`colour.models.eotf_inverse_BT2100_PQ`
-   :func:`colour.models.ootf_BT2100_PQ`
-   :func:`colour.models.ootf_inverse_BT2100_PQ`
-   :func:`colour.models.oetf_BT2100_HLG`
-   :func:`colour.models.oetf_inverse_BT2100_HLG`
-   :func:`colour.models.eotf_BT2100_HLG_1`
-   :func:`colour.models.eotf_BT2100_HLG_2`
-   :attr:`colour.models.BT2100_HLG_EOTF_METHODS`
-   :func:`colour.models.eotf_BT2100_HLG`
-   :func:`colour.models.eotf_inverse_BT2100_HLG_1`
-   :func:`colour.models.eotf_inverse_BT2100_HLG_2`
-   :attr:`colour.models.BT2100_HLG_EOTF_INVERSE_METHODS`
-   :func:`colour.models.eotf_inverse_BT2100_HLG`
-   :func:`colour.models.ootf_BT2100_HLG`
-   :func:`colour.models.ootf_inverse_BT2100_HLG`
-   :func:`colour.models.ootf_BT2100_HLG_1`
-   :func:`colour.models.ootf_BT2100_HLG_2`
-   :attr:`colour.models.BT2100_HLG_OOTF_METHODS`
-   :func:`colour.models.ootf_BT2100_HLG`
-   :func:`colour.models.ootf_inverse_BT2100_HLG_1`
-   :func:`colour.models.ootf_inverse_BT2100_HLG_2`
-   :attr:`colour.models.BT2100_HLG_OOTF_INVERSE_METHODS`
-   :func:`colour.models.ootf_inverse_BT2100_HLG`

References
----------
-   :cite:`Borer2017a` : Borer, T. (2017). Private Discussion with Mansencal,
    T. and Shaw, N.
-   :cite:`InternationalTelecommunicationUnion2017` : International
    Telecommunication Union. (2017). Recommendation ITU-R BT.2100-1 - Image
    parameter values for high dynamic range television for use in production
    and international programme exchange.
    https://www.itu.int/dms_pubrec/itu-r/rec/bt/\
R-REC-BT.2100-1-201706-I!!PDF-E.pdf
-   :cite:`InternationalTelecommunicationUnion2018` : International
    Telecommunication Union. (2018). Recommendation ITU-R BT.2100-2 - Image
    parameter values for high dynamic range television for use in production
    and international programme exchange.
    https://www.itu.int/dms_pubrec/itu-r/rec/bt/\
R-REC-BT.2100-2-201807-I!!PDF-E.pdf
"""

from __future__ import annotations

import numpy as np

from colour.algebra import spow
from colour.hints import ArrayLike, Literal, NDArrayFloat
from colour.models.rgb.transfer_functions import (
    eotf_BT1886,
    eotf_inverse_BT1886,
    eotf_inverse_ST2084,
    eotf_ST2084,
    oetf_ARIBSTDB67,
    oetf_BT709,
    oetf_inverse_ARIBSTDB67,
    oetf_inverse_BT709,
)
from colour.models.rgb.transfer_functions.arib_std_b67 import (
    CONSTANTS_ARIBSTDB67,
)
from colour.utilities import (
    CanonicalMapping,
    Structure,
    as_float,
    as_float_array,
    as_float_scalar,
    domain_range_scale,
    filter_kwargs,
    from_range_1,
    optional,
    to_domain_1,
    tsplit,
    tstack,
    usage_warning,
    validate_method,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "oetf_BT2100_PQ",
    "oetf_inverse_BT2100_PQ",
    "eotf_BT2100_PQ",
    "eotf_inverse_BT2100_PQ",
    "ootf_BT2100_PQ",
    "ootf_inverse_BT2100_PQ",
    "WEIGHTS_BT2100_HLG",
    "CONSTANTS_BT2100_HLG",
    "gamma_function_BT2100_HLG",
    "oetf_BT2100_HLG",
    "oetf_inverse_BT2100_HLG",
    "black_level_lift_BT2100_HLG",
    "eotf_BT2100_HLG_1",
    "eotf_BT2100_HLG_2",
    "BT2100_HLG_EOTF_METHODS",
    "eotf_BT2100_HLG",
    "eotf_inverse_BT2100_HLG_1",
    "eotf_inverse_BT2100_HLG_2",
    "BT2100_HLG_EOTF_INVERSE_METHODS",
    "eotf_inverse_BT2100_HLG",
    "ootf_BT2100_HLG_1",
    "ootf_BT2100_HLG_2",
    "BT2100_HLG_OOTF_METHODS",
    "ootf_BT2100_HLG",
    "ootf_inverse_BT2100_HLG_1",
    "ootf_inverse_BT2100_HLG_2",
    "BT2100_HLG_OOTF_INVERSE_METHODS",
    "ootf_inverse_BT2100_HLG",
]


def oetf_BT2100_PQ(E: ArrayLike) -> NDArrayFloat:
    """
    Define *Recommendation ITU-R BT.2100* *Reference PQ* opto-electrical
    transfer function (OETF).

    The OETF maps relative scene linear light into the non-linear *PQ* signal
    value.

    Parameters
    ----------
    E
        :math:`E = {R_S, G_S, B_S; Y_S; or I_S}` is the signal determined by
        scene light and scaled by camera exposure.

    Returns
    -------
    :class:`numpy.ndarray`
        :math:`E'` is the resulting non-linear signal (:math:`R'`, :math:`G'`,
        :math:`B'`).

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``E``      | ``UN``                | ``UN``        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``E_p``    | ``UN``                | ``UN``        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Borer2017a`, :cite:`InternationalTelecommunicationUnion2017`

    Examples
    --------
    >>> oetf_BT2100_PQ(0.1)  # doctest: +ELLIPSIS
    0.7247698...
    """

    return eotf_inverse_ST2084(ootf_BT2100_PQ(E), 10000)


def oetf_inverse_BT2100_PQ(E_p: ArrayLike) -> NDArrayFloat:
    """
    Define *Recommendation ITU-R BT.2100* *Reference PQ* inverse
    opto-electrical transfer function (OETF).

    Parameters
    ----------
    E_p
        :math:`E'` is the resulting non-linear signal (:math:`R'`, :math:`G'`,
        :math:`B'`).

    Returns
    -------
    :class:`numpy.ndarray`
        :math:`E = {R_S, G_S, B_S; Y_S; or I_S}` is the signal determined by
        scene light and scaled by camera exposure.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``E_p``    | ``UN``                | ``UN``        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``E``      | ``UN``                | ``UN``        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Borer2017a`, :cite:`InternationalTelecommunicationUnion2017`

    Examples
    --------
    >>> oetf_inverse_BT2100_PQ(0.724769816665726)  # doctest: +ELLIPSIS
    0.0999999...
    """

    return ootf_inverse_BT2100_PQ(eotf_ST2084(E_p, 10000))


def eotf_BT2100_PQ(E_p: ArrayLike) -> NDArrayFloat:
    """
    Define *Recommendation ITU-R BT.2100* *Reference PQ* electro-optical
    transfer function (EOTF).

    The EOTF maps the non-linear *PQ* signal into display light.

    Parameters
    ----------
    E_p
        :math:`E'` denotes a non-linear colour value :math:`{R', G', B'}` or
        :math:`{L', M', S'}` in *PQ* space [0, 1].

    Returns
    -------
    :class:`numpy.ndarray`
        :math:`F_D` is the luminance of a displayed linear component
        :math:`{R_D, G_D, B_D}` or :math:`Y_D` or :math:`I_D`, in
        :math:`cd/m^2`.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``E_p``    | ``UN``                | ``UN``        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``F_D``    | ``UN``                | ``UN``        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Borer2017a`, :cite:`InternationalTelecommunicationUnion2017`

    Examples
    --------
    >>> eotf_BT2100_PQ(0.724769816665726)  # doctest: +ELLIPSIS
    779.9883608...
    """

    return eotf_ST2084(E_p, 10000)


def eotf_inverse_BT2100_PQ(F_D: ArrayLike) -> NDArrayFloat:
    """
    Define *Recommendation ITU-R BT.2100* *Reference PQ* inverse
    electro-optical transfer function (EOTF).

    Parameters
    ----------
    F_D
        :math:`F_D` is the luminance of a displayed linear component
        :math:`{R_D, G_D, B_D}` or :math:`Y_D` or :math:`I_D`, in
        :math:`cd/m^2`.

    Returns
    -------
    :class:`numpy.ndarray`
        :math:`E'` denotes a non-linear colour value :math:`{R', G', B'}` or
        :math:`{L', M', S'}` in *PQ* space [0, 1].

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``F_D``    | ``UN``                | ``UN``        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``E_p``    | ``UN``                | ``UN``        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Borer2017a`, :cite:`InternationalTelecommunicationUnion2017`

    Examples
    --------
    >>> eotf_inverse_BT2100_PQ(779.988360834085370)  # doctest: +ELLIPSIS
    0.7247698...
    """

    return eotf_inverse_ST2084(F_D, 10000)


def ootf_BT2100_PQ(E: ArrayLike) -> NDArrayFloat:
    """
    Define *Recommendation ITU-R BT.2100* *Reference PQ* opto-optical transfer
    function (OOTF / OOCF).

    The OOTF maps relative scene linear light to display linear light.

    Parameters
    ----------
    E
        :math:`E = {R_S, G_S, B_S; Y_S; or I_S}` is the signal determined by
        scene light and scaled by camera exposure.

    Returns
    -------
    :class:`numpy.ndarray`
        :math:`F_D` is the luminance of a displayed linear component
        (:math:`R_D`, :math:`G_D`, :math:`B_D`; :math:`Y_D`; or :math:`I_D`).

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``E``      | ``UN``                | ``UN``        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``F_D``    | ``UN``                | ``UN``        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Borer2017a`, :cite:`InternationalTelecommunicationUnion2017`

    Examples
    --------
    >>> ootf_BT2100_PQ(0.1)  # doctest: +ELLIPSIS
    779.9883608...
    """

    E = as_float_array(E)

    with domain_range_scale("ignore"):
        return 100 * eotf_BT1886(oetf_BT709(59.5208 * E))


def ootf_inverse_BT2100_PQ(F_D: ArrayLike) -> NDArrayFloat:
    """
    Define *Recommendation ITU-R BT.2100* *Reference PQ* inverse opto-optical
    transfer function (OOTF / OOCF).

    Parameters
    ----------
    F_D
        :math:`F_D` is the luminance of a displayed linear component
        (:math:`R_D`, :math:`G_D`, :math:`B_D`; :math:`Y_D`; or :math:`I_D`).

    Returns
    -------
    :class:`numpy.ndarray`
        :math:`E = {R_S, G_S, B_S; Y_S; or I_S}` is the signal determined by
        scene light and scaled by camera exposure.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``F_D``    | ``UN``                | ``UN``        |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``E``      | ``UN``                | ``UN``        |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Borer2017a`, :cite:`InternationalTelecommunicationUnion2017`

    Examples
    --------
    >>> ootf_inverse_BT2100_PQ(779.988360834115840)  # doctest: +ELLIPSIS
    0.1000000...
    """

    F_D = as_float_array(F_D)

    with domain_range_scale("ignore"):
        return oetf_inverse_BT709(eotf_inverse_BT1886(F_D / 100)) / 59.5208


WEIGHTS_BT2100_HLG: NDArrayFloat = np.array([0.2627, 0.6780, 0.0593])
"""Luminance weights for *Recommendation ITU-R BT.2100* *Reference HLG*."""

CONSTANTS_BT2100_HLG: Structure = Structure(
    a=CONSTANTS_ARIBSTDB67.a,
    b=1 - 4 * CONSTANTS_ARIBSTDB67.a,
    c=0.5 - CONSTANTS_ARIBSTDB67.a * np.log(4 * CONSTANTS_ARIBSTDB67.a),
)
"""
*Recommendation ITU-R BT.2100* *Reference HLG* constants expressed in their
analytical form in contrast to the *ARIB STD-B67 (Hybrid Log-Gamma)* numerical
reference.

References
----------
:cite:`InternationalTelecommunicationUnion2017`
"""


def gamma_function_BT2100_HLG(L_W: float = 1000) -> float:
    """
    Return the *Reference HLG* system gamma value for given display nominal
    peak luminance.

    Parameters
    ----------
    L_W
        :math:`L_W` is nominal peak luminance of the display in :math:`cd/m^2`
        for achromatic pixels.

    Returns
    -------
    :class:`float`
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

    return as_float_scalar(gamma)


def oetf_BT2100_HLG(
    E: ArrayLike, constants: Structure = CONSTANTS_BT2100_HLG
) -> NDArrayFloat:
    """
    Define *Recommendation ITU-R BT.2100* *Reference HLG* opto-electrical
    transfer function (OETF).

    The OETF maps relative scene linear light into the non-linear *HLG* signal
    value.

    Parameters
    ----------
    E
        :math:`E` is the signal for each colour component
        :math:`{R_S, G_S, B_S}` proportional to scene linear light and scaled
        by camera exposure.
    constants
        *Recommendation ITU-R BT.2100* *Reference HLG* constants.

    Returns
    -------
    :class:`numpy.ndarray`
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
    :cite:`Borer2017a`, :cite:`InternationalTelecommunicationUnion2017`

    Examples
    --------
    >>> oetf_BT2100_HLG(0.18 / 12)  # doctest: +ELLIPSIS
    0.2121320...
    """

    E = as_float_array(E)

    return oetf_ARIBSTDB67(12 * E, constants=constants)


def oetf_inverse_BT2100_HLG(
    E_p: ArrayLike, constants: Structure = CONSTANTS_BT2100_HLG
) -> NDArrayFloat:
    """
    Define *Recommendation ITU-R BT.2100* *Reference HLG* inverse
    opto-electrical transfer function (OETF).

    Parameters
    ----------
    E_p
        :math:`E'` is the resulting non-linear signal :math:`{R', G', B'}`.
    constants
        *Recommendation ITU-R BT.2100* *Reference HLG* constants.

    Returns
    -------
    :class:`numpy.ndarray`
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
    :cite:`Borer2017a`, :cite:`InternationalTelecommunicationUnion2017`

    Examples
    --------
    >>> oetf_inverse_BT2100_HLG(0.212132034355964)  # doctest: +ELLIPSIS
    0.0149999...
    """

    return oetf_inverse_ARIBSTDB67(E_p, constants=constants) / 12


def black_level_lift_BT2100_HLG(
    L_B: float = 0, L_W: float = 1000, gamma: float | None = None
) -> float:
    """
    Return the *Reference HLG* black level lift :math:`\\beta` for given
    display luminance for black, nominal peak luminance and system gamma value.

    Parameters
    ----------
    L_B
        :math:`L_B` is the display luminance for black in :math:`cd/m^2`.
    L_W
        :math:`L_W` is nominal peak luminance of the display in :math:`cd/m^2`
        for achromatic pixels.
    gamma
        System gamma value, 1.2 at the nominal display peak luminance of
        :math:`1000 cd/m^2`.

    Returns
    -------
    :class:`float`
        *Reference HLG* black level lift :math:`\\beta`.

    Examples
    --------
    >>> black_level_lift_BT2100_HLG()
    0.0
    >>> black_level_lift_BT2100_HLG(0.01)  # doctest: +ELLIPSIS
    0.0142964...
    >>> black_level_lift_BT2100_HLG(0.001, 2000)  # doctest: +ELLIPSIS
    0.0073009...
    >>> black_level_lift_BT2100_HLG(0.01, gamma=1.4)  # doctest: +ELLIPSIS
    0.0283691...
    """

    gamma = optional(gamma, gamma_function_BT2100_HLG(L_W))

    beta = np.sqrt(3 * spow((L_B / L_W), 1 / gamma))

    return as_float_scalar(beta)


def eotf_BT2100_HLG_1(
    E_p: ArrayLike,
    L_B: float = 0,
    L_W: float = 1000,
    gamma: float | None = None,
    constants: Structure = CONSTANTS_BT2100_HLG,
) -> NDArrayFloat:
    """
    Define *Recommendation ITU-R BT.2100* *Reference HLG* electro-optical
    transfer function (EOTF) as given in *ITU-R BT.2100-1*.

    The EOTF maps the non-linear *HLG* signal into display light.

    Parameters
    ----------
    E_p
        :math:`E'` is the non-linear signal :math:`{R', G', B'}` as defined for
        the OETF.
    L_B
        :math:`L_B` is the display luminance for black in :math:`cd/m^2`.
    L_W
        :math:`L_W` is nominal peak luminance of the display in :math:`cd/m^2`
        for achromatic pixels.
    gamma
        System gamma value, 1.2 at the nominal display peak luminance of
        :math:`1000 cd/m^2`.
    constants
        *Recommendation ITU-R BT.2100* *Reference HLG* constants.

    Returns
    -------
    :class:`numpy.ndarray`
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
    :cite:`Borer2017a`, :cite:`InternationalTelecommunicationUnion2017`

    Examples
    --------
    >>> eotf_BT2100_HLG_1(0.212132034355964)  # doctest: +ELLIPSIS
    6.4760398...
    >>> eotf_BT2100_HLG_1(0.212132034355964, 0.01)  # doctest: +ELLIPSIS
    6.4859750...
    """

    return ootf_BT2100_HLG_1(
        oetf_inverse_ARIBSTDB67(E_p, constants=constants) / 12, L_B, L_W, gamma
    )


def eotf_BT2100_HLG_2(
    E_p: ArrayLike,
    L_B: float = 0,
    L_W: float = 1000,
    gamma: float | None = None,
    constants: Structure = CONSTANTS_BT2100_HLG,
) -> NDArrayFloat:
    """
    Define *Recommendation ITU-R BT.2100* *Reference HLG* electro-optical
    transfer function (EOTF) as given in *ITU-R BT.2100-2* with the
    modified black level behaviour.

    The EOTF maps the non-linear *HLG* signal into display light.

    Parameters
    ----------
    E_p
        :math:`E'` is the non-linear signal :math:`{R', G', B'}` as defined for
        the *HLG Reference* OETF.
    L_B
        :math:`L_B` is the display luminance for black in :math:`cd/m^2`.
    L_W
        :math:`L_W` is nominal peak luminance of the display in :math:`cd/m^2`
        for achromatic pixels.
    gamma
        System gamma value, 1.2 at the nominal display peak luminance of
        :math:`1000 cd/m^2`.
    constants
        *Recommendation ITU-R BT.2100* *Reference HLG* constants.

    Returns
    -------
    :class:`numpy.ndarray`
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
    :cite:`Borer2017a`, :cite:`InternationalTelecommunicationUnion2018`

    Examples
    --------
    >>> eotf_BT2100_HLG_2(0.212132034355964)  # doctest: +ELLIPSIS
    6.4760398...
    >>> eotf_BT2100_HLG_2(0.212132034355964, 0.01)  # doctest: +ELLIPSIS
    7.3321975...
    """

    E_p = as_float_array(E_p)

    beta = black_level_lift_BT2100_HLG(L_B, L_W, gamma)

    return ootf_BT2100_HLG_2(
        oetf_inverse_ARIBSTDB67((1 - beta) * E_p + beta, constants=constants)
        / 12,
        L_W,
        gamma,
    )


BT2100_HLG_EOTF_METHODS: CanonicalMapping = CanonicalMapping(
    {
        "ITU-R BT.2100-1": eotf_BT2100_HLG_1,
        "ITU-R BT.2100-2": eotf_BT2100_HLG_2,
    }
)
BT2100_HLG_EOTF_METHODS.__doc__ = """
Supported *Recommendation ITU-R BT.2100* *Reference HLG* electro-optical
transfer function (EOTF).

References
----------
:cite:`Borer2017a`, :cite:`InternationalTelecommunicationUnion2017`,
:cite:`InternationalTelecommunicationUnion2018`
"""


def eotf_BT2100_HLG(
    E_p: ArrayLike,
    L_B: float = 0,
    L_W: float = 1000,
    gamma: float | None = None,
    constants: Structure = CONSTANTS_BT2100_HLG,
    method: Literal["ITU-R BT.2100-1", "ITU-R BT.2100-2"]
    | str = "ITU-R BT.2100-2",
) -> NDArrayFloat:
    """
    Define *Recommendation ITU-R BT.2100* *Reference HLG* electro-optical
    transfer function (EOTF).

    The EOTF maps the non-linear *HLG* signal into display light.

    Parameters
    ----------
    E_p
        :math:`E'` denotes a non-linear colour value :math:`{R', G', B'}` or
        :math:`{L', M', S'}` in *HLG* space.
    L_B
        :math:`L_B` is the display luminance for black in :math:`cd/m^2`.
    L_W
        :math:`L_W` is nominal peak luminance of the display in :math:`cd/m^2`
        for achromatic pixels.
    gamma
        System gamma value, 1.2 at the nominal display peak luminance of
        :math:`1000 cd/m^2`.
    constants
        *Recommendation ITU-R BT.2100* *Reference HLG* constants.
    method
        Computation method.

    Returns
    -------
    :class:`numpy.ndarray`
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
    :cite:`Borer2017a`, :cite:`InternationalTelecommunicationUnion2017`,
    :cite:`InternationalTelecommunicationUnion2018`

    Examples
    --------
    >>> eotf_BT2100_HLG(0.212132034355964)  # doctest: +ELLIPSIS
    6.4760398...
    >>> eotf_BT2100_HLG(0.212132034355964, method="ITU-R BT.2100-1")
    ... # doctest: +ELLIPSIS
    6.4760398...
    >>> eotf_BT2100_HLG(0.212132034355964, 0.01)
    ... # doctest: +ELLIPSIS
    7.3321975...
    """

    method = validate_method(method, tuple(BT2100_HLG_EOTF_METHODS))

    return BT2100_HLG_EOTF_METHODS[method](E_p, L_B, L_W, gamma, constants)


def eotf_inverse_BT2100_HLG_1(
    F_D: ArrayLike,
    L_B: float = 0,
    L_W: float = 1000,
    gamma: float | None = None,
    constants: Structure = CONSTANTS_BT2100_HLG,
) -> NDArrayFloat:
    """
    Define *Recommendation ITU-R BT.2100* *Reference HLG* inverse
    electro-optical transfer function (EOTF) as given in
    *ITU-R BT.2100-1*.

    Parameters
    ----------
    F_D
        Luminance :math:`F_D` of a displayed linear component
        :math:`{R_D, G_D, B_D}` or :math:`Y_D` or :math:`I_D`, in
        :math:`cd/m^2`.
    L_B
        :math:`L_B` is the display luminance for black in :math:`cd/m^2`.
    L_W
        :math:`L_W` is nominal peak luminance of the display in :math:`cd/m^2`
        for achromatic pixels.
    gamma
        System gamma value, 1.2 at the nominal display peak luminance of
        :math:`1000 cd/m^2`.
    constants
        *Recommendation ITU-R BT.2100* *Reference HLG* constants.

    Returns
    -------
    :class:`numpy.ndarray`
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
    :cite:`Borer2017a`, :cite:`InternationalTelecommunicationUnion2017`

    Examples
    --------
    >>> eotf_inverse_BT2100_HLG_1(6.476039825649814)  # doctest: +ELLIPSIS
    0.2121320...
    >>> eotf_inverse_BT2100_HLG_1(6.485975065251558, 0.01)
    ... # doctest: +ELLIPSIS
    0.2121320...
    """

    return oetf_ARIBSTDB67(
        ootf_inverse_BT2100_HLG_1(F_D, L_B, L_W, gamma) * 12,
        constants=constants,
    )


def eotf_inverse_BT2100_HLG_2(
    F_D: ArrayLike,
    L_B: float = 0,
    L_W: float = 1000,
    gamma: float | None = None,
    constants: Structure = CONSTANTS_BT2100_HLG,
) -> NDArrayFloat:
    """
    Define *Recommendation ITU-R BT.2100* *Reference HLG* inverse
    electro-optical transfer function (EOTF) as given in
    *ITU-R BT.2100-2* with the modified black level behaviour.

    Parameters
    ----------
    F_D
        Luminance :math:`F_D` of a displayed linear component
        :math:`{R_D, G_D, B_D}` or :math:`Y_D` or :math:`I_D`, in
        :math:`cd/m^2`.
    L_B
        :math:`L_B` is the display luminance for black in :math:`cd/m^2`.
    L_W
        :math:`L_W` is nominal peak luminance of the display in :math:`cd/m^2`
        for achromatic pixels.
    gamma
        System gamma value, 1.2 at the nominal display peak luminance of
        :math:`1000 cd/m^2`.
    constants
        *Recommendation ITU-R BT.2100* *Reference HLG* constants.

    Returns
    -------
    :class:`numpy.ndarray`
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
    :cite:`Borer2017a`, :cite:`InternationalTelecommunicationUnion2018`

    Examples
    --------
    >>> eotf_inverse_BT2100_HLG_2(6.476039825649814)  # doctest: +ELLIPSIS
    0.2121320...
    >>> eotf_inverse_BT2100_HLG_2(7.332197528353875, 0.01)
    ... # doctest: +ELLIPSIS
    0.2121320...
    """

    beta = black_level_lift_BT2100_HLG(L_B, L_W, gamma)

    return (
        oetf_ARIBSTDB67(
            ootf_inverse_BT2100_HLG_2(F_D, L_W, gamma) * 12,
            constants=constants,
        )
        - beta
    ) / (1 - beta)


BT2100_HLG_EOTF_INVERSE_METHODS: CanonicalMapping = CanonicalMapping(
    {
        "ITU-R BT.2100-1": eotf_inverse_BT2100_HLG_1,
        "ITU-R BT.2100-2": eotf_inverse_BT2100_HLG_2,
    }
)
BT2100_HLG_EOTF_INVERSE_METHODS.__doc__ = """
Supported *Recommendation ITU-R BT.2100* *Reference HLG* inverse
electro-optical transfer function (EOTF).

References
----------
:cite:`Borer2017a`, :cite:`InternationalTelecommunicationUnion2017`,
:cite:`InternationalTelecommunicationUnion2018`
"""


def eotf_inverse_BT2100_HLG(
    F_D: ArrayLike,
    L_B: float = 0,
    L_W: float = 1000,
    gamma: float | None = None,
    constants: Structure = CONSTANTS_BT2100_HLG,
    method: Literal["ITU-R BT.2100-1", "ITU-R BT.2100-2"]
    | str = "ITU-R BT.2100-2",
) -> NDArrayFloat:
    """
    Define *Recommendation ITU-R BT.2100* *Reference HLG* inverse
    electro-optical transfer function (EOTF).

    Parameters
    ----------
    F_D
        Luminance :math:`F_D` of a displayed linear component
        :math:`{R_D, G_D, B_D}` or :math:`Y_D` or :math:`I_D`, in
        :math:`cd/m^2`.
    L_B
        :math:`L_B` is the display luminance for black in :math:`cd/m^2`.
    L_W
        :math:`L_W` is nominal peak luminance of the display in :math:`cd/m^2`
        for achromatic pixels.
    gamma
        System gamma value, 1.2 at the nominal display peak luminance of
        :math:`1000 cd/m^2`.
    constants
        *Recommendation ITU-R BT.2100* *Reference HLG* constants.
    method
        Computation method.

    Returns
    -------
    :class:`numpy.ndarray`
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
    :cite:`Borer2017a`, :cite:`InternationalTelecommunicationUnion2017`,
    :cite:`InternationalTelecommunicationUnion2018`

    Examples
    --------
    >>> eotf_inverse_BT2100_HLG(6.476039825649814)  # doctest: +ELLIPSIS
    0.2121320...
    >>> eotf_inverse_BT2100_HLG(6.476039825649814, method="ITU-R BT.2100-1")
    ... # doctest: +ELLIPSIS
    0.2121320...
    >>> eotf_inverse_BT2100_HLG(7.332197528353875, 0.01)  # doctest: +ELLIPSIS
    0.2121320...
    """

    method = validate_method(method, tuple(BT2100_HLG_EOTF_INVERSE_METHODS))

    return BT2100_HLG_EOTF_INVERSE_METHODS[method](
        F_D, L_B, L_W, gamma, constants
    )


def ootf_BT2100_HLG_1(
    E: ArrayLike,
    L_B: float = 0,
    L_W: float = 1000,
    gamma: float | None = None,
) -> NDArrayFloat:
    """
    Define *Recommendation ITU-R BT.2100* *Reference HLG* opto-optical
    transfer function (OOTF / OOCF) as given in *ITU-R BT.2100-1*.

    The OOTF maps relative scene linear light to display linear light.

    Parameters
    ----------
    E
        :math:`E` is the signal for each colour component
        :math:`{R_S, G_S, B_S}` proportional to scene linear light and scaled
        by camera exposure.
    L_B
        :math:`L_B` is the display luminance for black in :math:`cd/m^2`.
    L_W
        :math:`L_W` is nominal peak luminance of the display in :math:`cd/m^2`
        for achromatic pixels.
    gamma
        System gamma value, 1.2 at the nominal display peak luminance of
        :math:`1000 cd/m^2`.

    Returns
    -------
    :class:`numpy.ndarray`
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
    :cite:`Borer2017a`, :cite:`InternationalTelecommunicationUnion2017`

    Examples
    --------
    >>> ootf_BT2100_HLG_1(0.1)  # doctest: +ELLIPSIS
    63.0957344...
    >>> ootf_BT2100_HLG_1(0.1, 0.01)
    ... # doctest: +ELLIPSIS
    63.1051034...
    """

    E = to_domain_1(E)

    is_single_channel = np.atleast_1d(E).shape[-1] != 3

    if is_single_channel:
        usage_warning(
            '"Recommendation ITU-R BT.2100" "Reference HLG OOTF" uses '
            "RGB Luminance in computations and expects a vector input, thus "
            "the given input array will be stacked to compose a vector for "
            "internal computations but a single component will be output."
        )
        R_S = G_S = B_S = E
    else:
        R_S, G_S, B_S = tsplit(E)

    alpha = L_W - L_B
    beta = L_B

    Y_S = np.sum(WEIGHTS_BT2100_HLG * tstack([R_S, G_S, B_S]), axis=-1)

    gamma = optional(gamma, gamma_function_BT2100_HLG(L_W))

    R_D = alpha * R_S * np.abs(Y_S) ** (gamma - 1) + beta
    G_D = alpha * G_S * np.abs(Y_S) ** (gamma - 1) + beta
    B_D = alpha * B_S * np.abs(Y_S) ** (gamma - 1) + beta

    if is_single_channel:
        return as_float(from_range_1(R_D))
    else:
        RGB_D = tstack([R_D, G_D, B_D])

        return from_range_1(RGB_D)


def ootf_BT2100_HLG_2(
    E: ArrayLike,
    L_W: float = 1000,
    gamma: float | None = None,
) -> NDArrayFloat:
    """
    Define *Recommendation ITU-R BT.2100* *Reference HLG* opto-optical
    transfer function (OOTF / OOCF) as given in *ITU-R BT.2100-2*.

    The OOTF maps relative scene linear light to display linear light.

    Parameters
    ----------
    E
        :math:`E` is the signal for each colour component
        :math:`{R_S, G_S, B_S}` proportional to scene linear light and scaled
        by camera exposure.
    L_W
        :math:`L_W` is nominal peak luminance of the display in :math:`cd/m^2`
        for achromatic pixels.
    gamma
        System gamma value, 1.2 at the nominal display peak luminance of
        :math:`1000 cd/m^2`.

    Returns
    -------
    :class:`numpy.ndarray`
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
    :cite:`InternationalTelecommunicationUnion2018`

    Examples
    --------
    >>> ootf_BT2100_HLG_2(0.1)  # doctest: +ELLIPSIS
    63.0957344...
    """

    E = to_domain_1(E)

    is_single_channel = np.atleast_1d(E).shape[-1] != 3

    if is_single_channel:
        usage_warning(
            '"Recommendation ITU-R BT.2100" "Reference HLG OOTF" uses '
            "RGB Luminance in computations and expects a vector input, thus "
            "the given input array will be stacked to compose a vector for "
            "internal computations but a single component will be output."
        )
        R_S = G_S = B_S = E
    else:
        R_S, G_S, B_S = tsplit(E)

    alpha = L_W

    Y_S = np.sum(WEIGHTS_BT2100_HLG * tstack([R_S, G_S, B_S]), axis=-1)

    gamma = optional(gamma, gamma_function_BT2100_HLG(L_W))

    R_D = alpha * R_S * np.abs(Y_S) ** (gamma - 1)
    G_D = alpha * G_S * np.abs(Y_S) ** (gamma - 1)
    B_D = alpha * B_S * np.abs(Y_S) ** (gamma - 1)

    if is_single_channel:
        return as_float(from_range_1(R_D))
    else:
        RGB_D = tstack([R_D, G_D, B_D])

        return from_range_1(RGB_D)


BT2100_HLG_OOTF_METHODS: CanonicalMapping = CanonicalMapping(
    {
        "ITU-R BT.2100-1": ootf_BT2100_HLG_1,
        "ITU-R BT.2100-2": ootf_BT2100_HLG_2,
    }
)
BT2100_HLG_OOTF_METHODS.__doc__ = """
Supported *Recommendation ITU-R BT.2100* *Reference HLG* opto-optical transfer
function (OOTF / OOCF).

References
----------
:cite:`Borer2017a`, :cite:`InternationalTelecommunicationUnion2017`,
:cite:`InternationalTelecommunicationUnion2018`
"""


def ootf_BT2100_HLG(
    E: ArrayLike,
    L_B: float = 0,
    L_W: float = 1000,
    gamma: float | None = None,
    method: Literal["ITU-R BT.2100-1", "ITU-R BT.2100-2"]
    | str = "ITU-R BT.2100-2",
) -> NDArrayFloat:
    """
    Define *Recommendation ITU-R BT.2100* *Reference HLG* opto-optical
    transfer function (OOTF / OOCF).

    The OOTF maps relative scene linear light to display linear light.

    Parameters
    ----------
    E
        :math:`E` is the signal for each colour component
        :math:`{R_S, G_S, B_S}` proportional to scene linear light and scaled
        by camera exposure.
    L_B
        :math:`L_B` is the display luminance for black in :math:`cd/m^2`.
    L_W
        :math:`L_W` is nominal peak luminance of the display in :math:`cd/m^2`
        for achromatic pixels.
    gamma
        System gamma value, 1.2 at the nominal display peak luminance of
        :math:`1000 cd/m^2`.
    method
        Computation method.

    Returns
    -------
    :class:`numpy.ndarray`
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
    :cite:`Borer2017a`, :cite:`InternationalTelecommunicationUnion2017`

    Examples
    --------
    >>> ootf_BT2100_HLG(0.1)  # doctest: +ELLIPSIS
    63.0957344...
    >>> ootf_BT2100_HLG(0.1, 0.01, method="ITU-R BT.2100-1")
    ... # doctest: +ELLIPSIS
    63.1051034...
    """

    method = validate_method(method, tuple(BT2100_HLG_OOTF_METHODS))

    function = BT2100_HLG_OOTF_METHODS[method]

    return function(
        E,
        **filter_kwargs(function, **{"L_B": L_B, "L_W": L_W, "gamma": gamma}),
    )


def ootf_inverse_BT2100_HLG_1(
    F_D: ArrayLike,
    L_B: float = 0,
    L_W: float = 1000,
    gamma: float | None = None,
) -> NDArrayFloat:
    """
    Define *Recommendation ITU-R BT.2100* *Reference HLG* inverse opto-optical
    transfer function (OOTF / OOCF) as given in *ITU-R BT.2100-1*.

    Parameters
    ----------
    F_D
        :math:`F_D` is the luminance of a displayed linear component
        :math:`{R_D, G_D, or B_D}`, in :math:`cd/m^2`.
    L_B
        :math:`L_B` is the display luminance for black in :math:`cd/m^2`.
    L_W
        :math:`L_W` is nominal peak luminance of the display in :math:`cd/m^2`
        for achromatic pixels.
    gamma
        System gamma value, 1.2 at the nominal display peak luminance of
        :math:`1000 cd/m^2`.

    Returns
    -------
    :class:`numpy.ndarray`
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
    :cite:`Borer2017a`, :cite:`InternationalTelecommunicationUnion2017`

    Examples
    --------
    >>> ootf_inverse_BT2100_HLG_1(63.095734448019336)  # doctest: +ELLIPSIS
    0.1000000...
    >>> ootf_inverse_BT2100_HLG_1(63.105103490674857, 0.01)
    ... # doctest: +ELLIPSIS
    0.0999999...
    """

    F_D = to_domain_1(F_D)

    is_single_channel = np.atleast_1d(F_D).shape[-1] != 3

    if is_single_channel:
        usage_warning(
            '"Recommendation ITU-R BT.2100" "Reference HLG OOTF" uses '
            "RGB Luminance in computations and expects a vector input, thus "
            "the given input array will be stacked to compose a vector for "
            "internal computations but a single component will be output."
        )
        R_D = G_D = B_D = F_D
    else:
        R_D, G_D, B_D = tsplit(F_D)

    Y_D = np.sum(WEIGHTS_BT2100_HLG * tstack([R_D, G_D, B_D]), axis=-1)

    alpha = L_W - L_B
    beta = L_B

    gamma = optional(gamma, gamma_function_BT2100_HLG(L_W))

    Y_D_beta = np.abs((Y_D - beta) / alpha) ** ((1 - gamma) / gamma)

    R_S = np.where(
        beta == Y_D,
        0.0,
        Y_D_beta * (R_D - beta) / alpha,
    )
    G_S = np.where(
        beta == Y_D,
        0.0,
        Y_D_beta * (G_D - beta) / alpha,
    )
    B_S = np.where(
        beta == Y_D,
        0.0,
        Y_D_beta * (B_D - beta) / alpha,
    )

    if is_single_channel:
        return as_float(from_range_1(R_S))
    else:
        RGB_S = tstack([R_S, G_S, B_S])

        return from_range_1(RGB_S)


def ootf_inverse_BT2100_HLG_2(
    F_D: ArrayLike,
    L_W: float = 1000,
    gamma: float | None = None,
) -> NDArrayFloat:
    """
    Define *Recommendation ITU-R BT.2100* *Reference HLG* inverse opto-optical
    transfer function (OOTF / OOCF) as given in *ITU-R BT.2100-2*.

    Parameters
    ----------
    F_D
        :math:`F_D` is the luminance of a displayed linear component
        :math:`{R_D, G_D, or B_D}`, in :math:`cd/m^2`.
    L_W
        :math:`L_W` is nominal peak luminance of the display in :math:`cd/m^2`
        for achromatic pixels.
    gamma
        System gamma value, 1.2 at the nominal display peak luminance of
        :math:`1000 cd/m^2`.

    Returns
    -------
    :class:`numpy.ndarray`
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
    :cite:`InternationalTelecommunicationUnion2018`

    Examples
    --------
    >>> ootf_inverse_BT2100_HLG_2(63.095734448019336)  # doctest: +ELLIPSIS
    0.1000000...
    """

    F_D = to_domain_1(F_D)

    is_single_channel = np.atleast_1d(F_D).shape[-1] != 3

    if is_single_channel:
        usage_warning(
            '"Recommendation ITU-R BT.2100" "Reference HLG OOTF" uses '
            "RGB Luminance in computations and expects a vector input, thus "
            "the given input array will be stacked to compose a vector for "
            "internal computations but a single component will be output."
        )
        R_D = G_D = B_D = F_D
    else:
        R_D, G_D, B_D = tsplit(F_D)

    Y_D = np.sum(WEIGHTS_BT2100_HLG * tstack([R_D, G_D, B_D]), axis=-1)

    alpha = L_W

    gamma = optional(gamma, gamma_function_BT2100_HLG(L_W))

    Y_D_alpha = np.abs(Y_D / alpha) ** ((1 - gamma) / gamma)

    R_S = np.where(
        Y_D == 0,
        0.0,
        Y_D_alpha * R_D / alpha,
    )
    G_S = np.where(
        Y_D == 0,
        0.0,
        Y_D_alpha * G_D / alpha,
    )
    B_S = np.where(
        Y_D == 0,
        0.0,
        Y_D_alpha * B_D / alpha,
    )

    if is_single_channel:
        return as_float(from_range_1(R_S))
    else:
        RGB_S = tstack([R_S, G_S, B_S])

        return from_range_1(RGB_S)


BT2100_HLG_OOTF_INVERSE_METHODS: CanonicalMapping = CanonicalMapping(
    {
        "ITU-R BT.2100-1": ootf_inverse_BT2100_HLG_1,
        "ITU-R BT.2100-2": ootf_inverse_BT2100_HLG_2,
    }
)
BT2100_HLG_OOTF_INVERSE_METHODS.__doc__ = """
Supported *Recommendation ITU-R BT.2100* *Reference HLG* inverse opto-optical
transfer function (OOTF / OOCF).

References
----------
:cite:`Borer2017a`, :cite:`InternationalTelecommunicationUnion2017`,
:cite:`InternationalTelecommunicationUnion2018`
"""


def ootf_inverse_BT2100_HLG(
    F_D: ArrayLike,
    L_B: float = 0,
    L_W: float = 1000,
    gamma: float | None = None,
    method: Literal["ITU-R BT.2100-1", "ITU-R BT.2100-2"]
    | str = "ITU-R BT.2100-2",
) -> NDArrayFloat:
    """
    Define *Recommendation ITU-R BT.2100* *Reference HLG* inverse opto-optical
    transfer function (OOTF / OOCF).

    Parameters
    ----------
    F_D
        :math:`F_D` is the luminance of a displayed linear component
        :math:`{R_D, G_D, or B_D}`, in :math:`cd/m^2`.
    L_B
        :math:`L_B` is the display luminance for black in :math:`cd/m^2`.
    L_W
        :math:`L_W` is nominal peak luminance of the display in :math:`cd/m^2`
        for achromatic pixels.
    gamma
        System gamma value, 1.2 at the nominal display peak luminance of
        :math:`1000 cd/m^2`.
    method
        Computation method.

    Returns
    -------
    :class:`numpy.ndarray`
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
    :cite:`Borer2017a`, :cite:`InternationalTelecommunicationUnion2017`,
    :cite:`InternationalTelecommunicationUnion2018`

    Examples
    --------
    >>> ootf_inverse_BT2100_HLG(63.095734448019336)  # doctest: +ELLIPSIS
    0.1000000...
    >>> ootf_inverse_BT2100_HLG(
    ...     63.105103490674857, 0.01, method="ITU-R BT.2100-1"
    ... )
    ... # doctest: +ELLIPSIS
    0.0999999...
    """

    method = validate_method(method, tuple(BT2100_HLG_OOTF_INVERSE_METHODS))

    function = BT2100_HLG_OOTF_INVERSE_METHODS[method]

    return function(
        F_D,
        **filter_kwargs(function, **{"L_B": L_B, "L_W": L_W, "gamma": gamma}),
    )
