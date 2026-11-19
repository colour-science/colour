"""
*Transfer Matrix Method* (TMM) for Multilayer Optical Calculations
==================================================================

Implements the *Transfer Matrix Method* for computing optical properties of
multilayer planar stacks.

The *TMM* calculations handle two orthogonal polarisation states:

-   **s-polarisation (Transverse Electric)**: The electric field vector
    :math:`\\vec{E}_s` oscillates perpendicular to the plane of incidence.
    This is sometimes called TE-mode because the electric field is
    transverse to the plane of incidence.

-   **p-polarisation (TM - Transverse Magnetic)**: The electric field vector
    :math:`\\vec{E}_p` oscillates parallel to the plane of incidence.
    This is sometimes called TM polarisation because the magnetic field is
    transverse to the plane of incidence.

The plane of incidence is defined by the incident light ray and the surface
normal vector. For unpolarised light, the reflectance and transmittance are
the average of the s and p components: :math:`R = (R_s + R_p)/2`.

-   :func:`colour.phenomena.snell_law`
-   :func:`colour.phenomena.polarised_light_magnitude_elements`
-   :func:`colour.phenomena.polarised_light_reflection_amplitude`
-   :func:`colour.phenomena.polarised_light_reflection_coefficient`
-   :func:`colour.phenomena.polarised_light_transmission_amplitude`
-   :func:`colour.phenomena.polarised_light_transmission_coefficient`
-   :class:`colour.phenomena.TransferMatrixResult_TMM`
-   :func:`colour.phenomena.matrix_transfer_tmm`

References
----------
-   :cite:`Byrnes2016` : Byrnes, S. J. (2016). Multilayer optical
    calculations. arXiv:1603.02720 [Physics].
    http://arxiv.org/abs/1603.02720
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from colour.constants import DTYPE_COMPLEX_DEFAULT
from colour.hints import cast
from colour.utilities import (
    MixinDataclassArithmetic,
    array_namespace,
    as_array,
    as_complex_array,
    as_float_array,
    is_non_ndarray,
    tsplit,
    tstack,
    xp_as_array,
    xp_as_float_array,
    xp_astype,
    xp_atleast_1d,
    xp_atleast_2d,
    xp_broadcast_to,
    xp_degrees,
    xp_matrix_transpose,
    xp_radians,
)

if TYPE_CHECKING:
    from colour.hints import Any, ArrayLike, NDArrayComplex, NDArrayFloat, Tuple

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "snell_law",
    "polarised_light_magnitude_elements",
    "polarised_light_reflection_amplitude",
    "polarised_light_reflection_coefficient",
    "polarised_light_transmission_amplitude",
    "polarised_light_transmission_coefficient",
    "TransferMatrixResult",
    "matrix_transfer_tmm",
]


def snell_law(
    n_1: ArrayLike,
    n_2: ArrayLike,
    theta_i: ArrayLike,
) -> NDArrayFloat:
    """
    Compute the refraction angle using *Snell's Law*.

    Parameters
    ----------
    n_1
        Refractive index of the incident medium :math:`n_1`.
    n_2
        Refractive index of the refracting medium :math:`n_2`.
    theta_i
        Incident angle :math:`\\theta_i` in degrees.

    Returns
    -------
    :class:`numpy.ndarray`
        Refracted angle in degrees.

    Notes
    -----
    -   *Snell's Law* relates the angles of incidence and refraction when light
        passes through a boundary between two different media (*Equation 3*
        from :cite:`Byrnes2016`):

    .. math::

        n_i \\sin \\theta_i = n_j \\sin \\theta_j

    Where:

    -   :math:`n_i, n_j`: Refractive indices of the incident and refracting media
    -   :math:`\\theta_i, \\theta_j`: Angles of incidence and refraction

    References
    ----------
    :cite:`Byrnes2016`

    Examples
    --------
    >>> snell_law(1.0, 1.5, 30.0)  # doctest: +ELLIPSIS
    np.float64(19.4712206...)
    """

    theta_i = xp_radians(as_float_array(theta_i))

    xp = array_namespace(theta_i, n_1, n_2)

    theta_i = xp_as_float_array(
        theta_i, xp=xp, like=n_1 if is_non_ndarray(n_1) else n_2
    )
    n_1 = xp_as_float_array(
        cast("Any", n_1).real if hasattr(n_1, "real") else np.real(n_1),
        xp=xp,
        like=theta_i,
    )
    n_2 = xp_as_float_array(
        cast("Any", n_2).real if hasattr(n_2, "real") else np.real(n_2),
        xp=xp,
        like=theta_i,
    )

    # Apply Snell's law: n_i * sin(theta_i) = n_j * sin(theta_j) (Byrnes Eq. 3)
    return xp_degrees(xp.asin(n_1 * xp.sin(theta_i) / n_2))


def polarised_light_magnitude_elements(
    n_1: ArrayLike,
    n_2: ArrayLike,
    theta_i: ArrayLike,
    theta_t: ArrayLike,
) -> Tuple[NDArrayComplex, NDArrayComplex, NDArrayComplex, NDArrayComplex]:
    """
    Compute common magnitude elements for *Fresnel* equations.

    This function computes the common terms used in the *Fresnel* equations
    for both s-polarisation (perpendicular) and p-polarisation (parallel)
    components of light at a dielectric interface.

    Parameters
    ----------
    n_1
        Refractive index of the incident medium :math:`n_1`.
    n_2
        Refractive index of the transmitted medium :math:`n_2`.
    theta_i
        Incident angle :math:`\\theta_i` in degrees.
    theta_t
        Transmitted angle :math:`\\theta_t` in degrees.

    Returns
    -------
    :class:`tuple`
        Tuple of precomputed magnitude elements:
        :math:`(n_1 \\cos \\theta_i, n_1 \\cos \\theta_t, n_2 \\cos \\theta_i,
        n_2 \\cos \\theta_t)`

    Notes
    -----
    These magnitude elements are fundamental components in the *Fresnel* equations:

    -   :math:`n_1 \\cos \\theta_i`: Incident medium magnitude
    -   :math:`n_1 \\cos \\theta_t`: Incident medium magnitude (transmitted angle)
    -   :math:`n_2 \\cos \\theta_i`: Transmitted medium magnitude (incident angle)
    -   :math:`n_2 \\cos \\theta_t`: Transmitted medium magnitude

    These terms appear in all *Fresnel* amplitude and power coefficients for
    both reflection and transmission at dielectric interfaces.

    Examples
    --------
    >>> polarised_light_magnitude_elements(1.0, 1.5, 0.0, 0.0)  # doctest: +ELLIPSIS
    (np.complex128(1+0j), np.complex128(1+0j), np.complex128(1.5+0j), \
np.complex128(1.5+0j))
    """

    n_1 = as_complex_array(n_1)
    n_2 = as_complex_array(n_2)

    theta_i = as_array(theta_i)
    theta_t = as_array(theta_t)

    operands = (n_1, n_2, theta_i, theta_t)
    xp = array_namespace(*operands)
    like = next((a for a in operands if is_non_ndarray(a)), theta_i)

    n_1 = xp_as_array(n_1, xp=xp, like=like)
    n_2 = xp_as_array(n_2, xp=xp, like=n_1)
    n_1 = as_complex_array(n_1)
    n_2 = as_complex_array(n_2)
    theta_i = xp_as_array(theta_i, xp=xp, like=n_1)
    theta_t = xp_as_array(theta_t, xp=xp, like=n_1)
    complex_dtypes = (xp.complex64, xp.complex128)
    theta_i_rad = (
        theta_i * (np.pi / 180)
        if theta_i.dtype in complex_dtypes
        else xp_radians(theta_i, xp=xp)
    )
    theta_t_rad = (
        theta_t * (np.pi / 180)
        if theta_t.dtype in complex_dtypes
        else xp_radians(theta_t, xp=xp)
    )

    cos_theta_i = xp_astype(xp.cos(theta_i_rad), n_1.dtype, xp=xp)
    cos_theta_t = xp_astype(xp.cos(theta_t_rad), n_1.dtype, xp=xp)

    n_1_cos_theta_i = n_1 * cos_theta_i
    n_1_cos_theta_t = n_1 * cos_theta_t
    n_2_cos_theta_i = n_2 * cos_theta_i
    n_2_cos_theta_t = n_2 * cos_theta_t

    return n_1_cos_theta_i, n_1_cos_theta_t, n_2_cos_theta_i, n_2_cos_theta_t


def polarised_light_reflection_amplitude(
    n_1: ArrayLike,
    n_2: ArrayLike,
    theta_i: ArrayLike,
    theta_t: ArrayLike,
) -> NDArrayComplex:
    """
    Compute *Fresnel* reflection amplitude coefficients.

    This function computes the complex reflection amplitude coefficients for
    both s-polarisation (perpendicular) and p-polarisation (parallel) components
    of electromagnetic waves at a dielectric interface.

    Parameters
    ----------
    n_1
        Refractive index of the incident medium :math:`n_1`.
    n_2
        Refractive index of the transmitted medium :math:`n_2`.
    theta_i
        Incident angle :math:`\\theta_i` in degrees.
    theta_t
        Transmitted angle :math:`\\theta_t` in degrees.

    Returns
    -------
    :class:`numpy.ndarray`
        *Fresnel* reflection amplitude coefficients for s and p polarisations
        stacked along the last axis. The array contains :math:`[r_s, r_p]` where
        :math:`r_s` and :math:`r_p` are the complex reflection coefficients.

    Notes
    -----
    The *Fresnel* reflection amplitude coefficients are given by (*Equation 6*
    from :cite:`Byrnes2016`):

    .. math::

        r_s &= \\frac{n_1 \\cos \\theta_1 - n_2 \\cos \\theta_2}{n_1 \\cos \
\\theta_1 + n_2 \\cos \\theta_2} \\\\
        r_p &= \\frac{n_2 \\cos \\theta_1 - n_1 \\cos \\theta_2}{n_2 \\cos \
\\theta_1 + n_1 \\cos \\theta_2}

    Where:

    -   :math:`r_s`: s-polarisation reflection amplitude (electric field perpendicular
        to the plane of incidence)
    -   :math:`r_p`: p-polarisation reflection amplitude (electric field parallel
        to the plane of incidence)
    -   :math:`n_1, n_2`: Refractive indices of incident and transmitted media
    -   :math:`\\theta_1, \\theta_2`: Incident and transmitted angles

    Examples
    --------
    >>> polarised_light_reflection_amplitude(1.0, 1.5, 0.0, 0.0)
    array([-0.2+0.j, -0.2+0.j])
    """

    n_1_cos_theta_i, n_1_cos_theta_t, n_2_cos_theta_i, n_2_cos_theta_t = (
        polarised_light_magnitude_elements(n_1, n_2, theta_i, theta_t)
    )

    # Fresnel reflection amplitudes (Byrnes Eq. 6)
    r_s = (n_1_cos_theta_i - n_2_cos_theta_t) / (n_1_cos_theta_i + n_2_cos_theta_t)
    r_p = (n_1_cos_theta_t - n_2_cos_theta_i) / (n_1_cos_theta_t + n_2_cos_theta_i)

    return tstack([r_s, r_p], dtype=DTYPE_COMPLEX_DEFAULT)


def polarised_light_reflection_coefficient(
    n_1: ArrayLike,
    n_2: ArrayLike,
    theta_i: ArrayLike,
    theta_t: ArrayLike,
) -> NDArrayComplex:
    """
    Compute *Fresnel* reflection power coefficients (reflectance).

    This function computes the reflection power coefficients, which represent
    the fraction of incident power that is reflected at a dielectric interface
    for both s-polarisation (perpendicular) and p-polarisation (parallel) components.

    Parameters
    ----------
    n_1
        Refractive index of the incident medium :math:`n_1`.
    n_2
        Refractive index of the transmitted medium :math:`n_2`.
    theta_i
        Incident angle :math:`\\theta_i` in degrees.
    theta_t
        Transmitted angle :math:`\\theta_t` in degrees.

    Returns
    -------
    :class:`numpy.ndarray`
        *Fresnel* reflection power coefficients (reflectance) for s and p
        polarisations stacked along the last axis. The array contains
        :math:`[R_s, R_p]`.

    Notes
    -----
    The *Fresnel* reflection power coefficients (reflectance) are given by:

    .. math::

        R_s &= |r_s|^2 = \\left|\\frac{n_1 \\cos \\theta_i - n_2 \\cos \
\\theta_t}{n_1 \\cos \\theta_i + n_2 \\cos \\theta_t}\\right|^2 \\\\
        R_p &= |r_p|^2 = \\left|\\frac{n_1 \\cos \\theta_t - n_2 \\cos \
\\theta_i}{n_1 \\cos \\theta_t + n_2 \\cos \\theta_i}\\right|^2

    Where:

    -   :math:`R_s`: s-polarisation reflectance (fraction of incident power reflected)
    -   :math:`R_p`: p-polarisation reflectance (fraction of incident power reflected)
    -   :math:`r_s, r_p`: complex reflection amplitude coefficients
    -   The s-polarisation electric field is perpendicular to the plane of incidence
    -   The p-polarisation electric field is parallel to the plane of incidence

    The reflectance values satisfy: :math:`0 \\leq R_s, R_p \\leq 1`.

    References
    ----------
    :cite:`Byrnes2016`

    Examples
    --------
    >>> result = polarised_light_reflection_coefficient(1.0, 1.5, 0.0, 0.0)
    >>> result.real
    array([0.04, 0.04])
    """

    # Reflectance: R = |r|^2 (Byrnes Eq. 23)
    r = polarised_light_reflection_amplitude(n_1, n_2, theta_i, theta_t)

    xp = array_namespace(r)

    R = xp.abs(r) ** 2

    return as_complex_array(R)


def polarised_light_transmission_amplitude(
    n_1: ArrayLike,
    n_2: ArrayLike,
    theta_i: ArrayLike,
    theta_t: ArrayLike,
) -> NDArrayComplex:
    """
    Compute *Fresnel* transmission amplitude coefficients.

    This function computes the complex transmission amplitude coefficients for
    both s-polarisation (perpendicular) and p-polarisation (parallel) components
    of electromagnetic waves at a dielectric interface.

    Parameters
    ----------
    n_1
        Refractive index of the incident medium :math:`n_1`.
    n_2
        Refractive index of the transmitted medium :math:`n_2`.
    theta_i
        Incident angle :math:`\\theta_i` in degrees.
    theta_t
        Transmitted angle :math:`\\theta_t` in degrees.

    Returns
    -------
    :class:`numpy.ndarray`
        *Fresnel* transmission amplitude coefficients for s and p polarisations
        stacked along the last axis. The array contains :math:`[t_s, t_p]` where
        :math:`t_s` and :math:`t_p` are the complex transmission coefficients.

    Notes
    -----
    The *Fresnel* transmission amplitude coefficients are given by (*Equation 6*
    from :cite:`Byrnes2016`):

    .. math::

        t_s &= \\frac{2n_1 \\cos \\theta_1}{n_1 \\cos \\theta_1 + n_2 \\cos \
\\theta_2} \\\\
        t_p &= \\frac{2n_1 \\cos \\theta_1}{n_2 \\cos \\theta_1 + n_1 \\cos \
\\theta_2}

    Where:

    -   :math:`t_s`: s-polarisation transmission amplitude (electric field perpendicular
        to the plane of incidence)
    -   :math:`t_p`: p-polarisation transmission amplitude (electric field parallel
        to the plane of incidence)
    -   :math:`n_1, n_2`: Refractive indices of incident and transmitted media
    -   :math:`\\theta_1, \\theta_2`: Incident and transmitted angles

    Examples
    --------
    >>> polarised_light_transmission_amplitude(1.0, 1.5, 0.0, 0.0)
    array([0.8+0.j, 0.8+0.j])
    """

    n_1_cos_theta_i, n_1_cos_theta_t, n_2_cos_theta_i, n_2_cos_theta_t = (
        polarised_light_magnitude_elements(n_1, n_2, theta_i, theta_t)
    )

    two_n_1_cos_theta_i = 2 * n_1_cos_theta_i

    # Fresnel transmission amplitudes (Byrnes Eq. 6)
    t_s = two_n_1_cos_theta_i / (n_1_cos_theta_i + n_2_cos_theta_t)
    t_p = two_n_1_cos_theta_i / (n_2_cos_theta_i + n_1_cos_theta_t)

    return tstack([t_s, t_p], dtype=DTYPE_COMPLEX_DEFAULT)


def polarised_light_transmission_coefficient(
    n_1: ArrayLike,
    n_2: ArrayLike,
    theta_i: ArrayLike,
    theta_t: ArrayLike,
) -> NDArrayComplex:
    """
    Compute *Fresnel* transmission power coefficients (transmittance).

    This function computes the transmission power coefficients, which represent
    the fraction of incident power that is transmitted through a dielectric interface
    for both s-polarisation (perpendicular) and p-polarisation (parallel) components.

    Parameters
    ----------
    n_1
        Refractive index of the incident medium :math:`n_1`.
    n_2
        Refractive index of the transmitted medium :math:`n_2`.
    theta_i
        Incident angle :math:`\\theta_i` in degrees.
    theta_t
        Transmitted angle :math:`\\theta_t` in degrees.

    Returns
    -------
    :class:`numpy.ndarray`
        *Fresnel* transmission power coefficients (transmittance) for s and p
        polarisations stacked along the last axis. The array contains
        :math:`[T_s, T_p]`.

    Notes
    -----
    The *Fresnel* transmission power coefficients (transmittance) are given by:

    .. math::

        T_s &= \\frac{\\mathrm{Re}(n_2 \\cos \\theta_t)}
        {\\mathrm{Re}(n_1 \\cos \\theta_i)} |t_s|^2 \\\\
        T_p &= \\frac{\\mathrm{Re}(n_2 \\cos \\theta_t^*)}
        {\\mathrm{Re}(n_1 \\cos \\theta_i^*)} |t_p|^2

    Where:

    -   :math:`T_s`: s-polarisation transmittance (fraction of incident power
        transmitted)
    -   :math:`T_p`: p-polarisation transmittance (fraction of incident power
        transmitted)
    -   :math:`t_s, t_p`: complex transmission amplitude coefficients
    -   The s-polarisation electric field is perpendicular to the plane of incidence
    -   The p-polarisation electric field is parallel to the plane of incidence

    The ratios normalise transmitted power by the incident normal flux
    (Equations 21 and 22 in :cite:`Byrnes2016`). For non-absorbing media,
    :math:`R_s + T_s = 1` and :math:`R_p + T_p = 1`. For an absorbing
    incident medium, this normalisation can give :math:`T > 1`.

    References
    ----------
    :cite:`Byrnes2016`

    Examples
    --------
    >>> polarised_light_transmission_coefficient(1.0, 1.5, 0.0, 0.0)
    array([0.96+0.j, 0.96+0.j])
    """

    n_1 = as_complex_array(n_1)
    n_2 = as_complex_array(n_2)
    theta_i = as_array(theta_i)
    theta_t = as_array(theta_t)

    operands = (n_1, n_2, theta_i, theta_t)
    xp = array_namespace(*operands)
    like = next((a for a in operands if is_non_ndarray(a)), theta_i)

    n_1 = xp_as_array(n_1, xp=xp, like=like)
    n_2 = xp_as_array(n_2, xp=xp, like=like)
    theta_i = xp_as_array(theta_i, xp=xp, like=like)
    theta_t = xp_as_array(theta_t, xp=xp, like=like)

    # Unit refractive indices give the angle cosines directly.
    cos_theta_i, cos_theta_t, _, _ = polarised_light_magnitude_elements(
        1, 1, theta_i, theta_t
    )

    n_1_cos_theta_i = n_1 * cos_theta_i
    n_2_cos_theta_t = n_2 * cos_theta_t

    # Reuse the angle cosines for the amplitudes and normal power flux.
    t_s = 2 * n_1_cos_theta_i / (n_1_cos_theta_i + n_2_cos_theta_t)
    t_p = 2 * n_1_cos_theta_i / (n_2 * cos_theta_i + n_1 * cos_theta_t)
    flux_s = xp.real(n_2_cos_theta_t) / xp.real(n_1_cos_theta_i)
    flux_p = xp.real(n_2 * xp.conj(cos_theta_t)) / xp.real(n_1 * xp.conj(cos_theta_i))

    return as_complex_array(
        xp.stack([flux_s * xp.abs(t_s) ** 2, flux_p * xp.abs(t_p) ** 2], axis=-1)
    )


@dataclass
class TransferMatrixResult(MixinDataclassArithmetic):
    """
    Define the *Transfer Matrix Method* calculation results.

    Parameters
    ----------
    M_s
        Transfer matrix for s-polarisation :math:`M_s`, shape
        (..., wavelengths_count, 2, 2).
    M_p
        Transfer matrix for p-polarisation :math:`M_p`, shape
        (..., wavelengths_count, 2, 2).
    theta
        Propagation angles in each layer :math:`\\theta_j` (degrees), shape
        (..., n_layers+2). The angles are calculated from the real refractive
        indices at the first wavelength, from incident medium to substrate.
        Evanescent angles are NaN.
    n
        Complete multilayer stack :math:`n_j`, shape
        (..., n_layers+2, wavelengths_count). Includes
        [n_incident, n_layer_1, ..., n_layer_n, n_substrate].

    References
    ----------
    :cite:`Byrnes2016`
    """

    M_s: NDArrayComplex
    M_p: NDArrayComplex
    theta: NDArrayFloat
    n: NDArrayComplex


def matrix_transfer_tmm(
    n: ArrayLike,
    t: ArrayLike,
    theta: ArrayLike,
    wavelength: ArrayLike,
) -> TransferMatrixResult:
    """
    Calculate transfer matrices for multilayer thin film structures using the
    *Transfer Matrix Method*.

    This function constructs the transfer matrices for s-polarised and
    p-polarised light propagating through a multilayer structure. The transfer
    matrices encode the optical properties of the structure and are used to
    calculate reflectance and transmittance.

    Parameters
    ----------
    n
        Complete refractive index stack :math:`n_j` including incident medium,
        layers, and substrate. Shape: (media_count,) or
        (media_count, wavelengths_count). Can be complex for absorbing
        materials. The array should contain [n_incident, n_layer_1, ...,
        n_layer_n, n_substrate].
    t
        Thicknesses of each layer :math:`t_j` in nanometers (excluding incident
        and substrate). Shape: (layers_count,) or (thickness_count, layers_count).

        - **1D array** ``[t1, t2, ...]``: One thickness per layer for a single
          multilayer configuration. Shape: ``(layers_count,)``
        - **2D array** ``[[t1, t2, ...], [t1', t2', ...]]``: Multiple thickness
          configurations for outer product broadcasting. Shape:
          ``(thickness_count, layers_count)``

        Most users should use :func:`thin_film_tmm` or :func:`multilayer_tmm`
        instead, which provide simpler interfaces.
    theta
        Incident angle :math:`\\theta` in degrees. Scalar or array of shape
        (angles_count,) for angle broadcasting.
    wavelength
        Vacuum wavelength values :math:`\\lambda` in nanometers.

    Returns
    -------
    :class:`colour.TransferMatrixResult`
        Transfer matrix calculation results containing M_s, M_p, theta,
        and n arrays.

    Examples
    --------
    Single layer at one wavelength:

    >>> result = matrix_transfer_tmm(
    ...     n=[1.0, 1.5, 1.0],
    ...     t=[250],
    ...     theta=0,
    ...     wavelength=550,
    ... )
    >>> result.M_s.shape
    (1, 1, 1, 2, 2)
    >>> result.theta.shape
    (1, 3)

    Multiple wavelengths:

    >>> result = matrix_transfer_tmm(
    ...     n=[1.0, 1.5, 1.0],
    ...     t=[250],
    ...     theta=0,
    ...     wavelength=[400, 500, 600],
    ... )
    >>> result.M_s.shape
    (3, 1, 1, 2, 2)

    Multiple angles (angle broadcasting):

    >>> result = matrix_transfer_tmm(
    ...     n=[1.0, 1.5, 1.0],
    ...     t=[250],
    ...     theta=[0, 30, 45, 60],
    ...     wavelength=[400, 500, 600],
    ... )
    >>> result.M_s.shape
    (3, 4, 1, 2, 2)
    >>> result.theta.shape
    (4, 3)

    Notes
    -----
    -   The *Transfer Matrix Method* relates the field amplitudes across the entire
        multilayer structure (*Equations 10-15* from :cite:`Byrnes2016`):

    .. math::

        \\begin{pmatrix} v_n \\\\ w_n \\end{pmatrix} = M_n \\begin{pmatrix} \
v_{n+1} \\\\ w_{n+1} \\end{pmatrix}

    Where :math:`M_n` combines the layer propagation and interface matrices:

    .. math::

        M_n = L_n \\cdot I_{n,n+1} =
        \\begin{pmatrix}
        e^{-i\\delta_n} & 0 \\\\ 0 & e^{i\\delta_n}
        \\end{pmatrix}
        \\frac{1}{t_{n,n+1}}
        \\begin{pmatrix}
        1 & r_{n,n+1} \\\\ r_{n,n+1} & 1
        \\end{pmatrix}

    The overall transfer matrix :math:`\\tilde{M}` for the complete structure is:

    .. math::

        \\tilde{M} = \\frac{1}{t_{0,1}}
        \\begin{pmatrix}
        1 & r_{0,1} \\\\ r_{0,1} & 1
        \\end{pmatrix}
        M_1 M_2 \\cdots M_{N-2}

    From which the overall reflection and transmission coefficients are extracted:

    .. math::

        \\begin{pmatrix} 1 \\\\ r \\end{pmatrix} = \\tilde{M} \
\\begin{pmatrix} t \\\\ 0 \\end{pmatrix}

    .. math::

        t = \\frac{1}{\\tilde{M}_{00}}, \\quad r = \
\\frac{\\tilde{M}_{10}}{\\tilde{M}_{00}}

    Where:

    -   :math:`v_n, w_n`: Forward and backward field amplitudes in layer :math:`n`
    -   :math:`M_n`: Transfer matrix for layer :math:`n`
    -   :math:`L_n`: Layer propagation matrix
    -   :math:`I_{n,n+1}`: Interface matrix between layers :math:`n` and :math:`n+1`
    -   :math:`\\tilde{M}`: Overall transfer matrix
    -   :math:`r, t`: Overall reflection and transmission amplitude coefficients

    -   Supports complex refractive indices for absorbing materials.
    -   **Angle broadcasting**: All computations are vectorized across angles.
        The output always includes the angle dimension.
    -   The transfer matrices always have shape (angles_count, wavelengths_count, 2, 2),
        even for scalar theta (angles_count=1).

    References
    ----------
    :cite:`Byrnes2016`
    """

    n = as_complex_array(n)
    t = as_float_array(t)
    theta = as_float_array(theta)
    wavelength = as_float_array(wavelength)

    xp = array_namespace(wavelength, theta, t, n)

    operands = (wavelength, theta, t, n)
    like = next((a for a in operands if is_non_ndarray(a)), wavelength)
    wavelength = xp_as_float_array(wavelength, xp=xp, like=like)

    n = xp_as_array(n, xp=xp, like=wavelength) if isinstance(n, np.ndarray) else n
    t = xp_as_array(t, xp=xp, like=wavelength)
    theta = xp_atleast_1d(xp_as_array(theta, xp=xp, like=wavelength), xp=xp)
    wavelength = xp_atleast_1d(wavelength, xp=xp)

    wavelengths_count = wavelength.shape[0]

    if n.ndim == 1:
        n = xp_matrix_transpose(xp_atleast_2d(n, xp=xp), xp=xp)
        # ``broadcast_to`` is equivalent to ``tile`` for an ``(M, 1)`` ->
        # ``(M, W)`` expansion, works on every backend (*MPS* does not
        # support ``tile`` on complex tensors) and avoids the copy; ``n``
        # is only ever read downstream.
        n = xp_broadcast_to(n, (n.shape[0], wavelengths_count), xp=xp)

    # (1, layers_count)
    if t.ndim == 1:
        t = t[None, :]

    media_count = n.shape[0]
    layers_count = media_count - 2

    theta_radians = xp_radians(theta)
    sin_theta = xp.sin(theta_radians)[:, None, None]

    # Retain the real, first-wavelength angle metadata. An evanescent
    # angle has no real value.
    n_real = xp.real(n[:, 0])
    sin_theta_media = sin_theta[..., 0] * n_real[0, ...] / n_real[None, :]
    theta_media = xp_degrees(
        xp.asin(xp.where(xp.abs(sin_theta_media) <= 1, sin_theta_media, xp.nan))
    )

    # Conserve the transverse wavevector at each wavelength (Byrnes Eq. 3).
    sin_theta_media_complex = n[None, :1, :] * sin_theta / n[None, :, :]
    theta_media_complex = xp.asin(sin_theta_media_complex)
    cos_theta_media = xp.cos(theta_media_complex)

    # A real propagating wave uses the principal angle. Only use the
    # imaginary wavevector to select a branch when decay is present.
    propagating = (
        (xp.imag(n[None, :, :]) == 0)
        & (xp.imag(sin_theta_media_complex) == 0)
        & (xp.abs(xp.real(sin_theta_media_complex)) <= 1)
    )
    reverse = (~propagating) & (xp.imag(n[None, :, :] * cos_theta_media) < 0)
    theta_media_complex = xp.where(
        reverse, np.pi - theta_media_complex, theta_media_complex
    )

    n_1 = n[None, :-1, :]
    n_2 = n[None, 1:, :]
    theta_1 = theta_media_complex[:, :-1, :] * (180 / np.pi)
    theta_2 = theta_media_complex[:, 1:, :] * (180 / np.pi)

    r_media = polarised_light_reflection_amplitude(n_1, n_2, theta_1, theta_2)
    t_media = polarised_light_transmission_amplitude(n_1, n_2, theta_1, theta_2)
    r_media_s, r_media_p = xp.unstack(tsplit(r_media, dtype=DTYPE_COMPLEX_DEFAULT))
    t_media_s, t_media_p = xp.unstack(tsplit(t_media, dtype=DTYPE_COMPLEX_DEFAULT))

    # Phase accumulation: delta = d * k_z (Byrnes Eq. 8).
    k_z_layers = (n[None, :, :] * xp.cos(theta_media_complex))[:, 1:-1, :]

    # Compute phase: delta = (2π/λ) * d * k_z
    phase_factor = 2 * np.pi / wavelength[:, None, None, None]  # (W, 1, 1, 1)
    # Reshape k_z from (A, L, W) to (W, A, 1, L) for broadcasting with thickness
    k_z = xp.moveaxis(k_z_layers, (0, 1, 2), (1, 2, 0))[:, :, None, :]  # (W, A, 1, L)
    delta = phase_factor * t[None, None, :, :] * k_z  # (W, A, T, L)

    A = xp.exp(1j * delta)  # (W, A, T, L)

    # Layer matrices: M_n = L_n * I_{n,n+1} (Byrnes Eq. 10-11)
    # (W, A, T, L, 2, 2, 2) for [wavelengths, angles, thickness, layers, 2x2, pol]
    r_s = r_media_s[:, 1 : layers_count + 1, :]  # (A, L, W)
    r_p = r_media_p[:, 1 : layers_count + 1, :]  # (A, L, W)
    t_s = t_media_s[:, 1 : layers_count + 1, :]  # (A, L, W)
    t_p = t_media_p[:, 1 : layers_count + 1, :]  # (A, L, W)

    # Broadcast Fresnel coefficients from (A, L, W) to (W, A, 1, L)
    # (A,L,W) -> (W,A,L) -> (W,A,1,L)
    r_s_b = xp.moveaxis(r_s, (0, 1, 2), (1, 2, 0))[:, :, None, :]
    r_p_b = xp.moveaxis(r_p, (0, 1, 2), (1, 2, 0))[:, :, None, :]
    t_s_b = xp.moveaxis(t_s, (0, 1, 2), (1, 2, 0))[:, :, None, :]
    t_p_b = xp.moveaxis(t_p, (0, 1, 2), (1, 2, 0))[:, :, None, :]

    M_s_layer = xp.stack(
        [
            xp.stack([1 / (A * t_s_b), r_s_b / (A * t_s_b)], axis=-1),
            xp.stack([A * r_s_b / t_s_b, A / t_s_b], axis=-1),
        ],
        axis=-2,
    )
    M_p_layer = xp.stack(
        [
            xp.stack([1 / (A * t_p_b), r_p_b / (A * t_p_b)], axis=-1),
            xp.stack([A * r_p_b / t_p_b, A / t_p_b], axis=-1),
        ],
        axis=-2,
    )
    M = xp.stack([M_s_layer, M_p_layer], axis=-1)

    # Initial interface matrix (Byrnes Eq. 11)
    # Shape: (W, A, T, 2, 2)
    # Fresnel coefficients at incident → first layer interface
    t_s_01 = t_media_s[:, 0, :]  # (A, W)
    r_s_01 = r_media_s[:, 0, :]  # (A, W)
    inv_t_s = xp_matrix_transpose(1 / t_s_01, xp=xp)[:, :, None]  # (W, A, 1)
    r_over_t_s = xp_matrix_transpose(r_s_01 / t_s_01, xp=xp)[:, :, None]
    M_s = xp.stack(
        [
            xp.stack([inv_t_s, r_over_t_s], axis=-1),
            xp.stack([r_over_t_s, inv_t_s], axis=-1),
        ],
        axis=-2,
    )

    t_p_01 = t_media_p[:, 0, :]  # (A, W)
    r_p_01 = r_media_p[:, 0, :]  # (A, W)
    inv_t_p = xp_matrix_transpose(1 / t_p_01, xp=xp)[:, :, None]
    r_over_t_p = xp_matrix_transpose(r_p_01 / t_p_01, xp=xp)[:, :, None]
    M_p = xp.stack(
        [
            xp.stack([inv_t_p, r_over_t_p], axis=-1),
            xp.stack([r_over_t_p, inv_t_p], axis=-1),
        ],
        axis=-2,
    )

    # Overall transfer matrix: M_tilde = I_01 @ M_1 @ M_2 @ ... (Byrnes Eq. 12)
    for i in range(layers_count):
        M_s = xp.matmul(M_s, M[:, :, :, i, :, :, 0])
        M_p = xp.matmul(M_p, M[:, :, :, i, :, :, 1])

    return TransferMatrixResult(
        M_s=M_s,
        M_p=M_p,
        theta=theta_media,
        n=n,
    )
