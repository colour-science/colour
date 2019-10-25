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
from colour.utilities import (
    MixinDataclassArithmetic,
    as_complex_array,
    as_float_array,
    tsplit,
    tstack,
    zeros,
)

if TYPE_CHECKING:
    from colour.hints import ArrayLike, NDArrayComplex, NDArrayFloat, Tuple

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


def _tsplit_complex(a: ArrayLike) -> NDArrayComplex:
    """
    Split the specified stacked array along the last axis (tail)
    to produce an array of complex arrays.

    Convenience wrapper around :func:`colour.utilities.tsplit` that
    automatically uses ``DTYPE_COMPLEX_DEFAULT`` for complex number
    operations in *Transfer Matrix Method* calculations.

    Parameters
    ----------
    a
        Stacked array to split.

    Returns
    -------
    :class:`numpy.ndarray`
        Array of complex arrays.
    """

    return tsplit(a, dtype=DTYPE_COMPLEX_DEFAULT)  # type: ignore[arg-type]


def _tstack_complex(a: ArrayLike) -> NDArrayComplex:
    """
    Stack the specified array of arrays along the last axis (tail)
    to produce a stacked complex array.

    Convenience wrapper around :func:`colour.utilities.tstack` that
    automatically uses ``DTYPE_COMPLEX_DEFAULT`` for complex number
    operations in *Transfer Matrix Method* calculations.

    Parameters
    ----------
    a
        Array of arrays to stack along the last axis.

    Returns
    -------
    :class:`numpy.ndarray`
        Stacked complex array.

    References
    ----------
    :cite:`Byrnes2016`
    """

    return tstack(a, dtype=DTYPE_COMPLEX_DEFAULT)  # type: ignore[arg-type]


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
    19.4712206...
    """

    n_1 = np.real(as_complex_array(n_1))
    n_2 = np.real(as_complex_array(n_2))
    theta_i = np.radians(as_float_array(theta_i))

    # Apply Snell's law: n_i * sin(theta_i) = n_j * sin(theta_j) (Byrnes Eq. 3)
    return np.degrees(np.arcsin(n_1 * np.sin(theta_i) / n_2))


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
    >>> polarised_light_magnitude_elements(1.0, 1.5, 0.0, 0.0)
    ((1+0j), (1+0j), (1.5+0j), (1.5+0j))
    """

    n_1 = as_complex_array(n_1)
    n_2 = as_complex_array(n_2)

    cos_theta_i = np.cos(np.radians(as_float_array(theta_i)))
    cos_theta_t = np.cos(np.radians(as_float_array(theta_t)))

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

    return _tstack_complex([r_s, r_p])


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
    array([ 0.04,  0.04])
    """

    # Reflectance: R = |r|^2 (Byrnes Eq. 23)
    R = np.abs(polarised_light_reflection_amplitude(n_1, n_2, theta_i, theta_t)) ** 2

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
    array([ 0.8+0.j,  0.8+0.j])
    """

    n_1_cos_theta_i, n_1_cos_theta_t, n_2_cos_theta_i, n_2_cos_theta_t = (
        polarised_light_magnitude_elements(n_1, n_2, theta_i, theta_t)
    )

    two_n_1_cos_theta_i = 2 * n_1_cos_theta_i

    # Fresnel transmission amplitudes (Byrnes Eq. 6)
    t_s = two_n_1_cos_theta_i / (n_1_cos_theta_i + n_2_cos_theta_t)
    t_p = two_n_1_cos_theta_i / (n_2_cos_theta_i + n_1_cos_theta_t)

    return _tstack_complex([t_s, t_p])


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

        T_s &= \\frac{n_2 \\cos \\theta_t}{n_1 \\cos \\theta_i} |t_s|^2 = \
\\frac{n_2 \\cos \\theta_t}{n_1 \\cos \\theta_i} \\left|\\frac{2n_1 \\cos \
\\theta_i}{n_1 \\cos \\theta_i + n_2 \\cos \\theta_t}\\right|^2 \\\\
        T_p &= \\frac{n_2 \\cos \\theta_t}{n_1 \\cos \\theta_i} |t_p|^2 = \
\\frac{n_2 \\cos \\theta_t}{n_1 \\cos \\theta_i} \\left|\\frac{2n_1 \\cos \
\\theta_i}{n_2 \\cos \\theta_i + n_1 \\cos \\theta_t}\\right|^2

    Where:

    -   :math:`T_s`: s-polarisation transmittance (fraction of incident power
        transmitted)
    -   :math:`T_p`: p-polarisation transmittance (fraction of incident power
        transmitted)
    -   :math:`t_s, t_p`: complex transmission amplitude coefficients
    -   The s-polarisation electric field is perpendicular to the plane of incidence
    -   The p-polarisation electric field is parallel to the plane of incidence

    The refractive index factor
    :math:`\\frac{n_2 \\cos \\theta_t}{n_1 \\cos \\theta_i}` accounts for the
    change in beam cross-section and energy density in the transmission medium.

    **Energy Conservation**: For non-absorbing media:
    :math:`R_s + T_s = 1` and :math:`R_p + T_p = 1`, where :math:`R_s, R_p` are the
    corresponding reflectance coefficients.

    The transmittance values satisfy: :math:`0 \\leq T_s, T_p \\leq 1`.

    References
    ----------
    :cite:`Byrnes2016`

    Examples
    --------
    >>> polarised_light_transmission_coefficient(1.0, 1.5, 0.0, 0.0)
    array([ 0.96+0.j,  0.96+0.j])
    """

    n_1 = as_complex_array(n_1)
    n_2 = as_complex_array(n_2)

    n_1_cos_theta_i, _n_1_cos_theta_t, _n_2_cos_theta_i, n_2_cos_theta_t = (
        polarised_light_magnitude_elements(n_1, n_2, theta_i, theta_t)
    )

    # Transmittance with beam cross-section correction (Byrnes Eq. 21-22)
    T = (n_2_cos_theta_t / n_1_cos_theta_i)[..., None] * np.abs(
        polarised_light_transmission_amplitude(n_1, n_2, theta_i, theta_t)
    ) ** 2
    return as_complex_array(T)


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
        (..., n_layers+2). Includes [incident, layer_1, ..., layer_n, substrate].
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
    theta = np.atleast_1d(as_float_array(theta))
    wavelength = np.atleast_1d(as_float_array(wavelength))

    angles_count = theta.shape[0]
    wavelengths_count = wavelength.shape[0]

    # Convert 1D n to column vector and tile across wavelengths
    # (M,) -> (M, 1) -> (M, W)
    if n.ndim == 1:
        n = np.transpose(np.atleast_2d(n))
        n = np.tile(n, (1, wavelengths_count))

    # (1, layers_count)
    if t.ndim == 1:
        t = t[np.newaxis, :]

    media_count = n.shape[0]
    layers_count = media_count - 2

    thickness_count = t.shape[0]

    n_0 = n[0, 0] if n.ndim == 2 else n[0]

    # Snell's law: n_i * sin(theta_i) = n_j * sin(theta_j) (Byrnes Eq. 3)
    # Broadcasting: theta (A,) → theta_media (A, M)
    theta_media = snell_law(
        n_0, (n[:, 0] if n.ndim == 2 else n)[:, None], theta[None, :]
    ).T

    # Fresnel coefficients (Byrnes Eq. 6)
    # Broadcasting: n (M, W), theta_media (A, M) → coefficients (A, M-1, W)
    n_1 = n[:-1, :]  # (M-1, W)
    n_2 = n[1:, :]  # (M-1, W)
    theta_1 = theta_media[:, :-1]  # (A, M-1)
    theta_2 = theta_media[:, 1:]  # (A, M-1)

    r_media_s, r_media_p = _tsplit_complex(
        polarised_light_reflection_amplitude(
            n_1[None, :, :],  # (1, M-1, W)
            n_2[None, :, :],  # (1, M-1, W)
            theta_1[:, :, None],  # (A, M-1, 1)
            theta_2[:, :, None],  # (A, M-1, 1)
        )
    )  # Output: (A, M-1, W)

    t_media_s, t_media_p = _tsplit_complex(
        polarised_light_transmission_amplitude(
            n_1[None, :, :],  # (1, M-1, W)
            n_2[None, :, :],  # (1, M-1, W)
            theta_1[:, :, None],  # (A, M-1, 1)
            theta_2[:, :, None],  # (A, M-1, 1)
        )
    )  # Output: (A, M-1, W)

    # Phase accumulation: delta = d * k_z (Byrnes Eq. 8)
    # Broadcasting directly in (W, A, T, L) order
    n_previous = n[0:layers_count, :]  # (L, W) - Media before each layer
    n_layer = n[1 : layers_count + 1, :]  # (L, W) - Each layer's refractive index
    theta_layer = theta_media[:, 0:layers_count]  # (A, L)

    theta_radians = np.radians(theta_layer)[:, :, None]  # (A, L, 1)
    k_z_layers = np.sqrt(
        n_layer[None, :, :] ** 2
        - n_previous[None, :, :] ** 2 * np.sin(theta_radians) ** 2
    )  # (A, L, W)

    # Compute phase: delta = (2π/λ) * d * k_z
    phase_factor = 2 * np.pi / wavelength[:, None, None, None]  # (W, 1, 1, 1)
    # Reshape k_z from (A, L, W) to (W, A, 1, L) for broadcasting with thickness
    k_z = np.transpose(k_z_layers, (2, 0, 1))[:, :, None, :]  # (W, A, 1, L)
    delta = phase_factor * t[None, None, :, :] * k_z  # (W, A, T, L)

    A = np.exp(1j * delta)  # (W, A, T, L)

    # Layer matrices: M_n = L_n * I_{n,n+1} (Byrnes Eq. 10-11)
    # (W, A, T, L, 2, 2, 2) for [wavelengths, angles, thickness, layers, 2x2, pol]
    M = zeros(
        (wavelengths_count, angles_count, thickness_count, layers_count, 2, 2, 2),
        dtype=DTYPE_COMPLEX_DEFAULT,  # pyright: ignore
    )

    r_s = r_media_s[:, 1 : layers_count + 1, :]  # (A, L, W)
    r_p = r_media_p[:, 1 : layers_count + 1, :]  # (A, L, W)
    t_s = t_media_s[:, 1 : layers_count + 1, :]  # (A, L, W)
    t_p = t_media_p[:, 1 : layers_count + 1, :]  # (A, L, W)

    # Broadcast Fresnel coefficients from (A, L, W) to (W, A, 1, L)
    # (A,L,W) -> (W,A,L) -> (W,A,1,L)
    r_s_b = np.transpose(r_s, (2, 0, 1))[:, :, None, :]
    r_p_b = np.transpose(r_p, (2, 0, 1))[:, :, None, :]
    t_s_b = np.transpose(t_s, (2, 0, 1))[:, :, None, :]
    t_p_b = np.transpose(t_p, (2, 0, 1))[:, :, None, :]

    M[:, :, :, :, 0, 0, 0] = 1 / (A * t_s_b)
    M[:, :, :, :, 0, 1, 0] = r_s_b / (A * t_s_b)
    M[:, :, :, :, 1, 0, 0] = A * r_s_b / t_s_b
    M[:, :, :, :, 1, 1, 0] = A / t_s_b

    M[:, :, :, :, 0, 0, 1] = 1 / (A * t_p_b)
    M[:, :, :, :, 0, 1, 1] = r_p_b / (A * t_p_b)
    M[:, :, :, :, 1, 0, 1] = A * r_p_b / t_p_b
    M[:, :, :, :, 1, 1, 1] = A / t_p_b

    # Initial interface matrix (Byrnes Eq. 11)
    # Shape: (W, A, T, 2, 2)
    M_s = zeros(
        (wavelengths_count, angles_count, thickness_count, 2, 2),
        dtype=DTYPE_COMPLEX_DEFAULT,  # pyright: ignore
    )
    # Fresnel coefficients at incident → first layer interface
    t_s_01 = t_media_s[:, 0, :]  # (A, W)
    r_s_01 = r_media_s[:, 0, :]  # (A, W)
    M_s[:, :, :, 0, 0] = (1 / t_s_01).T[:, :, None]  # (W, A, 1)
    M_s[:, :, :, 0, 1] = (r_s_01 / t_s_01).T[:, :, None]
    M_s[:, :, :, 1, 0] = (r_s_01 / t_s_01).T[:, :, None]
    M_s[:, :, :, 1, 1] = (1 / t_s_01).T[:, :, None]

    M_p = zeros(
        (wavelengths_count, angles_count, thickness_count, 2, 2),
        dtype=DTYPE_COMPLEX_DEFAULT,  # pyright: ignore
    )
    t_p_01 = t_media_p[:, 0, :]  # (A, W)
    r_p_01 = r_media_p[:, 0, :]  # (A, W)
    M_p[:, :, :, 0, 0] = (1 / t_p_01).T[:, :, None]
    M_p[:, :, :, 0, 1] = (r_p_01 / t_p_01).T[:, :, None]
    M_p[:, :, :, 1, 0] = (r_p_01 / t_p_01).T[:, :, None]
    M_p[:, :, :, 1, 1] = (1 / t_p_01).T[:, :, None]

    # Overall transfer matrix: M_tilde = I_01 @ M_1 @ M_2 @ ... (Byrnes Eq. 12)
    for i in range(layers_count):
        M_s = np.matmul(M_s, M[:, :, :, i, :, :, 0])
        M_p = np.matmul(M_p, M[:, :, :, i, :, :, 1])

    return TransferMatrixResult(
        M_s=M_s,
        M_p=M_p,
        theta=theta_media,
        n=n,
    )
