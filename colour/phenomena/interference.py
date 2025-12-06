"""
Thin Film Interference
======================

Provides support for thin film interference calculations and visualization.

-   :func:`colour.phenomena.light_water_molar_refraction_Schiebener1990`
-   :func:`colour.phenomena.light_water_refractive_index_Schiebener1990`
-   :func:`colour.phenomena.thin_film_tmm`
-   :func:`colour.phenomena.multilayer_tmm`

References
----------
-   :cite:`Byrnes2016` : Byrnes, S. J. (2016). Multilayer optical
    calculations. arXiv:1603.02720 [Physics].
    http://arxiv.org/abs/1603.02720
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from colour.phenomena.tmm import matrix_transfer_tmm
from colour.utilities import as_float_array, tstack

if TYPE_CHECKING:
    from colour.hints import ArrayLike, NDArrayFloat

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "light_water_molar_refraction_Schiebener1990",
    "light_water_refractive_index_Schiebener1990",
    "thin_film_tmm",
    "multilayer_tmm",
]


def light_water_molar_refraction_Schiebener1990(
    wavelength: ArrayLike,
    temperature: ArrayLike = 294,
    density: ArrayLike = 1000,
) -> NDArrayFloat:
    """
    Calculate water molar refraction using Schiebener et al. (1990) model.

    Parameters
    ----------
    wavelength : array_like
        Wavelength values :math:`\\lambda` in nanometers.
    temperature : float, optional
        Water temperature :math:`T` in Kelvin. Default is 294 K (21°C).
    density : float, optional
        Water density :math:`\\rho` in kg/m³. Default is 1000 kg/m³.

    Returns
    -------
    :class:`numpy.ndarray`
        Molar refraction values.

    Examples
    --------
    >>> light_water_molar_refraction_Schiebener1990(589)  # doctest: +ELLIPSIS
    0.2062114...
    >>> light_water_molar_refraction_Schiebener1990([400, 500, 600])
    ... # doctest: +ELLIPSIS
    array([ 0.2119202...,  0.2081386...,  0.2060235...])

    References
    ----------
    :cite:`Schiebener1990`
    """

    wl = as_float_array(wavelength) / 589
    T = as_float_array(temperature) / 273.15
    p = as_float_array(density) / 1000

    a_0 = 0.243905091
    a_1 = 9.53518094 * 10**-3
    a_2 = -3.64358110 * 10**-3
    a_3 = 2.65666426 * 10**-4
    a_4 = 1.59189325 * 10**-3
    a_5 = 2.45733798 * 10**-3
    a_6 = 0.897478251
    a_7 = -1.63066183 * 10**-2
    wl_UV = 0.2292020
    wl_IR = 5.432937

    wl_2 = wl**2

    return (
        a_0
        + a_1 * p
        + a_2 * T
        + a_3 * wl_2 * T
        + a_4 / wl_2
        + (a_5 / (wl_2 - wl_UV**2))
        + (a_6 / (wl_2 - wl_IR**2))
        + a_7 * p**2
    )


def light_water_refractive_index_Schiebener1990(
    wavelength: ArrayLike,
    temperature: ArrayLike = 294,
    density: ArrayLike = 1000,
) -> NDArrayFloat:
    """
    Calculate water refractive index using Schiebener et al. (1990) model.

    Parameters
    ----------
    wavelength : array_like
        Wavelength values :math:`\\lambda` in nanometers.
    temperature : float, optional
        Water temperature :math:`T` in Kelvin. Default is 294 K (21°C).
    density : float, optional
        Water density :math:`\\rho` in kg/m³. Default is 1000 kg/m³.

    Returns
    -------
    :class:`numpy.ndarray`
        Refractive index values for water.

    Examples
    --------
    >>> light_water_refractive_index_Schiebener1990(
    ...     [400, 500, 600]
    ... )  # doctest: +ELLIPSIS
    array([ 1.3441433...,  1.3373637...,  1.3335851...])

    References
    ----------
    :cite:`Schiebener1990`
    """

    p_s = as_float_array(density) / 1000

    LL = light_water_molar_refraction_Schiebener1990(wavelength, temperature, density)

    return np.sqrt((2 * LL + 1 / p_s) / (1 / p_s - LL))


def thin_film_tmm(
    n: ArrayLike,
    t: ArrayLike,
    wavelength: ArrayLike,
    theta: ArrayLike = 0,
) -> tuple[NDArrayFloat, NDArrayFloat]:
    """
    Calculate thin film reflectance and transmittance using *Transfer Matrix Method*.

    Unified function that returns both R and T in a single call, matching the
    approach used by Byrnes' tmm and TMM-Fast packages. Supports **outer product
    broadcasting** and wavelength-dependent refractive index (dispersion).

    Parameters
    ----------
    n : array_like
        Complete refractive index stack :math:`n_j` for single-layer film. Shape:
        (3,) or (3, wavelengths_count). The array should contain
        [n_incident, n_film, n_substrate].

        For example: constant n ``[1.0, 1.5, 1.0]`` (air | film | air), or
        dispersive n ``[[1.0, 1.0, 1.0], [1.52, 1.51, 1.50], [1.0, 1.0, 1.0]]``
        for wavelength-dependent film refractive index.
    t : array_like
        Film thickness :math:`d` in nanometers. Can be:

        - **Scalar**: Single thickness value (e.g., ``250``) → shape
          ``(W, A, 1, 2)``
        - **1D array**: Multiple thickness values (e.g., ``[200, 250, 300]``)
          for **thickness sweeps** via outer product broadcasting → shape
          ``(W, A, T, 2)``

        When an array is provided, the function computes reflectance and
        transmittance for ALL combinations of thickness x wavelength x angle
        values.
    wavelength : array_like
        Wavelength values :math:`\\lambda` in nanometers. Can be scalar or array.
    theta : array_like, optional
        Incident angle :math:`\\theta` in degrees. Scalar or array of shape
        (angles_count,) for angle broadcasting. Default is 0 (normal incidence).

    Returns
    -------
    tuple
        (R, T) where:

        - **R**: Reflectance, :class:`numpy.ndarray`, shape **(W, A, T, 2)**
          for [R_s, R_p]
        - **T**: Transmittance, :class:`numpy.ndarray`, shape **(W, A, T, 2)**
          for [T_s, T_p]

        where **W** = number of wavelengths, **A** = number of angles,
        **T** = number of thicknesses (the **Spectroscopy Convention**)

    Examples
    --------
    Basic usage:

    >>> R, T = thin_film_tmm([1.0, 1.5, 1.0], 250, 555)
    >>> R.shape, T.shape
    ((1, 1, 1, 2), (1, 1, 1, 2))
    >>> R[0, 0, 0]  # [R_s, R_p] at 555nm, shape (W, A, T, 2) = (1, 1, 1, 2)
    ... # doctest: +ELLIPSIS
    array([ 0.1215919...,  0.1215919...])
    >>> np.allclose(R + T, 1.0)  # Energy conservation
    True

    Multiple wavelengths:

    >>> R, T = thin_film_tmm([1.0, 1.5, 1.0], 250, [400, 500, 600])
    >>> R.shape  # (W, A, T, 2) = (3 wavelengths, 1 angle, 1 thickness, 2 pols)
    (3, 1, 1, 2)

    **Thickness sweep** (outer product broadcasting):

    >>> R, T = thin_film_tmm([1.0, 1.5, 1.0], [200, 250, 300], [400, 500, 600])
    >>> R.shape  # (W, A, T, 2) = (3 wavelengths, 1 angle, 3 thicknesses, 2 pols)
    (3, 1, 3, 2)
    >>> # Access via R[wl_idx, ang_idx, thick_idx, pol_idx]
    >>> # R[0, 0, 0] = reflectance at λ=400nm, thickness=200nm
    >>> # R[1, 0, 1] = reflectance at λ=500nm, thickness=250nm

    **Angle broadcasting**:

    >>> R, T = thin_film_tmm([1.0, 1.5, 1.0], 250, [400, 500, 600], [0, 30, 45, 60])
    >>> R.shape  # (W, A, T, 2) = (3 wavelengths, 4 angles, 1 thickness, 2 pols)
    (3, 4, 1, 2)
    >>> # R[0, 0, 0] = reflectance at λ=400nm, θ=0°
    >>> # R[1, 2, 0] = reflectance at λ=500nm, θ=45°

    **Dispersion**: wavelength-dependent refractive index:

    >>> wavelengths = [400, 500, 600]
    >>> n_dispersive = [[1.0, 1.0, 1.0], [1.52, 1.51, 1.50], [1.0, 1.0, 1.0]]
    >>> R, T = thin_film_tmm(n_dispersive, 250, wavelengths)
    >>> R.shape  # (W, A, T, 2) = (3 wavelengths, 1 angle, 1 thickness, 2 pols)
    (3, 1, 1, 2)

    Notes
    -----
    -   **Thickness broadcasting** (outer product): When ``t`` is an array with
        multiple values, ALL combinations of thickness x wavelength x angle are
        computed. For example: 3 thicknesses x 5 wavelengths x 2 angles = 30
        total calculations, returned in shape ``(W, A, T, 2)`` = ``(5, 2, 3, 2)``.

        This differs from multilayer specification where thickness specifies
        one value per layer (e.g., ``[250, 150]`` for a 2-layer stack).

    -   **Spectroscopy Convention**: Output arrays use wavelength-first ordering
        ``(W, A, T, 2)`` which is natural for spectroscopy applications where
        you typically iterate over wavelengths in the outer loop.

    -   **Dispersion support**: If ``n`` is 2D, the second dimension must match the
        ``wavelength`` array length. Each wavelength uses its corresponding n value.

    -   **Energy conservation**: For non-absorbing media, R + T = 1.

    -   Supports complex refractive indices for absorbing materials (e.g., metals).

    -   For absorbing media: R + T < 1 (absorption A = 1 - R - T).

    References
    ----------
    :cite:`Byrnes2016`
    """

    t = np.atleast_1d(as_float_array(t))

    # Handle thickness broadcasting: reshape from (T,) to (T, 1) for single-layer
    t = t[:, np.newaxis] if len(t) > 1 else t

    return multilayer_tmm(n=n, t=t, wavelength=wavelength, theta=theta)


def multilayer_tmm(
    n: ArrayLike,
    t: ArrayLike,
    wavelength: ArrayLike,
    theta: ArrayLike = 0,
) -> tuple[NDArrayFloat, NDArrayFloat]:
    """
    Calculate multilayer reflectance and transmittance using *Transfer Matrix Method*.

    Unified function that returns both R and T in a single call, eliminating duplication
    and matching industry-standard TMM implementations. Computes both values from the
    same transfer matrix for efficiency.

    Parameters
    ----------
    n : array_like
        Complete refractive index stack :math:`n_j` including incident medium,
        layers, and substrate. Shape: (media_count,) or
        (media_count, wavelengths_count). Can be complex for absorbing
        materials. The array should contain [n_incident, n_layer_1, ...,
        n_layer_n, n_substrate].

        For example: single layer ``[1.0, 1.5, 1.0]`` (air | film | air),
        two layers ``[1.0, 1.5, 2.0, 1.0]`` (air | film1 | film2 | air), or
        with dispersion ``[[1.0, 1.0, 1.0], [1.52, 1.51, 1.50], [1.0, 1.0, 1.0]]``
        for wavelength-dependent n.
    t : array_like
        Thicknesses of each layer :math:`t_j` in nanometers (excluding incident
        and substrate). Shape: (layers_count,).

        **Important**: This parameter specifies ONE thickness value per layer
        in the multilayer stack. It does NOT perform thickness sweeps.

        For example: single layer ``[250]``, two-layer stack ``[250, 150]``,
        three-layer stack ``[100, 200, 100]``.

        **For thickness sweeps**, use :func:`thin_film_tmm` with an array of
        thickness values (e.g., ``[200, 250, 300]``), which computes all
        combinations via outer product broadcasting.
    wavelength : array_like
        Wavelength values :math:`\\lambda` in nanometers.
    theta : array_like, optional
        Incident angle :math:`\\theta` in degrees. Scalar or array of shape
        (angles_count,) for angle broadcasting. Default is 0 (normal incidence).

    Returns
    -------
    tuple
        (R, T) where:

        - **R**: Reflectance, :class:`numpy.ndarray`, shape **(W, A, 1, 2)**
          for [R_s, R_p]
        - **T**: Transmittance, :class:`numpy.ndarray`, shape **(W, A, 1, 2)**
          for [T_s, T_p]

        where **W** = number of wavelengths, **A** = number of angles
        (the **Spectroscopy Convention**). The thickness dimension is always 1
        for multilayer stacks.

    Examples
    --------
    Single layer:

    >>> R, T = multilayer_tmm([1.0, 1.5, 1.0], [250], 555)
    >>> R.shape, T.shape
    ((1, 1, 1, 2), (1, 1, 1, 2))
    >>> np.allclose(R + T, 1.0)  # Energy conservation
    True

    Two-layer stack:

    >>> R, T = multilayer_tmm([1.0, 1.5, 2.0, 1.0], [250, 150], 555)
    >>> R.shape  # (W, A, T, 2) = (1 wavelength, 1 angle, 1 thickness, 2 pols)
    (1, 1, 1, 2)

    Multiple wavelengths:

    >>> R, T = multilayer_tmm([1.0, 1.5, 1.0], [250], [400, 500, 600])
    >>> R.shape  # (W, A, T, 2) = (3 wavelengths, 1 angle, 1 thickness, 2 pols)
    (3, 1, 1, 2)

    Multiple angles (angle broadcasting):

    >>> R, T = multilayer_tmm([1.0, 1.5, 1.0], [250], [400, 500, 600], [0, 30, 45, 60])
    >>> R.shape  # (W, A, T, 2) = (3 wavelengths, 4 angles, 1 thickness, 2 pols)
    (3, 4, 1, 2)

    Notes
    -----
    -   The reflectance is calculated from (*Equation 15* from :cite:`Byrnes2016`):

    .. math::

        r = \\frac{\\tilde{M}_{10}}{\\tilde{M}_{00}}, \\quad R = |r|^2

    -   The transmittance is calculated from (*Equations 14 and 21-22* from
        :cite:`Byrnes2016`):

    .. math::

        t = \\frac{1}{\\tilde{M}_{00}}

    .. math::

        T = |t|^2 \\frac{\\text{Re}[n_{\\text{substrate}} \
\\cos \\theta_{\\text{final}}]}{\\text{Re}[n_{\\text{incident}} \\cos \\theta_i]}

    Where:

    -   :math:`\\tilde{M}`: Overall transfer matrix for the multilayer stack
    -   :math:`r, t`: Complex reflection and transmission amplitude coefficients
    -   :math:`R, T`: Reflectance and transmittance (fraction of incident power)

    -   **Energy conservation**: For non-absorbing media, R + T = 1.
    -   Supports complex refractive indices for absorbing materials (e.g., metals).
    -   For absorbing media: R + T < 1 (absorption A = 1 - R - T).

    References
    ----------
    :cite:`Byrnes2016`
    """

    theta = np.atleast_1d(as_float_array(theta))

    result = matrix_transfer_tmm(
        n=n,
        t=np.atleast_1d(as_float_array(t)),
        theta=theta,
        wavelength=wavelength,
    )

    # Extract n_incident and n_substrate from result.n
    # result.n has shape (media_count, wavelengths_count)
    n_incident = result.n[0, 0]
    n_substrate = result.n[-1, 0]

    # T = thickness_count, A = angles_count, W = wavelengths_count
    # Reflectance (Byrnes Eq. 15, 23)
    r_s = np.abs(result.M_s[:, :, :, 1, 0] / result.M_s[:, :, :, 0, 0]) ** 2
    r_p = np.abs(result.M_p[:, :, :, 1, 0] / result.M_p[:, :, :, 0, 0]) ** 2

    # Transmittance (Byrnes Eq. 14, 21-22)
    t_s = 1 / result.M_s[:, :, :, 0, 0]
    t_p = 1 / result.M_p[:, :, :, 0, 0]

    # Transmittance correction factor: Re[n_f cos(θ_f) / n_i cos(θ_i)]
    # result.theta has shape (A, M) where M = media_count
    cos_theta_i = np.cos(np.radians(theta))[:, None]  # (A, 1)
    cos_theta_f = np.cos(np.radians(result.theta[:, -1]))[:, None]  # (A, 1)
    transmittance_correction = np.real(
        (n_substrate * cos_theta_f) / (n_incident * cos_theta_i)
    )  # (A, 1)

    # Broadcast to thickness dimension: (1, A, 1)
    transmittance_correction = transmittance_correction[None, :, :]

    t_s = np.abs(t_s) ** 2 * transmittance_correction  # (T, A, W)
    t_p = np.abs(t_p) ** 2 * transmittance_correction  # (T, A, W)

    # Stack results: (T, A, W, 2)
    R = tstack([r_s, r_p])
    T = tstack([t_s, t_p])

    return R, T
