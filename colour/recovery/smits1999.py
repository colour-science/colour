"""
Smits (1999) - Reflectance Recovery
===================================

Define objects for reflectance recovery using the *Smits (1999)* method.

References
----------
-   :cite:`Smits1999a` : Smits, B. (1999). An RGB-to-Spectrum Conversion for
    Reflectances. Journal of Graphics Tools, 4(4), 11-22.
    doi:10.1080/10867651.1999.10487511
"""

from __future__ import annotations

import typing
from typing import TYPE_CHECKING

from colour.colorimetry import (
    CCS_ILLUMINANTS,
    MultiSpectralDistributions,
    SpectralDistribution,
)
from colour.models import RGB_Colourspace, RGB_COLOURSPACE_sRGB, XYZ_to_RGB
from colour.recovery import MSDS_SMITS1999
from colour.utilities import (
    array_namespace,
    as_float_array,
    optional,
    to_domain_1,
    xp_as_float_array,
    xp_atleast_2d,
    xp_matrix_transpose,
    xp_reshape,
)

if TYPE_CHECKING:
    from colour.hints import ArrayLike, Domain1, Literal, NDArrayFloat, Range1

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "PRIMARIES_SMITS1999",
    "WHITEPOINT_NAME_SMITS1999",
    "CCS_WHITEPOINT_SMITS1999",
    "RGB_COLOURSPACE_SMITS1999",
    "XYZ_to_RGB_Smits1999",
    "RGB_to_msds_Smits1999",
    "RGB_to_sd_Smits1999",
]

PRIMARIES_SMITS1999: NDArrayFloat = RGB_COLOURSPACE_sRGB.primaries
"""*Smits (1999)* method implementation colourspace primaries."""

WHITEPOINT_NAME_SMITS1999 = "E"
"""*Smits (1999)* method implementation colourspace whitepoint name."""

CCS_WHITEPOINT_SMITS1999: NDArrayFloat = CCS_ILLUMINANTS[
    "CIE 1931 2 Degree Standard Observer"
][WHITEPOINT_NAME_SMITS1999]
"""*Smits (1999)* method implementation colourspace whitepoint."""

RGB_COLOURSPACE_SMITS1999 = RGB_Colourspace(
    "Smits 1999",
    PRIMARIES_SMITS1999,
    CCS_WHITEPOINT_SMITS1999,
    WHITEPOINT_NAME_SMITS1999,
)
RGB_COLOURSPACE_SMITS1999.__doc__ = """
*Smits (1999)* colourspace.

References
----------
:cite:`Smits1999a`,
"""


def XYZ_to_RGB_Smits1999(XYZ: Domain1) -> Range1:
    """
    Convert from *CIE XYZ* tristimulus values to *RGB* colourspace using
    the conditions required by the current *Smits (1999)* method
    implementation.

    Parameters
    ----------
    XYZ
        *CIE XYZ* tristimulus values.

    Returns
    -------
    :class:`numpy.ndarray`
        *RGB* colour array.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``XYZ``    | 1                     | 1             |
    +------------+-----------------------+---------------+

    +------------+-----------------------+---------------+
    | **Range**  | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``RGB``    | 1                     | 1             |
    +------------+-----------------------+---------------+

    Examples
    --------
    >>> import numpy as np
    >>> XYZ = np.array([0.21781186, 0.12541048, 0.04697113])
    >>> XYZ_to_RGB_Smits1999(XYZ)  # doctest: +ELLIPSIS
    array([0.4063959..., 0.0275289..., 0.0398219...])
    """

    return XYZ_to_RGB(XYZ, RGB_COLOURSPACE_SMITS1999)


@typing.overload
def RGB_to_msds_Smits1999(
    RGB: ArrayLike,
    basis: MultiSpectralDistributions | None = None,
    *,
    as_array: Literal[False] = False,
) -> MultiSpectralDistributions: ...


@typing.overload
def RGB_to_msds_Smits1999(
    RGB: ArrayLike,
    basis: MultiSpectralDistributions | None = None,
    *,
    as_array: Literal[True],
) -> NDArrayFloat: ...


def RGB_to_msds_Smits1999(
    RGB: ArrayLike,
    basis: MultiSpectralDistributions | None = None,
    *,
    as_array: bool = False,
) -> MultiSpectralDistributions | NDArrayFloat:
    """
    Recover the multi-spectral distributions from the specified *RGB*
    colourspace array using the *Smits (1999)* decomposition algorithm.

    This is a vectorised implementation supporting multi-dimensional arrays.

    Parameters
    ----------
    RGB
        *RGB* colourspace array to recover spectral values from. The last
        dimension must be size 3.
    basis
        Multi-spectral distributions basis with signals: white, cyan, magenta,
        yellow, red, green, blue. Defaults to :attr:`MSDS_SMITS1999`.
    as_array
        Whether to return raw spectral values as a
        :class:`numpy.ndarray` of shape
        ``(*RGB.shape[:-1], wavelengths)`` instead of a
        :class:`MultiSpectralDistributions` instance. Defaults to *False*.

    Returns
    -------
    :class:`MultiSpectralDistributions` or :class:`numpy.ndarray`
        Recovered multi-spectral distributions, or the underlying
        spectral values when ``as_array=True``.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``RGB``    | 1                     | 1             |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Smits1999a`

    Examples
    --------
    >>> import numpy as np
    >>> RGB = np.array(
    ...     [
    ...         [0.45623196, 0.03080455, 0.04093343],
    ...         [0.05438271, 0.29877169, 0.07188444],
    ...         [0.01863137, 0.05139773, 0.28887675],
    ...     ]
    ... )
    >>> RGB_to_msds_Smits1999(RGB, as_array=True).shape
    (3, 10)
    >>> float(RGB_to_msds_Smits1999(RGB, as_array=True)[0, 0])
    ... # doctest: +ELLIPSIS
    0.0829...
    """

    basis = optional(basis, MSDS_SMITS1999)

    RGB = to_domain_1(as_float_array(RGB))

    xp = array_namespace(RGB)

    shape = RGB.shape
    RGB = xp_atleast_2d(xp_reshape(RGB, (-1, 3), xp=xp), xp=xp)

    label_to_index = {label: index for index, label in enumerate(basis.labels)}
    basis_values = xp_as_float_array(basis.values, xp=xp, like=RGB)

    # *Smits (1999)* decomposes any *RGB* into three basis spectra
    # weighted by the sorted ``(min, mid, max)`` channel deltas. The
    # paper expresses this as a six-branch ``if`` ladder over the
    # orderings of *R*, *G*, *B*:
    #
    #   min  max  | secondary  primary  | spectrum
    #   ---- ---- | ---------- -------- | ---------------------------------
    #   R    B    | cyan       blue     | white*R + cyan*(G - R) + blue*(B - G)
    #   R    G    | cyan       green    | white*R + cyan*(B - R) + green*(G - B)
    #   G    B    | magenta    blue     | white*G + magenta*(R - G) + blue*(B - R)
    #   G    R    | magenta    red      | white*G + magenta*(B - G) + red*(R - B)
    #   B    G    | yellow     green    | white*B + yellow*(R - B) + green*(G - R)
    #   B    R    | yellow     red      | white*B + yellow*(G - B) + red*(R - G)
    #
    # Every branch reduces to ``white*min + secondary*(mid - min) +
    # primary*(max - mid)``; ``argmin`` / ``argmax`` over the stacked
    # candidate bases collapse the six-way switch to a single gather.
    white = basis_values[:, label_to_index["white"]]
    secondary_bases = xp.stack(
        [
            basis_values[:, label_to_index["cyan"]],
            basis_values[:, label_to_index["magenta"]],
            basis_values[:, label_to_index["yellow"]],
        ]
    )
    primary_bases = xp.stack(
        [
            basis_values[:, label_to_index["red"]],
            basis_values[:, label_to_index["green"]],
            basis_values[:, label_to_index["blue"]],
        ]
    )

    min_channel = xp.min(RGB, axis=-1)
    max_channel = xp.max(RGB, axis=-1)
    mid_channel = xp.sum(RGB, axis=-1) - min_channel - max_channel
    secondary = secondary_bases[xp.argmin(RGB, axis=-1)]
    primary = primary_bases[xp.argmax(RGB, axis=-1)]

    spectra = (
        white * min_channel[..., None]
        + secondary * (mid_channel - min_channel)[..., None]
        + primary * (max_channel - mid_channel)[..., None]
    )

    spectra = xp_reshape(spectra, [*list(shape[:-1]), len(white)], xp=xp)

    if as_array:
        return spectra

    # ``MultiSpectralDistributions`` expects ``(n_wavelengths,
    # n_samples)``; flatten any leading shape into a single sample axis.
    msds_values = xp_matrix_transpose(
        xp_reshape(spectra, (-1, len(white)), xp=xp), xp=xp
    )

    return MultiSpectralDistributions(msds_values, basis.wavelengths)


def RGB_to_sd_Smits1999(
    RGB: Domain1,
    basis: MultiSpectralDistributions | None = None,
    name: str | None = None,
) -> SpectralDistribution:
    """
    Generate a spectral distribution from *RGB* values using the
    *Smits (1999)* decomposition algorithm.

    Parameters
    ----------
    RGB
        *RGB* colourspace array to recover the spectral distribution from.
    basis
        Multi-spectral distributions basis with signals: white, cyan, magenta,
        yellow, red, green, blue. Defaults to :attr:`MSDS_SMITS1999`.
    name
        Name for the resulting spectral distribution.

    Returns
    -------
    :class:`colour.SpectralDistribution`
        Recovered spectral distribution.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``RGB``    | 1                     | 1             |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Smits1999a`

    Examples
    --------
    >>> import numpy as np
    >>> from colour import MSDS_CMFS, SDS_ILLUMINANTS, SpectralShape
    >>> from colour.colorimetry import sd_to_XYZ_integration
    >>> from colour.utilities import numpy_print_options
    >>> XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
    >>> RGB = XYZ_to_RGB_Smits1999(XYZ)
    >>> cmfs = (
    ...     MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
    ...     .copy()
    ...     .align(SpectralShape(360, 780, 10))
    ... )
    >>> illuminant = SDS_ILLUMINANTS["E"].copy().align(cmfs.shape)
    >>> sd = RGB_to_sd_Smits1999(RGB)
    >>> with numpy_print_options(suppress=True):
    ...     sd  # doctest: +ELLIPSIS
    SpectralDistribution([[380.        ,   0.0787830...],
                          [417.7778    ,   0.0622018...],
                          [455.5556    ,   0.0446206...],
                          [493.3333    ,   0.0352220...],
                          [531.1111    ,   0.0324149...],
                          [568.8889    ,   0.0330105...],
                          [606.6667    ,   0.3207115...],
                          [644.4444    ,   0.3836164...],
                          [682.2222    ,   0.3836164...],
                          [720.        ,   0.3835649...]],
                         LinearInterpolator,
                         {},
                         Extrapolator,
                         {'method': 'Constant', 'left': None, 'right': None})
    >>> sd_to_XYZ_integration(sd, cmfs, illuminant) / 100  # doctest: +ELLIPSIS
    array([0.1894770..., 0.1126470..., 0.0474420...])
    """

    basis = optional(basis, MSDS_SMITS1999)
    name = optional(name, f"Smits (1999) - {RGB!r}")

    values = RGB_to_msds_Smits1999(RGB, basis, as_array=True)

    return SpectralDistribution(
        values,
        basis.wavelengths,
        name=name,
        interpolator=basis.interpolator,
        interpolator_kwargs=basis.interpolator_kwargs,
    )
