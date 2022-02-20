"""
Rösch-MacAdam colour solid - Visible Spectrum Volume Computations
=================================================================

Defines the objects related to *Rösch-MacAdam* colour solid, visible spectrum
volume computations.

References
----------
-   :cite:`Lindbloom2015` : Lindbloom, B. (2015). About the Lab Gamut.
    Retrieved August 20, 2018, from
    http://www.brucelindbloom.com/LabGamutDisplayHelp.html
-   :cite:`Mansencal2018` : Mansencal, T. (2018). How is the visible gamut
    bounded? Retrieved August 19, 2018, from
    https://stackoverflow.com/a/48396021/931625
-   :cite:`Martinez-Verdu2007` : Martínez-Verdú, F., Perales, E., Chorro, E.,
    de Fez, D., Viqueira, V., & Gilabert, E. (2007). Computation and
    visualization of the MacAdam limits for any lightness, hue angle, and light
    source. Journal of the Optical Society of America A, 24(6), 1501.
    doi:10.1364/JOSAA.24.001501
"""

from __future__ import annotations

import numpy as np

from colour.colorimetry import (
    MultiSpectralDistributions,
    SpectralDistribution,
    SpectralShape,
    handle_spectral_arguments,
    msds_to_XYZ,
)
from colour.constants import DEFAULT_FLOAT_DTYPE
from colour.hints import (
    Any,
    ArrayLike,
    Boolean,
    Dict,
    Floating,
    Integer,
    Literal,
    NDArray,
    Optional,
    Union,
)
from colour.volume import is_within_mesh_volume
from colour.utilities import CACHE_REGISTRY, zeros, validate_method

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "SPECTRAL_SHAPE_OUTER_SURFACE_XYZ",
    "generate_pulse_waves",
    "XYZ_outer_surface",
    "solid_RoschMacAdam",
    "is_within_visible_spectrum",
]

SPECTRAL_SHAPE_OUTER_SURFACE_XYZ: SpectralShape = SpectralShape(360, 780, 5)
"""
Default spectral shape according to *ASTM E308-15* practise shape but using an
interval of 5.
"""

_CACHE_OUTER_SURFACE_XYZ: Dict = CACHE_REGISTRY.register_cache(
    f"{__name__}._CACHE_OUTER_SURFACE_XYZ"
)

_CACHE_OUTER_SURFACE_XYZ_POINTS: Dict = CACHE_REGISTRY.register_cache(
    f"{__name__}._CACHE_OUTER_SURFACE_XYZ_POINTS"
)


def generate_pulse_waves(
    bins: Integer,
    pulse_order: Union[Literal["Bins", "Pulse Wave Width"], str] = "Bins",
    filter_jagged_pulses: Boolean = False,
) -> NDArray:
    """
    Generate the pulse waves of given number of bins necessary to totally
    stimulate the colour matching functions and produce the *Rösch-MacAdam*
    colour solid.

    Assuming 5 bins, a first set of SPDs would be as follows::

        1 0 0 0 0
        0 1 0 0 0
        0 0 1 0 0
        0 0 0 1 0
        0 0 0 0 1

    The second one::

        1 1 0 0 0
        0 1 1 0 0
        0 0 1 1 0
        0 0 0 1 1
        1 0 0 0 1

    The third:

        1 1 1 0 0
        0 1 1 1 0
        0 0 1 1 1
        1 0 0 1 1
        1 1 0 0 1

    Etc...

    Parameters
    ----------
    bins
        Number of bins of the pulse waves.
    pulse_order
        Method for ordering the pulse waves. *Bins* is the default order, with
        *Pulse Wave Width* ordering, instead of iterating over the pulse wave
        widths first, iteration occurs over the bins, producing blocks of pulse
        waves with increasing width.
    filter_jagged_pulses
        Whether to filter jagged pulses. When ``pulse_order`` is set to
        *Pulse Wave Width*, the pulses are ordered by increasing width. Because
        of the discrete nature of the underlying signal, the resulting pulses
        will be jagged. For example assuming 5 bins, the center block with
        the two extreme values added would be as follows::

            0 0 0 0 0
            0 0 1 0 0
            0 0 1 1 0 <--
            0 1 1 1 0
            0 1 1 1 1 <--
            1 1 1 1 1

        Setting the ``filter_jagged_pulses`` parameter to `True` will result
        in the removal of the two marked pulse waves above thus avoiding jagged
        lines when plotting and having to resort to excessive ``bins`` values.

    Returns
    -------
    :class:`numpy.ndarray`
        Pulse waves.

    References
    ----------
    :cite:`Lindbloom2015`, :cite:`Mansencal2018`, :cite:`Martinez-Verdu2007`

    Examples
    --------
    >>> generate_pulse_waves(5)
    array([[ 0.,  0.,  0.,  0.,  0.],
           [ 1.,  0.,  0.,  0.,  0.],
           [ 0.,  1.,  0.,  0.,  0.],
           [ 0.,  0.,  1.,  0.,  0.],
           [ 0.,  0.,  0.,  1.,  0.],
           [ 0.,  0.,  0.,  0.,  1.],
           [ 1.,  1.,  0.,  0.,  0.],
           [ 0.,  1.,  1.,  0.,  0.],
           [ 0.,  0.,  1.,  1.,  0.],
           [ 0.,  0.,  0.,  1.,  1.],
           [ 1.,  0.,  0.,  0.,  1.],
           [ 1.,  1.,  1.,  0.,  0.],
           [ 0.,  1.,  1.,  1.,  0.],
           [ 0.,  0.,  1.,  1.,  1.],
           [ 1.,  0.,  0.,  1.,  1.],
           [ 1.,  1.,  0.,  0.,  1.],
           [ 1.,  1.,  1.,  1.,  0.],
           [ 0.,  1.,  1.,  1.,  1.],
           [ 1.,  0.,  1.,  1.,  1.],
           [ 1.,  1.,  0.,  1.,  1.],
           [ 1.,  1.,  1.,  0.,  1.],
           [ 1.,  1.,  1.,  1.,  1.]])
    >>> generate_pulse_waves(5, 'Pulse Wave Width')
    array([[ 0.,  0.,  0.,  0.,  0.],
           [ 1.,  0.,  0.,  0.,  0.],
           [ 1.,  1.,  0.,  0.,  0.],
           [ 1.,  1.,  0.,  0.,  1.],
           [ 1.,  1.,  1.,  0.,  1.],
           [ 0.,  1.,  0.,  0.,  0.],
           [ 0.,  1.,  1.,  0.,  0.],
           [ 1.,  1.,  1.,  0.,  0.],
           [ 1.,  1.,  1.,  1.,  0.],
           [ 0.,  0.,  1.,  0.,  0.],
           [ 0.,  0.,  1.,  1.,  0.],
           [ 0.,  1.,  1.,  1.,  0.],
           [ 0.,  1.,  1.,  1.,  1.],
           [ 0.,  0.,  0.,  1.,  0.],
           [ 0.,  0.,  0.,  1.,  1.],
           [ 0.,  0.,  1.,  1.,  1.],
           [ 1.,  0.,  1.,  1.,  1.],
           [ 0.,  0.,  0.,  0.,  1.],
           [ 1.,  0.,  0.,  0.,  1.],
           [ 1.,  0.,  0.,  1.,  1.],
           [ 1.,  1.,  0.,  1.,  1.],
           [ 1.,  1.,  1.,  1.,  1.]])
    >>> generate_pulse_waves(5, 'Pulse Wave Width', True)
    array([[ 0.,  0.,  0.,  0.,  0.],
           [ 1.,  0.,  0.,  0.,  0.],
           [ 1.,  1.,  0.,  0.,  1.],
           [ 0.,  1.,  0.,  0.,  0.],
           [ 1.,  1.,  1.,  0.,  0.],
           [ 0.,  0.,  1.,  0.,  0.],
           [ 0.,  1.,  1.,  1.,  0.],
           [ 0.,  0.,  0.,  1.,  0.],
           [ 0.,  0.,  1.,  1.,  1.],
           [ 0.,  0.,  0.,  0.,  1.],
           [ 1.,  0.,  0.,  1.,  1.],
           [ 1.,  1.,  1.,  1.,  1.]])
    """

    pulse_order = validate_method(
        pulse_order,
        ["Bins", "Pulse Wave Width"],
        '"{0}" pulse order is invalid, it must be one of {1}!',
    )

    square_waves = []
    square_waves_basis = np.tril(
        np.ones((bins, bins), dtype=DEFAULT_FLOAT_DTYPE)
    )[0:-1, :]

    if pulse_order.lower() == "bins":
        for square_wave_basis in square_waves_basis:
            for i in range(bins):
                square_waves.append(np.roll(square_wave_basis, i))
    else:
        for i in range(bins):
            for j, square_wave_basis in enumerate(square_waves_basis):
                square_waves.append(np.roll(square_wave_basis, i - j // 2))

        if filter_jagged_pulses:
            square_waves = square_waves[::2]

    return np.vstack(
        [
            zeros(bins),
            np.vstack(square_waves),
            np.ones(bins, dtype=DEFAULT_FLOAT_DTYPE),
        ]
    )


def XYZ_outer_surface(
    cmfs: Optional[MultiSpectralDistributions] = None,
    illuminant: Optional[SpectralDistribution] = None,
    point_order: Union[Literal["Bins", "Pulse Wave Width"], str] = "Bins",
    filter_jagged_points: Boolean = False,
    **kwargs: Any,
) -> NDArray:
    """
    Generate the *Rösch-MacAdam* colour solid, i.e. *CIE XYZ* colourspace
    outer surface, for given colour matching functions using multi-spectral
    conversion of pulse waves to *CIE XYZ* tristimulus values.

    Parameters
    ----------
    cmfs
        Standard observer colour matching functions, default to the
        *CIE 1931 2 Degree Standard Observer*.
    illuminant
        Illuminant spectral distribution, default to *CIE Illuminant E*.
    point_order
        Method for ordering the underlying pulse waves used to generate the
        *Rösch-MacAdam* colour solid. *Bins* is the default order, with
        *Pulse Wave Width* ordering, instead of iterating over the pulse wave
        widths first, iteration occurs over the bins, producing blocks of pulse
        waves with increasing width.
    filter_jagged_points
        Whether to filter the underlying jagged pulses. When ``point_order`` is
        set to *Pulse Wave Width*, the pulses are ordered by increasing width.
        Because of the discrete nature of the underlying signal, the resulting
        pulses will be jagged. For example assuming 5 bins, the center block
        with the two extreme values added would be as follows::

            0 0 0 0 0
            0 0 1 0 0
            0 0 1 1 0 <--
            0 1 1 1 0
            0 1 1 1 1 <--
            1 1 1 1 1

        Setting the ``filter_jagged_points`` parameter to `True` will result
        in the removal of the two marked pulse waves above thus avoiding jagged
        lines when plotting and having to resort to excessive ``bins`` values.

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.msds_to_XYZ`},
        See the documentation of the previously listed definition.

    Returns
    -------
    :class:`numpy.ndarray`
        *Rösch-MacAdam* colour solid, *CIE XYZ* outer surface tristimulus
        values.

    References
    ----------
    :cite:`Lindbloom2015`, :cite:`Mansencal2018`, :cite:`Martinez-Verdu2007`

    Examples
    --------
    >>> from colour import MSDS_CMFS, SPECTRAL_SHAPE_DEFAULT
    >>> shape = SpectralShape(
    ...     SPECTRAL_SHAPE_DEFAULT.start, SPECTRAL_SHAPE_DEFAULT.end, 84
    ... )
    >>> cmfs = MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
    >>> XYZ_outer_surface(cmfs.copy().align(shape))  # doctest: +ELLIPSIS
    array([[  0.0000000...e+00,   0.0000000...e+00,   0.0000000...e+00],
           [  9.6361381...e-05,   2.9056776...e-06,   4.4961226...e-04],
           [  2.5910529...e-01,   2.1031298...e-02,   1.3207468...e+00],
           [  1.0561021...e-01,   6.2038243...e-01,   3.5423571...e-02],
           [  7.2647980...e-01,   3.5460869...e-01,   2.1005149...e-04],
           [  1.0971874...e-02,   3.9635453...e-03,   0.0000000...e+00],
           [  3.0792572...e-05,   1.1119762...e-05,   0.0000000...e+00],
           [  2.5920165...e-01,   2.1034203...e-02,   1.3211965...e+00],
           [  3.6471551...e-01,   6.4141373...e-01,   1.3561704...e+00],
           [  8.3209002...e-01,   9.7499113...e-01,   3.5633622...e-02],
           [  7.3745167...e-01,   3.5857224...e-01,   2.1005149...e-04],
           [  1.1002667...e-02,   3.9746651...e-03,   0.0000000...e+00],
           [  1.2715395...e-04,   1.4025439...e-05,   4.4961226...e-04],
           [  3.6481187...e-01,   6.4141663...e-01,   1.3566200...e+00],
           [  1.0911953...e+00,   9.9602242...e-01,   1.3563805...e+00],
           [  8.4306189...e-01,   9.7895467...e-01,   3.5633622...e-02],
           [  7.3748247...e-01,   3.5858336...e-01,   2.1005149...e-04],
           [  1.1099028...e-02,   3.9775708...e-03,   4.4961226...e-04],
           [  2.5923244...e-01,   2.1045323...e-02,   1.3211965...e+00],
           [  1.0912916...e+00,   9.9602533...e-01,   1.3568301...e+00],
           [  1.1021671...e+00,   9.9998597...e-01,   1.3563805...e+00],
           [  8.4309268...e-01,   9.7896579...e-01,   3.5633622...e-02],
           [  7.3757883...e-01,   3.5858626...e-01,   6.5966375...e-04],
           [  2.7020432...e-01,   2.5008868...e-02,   1.3211965...e+00],
           [  3.6484266...e-01,   6.4142775...e-01,   1.3566200...e+00],
           [  1.1022635...e+00,   9.9998888...e-01,   1.3568301...e+00],
           [  1.1021979...e+00,   9.9999709...e-01,   1.3563805...e+00],
           [  8.4318905...e-01,   9.7896870...e-01,   3.6083235...e-02],
           [  9.9668412...e-01,   3.7961756...e-01,   1.3214065...e+00],
           [  3.7581454...e-01,   6.4539130...e-01,   1.3566200...e+00],
           [  1.0913224...e+00,   9.9603645...e-01,   1.3568301...e+00],
           [  1.1022943...e+00,   1.0000000...e+00,   1.3568301...e+00]])
    """

    cmfs, illuminant = handle_spectral_arguments(
        cmfs,
        illuminant,
        "CIE 1931 2 Degree Standard Observer",
        "E",
        SPECTRAL_SHAPE_OUTER_SURFACE_XYZ,
    )

    settings = dict(kwargs)
    settings.update({"shape": cmfs.shape})

    key = (
        hash(cmfs),
        hash(illuminant),
        point_order,
        filter_jagged_points,
        str(settings),
    )
    XYZ = _CACHE_OUTER_SURFACE_XYZ.get(key)

    if XYZ is None:
        pulse_waves = generate_pulse_waves(
            len(cmfs.wavelengths), point_order, filter_jagged_points
        )
        XYZ = (
            msds_to_XYZ(
                pulse_waves, cmfs, illuminant, method="Integration", **settings
            )
            / 100
        )

        _CACHE_OUTER_SURFACE_XYZ[key] = XYZ

    return XYZ


solid_RoschMacAdam = XYZ_outer_surface


def is_within_visible_spectrum(
    XYZ: ArrayLike,
    cmfs: Optional[MultiSpectralDistributions] = None,
    illuminant: Optional[SpectralDistribution] = None,
    tolerance: Optional[Floating] = None,
    **kwargs: Any,
) -> NDArray:
    """
    Return whether given *CIE XYZ* tristimulus values are within the visible
    spectrum volume, i.e. *Rösch-MacAdam* colour solid, for given colour
    matching functions and illuminant.

    Parameters
    ----------
    XYZ
        *CIE XYZ* tristimulus values.
    cmfs
        Standard observer colour matching functions, default to the
        *CIE 1931 2 Degree Standard Observer*.
    illuminant
        Illuminant spectral distribution, default to *CIE Illuminant E*.
    tolerance
        Tolerance allowed in the inside-triangle check.

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.msds_to_XYZ`},
        See the documentation of the previously listed definition.

    Returns
    -------
    :class:`numpy.ndarray`
        Are *CIE XYZ* tristimulus values within the visible spectrum volume,
        i.e. *Rösch-MacAdam* colour solid.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``XYZ``    | [0, 1]                | [0, 1]        |
    +------------+-----------------------+---------------+

    Examples
    --------
    >>> import numpy as np
    >>> is_within_visible_spectrum(np.array([0.3205, 0.4131, 0.51]))
    array(True, dtype=bool)
    >>> a = np.array([[0.3205, 0.4131, 0.51],
    ...               [-0.0005, 0.0031, 0.001]])
    >>> is_within_visible_spectrum(a)
    array([ True, False], dtype=bool)
    """

    cmfs, illuminant = handle_spectral_arguments(
        cmfs,
        illuminant,
        "CIE 1931 2 Degree Standard Observer",
        "E",
        SPECTRAL_SHAPE_OUTER_SURFACE_XYZ,
    )

    key = (hash(cmfs), hash(illuminant), str(kwargs))
    vertices = _CACHE_OUTER_SURFACE_XYZ_POINTS.get(key)

    if vertices is None:
        _CACHE_OUTER_SURFACE_XYZ_POINTS[key] = vertices = solid_RoschMacAdam(
            cmfs, illuminant, **kwargs
        )

    return is_within_mesh_volume(XYZ, vertices, tolerance)
