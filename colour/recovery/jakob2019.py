"""
Jakob and Hanika (2019) - Reflectance Recovery
==============================================

Define the objects for reflectance recovery, i.e., spectral upsampling, using
*Jakob and Hanika (2019)* method.

-   :func:`colour.recovery.sd_Jakob2019`
-   :func:`colour.recovery.find_coefficients_Jakob2019`
-   :func:`colour.recovery.XYZ_to_sd_Jakob2019`
-   :class:`colour.recovery.LUT3D_Jakob2019`

References
----------
-   :cite:`Jakob2019` : Jakob, W., & Hanika, J. (2019). A Low-Dimensional
    Function Space for Efficient Spectral Upsampling. Computer Graphics Forum,
    38(2), 147-155. doi:10.1111/cgf.13626
"""

from __future__ import annotations

import itertools
import struct
import typing

import numpy as np

from colour.algebra import smoothstep_function, spow
from colour.colorimetry import (
    MultiSpectralDistributions,
    SpectralDistribution,
    SpectralShape,
    handle_spectral_arguments,
    intermediate_lightness_function_CIE1976,
    sd_to_XYZ_integration,
)
from colour.constants import DTYPE_INT_DEFAULT
from colour.difference import JND_CIE1976

if typing.TYPE_CHECKING:
    from scipy.interpolate import RegularGridInterpolator

    from colour.hints import (
        Callable,
        Literal,
        PathLike,
        Tuple,
    )

from colour.hints import ArrayLike, Domain1, NDArrayFloat, cast
from colour.models import RGB_Colourspace, RGB_to_XYZ, XYZ_to_Lab, XYZ_to_xy
from colour.utilities import (
    array_namespace,
    as_float_array,
    as_float_scalar,
    as_ndarray,
    domain_range_scale,
    full,
    index_along_last_axis,
    is_tqdm_installed,
    message_box,
    optional,
    required,
    to_domain_1,
    tsplit,
    xp_as_float_array,
    xp_astype,
    xp_linspace,
    xp_matrix_transpose,
    xp_reshape,
    zeros,
)

from ._warnings import warn_if_optimisation_detaches

if is_tqdm_installed():
    from tqdm import tqdm
else:  # pragma: no cover
    from unittest import mock

    tqdm = mock.MagicMock()

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "SPECTRAL_SHAPE_JAKOB2019",
    "StopMinimizationEarlyError",
    "sd_Jakob2019",
    "error_function",
    "dimensionalise_coefficients",
    "lightness_scale",
    "find_coefficients_Jakob2019",
    "XYZ_to_sd_Jakob2019",
    "LUT3D_Jakob2019",
]

SPECTRAL_SHAPE_JAKOB2019: SpectralShape = SpectralShape(360, 780, 5)
"""Spectral shape for *Jakob and Hanika (2019)* method."""


class StopMinimizationEarlyError(Exception):
    """
    Define an exception to halt :func:`scipy.optimize.minimize` when the
    minimized function value becomes sufficiently small.

    *SciPy* does not currently provide a native mechanism for early
    termination based on function value thresholds.

    Attributes
    ----------
    -   :attr:`~colour.recovery.jakob2019.StopMinimizationEarlyError.coefficients`
    -   :attr:`~colour.recovery.jakob2019.StopMinimizationEarlyError.error`
    """

    def __init__(self, coefficients: ArrayLike, error: float) -> None:
        self._coefficients = as_float_array(coefficients)
        self._error = as_float_scalar(error)

    @property
    def coefficients(self) -> NDArrayFloat:
        """
        Getter for the *Jakob and Hanika (2019)* exception coefficients.

        Returns
        -------
        :class:`numpy.ndarray`
            *Jakob and Hanika (2019)* exception coefficients.
        """

        return self._coefficients

    @property
    def error(self) -> float:
        """
        Getter for the *Jakob and Hanika (2019)* spectral upsampling error
        value.

        Returns
        -------
        :class:`float`
            *Jakob and Hanika (2019)* spectral upsampling error value
            representing the quality of the coefficient fitting process.
        """

        return self._error


def sd_Jakob2019(
    coefficients: ArrayLike, shape: SpectralShape = SPECTRAL_SHAPE_JAKOB2019
) -> SpectralDistribution:
    """
    Generate a spectral distribution using the spectral model specified by
    *Jakob and Hanika (2019)*.

    Parameters
    ----------
    coefficients
        Dimensionless coefficients for the *Jakob and Hanika (2019)*
        reflectance spectral model.
    shape
        Shape used by the spectral distribution.

    Returns
    -------
    :class:`colour.SpectralDistribution`
        *Jakob and Hanika (2019)* spectral distribution.

    References
    ----------
    :cite:`Jakob2019`

    Examples
    --------
    >>> from colour.utilities import numpy_print_options
    >>> with numpy_print_options(suppress=True):
    ...     sd_Jakob2019([-9e-05, 8.5e-02, -20], SpectralShape(400, 700, 20))
    ...     # doctest: +ELLIPSIS
    SpectralDistribution([[400.        ,   0.3143046...],
                          [420.        ,   0.4133320...],
                          [440.        ,   0.4880034...],
                          [460.        ,   0.5279562...],
                          [480.        ,   0.5319346...],
                          [500.        ,   0.5      ...],
                          [520.        ,   0.4326202...],
                          [540.        ,   0.3373544...],
                          [560.        ,   0.2353056...],
                          [580.        ,   0.1507665...],
                          [600.        ,   0.0931332...],
                          [620.        ,   0.0577434...],
                          [640.        ,   0.0367011...],
                          [660.        ,   0.0240879...],
                          [680.        ,   0.0163316...],
                          [700.        ,   0.0114118...]],
                         SpragueInterpolator,
                         {},
                         Extrapolator,
                         {'method': 'Constant', 'left': None, 'right': None})
    """

    coefficients = as_float_array(coefficients)
    c_0, c_1, c_2 = coefficients

    xp = array_namespace(coefficients)

    wl = shape.wavelengths
    U = c_0 * wl**2 + c_1 * wl + c_2
    R = 1 / 2 + U / (2 * xp.sqrt(1 + U**2))

    name = f"{coefficients!r} (COEFF) - Jakob (2019)"

    return SpectralDistribution(R, wl, name=name)


@typing.overload
def error_function(
    coefficients: ArrayLike,
    target: ArrayLike,
    cmfs: MultiSpectralDistributions,
    illuminant: SpectralDistribution,
    max_error: float | None = ...,
    additional_data: Literal[True] = True,
) -> Tuple[float, NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat]: ...
@typing.overload
def error_function(
    coefficients: ArrayLike,
    target: ArrayLike,
    cmfs: MultiSpectralDistributions,
    illuminant: SpectralDistribution,
    max_error: float | None = ...,
    *,
    additional_data: Literal[False],
) -> Tuple[float, NDArrayFloat]: ...
@typing.overload
def error_function(
    coefficients: ArrayLike,
    target: ArrayLike,
    cmfs: MultiSpectralDistributions,
    illuminant: SpectralDistribution,
    max_error: float | None,
    additional_data: Literal[False],
) -> Tuple[float, NDArrayFloat]: ...
def error_function(
    coefficients: ArrayLike,
    target: ArrayLike,
    cmfs: MultiSpectralDistributions,
    illuminant: SpectralDistribution,
    max_error: float | None = None,
    additional_data: bool = False,
) -> (
    Tuple[float, NDArrayFloat]
    | Tuple[float, NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat]
):
    """
    Compute :math:`\\Delta E_{76}` between the target colour and the colour
    defined by the specified spectral model, along with its gradient.

    Parameters
    ----------
    coefficients
        Dimensionless coefficients for *Jakob and Hanika (2019)*
        reflectance spectral model.
    target
        *CIE L\\*a\\*b\\** colourspace array of the target colour.
    cmfs
        Standard observer colour matching functions.
    illuminant
        Illuminant spectral distribution.
    max_error
        Raise ``StopMinimizationEarlyError`` if the error is smaller than
        this. The default is *None* and the function doesn't raise
        anything.
    additional_data
        If *True*, some intermediate calculations are returned, for use in
        correctness tests: R, XYZ and Lab.

    Returns
    -------
    :class:`tuple` or :class:`tuple`
        Tuple of computed :math:`\\Delta E_{76}` error and gradient of
        error, i.e., the first derivatives of error with respect to the
        input coefficients or tuple of computed :math:`\\Delta E_{76}`
        error, gradient of error, computed spectral reflectance, *CIE XYZ*
        tristimulus values corresponding to ``R`` and *CIE L\\*a\\*b\\**
        colourspace array corresponding to ``XYZ``.

    Raises
    ------
    StopMinimizationEarlyError
        Raised when the error is below ``max_error``.
    """

    target = as_float_array(target)

    c_0, c_1, c_2 = as_float_array(coefficients)

    xp = array_namespace(target)

    # ``xp_linspace`` anchors the dtype to the *Colour* default rather than
    # the backend default, which would otherwise be float32 for stock
    # *PyTorch* and degrade the optimisation.
    wv = cast(
        "NDArrayFloat", xp_linspace(0, 1, num=len(cmfs.shape), xp=xp, like=target)
    )

    U = c_0 * wv**2 + c_1 * wv + c_2
    t1 = xp.sqrt(1 + U**2)
    R = 1 / 2 + U / (2 * t1)

    t2 = 1 / (2 * t1) - U**2 / (2 * t1**3)
    # ``xp.stack`` is used rather than ``xp_as_float_array`` because the
    # elements are already backend arrays and a backend such as *PyTorch*
    # rejects a list of tensors passed to ``asarray``.
    dR = xp.stack([wv**2 * t2, wv * t2, t2])

    XYZ = sd_to_XYZ_integration(R, cmfs, illuminant, shape=cmfs.shape) / 100
    dXYZ = xp_matrix_transpose(
        sd_to_XYZ_integration(dR, cmfs, illuminant, shape=cmfs.shape) / 100,
        xp=xp,
    )

    XYZ_n = sd_to_XYZ_integration(illuminant, cmfs)
    XYZ_n = XYZ_n / XYZ_n[1]
    XYZ_n = xp_as_float_array(XYZ_n, xp=xp, like=XYZ)
    XYZ_XYZ_n = XYZ / XYZ_n

    XYZ_f = intermediate_lightness_function_CIE1976(XYZ, XYZ_n)
    dXYZ_f = xp.where(
        XYZ_XYZ_n[..., None] > (24 / 116) ** 3,
        1 / (3 * spow(XYZ_n[..., None], 1 / 3) * spow(XYZ[..., None], 2 / 3)) * dXYZ,
        (841 / 108) * dXYZ / XYZ_n[..., None],
    )

    def intermediate_XYZ_to_Lab(
        XYZ_i: NDArrayFloat, offset: float | None = 16
    ) -> NDArrayFloat:
        """
        Return the final intermediate value for the *CIE Lab* to *CIE XYZ*
        conversion.
        """

        return xp.stack(
            [
                116 * XYZ_i[1] - offset,
                500 * (XYZ_i[0] - XYZ_i[1]),
                200 * (XYZ_i[1] - XYZ_i[2]),
            ]
        )

    Lab_i = intermediate_XYZ_to_Lab(XYZ_f)
    dLab_i = intermediate_XYZ_to_Lab(dXYZ_f, 0)

    error = as_float_scalar(xp.sqrt(xp.sum((Lab_i - target) ** 2)))
    if max_error is not None and error <= max_error:
        raise StopMinimizationEarlyError(coefficients, error)

    derror = xp.sum(dLab_i * (Lab_i[..., None] - target[..., None]), axis=0) / error

    if additional_data:
        return error, derror, R, XYZ, Lab_i

    return error, derror


def dimensionalise_coefficients(
    coefficients: ArrayLike, shape: SpectralShape
) -> NDArrayFloat:
    """
    Rescale dimensionless coefficients to the specified spectral shape.

    A dimensionless form of the reflectance spectral model is used in the
    optimisation process. Instead of the usual spectral shape, specified in
    nanometres, it is normalised to the [0, 1] range. A side effect is
    that computed coefficients work only with the normalised range and
    need to be rescaled to regain units and be compatible with standard
    shapes.

    Parameters
    ----------
    coefficients
        Dimensionless coefficients.
    shape
        Spectral distribution shape used in calculations.

    Returns
    -------
    :class:`numpy.ndarray`
        Dimensionful coefficients, with units of
        :math:`\\frac{1}{\\mathrm{nm}^2}`, :math:`\\frac{1}{\\mathrm{nm}}`
        and 1, respectively.
    """

    xp = array_namespace(coefficients)

    cp_0, cp_1, cp_2 = tsplit(coefficients)

    span = shape.end - shape.start

    c_0 = cp_0 / span**2
    c_1 = cp_1 / span - 2 * cp_0 * shape.start / span**2
    c_2 = cp_0 * shape.start**2 / span**2 - cp_1 * shape.start / span + cp_2

    return xp_as_float_array([c_0, c_1, c_2], xp=xp)


def lightness_scale(steps: int) -> NDArrayFloat:
    """
    Generate a non-linear lightness scale as described in *Jakob and Hanika
    (2019)*.

    The scale reduces spacing between very dark and very bright (and
    saturated) colours, providing finer resolution in regions where
    coefficients change rapidly.

    Parameters
    ----------
    steps
        Number of samples along the non-linear lightness scale.

    Returns
    -------
    :class:`numpy.ndarray`
        Non-linear lightness scale array.

    Examples
    --------
    >>> lightness_scale(5)  # doctest: +ELLIPSIS
    array([0.        , 0.0656127..., 0.5       , 0.9343872..., 1.        ])
    """

    xp = array_namespace()

    linear = xp.linspace(0, 1, steps)

    return smoothstep_function(smoothstep_function(linear))


@required("SciPy")
def find_coefficients_Jakob2019(
    XYZ: ArrayLike,
    cmfs: MultiSpectralDistributions | None = None,
    illuminant: SpectralDistribution | None = None,
    coefficients_0: ArrayLike = (0, 0, 0),
    max_error: float = JND_CIE1976 / 100,
    dimensionalise: bool = True,
) -> Tuple[NDArrayFloat, float]:
    """
    Find the coefficients for the *Jakob and Hanika (2019)* reflectance
    spectral model.

    Parameters
    ----------
    XYZ
        *CIE XYZ* tristimulus values to find the coefficients for.
    cmfs
        Standard observer colour matching functions, default to the
        *CIE 1931 2 Degree Standard Observer*.
    illuminant
        Illuminant spectral distribution, default to
        *CIE Standard Illuminant D65*.
    coefficients_0
        Starting coefficients for the solver.
    max_error
        Maximal acceptable error. Set higher to save computational time.
        If *None*, the solver will keep going until it is very close to
        the minimum. The default is ``ACCEPTABLE_DELTA_E``.
    dimensionalise
        If *True*, returned coefficients are dimensionful and will not
        work correctly if fed back as ``coefficients_0``. The default
        is *True*.

    Returns
    -------
    :class:`tuple`
        Tuple of computed coefficients that best fit the specified
        colour and :math:`\\Delta E_{76}` between the target colour and
        the colour corresponding to the computed coefficients.

    References
    ----------
    :cite:`Jakob2019`

    Examples
    --------
    >>> XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
    >>> find_coefficients_Jakob2019(XYZ)  # doctest: +ELLIPSIS
    (array([ 1.3723791...e-04, -1.3514399...e-01,  3.0838973...e+01]), \
0.0141941...)
    """

    from scipy.optimize import minimize  # noqa: PLC0415

    XYZ = as_float_array(XYZ)
    coefficients_0 = as_float_array(coefficients_0)

    cmfs, illuminant = handle_spectral_arguments(
        cmfs, illuminant, shape_default=SPECTRAL_SHAPE_JAKOB2019
    )

    def optimize(
        target_o: NDArrayFloat, coefficients_0_o: NDArrayFloat
    ) -> Tuple[NDArrayFloat, float | np.float64]:
        """Minimise the error function using *L-BFGS-B* method."""

        try:
            result = minimize(
                error_function,
                coefficients_0_o,
                (target_o, cmfs, illuminant, max_error),
                method="L-BFGS-B",
                jac=True,
            )
        except StopMinimizationEarlyError as error:
            return error.coefficients, error.error
        else:
            return result.x, result.fun

    xy_n = XYZ_to_xy(sd_to_XYZ_integration(illuminant, cmfs))

    XYZ_g = full(3, 0.5)
    coefficients_g = zeros(3)

    divisions = 3
    while divisions < 10:
        XYZ_r = XYZ_g
        coefficient_r = coefficients_g
        keep_divisions = False

        coefficients_0 = coefficient_r
        for i in range(1, divisions):
            XYZ_i = (XYZ - XYZ_r) * i / (divisions - 1) + XYZ_r
            Lab_i = XYZ_to_Lab(XYZ_i)

            coefficients_0, error = optimize(Lab_i, coefficients_0)

            if error > max_error:
                break
            XYZ_g = XYZ_i
            coefficients_g = coefficients_0
            keep_divisions = True
        else:
            break

        if not keep_divisions:
            divisions += 2

    target = XYZ_to_Lab(XYZ, xy_n)
    coefficients, error = optimize(target, coefficients_0)

    if dimensionalise:
        coefficients = dimensionalise_coefficients(coefficients, cmfs.shape)

    return coefficients, float(error)


@typing.overload
def XYZ_to_sd_Jakob2019(
    XYZ: Domain1,
    cmfs: MultiSpectralDistributions | None = ...,
    illuminant: SpectralDistribution | None = ...,
    optimisation_kwargs: dict | None = ...,
    additional_data: Literal[True] = True,
) -> Tuple[SpectralDistribution, float]: ...


@typing.overload
def XYZ_to_sd_Jakob2019(
    XYZ: Domain1,
    cmfs: MultiSpectralDistributions | None = ...,
    illuminant: SpectralDistribution | None = ...,
    optimisation_kwargs: dict | None = ...,
    *,
    additional_data: Literal[False],
) -> SpectralDistribution: ...


@typing.overload
def XYZ_to_sd_Jakob2019(
    XYZ: Domain1,
    cmfs: MultiSpectralDistributions | None,
    illuminant: SpectralDistribution | None,
    optimisation_kwargs: dict | None,
    additional_data: Literal[False],
) -> SpectralDistribution: ...


def XYZ_to_sd_Jakob2019(
    XYZ: Domain1,
    cmfs: MultiSpectralDistributions | None = None,
    illuminant: SpectralDistribution | None = None,
    optimisation_kwargs: dict | None = None,
    additional_data: bool = False,
) -> Tuple[SpectralDistribution, float] | SpectralDistribution:
    """
    Recover the spectral distribution from the specified *CIE XYZ* tristimulus
    values using *Jakob and Hanika (2019)* method.

    Parameters
    ----------
    XYZ
        *CIE XYZ* tristimulus values to recover the spectral distribution
        from.
    cmfs
        Standard observer colour matching functions, default to the
        *CIE 1931 2 Degree Standard Observer*.
    illuminant
        Illuminant spectral distribution, default to
        *CIE Standard Illuminant D65*.
    optimisation_kwargs
        Parameters for :func:`colour.recovery.find_coefficients_Jakob2019`
        definition.
    additional_data
        If *True*, ``error`` will be returned alongside the recovered
        spectral distribution.

    Returns
    -------
    :class:`tuple` or :class:`colour.SpectralDistribution`
        Tuple of recovered spectral distribution and :math:`\\Delta E_{76}`
        between the target colour and the colour corresponding to the
        computed coefficients or recovered spectral distribution.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``XYZ``    | 1                     | 1             |
    +------------+-----------------------+---------------+

    References
    ----------
    :cite:`Jakob2019`

    Examples
    --------
    >>> from colour import (
    ...     CCS_ILLUMINANTS,
    ...     MSDS_CMFS,
    ...     SDS_ILLUMINANTS,
    ...     XYZ_to_sRGB,
    ... )
    >>> from colour.colorimetry import sd_to_XYZ_integration
    >>> from colour.utilities import numpy_print_options  # noqa: PLC0415
    >>> XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
    >>> cmfs = (
    ...     MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
    ...     .copy()
    ...     .align(SpectralShape(360, 780, 10))
    ... )
    >>> illuminant = SDS_ILLUMINANTS["D65"].copy().align(cmfs.shape)
    >>> sd = XYZ_to_sd_Jakob2019(XYZ, cmfs, illuminant)
    >>> with numpy_print_options(suppress=True):
    ...     sd  # doctest: +ELLIPSIS
    SpectralDistribution([[360.        ,   0.4893773...],
                          [370.        ,   0.3258214...],
                          [380.        ,   0.2147792...],
                          [390.        ,   0.1482413...],
                          [400.        ,   0.1086169...],
                          [410.        ,   0.0841255...],
                          [420.        ,   0.0683114...],
                          [430.        ,   0.0577144...],
                          [440.        ,   0.0504267...],
                          [450.        ,   0.0453552...],
                          [460.        ,   0.0418520...],
                          [470.        ,   0.0395259...],
                          [480.        ,   0.0381430...],
                          [490.        ,   0.0375741...],
                          [500.        ,   0.0377685...],
                          [510.        ,   0.0387432...],
                          [520.        ,   0.0405871...],
                          [530.        ,   0.0434783...],
                          [540.        ,   0.0477225...],
                          [550.        ,   0.0538256...],
                          [560.        ,   0.0626314...],
                          [570.        ,   0.0755869...],
                          [580.        ,   0.0952675...],
                          [590.        ,   0.1264265...],
                          [600.        ,   0.1779272...],
                          [610.        ,   0.2649393...],
                          [620.        ,   0.4039779...],
                          [630.        ,   0.5832105...],
                          [640.        ,   0.7445440...],
                          [650.        ,   0.8499970...],
                          [660.        ,   0.9094792...],
                          [670.        ,   0.9425378...],
                          [680.        ,   0.9616376...],
                          [690.        ,   0.9732481...],
                          [700.        ,   0.9806562...],
                          [710.        ,   0.9855873...],
                          [720.        ,   0.9889903...],
                          [730.        ,   0.9914117...],
                          [740.        ,   0.9931801...],
                          [750.        ,   0.9945009...],
                          [760.        ,   0.9955066...],
                          [770.        ,   0.9962855...],
                          [780.        ,   0.9968976...]],
                         SpragueInterpolator,
                         {},
                         Extrapolator,
                         {'method': 'Constant', 'left': None, 'right': None})
    >>> sd_to_XYZ_integration(sd, cmfs, illuminant) / 100  # doctest: +ELLIPSIS
    array([0.2066217..., 0.1220128..., 0.0513958...])
    """

    warn_if_optimisation_detaches(XYZ, "Jakob 2019")

    XYZ = to_domain_1(XYZ)

    XYZ = as_ndarray(XYZ)

    cmfs, illuminant = handle_spectral_arguments(
        cmfs, illuminant, shape_default=SPECTRAL_SHAPE_JAKOB2019
    )

    optimisation_kwargs = optional(optimisation_kwargs, {})

    with domain_range_scale("ignore"):
        coefficients, error = find_coefficients_Jakob2019(
            XYZ, cmfs, illuminant, **optimisation_kwargs
        )

    sd = sd_Jakob2019(coefficients, cmfs.shape)
    sd.name = f"{XYZ} (XYZ) - Jakob (2019)"

    if additional_data:
        return sd, error

    return sd


class LUT3D_Jakob2019:
    """
    Define a class for working with pre-computed lookup tables for the
    *Jakob and Hanika (2019)* spectral upsampling method. This class
    enables significant time savings by performing expensive numerical
    optimisation ahead of time and storing the results in a file.

    The file format is compatible with the code and *\\*.coeff* files in the
    supplemental material published alongside the article. These files are
    directly available from
    `Colour - Datasets <https://github.com/colour-science/colour-datasets>`__
    under the record *4050598*.

    Attributes
    ----------
    -   :attr:`~colour.recovery.LUT3D_Jakob2019.size`
    -   :attr:`~colour.recovery.LUT3D_Jakob2019.lightness_scale`
    -   :attr:`~colour.recovery.LUT3D_Jakob2019.coefficients`
    -   :attr:`~colour.recovery.LUT3D_Jakob2019.interpolator`

    Methods
    -------
    -   :meth:`~colour.recovery.LUT3D_Jakob2019.__init__`
    -   :meth:`~colour.recovery.LUT3D_Jakob2019.generate`
    -   :meth:`~colour.recovery.LUT3D_Jakob2019.RGB_to_coefficients`
    -   :meth:`~colour.recovery.LUT3D_Jakob2019.RGB_to_sd`
    -   :meth:`~colour.recovery.LUT3D_Jakob2019.read`
    -   :meth:`~colour.recovery.LUT3D_Jakob2019.write`

    References
    ----------
    :cite:`Jakob2019`

    Examples
    --------
    >>> import os
    >>> import colour
    >>> from colour import CCS_ILLUMINANTS, SDS_ILLUMINANTS, MSDS_CMFS
    >>> from colour.colorimetry import sd_to_XYZ_integration
    >>> from colour.models import RGB_COLOURSPACE_sRGB
    >>> from colour.utilities import numpy_print_options
    >>> cmfs = (
    ...     MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
    ...     .copy()
    ...     .align(SpectralShape(360, 780, 10))
    ... )
    >>> illuminant = SDS_ILLUMINANTS["D65"].copy().align(cmfs.shape)
    >>> LUT = LUT3D_Jakob2019()
    >>> LUT.generate(RGB_COLOURSPACE_sRGB, cmfs, illuminant, 3, lambda x: x)
    >>> path = os.path.join(
    ...     colour.__path__[0],
    ...     "recovery",
    ...     "tests",
    ...     "resources",
    ...     "sRGB_Jakob2019.coeff",
    ... )
    >>> LUT.write(path)  # doctest: +SKIP
    >>> LUT.read(path)  # doctest: +SKIP
    >>> RGB = np.array([0.70573936, 0.19248266, 0.22354169])
    >>> with numpy_print_options(suppress=True):
    ...     LUT.RGB_to_sd(RGB, cmfs.shape)  # doctest: +ELLIPSIS
    SpectralDistribution([[360.        ,   0.7666803...],
                          [370.        ,   0.6251547...],
                          [380.        ,   0.4584310...],
                          [390.        ,   0.3161633...],
                          [400.        ,   0.2196155...],
                          [410.        ,   0.1596575...],
                          [420.        ,   0.1225525...],
                          [430.        ,   0.0989784...],
                          [440.        ,   0.0835782...],
                          [450.        ,   0.0733535...],
                          [460.        ,   0.0666049...],
                          [470.        ,   0.0623569...],
                          [480.        ,   0.06006  ...],
                          [490.        ,   0.0594383...],
                          [500.        ,   0.0604201...],
                          [510.        ,   0.0631195...],
                          [520.        ,   0.0678648...],
                          [530.        ,   0.0752834...],
                          [540.        ,   0.0864790...],
                          [550.        ,   0.1033773...],
                          [560.        ,   0.1293883...],
                          [570.        ,   0.1706018...],
                          [580.        ,   0.2374178...],
                          [590.        ,   0.3439472...],
                          [600.        ,   0.4950548...],
                          [610.        ,   0.6604253...],
                          [620.        ,   0.7914669...],
                          [630.        ,   0.8738724...],
                          [640.        ,   0.9213216...],
                          [650.        ,   0.9486880...],
                          [660.        ,   0.9650550...],
                          [670.        ,   0.9752838...],
                          [680.        ,   0.9819499...],
                          [690.        ,   0.9864585...],
                          [700.        ,   0.9896073...],
                          [710.        ,   0.9918680...],
                          [720.        ,   0.9935302...],
                          [730.        ,   0.9947778...],
                          [740.        ,   0.9957312...],
                          [750.        ,   0.9964714...],
                          [760.        ,   0.9970543...],
                          [770.        ,   0.9975190...],
                          [780.        ,   0.9978936...]],
                         SpragueInterpolator,
                         {},
                         Extrapolator,
                         {'method': 'Constant', 'left': None, 'right': None})
    """

    def __init__(self) -> None:
        from scipy.interpolate import RegularGridInterpolator  # noqa: PLC0415

        self._interpolator = RegularGridInterpolator((), np.array([]))

        self._size: int = 0
        self._lightness_scale: NDArrayFloat = np.array([])
        self._coefficients: NDArrayFloat = np.array([])

    @property
    def size(self) -> int:
        """
        Getter for the *Jakob and Hanika (2019)* interpolator size.

        The size represents the sample count on one side of the 3D lookup
        table used for spectral upsampling.

        Returns
        -------
        :class:`int`
            *Jakob and Hanika (2019)* interpolator size.
        """

        return self._size

    @property
    def lightness_scale(self) -> NDArrayFloat:
        """
        Getter for the *Jakob and Hanika (2019)* interpolator lightness
        scale.

        Returns
        -------
        :class:`numpy.ndarray`
            *Jakob and Hanika (2019)* interpolator lightness scale.
        """

        return self._lightness_scale

    @property
    def coefficients(self) -> NDArrayFloat:
        """
        Getter for the *Jakob and Hanika (2019)* interpolator coefficients.

        Returns
        -------
        :class:`numpy.ndarray`
            *Jakob and Hanika (2019)* interpolator coefficients.
        """

        return self._coefficients

    @property
    def interpolator(self) -> RegularGridInterpolator:
        """
        Getter for the *Jakob and Hanika (2019)* interpolator.

        Returns
        -------
        :class:`scipy.interpolate.RegularGridInterpolator`
            *Jakob and Hanika (2019)* interpolator.
        """

        return self._interpolator

    def _create_interpolator(self) -> None:
        """
        Create a :class:`scipy.interpolate.RegularGridInterpolator` class
        instance for read or generated coefficients.
        """

        from scipy.interpolate import RegularGridInterpolator  # noqa: PLC0415

        xp = array_namespace()

        samples = xp.linspace(0, 1, self._size)
        axes = ([0, 1, 2], self._lightness_scale, samples, samples)

        self._interpolator = RegularGridInterpolator(
            axes, self._coefficients, bounds_error=False
        )

    def generate(
        self,
        colourspace: RGB_Colourspace,
        cmfs: MultiSpectralDistributions | None = None,
        illuminant: SpectralDistribution | None = None,
        size: int = 64,
        print_callable: Callable = print,
    ) -> None:
        """
        Generate the lookup table data for the specified *RGB* colourspace,
        colour matching functions, illuminant and resolution.

        Parameters
        ----------
        colourspace
            The *RGB* colourspace to create a lookup table for.
        cmfs
            Standard observer colour matching functions, default to the
            *CIE 1931 2 Degree Standard Observer*.
        illuminant
            Illuminant spectral distribution, default to
            *CIE Standard Illuminant D65*.
        size
            The resolution of the lookup table. Higher values will decrease
            errors but at the cost of a much longer run time. The published
            *\\*.coeff* files have a resolution of 64.
        print_callable
            Callable used to print progress and diagnostic information.

        Examples
        --------
        >>> from colour import MSDS_CMFS, SDS_ILLUMINANTS
        >>> from colour.models import RGB_COLOURSPACE_sRGB
        >>> from colour.utilities import numpy_print_options
        >>> cmfs = (
        ...     MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
        ...     .copy()
        ...     .align(SpectralShape(360, 780, 10))
        ... )
        >>> illuminant = SDS_ILLUMINANTS["D65"].copy().align(cmfs.shape)
        >>> LUT = LUT3D_Jakob2019()
        >>> print(LUT.interpolator)  # doctest: +ELLIPSIS
        <scipy...RegularGridInterpolator object at 0x...>
        >>> LUT.generate(RGB_COLOURSPACE_sRGB, cmfs, illuminant, 3)
        ======================================================================\
=========
        *                                                                     \
        *
        *   "Jakob et al. (2018)" LUT Optimisation                            \
        *
        *                                                                     \
        *
        ======================================================================\
=========
        <BLANKLINE>
        Optimising 27 coefficients...
        <BLANKLINE>
        >>> print(LUT.interpolator)
        ... # doctest: +ELLIPSIS
        <scipy.interpolate...RegularGridInterpolator object at 0x...>
        """

        cmfs, illuminant = handle_spectral_arguments(
            cmfs, illuminant, shape_default=SPECTRAL_SHAPE_JAKOB2019
        )
        shape = cmfs.shape

        xy_n = XYZ_to_xy(sd_to_XYZ_integration(illuminant, cmfs))

        # It could be interesting to have different resolutions for lightness
        # and chromaticity, but the current file format doesn't allow it.
        lightness_steps = size
        chroma_steps = size

        xp = array_namespace()

        self._lightness_scale = lightness_scale(lightness_steps)
        self._coefficients = xp.zeros(
            (3, chroma_steps, chroma_steps, lightness_steps, 3)
        )

        cube_indexes = itertools.product(
            range(3), range(chroma_steps), range(chroma_steps)
        )
        total_coefficients = chroma_steps**2 * 3

        # First, create a list of all the fully bright colours with the order
        # matching cube_indexes.
        samples = xp.linspace(0, 1, chroma_steps)
        mg = xp.meshgrid(
            xp_as_float_array([1.0], xp=xp), samples, samples, indexing="ij"
        )
        # ``mg`` stacks three rank-3 axes; reversing all four axes mirrors
        # the original ``np.transpose`` over a meshgrid stack.
        mg_t = xp.permute_dims(xp.stack(mg), (3, 2, 1, 0))
        ij = xp_reshape(mg_t, (-1, 3), xp=xp)
        chromas = xp.concat(
            [
                ij,
                xp.roll(ij, 1, axis=1),
                xp.roll(ij, 2, axis=1),
            ]
        )

        message_box(
            '"Jakob et al. (2018)" LUT Optimisation',
            print_callable=print_callable,
        )

        print_callable(f"\nOptimising {total_coefficients} coefficients...\n")

        def optimize(
            ijkL: ArrayLike, coefficients_0: ArrayLike, chroma: NDArrayFloat
        ) -> NDArrayFloat:
            """
            Solve for a specific lightness and stores the result in the
            appropriate cell.
            """

            i, j, k, L = tsplit(ijkL, dtype=DTYPE_INT_DEFAULT)

            RGB = self._lightness_scale[L] * chroma

            XYZ = RGB_to_XYZ(RGB, colourspace, xy_n)

            coefficients, _error = find_coefficients_Jakob2019(
                XYZ, cmfs, illuminant, coefficients_0, dimensionalise=False
            )

            self._coefficients[i, L, j, k, :] = dimensionalise_coefficients(
                coefficients, shape
            )

            return coefficients

        with tqdm(total=total_coefficients) as progress:
            for ijk, chroma in zip(cube_indexes, chromas, strict=True):
                progress.update()

                # Starts from somewhere in the middle, similarly to how
                # feedback works in "colour.recovery.\
                # find_coefficients_Jakob2019" definition.
                L_middle = lightness_steps // 3
                coefficients_middle = optimize(
                    xp.concat([ijk, xp_as_float_array([L_middle], xp=xp)], axis=0),
                    zeros(3),
                    chroma,
                )

                # Down the lightness scale.
                coefficients_0 = coefficients_middle
                for L in reversed(range(L_middle)):
                    coefficients_0 = optimize(
                        xp.concat([ijk, xp_as_float_array([L], xp=xp)], axis=0),
                        coefficients_0,
                        chroma,
                    )

                # Up the lightness scale.
                coefficients_0 = coefficients_middle
                for L in range(L_middle + 1, lightness_steps):
                    coefficients_0 = optimize(
                        xp.concat([ijk, xp_as_float_array([L], xp=xp)], axis=0),
                        coefficients_0,
                        chroma,
                    )

        self._size = size
        self._create_interpolator()

    def RGB_to_coefficients(self, RGB: ArrayLike) -> NDArrayFloat:
        """
        Look up the specified *RGB* colourspace array and return the
        corresponding coefficients.

        Interpolation is used for colours not on the table grid.

        Parameters
        ----------
        RGB
            *RGB* colourspace array.

        Returns
        -------
        :class:`numpy.ndarray`
            Corresponding coefficients that can be passed to
            :func:`colour.recovery.jakob2019.sd_Jakob2019` to obtain a
            spectral distribution.

        Raises
        ------
        RuntimeError
            If the pre-computed lookup table has not been generated or read.

        Examples
        --------
        >>> from colour import MSDS_CMFS, SDS_ILLUMINANTS
        >>> from colour.models import RGB_COLOURSPACE_sRGB
        >>> cmfs = (
        ...     MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
        ...     .copy()
        ...     .align(SpectralShape(360, 780, 10))
        ... )
        >>> illuminant = SDS_ILLUMINANTS["D65"].copy().align(cmfs.shape)
        >>> LUT = LUT3D_Jakob2019()
        >>> LUT.generate(RGB_COLOURSPACE_sRGB, cmfs, illuminant, 3, lambda x: x)
        >>> RGB = np.array([0.70573936, 0.19248266, 0.22354169])
        >>> LUT.RGB_to_coefficients(RGB)  # doctest: +ELLIPSIS
        array([ 1.5013448...e-04, -1.4679754...e-01,  3.4020219...e+01])
        """

        if len(self._interpolator.grid) == 0:
            error = "The pre-computed lookup table has not been read or generated!"

            raise RuntimeError(error)

        RGB = as_float_array(RGB)

        xp = array_namespace(RGB)

        value_max = xp.max(RGB, axis=-1)
        chroma = RGB / (value_max[..., None] + 1e-10)

        i_m = xp.argmax(RGB, axis=-1)
        i_1 = index_along_last_axis(RGB, i_m)
        i_2 = index_along_last_axis(chroma, (i_m + 2) % 3)
        i_3 = index_along_last_axis(chroma, (i_m + 1) % 3)

        # Trilinear gather over the LUT grid, replacing
        # ``scipy.interpolate.RegularGridInterpolator(bounds_error=False)``
        # to keep the path backend-agnostic. The dominant-channel axis
        # is integer-indexed (``i_m``); the lightness axis uses the
        # non-uniform ``_lightness_scale``; the two chroma axes are
        # uniform on ``[0, 1]`` with ``size`` samples. Out-of-range
        # queries are *NaN*-filled to match the *scipy* default.
        coefficients = xp_as_float_array(self._coefficients, xp=xp, like=RGB)
        lightness_scale = xp_as_float_array(self._lightness_scale, xp=xp, like=RGB)

        size = self._size
        lightness_steps = lightness_scale.shape[0]

        outside_lookup_domain = (
            (i_1 < lightness_scale[0])
            | (i_1 > lightness_scale[-1])
            | (i_2 < 0)
            | (i_2 > 1)
            | (i_3 < 0)
            | (i_3 > 1)
        )

        j = xp.searchsorted(lightness_scale, i_1) - 1
        j = xp.clip(j, 0, lightness_steps - 2)
        v_lo = xp.take(lightness_scale, j, axis=0)
        v_hi = xp.take(lightness_scale, j + 1, axis=0)
        f_v = (i_1 - v_lo) / (v_hi - v_lo)

        last = size - 1
        f_2 = i_2 * last
        k_2 = xp.clip(xp_astype(xp.floor(f_2), DTYPE_INT_DEFAULT, xp=xp), 0, last - 1)
        f_2 = f_2 - xp_astype(k_2, f_2.dtype, xp=xp)

        f_3 = i_3 * last
        k_3 = xp.clip(xp_astype(xp.floor(f_3), DTYPE_INT_DEFAULT, xp=xp), 0, last - 1)
        f_3 = f_3 - xp_astype(k_3, f_3.dtype, xp=xp)

        # 8 corners of the unit cube: ``coefficients[i_m, j+a, k_2+b,
        # k_3+c, :]`` for ``a, b, c`` in ``{0, 1}``. Advanced indexing
        # broadcasts the per-pixel index arrays into the leading
        # ``(N,)`` shape of the result.
        c000 = coefficients[i_m, j, k_2, k_3, :]
        c100 = coefficients[i_m, j + 1, k_2, k_3, :]
        c010 = coefficients[i_m, j, k_2 + 1, k_3, :]
        c110 = coefficients[i_m, j + 1, k_2 + 1, k_3, :]
        c001 = coefficients[i_m, j, k_2, k_3 + 1, :]
        c101 = coefficients[i_m, j + 1, k_2, k_3 + 1, :]
        c011 = coefficients[i_m, j, k_2 + 1, k_3 + 1, :]
        c111 = coefficients[i_m, j + 1, k_2 + 1, k_3 + 1, :]

        f_v = f_v[..., None]
        f_2 = f_2[..., None]
        f_3 = f_3[..., None]

        c00 = c000 * (1 - f_v) + c100 * f_v
        c01 = c001 * (1 - f_v) + c101 * f_v
        c10 = c010 * (1 - f_v) + c110 * f_v
        c11 = c011 * (1 - f_v) + c111 * f_v

        c0 = c00 * (1 - f_2) + c10 * f_2
        c1 = c01 * (1 - f_2) + c11 * f_2

        return xp.where(
            outside_lookup_domain[..., None],
            float("nan"),
            c0 * (1 - f_3) + c1 * f_3,
        )

    def RGB_to_sd(
        self, RGB: ArrayLike, shape: SpectralShape = SPECTRAL_SHAPE_JAKOB2019
    ) -> SpectralDistribution:
        """
        Look up a specified *RGB* colourspace array and return the
        corresponding spectral distribution.

        Parameters
        ----------
        RGB
            *RGB* colourspace array.
        shape
            Shape used by the spectral distribution.

        Returns
        -------
        :class:`colour.SpectralDistribution`
            Spectral distribution corresponding with the *RGB* colourspace
            array.

        Examples
        --------
        >>> from colour import MSDS_CMFS, SDS_ILLUMINANTS
        >>> from colour.models import RGB_COLOURSPACE_sRGB
        >>> from colour.utilities import numpy_print_options
        >>> cmfs = (
        ...     MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
        ...     .copy()
        ...     .align(SpectralShape(360, 780, 10))
        ... )
        >>> illuminant = SDS_ILLUMINANTS["D65"].copy().align(cmfs.shape)
        >>> LUT = LUT3D_Jakob2019()
        >>> LUT.generate(RGB_COLOURSPACE_sRGB, cmfs, illuminant, 3, lambda x: x)
        >>> RGB = np.array([0.70573936, 0.19248266, 0.22354169])
        >>> with numpy_print_options(suppress=True):
        ...     LUT.RGB_to_sd(RGB, cmfs.shape)  # doctest: +ELLIPSIS
        SpectralDistribution([[360.        ,   0.7666803...],
                              [370.        ,   0.6251547...],
                              [380.        ,   0.4584310...],
                              [390.        ,   0.3161633...],
                              [400.        ,   0.2196155...],
                              [410.        ,   0.1596575...],
                              [420.        ,   0.1225525...],
                              [430.        ,   0.0989784...],
                              [440.        ,   0.0835782...],
                              [450.        ,   0.0733535...],
                              [460.        ,   0.0666049...],
                              [470.        ,   0.0623569...],
                              [480.        ,   0.06006  ...],
                              [490.        ,   0.0594383...],
                              [500.        ,   0.0604201...],
                              [510.        ,   0.0631195...],
                              [520.        ,   0.0678648...],
                              [530.        ,   0.0752834...],
                              [540.        ,   0.0864790...],
                              [550.        ,   0.1033773...],
                              [560.        ,   0.1293883...],
                              [570.        ,   0.1706018...],
                              [580.        ,   0.2374178...],
                              [590.        ,   0.3439472...],
                              [600.        ,   0.4950548...],
                              [610.        ,   0.6604253...],
                              [620.        ,   0.7914669...],
                              [630.        ,   0.8738724...],
                              [640.        ,   0.9213216...],
                              [650.        ,   0.9486880...],
                              [660.        ,   0.9650550...],
                              [670.        ,   0.9752838...],
                              [680.        ,   0.9819499...],
                              [690.        ,   0.9864585...],
                              [700.        ,   0.9896073...],
                              [710.        ,   0.9918680...],
                              [720.        ,   0.9935302...],
                              [730.        ,   0.9947778...],
                              [740.        ,   0.9957312...],
                              [750.        ,   0.9964714...],
                              [760.        ,   0.9970543...],
                              [770.        ,   0.9975190...],
                              [780.        ,   0.9978936...]],
                             SpragueInterpolator,
                             {},
                             Extrapolator,
                             {'method': 'Constant', 'left': None, 'right': None})
        """

        sd = sd_Jakob2019(self.RGB_to_coefficients(RGB), shape)
        sd.name = f"{RGB!r} (RGB) - Jakob (2019)"

        return sd

    def read(self, path: str | PathLike) -> LUT3D_Jakob2019:
        """
        Load a lookup table from a *\\*.coeff* file.

        Parameters
        ----------
        path
            Path to the file.

        Returns
        -------
        LUT3D_Jakob2019
            *Jakob and Hanika (2019)* lookup table.

        Examples
        --------
        >>> import os
        >>> import colour
        >>> from colour import MSDS_CMFS, SDS_ILLUMINANTS
        >>> from colour.models import RGB_COLOURSPACE_sRGB
        >>> from colour.utilities import numpy_print_options
        >>> cmfs = (
        ...     MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
        ...     .copy()
        ...     .align(SpectralShape(360, 780, 10))
        ... )
        >>> illuminant = SDS_ILLUMINANTS["D65"].copy().align(cmfs.shape)
        >>> LUT = LUT3D_Jakob2019()
        >>> LUT.generate(RGB_COLOURSPACE_sRGB, cmfs, illuminant, 3, lambda x: x)
        >>> path = os.path.join(
        ...     colour.__path__[0],
        ...     "recovery",
        ...     "tests",
        ...     "resources",
        ...     "sRGB_Jakob2019.coeff",
        ... )
        >>> LUT.write(path)  # doctest: +SKIP
        >>> LUT.read(path)  # doctest: +SKIP
        """

        path = str(path)

        with open(path, "rb") as coeff_file:
            if coeff_file.read(4).decode("ISO-8859-1") != "SPEC":
                error = "Bad magic number, this is likely not the right file type!"

                raise ValueError(error)

            self._size = struct.unpack("i", coeff_file.read(4))[0]
            self._lightness_scale = np.fromfile(
                coeff_file, count=self._size, dtype=np.float32
            )
            self._coefficients = np.fromfile(
                coeff_file, count=3 * (self._size**3) * 3, dtype=np.float32
            )

            xp = array_namespace(self._coefficients)

            self._coefficients = xp.reshape(
                self._coefficients, (3, self._size, self._size, self._size, 3)
            )

        self._create_interpolator()

        return self

    def write(self, path: str | PathLike) -> bool:
        """
        Write the lookup table to a *\\*.coeff* file.

        Parameters
        ----------
        path
            Path to the file.

        Returns
        -------
        :class:`bool`
            Definition success.

        Examples
        --------
        >>> import os
        >>> import colour
        >>> from colour import MSDS_CMFS, SDS_ILLUMINANTS
        >>> from colour.models import RGB_COLOURSPACE_sRGB
        >>> from colour.utilities import numpy_print_options
        >>> cmfs = (
        ...     MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
        ...     .copy()
        ...     .align(SpectralShape(360, 780, 10))
        ... )
        >>> illuminant = SDS_ILLUMINANTS["D65"].copy().align(cmfs.shape)
        >>> LUT = LUT3D_Jakob2019()
        >>> LUT.generate(RGB_COLOURSPACE_sRGB, cmfs, illuminant, 3, lambda x: x)
        >>> path = os.path.join(
        ...     colour.__path__[0],
        ...     "recovery",
        ...     "tests",
        ...     "resources",
        ...     "sRGB_Jakob2019.coeff",
        ... )
        >>> LUT.write(path)  # doctest: +SKIP
        >>> LUT.read(path)  # doctest: +SKIP
        """

        path = str(path)

        with open(path, "wb") as coeff_file:
            coeff_file.write(b"SPEC")
            coeff_file.write(struct.pack("i", self._coefficients.shape[1]))
            np.float32(self._lightness_scale).tofile(coeff_file)
            np.float32(self._coefficients).tofile(coeff_file)

        return True
