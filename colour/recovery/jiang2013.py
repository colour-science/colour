"""
Jiang et al. (2013) - Camera RGB Sensitivities Recovery
=======================================================

Define the objects for camera *RGB* sensitivities recovery using the
*Jiang, Liu, Gu and Süsstrunk (2013)* method.

-   :func:`colour.recovery.PCA_Jiang2013`
-   :func:`colour.recovery.RGB_to_sd_camera_sensitivity_Jiang2013`
-   :func:`colour.recovery.RGB_to_msds_camera_sensitivities_Jiang2013`

References
----------
-   :cite:`Jiang2013` : Jiang, J., Liu, D., Gu, J., & Susstrunk, S. (2013).
    What is the space of spectral sensitivity functions for digital color
    cameras? 2013 IEEE Workshop on Applications of Computer Vision (WACV),
    168-179. doi:10.1109/WACV.2013.6475015
"""

from __future__ import annotations

import typing

import numpy as np

from colour.algebra import eigen_decomposition
from colour.characterisation import RGB_CameraSensitivities
from colour.colorimetry import (
    MultiSpectralDistributions,
    SpectralDistribution,
    SpectralShape,
    reshape_msds,
    reshape_sd,
)

if typing.TYPE_CHECKING:
    from colour.hints import (
        ArrayLike,
        Domain1,
        Literal,
        Mapping,
        NDArrayFloat,
        Tuple,
    )

from colour.hints import cast
from colour.recovery import BASIS_FUNCTIONS_DYER2017
from colour.utilities import as_float_array, optional, runtime_warning, tsplit

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "PCA_Jiang2013",
    "RGB_to_sd_camera_sensitivity_Jiang2013",
    "RGB_to_msds_camera_sensitivities_Jiang2013",
]


@typing.overload
def PCA_Jiang2013(
    msds_camera_sensitivities: Mapping[str, MultiSpectralDistributions],
    eigen_w_v_count: int | None = ...,
    additional_data: Literal[True] = True,
) -> Tuple[
    Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat],
    Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat],
]: ...


@typing.overload
def PCA_Jiang2013(
    msds_camera_sensitivities: Mapping[str, MultiSpectralDistributions],
    eigen_w_v_count: int | None = ...,
    *,
    additional_data: Literal[False],
) -> Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]: ...


@typing.overload
def PCA_Jiang2013(
    msds_camera_sensitivities: Mapping[str, MultiSpectralDistributions],
    eigen_w_v_count: int | None,
    additional_data: Literal[False],
) -> Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]: ...


def PCA_Jiang2013(
    msds_camera_sensitivities: Mapping[str, MultiSpectralDistributions],
    eigen_w_v_count: int | None = None,
    additional_data: bool = False,
) -> (
    Tuple[
        Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat],
        Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat],
    ]
    | Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]
):
    """
    Perform *Principal Component Analysis* (PCA) on specified camera *RGB*
    sensitivities.

    Parameters
    ----------
    msds_camera_sensitivities
         Camera *RGB* sensitivities.
    eigen_w_v_count
        Eigen-values :math:`w` and eigen-vectors :math:`v` count.
    additional_data
        Whether to return both the eigen-values :math:`w` and
        eigen-vectors :math:`v`.

    Returns
    -------
    :class:`tuple`
        Tuple of camera *RGB* sensitivities eigen-values :math:`w` and
        eigen-vectors :math:`v` or tuple of camera *RGB* sensitivities
        eigen-vectors :math:`v`.

    Examples
    --------
    >>> from colour.colorimetry import SpectralShape
    >>> from colour.characterisation import MSDS_CAMERA_SENSITIVITIES
    >>> shape = SpectralShape(400, 700, 10)
    >>> camera_sensitivities = {
    ...     camera: msds.copy().align(shape)
    ...     for camera, msds in MSDS_CAMERA_SENSITIVITIES.items()
    ... }
    >>> np.array(PCA_Jiang2013(camera_sensitivities)).shape
    (3, 31, 31)
    """

    R_sensitivities, G_sensitivities, B_sensitivities = [], [], []

    def normalised_sensitivity(
        msds: MultiSpectralDistributions, channel: str
    ) -> NDArrayFloat:
        """Generate a normalised camera *RGB* sensitivity."""

        sensitivity = cast("SpectralDistribution", msds.signals[channel].copy())

        return sensitivity.normalise().values

    for msds in msds_camera_sensitivities.values():
        R_sensitivities.append(normalised_sensitivity(msds, msds.labels[0]))
        G_sensitivities.append(normalised_sensitivity(msds, msds.labels[1]))
        B_sensitivities.append(normalised_sensitivity(msds, msds.labels[2]))

    R_w_v = eigen_decomposition(
        np.vstack(R_sensitivities), eigen_w_v_count, covariance_matrix=True
    )
    G_w_v = eigen_decomposition(
        np.vstack(G_sensitivities), eigen_w_v_count, covariance_matrix=True
    )
    B_w_v = eigen_decomposition(
        np.vstack(B_sensitivities), eigen_w_v_count, covariance_matrix=True
    )

    if additional_data:
        return (
            (R_w_v[1], G_w_v[1], B_w_v[1]),
            (R_w_v[0], G_w_v[0], B_w_v[0]),
        )

    return R_w_v[1], G_w_v[1], B_w_v[1]


def RGB_to_sd_camera_sensitivity_Jiang2013(
    RGB: Domain1,
    illuminant: SpectralDistribution,
    reflectances: MultiSpectralDistributions,
    eigen_w: ArrayLike,
    shape: SpectralShape | None = None,
) -> SpectralDistribution:
    """
    Recover a single camera *RGB* sensitivity for the specified camera *RGB*
    values using *Jiang et al. (2013)* method.

    Parameters
    ----------
    RGB
        Camera *RGB* values corresponding with ``reflectances``.
    illuminant
        Illuminant spectral distribution used to produce the camera *RGB*
        values.
    reflectances
        Reflectance spectral distributions used to produce the camera
        *RGB* values.
    eigen_w
        Eigen-vectors :math:`v` for the particular camera *RGB*
        sensitivity being recovered.
    shape
        Spectral shape of the recovered camera *RGB* sensitivity,
        ``illuminant`` and ``reflectances`` will be aligned to it if
        passed, otherwise, ``illuminant`` shape is used.

    Returns
    -------
    :class:`colour.RGB_CameraSensitivities`
        Recovered camera *RGB* sensitivities.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``RGB``    | 1                     | 1             |
    +------------+-----------------------+---------------+

    Examples
    --------
    >>> from colour.colorimetry import (
    ...     SDS_ILLUMINANTS,
    ...     msds_to_XYZ,
    ...     sds_and_msds_to_msds,
    ... )
    >>> from colour.characterisation import (
    ...     MSDS_CAMERA_SENSITIVITIES,
    ...     SDS_COLOURCHECKERS,
    ... )
    >>> from colour.recovery import SPECTRAL_SHAPE_BASIS_FUNCTIONS_DYER2017
    >>> illuminant = SDS_ILLUMINANTS["D65"]
    >>> sensitivities = MSDS_CAMERA_SENSITIVITIES["Nikon 5100 (NPL)"]
    >>> reflectances = [
    ...     sd.copy().align(SPECTRAL_SHAPE_BASIS_FUNCTIONS_DYER2017)
    ...     for sd in SDS_COLOURCHECKERS["BabelColor Average"].values()
    ... ]
    >>> reflectances = sds_and_msds_to_msds(reflectances)
    >>> R, G, B = (
    ...     tsplit(
    ...         msds_to_XYZ(
    ...             reflectances,
    ...             method="Integration",
    ...             cmfs=sensitivities,
    ...             illuminant=illuminant,
    ...             k=1,
    ...             shape=SPECTRAL_SHAPE_BASIS_FUNCTIONS_DYER2017,
    ...         )
    ...     )
    ...     / 100
    ... )
    >>> R_w, G_w, B_w = tsplit(np.moveaxis(BASIS_FUNCTIONS_DYER2017, 0, 1))
    >>> RGB_to_sd_camera_sensitivity_Jiang2013(
    ...     R,
    ...     illuminant,
    ...     reflectances,
    ...     R_w,
    ...     SPECTRAL_SHAPE_BASIS_FUNCTIONS_DYER2017,
    ... )  # doctest: +ELLIPSIS
    SpectralDistribution([[ 4.00000000e+02,  7.20665029e-06],
                          [ 4.10000000e+02, -8.96986938e-06],
                          [ 4.20000000e+02,  4.68719619e-05],
                          [ 4.30000000e+02,  7.76949719e-05],
                          [ 4.40000000e+02,  6.93355114e-05],
                          [ 4.50000000e+02,  5.31349471e-05],
                          [ 4.60000000e+02,  4.48199586e-05],
                          [ 4.70000000e+02,  4.63937913e-05],
                          [ 4.80000000e+02,  5.18666681e-05],
                          [ 4.90000000e+02,  4.38283172e-05],
                          [ 5.00000000e+02,  4.20012318e-05],
                          [ 5.10000000e+02,  5.40655441e-05],
                          [ 5.20000000e+02,  9.64451417e-05],
                          [ 5.30000000e+02,  1.42771129e-04],
                          [ 5.40000000e+02,  7.99507187e-05],
                          [ 5.50000000e+02,  4.64298137e-05],
                          [ 5.60000000e+02,  5.34238406e-05],
                          [ 5.70000000e+02,  1.05193839e-04],
                          [ 5.80000000e+02,  5.28894436e-04],
                          [ 5.90000000e+02,  9.78511673e-04],
                          [ 6.00000000e+02,  9.96003826e-04],
                          [ 6.10000000e+02,  8.38408920e-04],
                          [ 6.20000000e+02,  6.91808589e-04],
                          [ 6.30000000e+02,  5.69678548e-04],
                          [ 6.40000000e+02,  4.29303089e-04],
                          [ 6.50000000e+02,  3.02412675e-04],
                          [ 6.60000000e+02,  2.32300470e-04],
                          [ 6.70000000e+02,  1.37219431e-04],
                          [ 6.80000000e+02,  4.09448851e-05],
                          [ 6.90000000e+02, -4.42234757e-06],
                          [ 7.00000000e+02, -6.14277697e-06]],
                         SpragueInterpolator,
                         {},
                         Extrapolator,
                         {'method': 'Constant', 'left': None, 'right': None})
    """

    RGB = as_float_array(RGB)
    shape = optional(shape, illuminant.shape)

    if illuminant.shape != shape:
        runtime_warning(f'Aligning "{illuminant.name}" illuminant shape to "{shape}".')
        illuminant = reshape_sd(illuminant, shape, copy=False)

    if reflectances.shape != shape:
        runtime_warning(
            f'Aligning "{reflectances.name}" reflectances shape to "{shape}".'
        )
        reflectances = reshape_msds(reflectances, shape, copy=False)

    S = np.diag(illuminant.values)
    R = np.transpose(reflectances.values)

    A = np.dot(np.dot(R, S), eigen_w)

    X = np.linalg.lstsq(A, RGB, rcond=None)[0]
    X = np.dot(eigen_w, X)

    return SpectralDistribution(X, shape.wavelengths)


def RGB_to_msds_camera_sensitivities_Jiang2013(
    RGB: Domain1,
    illuminant: SpectralDistribution,
    reflectances: MultiSpectralDistributions,
    basis_functions: ArrayLike = BASIS_FUNCTIONS_DYER2017,
    shape: SpectralShape | None = None,
) -> MultiSpectralDistributions:
    """
    Recover the camera *RGB* sensitivities for the specified camera *RGB*
    values using *Jiang et al. (2013)* method.

    Parameters
    ----------
    RGB
        Camera *RGB* values corresponding with ``reflectances``.
    illuminant
        Illuminant spectral distribution used to produce the camera *RGB*
        values.
    reflectances
        Reflectance spectral distributions used to produce the camera
        *RGB* values.
    basis_functions
        Basis functions for the method. The default is to use the
        built-in *sRGB* basis functions, i.e.,
        :attr:`colour.recovery.BASIS_FUNCTIONS_DYER2017`.
    shape
        Spectral shape of the recovered camera *RGB* sensitivities.
        The ``illuminant`` and ``reflectances`` will be aligned to it if
        passed, otherwise, the ``illuminant`` shape is used.

    Returns
    -------
    :class:`colour.RGB_CameraSensitivities`
        Recovered camera *RGB* sensitivities.

    Notes
    -----
    +------------+-----------------------+---------------+
    | **Domain** | **Scale - Reference** | **Scale - 1** |
    +============+=======================+===============+
    | ``RGB``    | 1                     | 1             |
    +------------+-----------------------+---------------+

    Examples
    --------
    >>> from colour.colorimetry import (
    ...     SDS_ILLUMINANTS,
    ...     msds_to_XYZ,
    ...     sds_and_msds_to_msds,
    ... )
    >>> from colour.characterisation import (
    ...     MSDS_CAMERA_SENSITIVITIES,
    ...     SDS_COLOURCHECKERS,
    ... )
    >>> from colour.recovery import SPECTRAL_SHAPE_BASIS_FUNCTIONS_DYER2017
    >>> illuminant = SDS_ILLUMINANTS["D65"]
    >>> sensitivities = MSDS_CAMERA_SENSITIVITIES["Nikon 5100 (NPL)"]
    >>> reflectances = [
    ...     sd.copy().align(SPECTRAL_SHAPE_BASIS_FUNCTIONS_DYER2017)
    ...     for sd in SDS_COLOURCHECKERS["BabelColor Average"].values()
    ... ]
    >>> reflectances = sds_and_msds_to_msds(reflectances)
    >>> RGB = (
    ...     msds_to_XYZ(
    ...         reflectances,
    ...         method="Integration",
    ...         cmfs=sensitivities,
    ...         illuminant=illuminant,
    ...         k=1,
    ...         shape=SPECTRAL_SHAPE_BASIS_FUNCTIONS_DYER2017,
    ...     )
    ...     / 100
    ... )
    >>> RGB_to_msds_camera_sensitivities_Jiang2013(
    ...     RGB,
    ...     illuminant,
    ...     reflectances,
    ...     BASIS_FUNCTIONS_DYER2017,
    ...     SPECTRAL_SHAPE_BASIS_FUNCTIONS_DYER2017,
    ... ).values  # doctest: +ELLIPSIS
    array([[ 7.04378461e-03,  9.21260449e-03, -7.64080878e-03],
           [-8.76715607e-03,  1.12726694e-02,  6.37434190e-03],
           [ 4.58126856e-02,  7.18000418e-02,  4.00001696e-01],
           [ 7.59391152e-02,  1.15620933e-01,  7.11521550e-01],
           [ 6.77685732e-02,  1.53406449e-01,  8.52668310e-01],
           [ 5.19341313e-02,  1.88575472e-01,  9.38957846e-01],
           [ 4.38070562e-02,  2.61086603e-01,  9.72130729e-01],
           [ 4.53453213e-02,  3.75440392e-01,  9.61450686e-01],
           [ 5.06945146e-02,  4.47658155e-01,  8.86481146e-01],
           [ 4.28378252e-02,  4.50713447e-01,  7.51770770e-01],
           [ 4.10520309e-02,  6.16577286e-01,  5.52730730e-01],
           [ 5.28436974e-02,  7.80199548e-01,  3.82269175e-01],
           [ 9.42655432e-02,  9.17674257e-01,  2.40354614e-01],
           [ 1.39544593e-01,  1.00000000e+00,  1.55374812e-01],
           [ 7.81438836e-02,  9.27720273e-01,  1.04409358e-01],
           [ 4.53805297e-02,  8.56701565e-01,  6.51222854e-02],
           [ 5.22164960e-02,  7.52322921e-01,  3.42954473e-02],
           [ 1.02816526e-01,  6.25809730e-01,  2.09495104e-02],
           [ 5.16941760e-01,  4.92746166e-01,  1.48524616e-02],
           [ 9.56397935e-01,  3.43364817e-01,  1.08983186e-02],
           [ 9.73494777e-01,  2.08587708e-01,  7.00494396e-03],
           [ 8.19461415e-01,  1.11784838e-01,  4.47180002e-03],
           [ 6.76174158e-01,  6.59071962e-02,  4.10135388e-03],
           [ 5.56804177e-01,  4.46268353e-02,  4.18528982e-03],
           [ 4.19601114e-01,  3.33671033e-02,  4.49165886e-03],
           [ 2.95578342e-01,  2.39487762e-02,  4.45932739e-03],
           [ 2.27050628e-01,  1.87787770e-02,  4.31697313e-03],
           [ 1.34118359e-01,  1.06954985e-02,  3.41192651e-03],
           [ 4.00195568e-02,  5.55512389e-03,  1.36794925e-03],
           [-4.32240535e-03,  2.49731193e-03,  3.80303275e-04],
           [-6.00395414e-03,  1.54678227e-03,  5.40394352e-04]])
    """

    R, G, B = tsplit(np.reshape(RGB, [-1, 3]))
    basis_functions = as_float_array(basis_functions)
    shape = optional(shape, illuminant.shape)

    R_w, G_w, B_w = tsplit(np.moveaxis(basis_functions, 0, 1))

    if illuminant.shape != shape:
        runtime_warning(f'Aligning "{illuminant.name}" illuminant shape to "{shape}".')
        illuminant = reshape_sd(illuminant, shape, copy=False)

    if reflectances.shape != shape:
        runtime_warning(
            f'Aligning "{reflectances.name}" reflectances shape to "{shape}".'
        )
        reflectances = reshape_msds(reflectances, shape, copy=False)

    S_R = RGB_to_sd_camera_sensitivity_Jiang2013(
        R, illuminant, reflectances, R_w, shape
    )
    S_G = RGB_to_sd_camera_sensitivity_Jiang2013(
        G, illuminant, reflectances, G_w, shape
    )
    S_B = RGB_to_sd_camera_sensitivity_Jiang2013(
        B, illuminant, reflectances, B_w, shape
    )

    msds_camera_sensitivities = RGB_CameraSensitivities([S_R, S_G, S_B])

    msds_camera_sensitivities /= np.max(msds_camera_sensitivities.values)

    return msds_camera_sensitivities
