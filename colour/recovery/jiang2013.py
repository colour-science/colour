"""
Jiang et al. (2013) - Camera RGB Sensitivities Recovery
=======================================================

Defines the objects for camera *RGB* sensitivities recovery using
*Jiang, Liu, Gu and Süsstrunk (2013)* method:

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
from colour.hints import (
    ArrayLike,
    Mapping,
    NDArrayFloat,
    Tuple,
    cast,
)
from colour.recovery import BASIS_FUNCTIONS_DYER2017
from colour.utilities import as_float_array, optional, tsplit, runtime_warning


__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "PCA_Jiang2013",
    "RGB_to_sd_camera_sensitivity_Jiang2013",
    "RGB_to_msds_camera_sensitivities_Jiang2013",
]


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
    Perform the *Principal Component Analysis* (PCA) on given camera *RGB*
    sensitivities.

    Parameters
    ----------
    msds_camera_sensitivities
         Camera *RGB* sensitivities.
    eigen_w_v_count
        Eigen-values :math:`w` and eigen-vectors :math:`v` count.
    additional_data
        Whether to return both the eigen-values :math:`w` and eigen-vectors
        :math:`v`.

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
        """Return a normalised camera *RGB* sensitivity."""

        sensitivity = cast(SpectralDistribution, msds.signals[channel].copy())

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
    else:
        return R_w_v[1], G_w_v[1], B_w_v[1]


def RGB_to_sd_camera_sensitivity_Jiang2013(
    RGB: ArrayLike,
    illuminant: SpectralDistribution,
    reflectances: MultiSpectralDistributions,
    eigen_w: ArrayLike,
    shape: SpectralShape | None = None,
) -> SpectralDistribution:
    """
    Recover a single camera *RGB* sensitivity for given camera *RGB* values
    using *Jiang et al. (2013)* method.

    Parameters
    ----------
    RGB
        Camera *RGB* values corresponding with ``reflectances``.
    illuminant
        Illuminant spectral distribution used to produce the camera *RGB*
        values.
    reflectances
        Reflectance spectral distributions used to produce the camera *RGB*
        values.
    eigen_w
        Eigen-vectors :math:`v` for the particular camera *RGB* sensitivity
        being recovered.
    shape
        Spectral shape of the recovered camera *RGB* sensitivity,
        ``illuminant`` and ``reflectances`` will be aligned to it if passed,
        otherwise, ``illuminant`` shape is used.

    Returns
    -------
    :class:`colour.RGB_CameraSensitivities`
        Recovered camera *RGB* sensitivities.

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
    >>> R, G, B = tsplit(
    ...     msds_to_XYZ(
    ...         reflectances,
    ...         method="Integration",
    ...         cmfs=sensitivities,
    ...         illuminant=illuminant,
    ...         k=1,
    ...         shape=SPECTRAL_SHAPE_BASIS_FUNCTIONS_DYER2017,
    ...     )
    ... )
    >>> R_w, G_w, B_w = tsplit(np.moveaxis(BASIS_FUNCTIONS_DYER2017, 0, 1))
    >>> RGB_to_sd_camera_sensitivity_Jiang2013(
    ...     R,
    ...     illuminant,
    ...     reflectances,
    ...     R_w,
    ...     SPECTRAL_SHAPE_BASIS_FUNCTIONS_DYER2017,
    ... )  # doctest: +ELLIPSIS
    SpectralDistribution([[  4.00000000e+02,   7.3976716...e-04],
                          [  4.10000000e+02,  -8.7040243...e-04],
                          [  4.20000000e+02,   4.6893657...e-03],
                          [  4.30000000e+02,   7.7522012...e-03],
                          [  4.40000000e+02,   6.9238417...e-03],
                          [  4.50000000e+02,   5.3089422...e-03],
                          [  4.60000000e+02,   4.4780109...e-03],
                          [  4.70000000e+02,   4.6386816...e-03],
                          [  4.80000000e+02,   5.1897663...e-03],
                          [  4.90000000e+02,   4.3906620...e-03],
                          [  5.00000000e+02,   4.2189259...e-03],
                          [  5.10000000e+02,   5.4270976...e-03],
                          [  5.20000000e+02,   9.6722601...e-03],
                          [  5.30000000e+02,   1.4272520...e-02],
                          [  5.40000000e+02,   7.9609053...e-03],
                          [  5.50000000e+02,   4.5917460...e-03],
                          [  5.60000000e+02,   5.2723695...e-03],
                          [  5.70000000e+02,   1.0479224...e-02],
                          [  5.80000000e+02,   5.3101298...e-02],
                          [  5.90000000e+02,   9.8185490...e-02],
                          [  6.00000000e+02,   9.9775094...e-02],
                          [  6.10000000e+02,   8.3935824...e-02],
                          [  6.20000000e+02,   6.9216733...e-02],
                          [  6.30000000e+02,   5.6902763...e-02],
                          [  6.40000000e+02,   4.2810635...e-02],
                          [  6.50000000e+02,   3.0064003...e-02],
                          [  6.60000000e+02,   2.3093789...e-02],
                          [  6.70000000e+02,   1.3756855...e-02],
                          [  6.80000000e+02,   4.1785101...e-03],
                          [  6.90000000e+02,  -3.8014848...e-04],
                          [  7.00000000e+02,  -5.7544253...e-04]],
                         SpragueInterpolator,
                         {},
                         Extrapolator,
                         {'method': 'Constant', 'left': None, 'right': None})
    """

    RGB = as_float_array(RGB)
    shape = optional(shape, illuminant.shape)

    if illuminant.shape != shape:
        runtime_warning(
            f'Aligning "{illuminant.name}" illuminant shape to "{shape}".'
        )
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
    RGB: ArrayLike,
    illuminant: SpectralDistribution,
    reflectances: MultiSpectralDistributions,
    basis_functions=BASIS_FUNCTIONS_DYER2017,
    shape: SpectralShape | None = None,
) -> MultiSpectralDistributions:
    """
    Recover the camera *RGB* sensitivities for given camera *RGB* values using
    *Jiang et al. (2013)* method.

    Parameters
    ----------
    RGB
        Camera *RGB* values corresponding with ``reflectances``.
    illuminant
        Illuminant spectral distribution used to produce the camera *RGB*
        values.
    reflectances
        Reflectance spectral distributions used to produce the camera *RGB*
        values.
    basis_functions
        Basis functions for the method. The default is to use the built-in
        *sRGB* basis functions, i.e.
        :attr:`colour.recovery.BASIS_FUNCTIONS_DYER2017`.
    shape
        Spectral shape of the recovered camera *RGB* sensitivities,
        ``illuminant`` and ``reflectances`` will be aligned to it if passed,
        otherwise, ``illuminant`` shape is used.

    Returns
    -------
    :class:`colour.RGB_CameraSensitivities`
        Recovered camera *RGB* sensitivities.

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
    >>> RGB = msds_to_XYZ(
    ...     reflectances,
    ...     method="Integration",
    ...     cmfs=sensitivities,
    ...     illuminant=illuminant,
    ...     k=1,
    ...     shape=SPECTRAL_SHAPE_BASIS_FUNCTIONS_DYER2017,
    ... )
    >>> RGB_to_msds_camera_sensitivities_Jiang2013(
    ...     RGB,
    ...     illuminant,
    ...     reflectances,
    ...     BASIS_FUNCTIONS_DYER2017,
    ...     SPECTRAL_SHAPE_BASIS_FUNCTIONS_DYER2017,
    ... ).values  # doctest: +ELLIPSIS
    array([[  7.2281577...e-03,   9.2250648...e-03,  -9.8836897...e-03],
           [ -8.5045760...e-03,   1.1277748...e-02,   3.8624865...e-03],
           [  4.5819113...e-02,   7.1552094...e-02,   4.0406829...e-01],
           [  7.5745635...e-02,   1.1530030...e-01,   7.1177452...e-01],
           [  6.7651854...e-02,   1.5311354...e-01,   8.5161378...e-01],
           [  5.1872905...e-02,   1.8828774...e-01,   9.3658053...e-01],
           [  4.3753995...e-02,   2.6093723...e-01,   9.7049828...e-01],
           [  4.5323885...e-02,   3.7531459...e-01,   9.5883525...e-01],
           [  5.0708454...e-02,   4.4750685...e-01,   8.8451412...e-01],
           [  4.2900523...e-02,   4.5047800...e-01,   7.5069924...e-01],
           [  4.1222513...e-02,   6.1672868...e-01,   5.5327277...e-01],
           [  5.3027385...e-02,   7.8015416...e-01,   3.8368507...e-01],
           [  9.4506252...e-02,   9.1751657...e-01,   2.4143664...e-01],
           [  1.3945472...e-01,   1.0000000...e+00,   1.5616071...e-01],
           [  7.7784852...e-02,   9.2719372...e-01,   1.0462050...e-01],
           [  4.4865285...e-02,   8.5627976...e-01,   6.5035086...e-02],
           [  5.1515558...e-02,   7.5193757...e-01,   3.3979292...e-02],
           [  1.0239098...e-01,   6.2562412...e-01,   2.0583993...e-02],
           [  5.1884509...e-01,   4.9264953...e-01,   1.4571020...e-02],
           [  9.5935619...e-01,   3.4322427...e-01,   1.0656116...e-02],
           [  9.7488799...e-01,   2.0857245...e-01,   6.8892462...e-03],
           [  8.2012477...e-01,   1.1178699...e-01,   4.3808407...e-03],
           [  6.7630666...e-01,   6.5977834...e-02,   4.0420907...e-03],
           [  5.5598866...e-01,   4.4719007...e-02,   4.2502316...e-03],
           [  4.1829651...e-01,   3.3471790...e-02,   4.6139542...e-03],
           [  2.9375101...e-01,   2.4044889...e-02,   4.7376860...e-03],
           [  2.2564606...e-01,   1.8870707...e-02,   4.6336440...e-03],
           [  1.3441624...e-01,   1.0702974...e-02,   3.4919622...e-03],
           [  4.0827617...e-02,   5.5529047...e-03,   1.3990786...e-03],
           [ -3.7143757...e-03,   2.5093564...e-03,   3.9765262...e-04],
           [ -5.6225656...e-03,   1.5643397...e-03,   5.8472693...e-04]])
    """

    R, G, B = tsplit(np.reshape(RGB, [-1, 3]))
    shape = optional(shape, illuminant.shape)

    R_w, G_w, B_w = tsplit(np.moveaxis(basis_functions, 0, 1))

    if illuminant.shape != shape:
        runtime_warning(
            f'Aligning "{illuminant.name}" illuminant shape to "{shape}".'
        )
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
