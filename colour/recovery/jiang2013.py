"""
Jiang et al. (2013) - Camera RGB Sensitivities Recovery
=======================================================

Define the objects for camera *RGB* sensitivities recovery using
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
    SpectralDistribution([[  4.00000000e+02,   7.2066502...e-04],
                          [  4.10000000e+02,  -8.9698693...e-04],
                          [  4.20000000e+02,   4.6871961...e-03],
                          [  4.30000000e+02,   7.7694971...e-03],
                          [  4.40000000e+02,   6.9335511...e-03],
                          [  4.50000000e+02,   5.3134947...e-03],
                          [  4.60000000e+02,   4.4819958...e-03],
                          [  4.70000000e+02,   4.6393791...e-03],
                          [  4.80000000e+02,   5.1866668...e-03],
                          [  4.90000000e+02,   4.3828317...e-03],
                          [  5.00000000e+02,   4.2001231...e-03],
                          [  5.10000000e+02,   5.4065544...e-03],
                          [  5.20000000e+02,   9.6445141...e-03],
                          [  5.30000000e+02,   1.4277112...e-02],
                          [  5.40000000e+02,   7.9950718...e-03],
                          [  5.50000000e+02,   4.6429813...e-03],
                          [  5.60000000e+02,   5.3423840...e-03],
                          [  5.70000000e+02,   1.0519383...e-02],
                          [  5.80000000e+02,   5.2889443...e-02],
                          [  5.90000000e+02,   9.7851167...e-02],
                          [  6.00000000e+02,   9.9600382...e-02],
                          [  6.10000000e+02,   8.3840892...e-02],
                          [  6.20000000e+02,   6.9180858...e-02],
                          [  6.30000000e+02,   5.6967854...e-02],
                          [  6.40000000e+02,   4.2930308...e-02],
                          [  6.50000000e+02,   3.0241267...e-02],
                          [  6.60000000e+02,   2.3230047...e-02],
                          [  6.70000000e+02,   1.3721943...e-02],
                          [  6.80000000e+02,   4.0944885...e-03],
                          [  6.90000000e+02,  -4.4223475...e-04],
                          [  7.00000000e+02,  -6.1427769...e-04]],
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
        *sRGB* basis functions, i.e.,
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
    array([[  7.0437846...e-03,   9.2126044...e-03,  -7.6408087...e-03],
           [ -8.7671560...e-03,   1.1272669...e-02,   6.3743419...e-03],
           [  4.5812685...e-02,   7.1800041...e-02,   4.0000169...e-01],
           [  7.5939115...e-02,   1.1562093...e-01,   7.1152155...e-01],
           [  6.7768573...e-02,   1.5340644...e-01,   8.5266831...e-01],
           [  5.1934131...e-02,   1.8857547...e-01,   9.3895784...e-01],
           [  4.3807056...e-02,   2.6108660...e-01,   9.7213072...e-01],
           [  4.5345321...e-02,   3.7544039...e-01,   9.6145068...e-01],
           [  5.0694514...e-02,   4.4765815...e-01,   8.8648114...e-01],
           [  4.2837825...e-02,   4.5071344...e-01,   7.5177077...e-01],
           [  4.1052030...e-02,   6.1657728...e-01,   5.5273073...e-01],
           [  5.2843697...e-02,   7.8019954...e-01,   3.8226917...e-01],
           [  9.4265543...e-02,   9.1767425...e-01,   2.4035461...e-01],
           [  1.3954459...e-01,   1.0000000...e+00,   1.5537481...e-01],
           [  7.8143883...e-02,   9.2772027...e-01,   1.0440935...e-01],
           [  4.5380529...e-02,   8.5670156...e-01,   6.5122285...e-02],
           [  5.2216496...e-02,   7.5232292...e-01,   3.4295447...e-02],
           [  1.0281652...e-01,   6.2580973...e-01,   2.0949510...e-02],
           [  5.1694176...e-01,   4.9274616...e-01,   1.4852461...e-02],
           [  9.5639793...e-01,   3.4336481...e-01,   1.0898318...e-02],
           [  9.7349477...e-01,   2.0858770...e-01,   7.0049439...e-03],
           [  8.1946141...e-01,   1.1178483...e-01,   4.4718000...e-03],
           [  6.7617415...e-01,   6.5907196...e-02,   4.1013538...e-03],
           [  5.5680417...e-01,   4.4626835...e-02,   4.1852898...e-03],
           [  4.1960111...e-01,   3.3367103...e-02,   4.4916588...e-03],
           [  2.9557834...e-01,   2.3948776...e-02,   4.4593273...e-03],
           [  2.2705062...e-01,   1.8778777...e-02,   4.3169731...e-03],
           [  1.3411835...e-01,   1.0695498...e-02,   3.4119265...e-03],
           [  4.0019556...e-02,   5.5551238...e-03,   1.3679492...e-03],
           [ -4.3224053...e-03,   2.4973119...e-03,   3.8030327...e-04],
           [ -6.0039541...e-03,   1.5467822...e-03,   5.4039435...e-04]])
    """

    R, G, B = tsplit(np.reshape(RGB, [-1, 3]))
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
