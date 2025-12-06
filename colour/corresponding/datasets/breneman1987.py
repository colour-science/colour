"""
Breneman Corresponding Chromaticities Dataset
=============================================

Define the *Breneman (1987)* results for corresponding chromaticities
experiments.

References
----------
-   :cite:`Breneman1987b` : Breneman, E. J. (1987). Corresponding
    chromaticities for different states of adaptation to complex visual fields.
    Journal of the Optical Society of America A, 4(6), 1115.
    doi:10.1364/JOSAA.4.001115
"""

from __future__ import annotations

import typing
from dataclasses import dataclass, field

import numpy as np

if typing.TYPE_CHECKING:
    from colour.hints import NDArrayFloat, Tuple

from colour.utilities import MixinDataclassIterable, as_float_array
from colour.utilities.documentation import DocstringDict, is_documentation_building

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "BrenemanExperimentResult",
    "PrimariesChromaticityCoordinates",
    "BRENEMAN_EXPERIMENT_1_RESULTS",
    "BRENEMAN_EXPERIMENT_2_RESULTS",
    "BRENEMAN_EXPERIMENT_3_RESULTS",
    "BRENEMAN_EXPERIMENT_4_RESULTS",
    "BRENEMAN_EXPERIMENT_5_RESULTS",
    "BRENEMAN_EXPERIMENT_6_RESULTS",
    "BRENEMAN_EXPERIMENT_7_RESULTS",
    "BRENEMAN_EXPERIMENT_10_RESULTS",
    "BRENEMAN_EXPERIMENT_8_RESULTS",
    "BRENEMAN_EXPERIMENT_9_RESULTS",
    "BRENEMAN_EXPERIMENT_11_RESULTS",
    "BRENEMAN_EXPERIMENT_12_RESULTS",
    "BRENEMAN_EXPERIMENT_PRIMARIES_CHROMATICITIES",
    "BRENEMAN_EXPERIMENTS",
]


@dataclass(frozen=True)
class BrenemanExperimentResult(MixinDataclassIterable):
    """
    Experiment result.

    Parameters
    ----------
    name
        Test colour name.
    uv_t
        Chromaticity coordinates :math:`uv_t^p` of test colour.
    uv_m
        Chromaticity coordinates :math:`uv_m^p` of matching colour.
    s_uv
        Interobserver variation (:math:`x10^3`) :math:`\\sigma_uv^p`.
    d_uv_i
        Deviation of individual linear transformation (:math:`x10^3`)
        :math:`\\delta_uv_i^p`.
    d_uv_g
        Deviation of individual linear transformation (:math:`x10^3`)
        :math:`\\delta_uv_g^p`.
    """

    name: str
    uv_t: NDArrayFloat
    uv_m: NDArrayFloat
    s_uv: NDArrayFloat | None = field(default_factory=lambda: None)
    d_uv_i: NDArrayFloat | None = field(default_factory=lambda: None)
    d_uv_g: NDArrayFloat | None = field(default_factory=lambda: None)

    def __post_init__(self) -> None:
        """Post-initialise the class."""

        object.__setattr__(self, "uv_t", as_float_array(self.uv_t))
        object.__setattr__(self, "uv_m", as_float_array(self.uv_m))

        if self.s_uv is not None:
            object.__setattr__(self, "s_uv", as_float_array(self.s_uv))

        if self.d_uv_i is not None:
            object.__setattr__(self, "d_uv_i", as_float_array(self.d_uv_i))

        if self.d_uv_g is not None:
            object.__setattr__(self, "d_uv_g", as_float_array(self.d_uv_g))


@dataclass(frozen=True)
class PrimariesChromaticityCoordinates(MixinDataclassIterable):
    """
    Chromaticity coordinates of the primaries.

    Parameters
    ----------
    experiment
        Experiment.
    illuminants
        Chromaticity coordinates :math:`uv_t^p` of test colour.
    Y
        White luminance :math:`Y` in :math:`cd/m^2`.
    P_uvp
        Chromaticity coordinates :math:`uv^p` of primary :math:`P`.
    D_uvp
        Chromaticity coordinates :math:`uv^p` of primary :math:`D`.
    T_uvp
        Chromaticity coordinates :math:`uv^p` of primary :math:`T`.
    """

    experiment: int
    illuminants: Tuple[str, str]
    Y: float
    P_uvp: NDArrayFloat | None = field(default_factory=lambda: None)
    D_uvp: NDArrayFloat | None = field(default_factory=lambda: None)
    T_uvp: NDArrayFloat | None = field(default_factory=lambda: None)

    def __post_init__(self) -> None:
        """Post-initialise the class."""

        object.__setattr__(self, "Y", as_float_array(self.Y))

        if self.P_uvp is not None:
            object.__setattr__(self, "P_uvp", as_float_array(self.P_uvp))

        if self.D_uvp is not None:
            object.__setattr__(self, "D_uvp", as_float_array(self.D_uvp))

        if self.T_uvp is not None:
            object.__setattr__(self, "T_uvp", as_float_array(self.T_uvp))


BRENEMAN_EXPERIMENT_1_RESULTS = (
    BrenemanExperimentResult(
        "Illuminant", np.array([0.259, 0.526]), np.array([0.200, 0.475])
    ),
    BrenemanExperimentResult(
        "Gray",
        np.array([0.259, 0.524]),
        np.array([0.199, 0.487]),
        np.array([4, 4]),
        np.array([2, 3]),
        np.array([0, 0]),
    ),
    BrenemanExperimentResult(
        "Red",
        np.array([0.459, 0.522]),
        np.array([0.420, 0.509]),
        np.array([19, 4]),
        np.array([-10, -7]),
        np.array([-19, -3]),
    ),
    BrenemanExperimentResult(
        "Skin",
        np.array([0.307, 0.526]),
        np.array([0.249, 0.497]),
        np.array([7, 4]),
        np.array([-1, 1]),
        np.array([-6, -1]),
    ),
    BrenemanExperimentResult(
        "Orange",
        np.array([0.360, 0.544]),
        np.array([0.302, 0.548]),
        np.array([12, 1]),
        np.array([1, -2]),
        np.array([-7, -6]),
    ),
    BrenemanExperimentResult(
        "Brown",
        np.array([0.350, 0.541]),
        np.array([0.290, 0.537]),
        np.array([11, 4]),
        np.array([3, 0]),
        np.array([-5, -3]),
    ),
    BrenemanExperimentResult(
        "Yellow",
        np.array([0.318, 0.550]),
        np.array([0.257, 0.554]),
        np.array([8, 2]),
        np.array([0, 2]),
        np.array([-5, -5]),
    ),
    BrenemanExperimentResult(
        "Foliage",
        np.array([0.258, 0.542]),
        np.array([0.192, 0.529]),
        np.array([4, 6]),
        np.array([3, 2]),
        np.array([3, -6]),
    ),
    BrenemanExperimentResult(
        "Green",
        np.array([0.193, 0.542]),
        np.array([0.129, 0.521]),
        np.array([7, 5]),
        np.array([3, 2]),
        np.array([9, -7]),
    ),
    BrenemanExperimentResult(
        "Blue-green",
        np.array([0.180, 0.516]),
        np.array([0.133, 0.469]),
        np.array([4, 6]),
        np.array([-3, -2]),
        np.array([2, -5]),
    ),
    BrenemanExperimentResult(
        "Blue",
        np.array([0.186, 0.445]),
        np.array([0.158, 0.340]),
        np.array([13, 33]),
        np.array([2, 7]),
        np.array([1, 13]),
    ),
    BrenemanExperimentResult(
        "Sky",
        np.array([0.226, 0.491]),
        np.array([0.178, 0.426]),
        np.array([3, 14]),
        np.array([1, -3]),
        np.array([0, -1]),
    ),
    BrenemanExperimentResult(
        "Purple",
        np.array([0.278, 0.456]),
        np.array([0.231, 0.365]),
        np.array([4, 25]),
        np.array([0, 2]),
        np.array([-5, 7]),
    ),
)
"""
*Breneman (1987)* experiment 1 results.

Notes
-----
-   Illuminants : *A*, *D65*
-   White Luminance : 1500 :math:`cd/m^2`
-   Observers Count : 7
"""


BRENEMAN_EXPERIMENT_2_RESULTS = (
    BrenemanExperimentResult(
        "Illuminant", np.array([0.222, 0.521]), np.array([0.204, 0.479])
    ),
    BrenemanExperimentResult(
        "Gray",
        np.array([0.227, 0.517]),
        np.array([0.207, 0.486]),
        np.array([2, 5]),
        np.array([-1, 0]),
        np.array([0, 0]),
    ),
    BrenemanExperimentResult(
        "Red",
        np.array([0.464, 0.520]),
        np.array([0.449, 0.511]),
        np.array([22, 3]),
        np.array([-8, -8]),
        np.array([-7, -2]),
    ),
    BrenemanExperimentResult(
        "Skin",
        np.array([0.286, 0.526]),
        np.array([0.263, 0.505]),
        np.array([7, 2]),
        np.array([0, -1]),
        np.array([0, -1]),
    ),
    BrenemanExperimentResult(
        "Orange",
        np.array([0.348, 0.546]),
        np.array([0.322, 0.545]),
        np.array([13, 3]),
        np.array([3, -1]),
        np.array([3, -2]),
    ),
    BrenemanExperimentResult(
        "Brown",
        np.array([0.340, 0.543]),
        np.array([0.316, 0.537]),
        np.array([11, 3]),
        np.array([1, 1]),
        np.array([0, 0]),
    ),
    BrenemanExperimentResult(
        "Yellow",
        np.array([0.288, 0.554]),
        np.array([0.265, 0.553]),
        np.array([5, 2]),
        np.array([-2, 2]),
        np.array([-1, -2]),
    ),
    BrenemanExperimentResult(
        "Foliage",
        np.array([0.244, 0.547]),
        np.array([0.221, 0.538]),
        np.array([4, 3]),
        np.array([-2, 1]),
        np.array([0, -3]),
    ),
    BrenemanExperimentResult(
        "Green",
        np.array([0.156, 0.548]),
        np.array([0.135, 0.532]),
        np.array([4, 3]),
        np.array([-1, 3]),
        np.array([3, -4]),
    ),
    BrenemanExperimentResult(
        "Blue-green",
        np.array([0.159, 0.511]),
        np.array([0.145, 0.472]),
        np.array([9, 7]),
        np.array([-1, 2]),
        np.array([2, 1]),
    ),
    BrenemanExperimentResult(
        "Blue",
        np.array([0.160, 0.406]),
        np.array([0.163, 0.331]),
        np.array([23, 31]),
        np.array([2, -3]),
        np.array([-1, 3]),
    ),
    BrenemanExperimentResult(
        "Sky",
        np.array([0.190, 0.481]),
        np.array([0.176, 0.431]),
        np.array([5, 24]),
        np.array([2, -2]),
        np.array([2, 0]),
    ),
    BrenemanExperimentResult(
        "Purple",
        np.array([0.258, 0.431]),
        np.array([0.244, 0.349]),
        np.array([4, 19]),
        np.array([-3, 13]),
        np.array([-4, 19]),
    ),
)
"""
*Breneman (1987)* experiment 2 results.

Notes
-----
-   Illuminants : *Projector*, *D55*
-   White Luminance : 1500 :math:`cd/m^2`
-   Observers Count : 7
"""


BRENEMAN_EXPERIMENT_3_RESULTS = (
    BrenemanExperimentResult(
        "Illuminant", np.array([0.223, 0.521]), np.array([0.206, 0.478])
    ),
    BrenemanExperimentResult(
        "Gray",
        np.array([0.228, 0.517]),
        np.array([0.211, 0.494]),
        np.array([1, 3]),
        np.array([0, 2]),
        np.array([0, 0]),
    ),
    BrenemanExperimentResult(
        "Red",
        np.array([0.462, 0.519]),
        np.array([0.448, 0.505]),
        np.array([11, 4]),
        np.array([-3, 6]),
        np.array([-4, 6]),
    ),
    BrenemanExperimentResult(
        "Skin",
        np.array([0.285, 0.524]),
        np.array([0.267, 0.507]),
        np.array([6, 3]),
        np.array([-1, 1]),
        np.array([-2, 1]),
    ),
    BrenemanExperimentResult(
        "Orange",
        np.array([0.346, 0.546]),
        np.array([0.325, 0.541]),
        np.array([11, 3]),
        np.array([1, -2]),
        np.array([2, 3]),
    ),
    BrenemanExperimentResult(
        "Brown",
        np.array([0.338, 0.543]),
        np.array([0.321, 0.532]),
        np.array([9, 6]),
        np.array([-3, 2]),
        np.array([-3, 7]),
    ),
    BrenemanExperimentResult(
        "Yellow",
        np.array([0.287, 0.554]),
        np.array([0.267, 0.548]),
        np.array([4, 5]),
        np.array([1, -2]),
        np.array([0, 5]),
    ),
    BrenemanExperimentResult(
        "Foliage",
        np.array([0.244, 0.547]),
        np.array([0.226, 0.531]),
        np.array([3, 6]),
        np.array([-1, 3]),
        np.array([-2, 8]),
    ),
    BrenemanExperimentResult(
        "Green",
        np.array([0.157, 0.548]),
        np.array([0.141, 0.528]),
        np.array([9, 6]),
        np.array([2, 2]),
        np.array([0, 6]),
    ),
    BrenemanExperimentResult(
        "Blue-green",
        np.array([0.160, 0.510]),
        np.array([0.151, 0.486]),
        np.array([8, 5]),
        np.array([-2, -1]),
        np.array([-2, -5]),
    ),
    BrenemanExperimentResult(
        "Blue",
        np.array([0.162, 0.407]),
        np.array([0.158, 0.375]),
        np.array([6, 7]),
        np.array([1, -6]),
        np.array([4, -23]),
    ),
    BrenemanExperimentResult(
        "Sky",
        np.array([0.191, 0.482]),
        np.array([0.179, 0.452]),
        np.array([4, 5]),
        np.array([0, 1]),
        np.array([1, -7]),
    ),
    BrenemanExperimentResult(
        "Purple",
        np.array([0.258, 0.432]),
        np.array([0.238, 0.396]),
        np.array([4, 8]),
        np.array([5, 3]),
        np.array([4, -11]),
    ),
)
"""
*Breneman (1987)* experiment 3 results.

Notes
-----
-   Illuminants : *Projector*, *D55*
-   White Luminance : 75 :math:`cd/m^2`
-   Observers Count : 7
"""


BRENEMAN_EXPERIMENT_4_RESULTS = (
    BrenemanExperimentResult(
        "Illuminant", np.array([0.258, 0.523]), np.array([0.199, 0.467])
    ),
    BrenemanExperimentResult(
        "Gray",
        np.array([0.257, 0.524]),
        np.array([0.205, 0.495]),
        np.array([2, 2]),
        np.array([0, 4]),
        np.array([0, 0]),
    ),
    BrenemanExperimentResult(
        "Red",
        np.array([0.460, 0.521]),
        np.array([0.416, 0.501]),
        np.array([11, 6]),
        np.array([-6, 4]),
        np.array([-6, 9]),
    ),
    BrenemanExperimentResult(
        "Skin",
        np.array([0.308, 0.526]),
        np.array([0.253, 0.503]),
        np.array([7, 3]),
        np.array([-1, 1]),
        np.array([-1, 0]),
    ),
    BrenemanExperimentResult(
        "Orange",
        np.array([0.360, 0.544]),
        np.array([0.303, 0.541]),
        np.array([14, 5]),
        np.array([1, -4]),
        np.array([1, 2]),
    ),
    BrenemanExperimentResult(
        "Brown",
        np.array([0.350, 0.541]),
        np.array([0.296, 0.527]),
        np.array([11, 7]),
        np.array([-2, 4]),
        np.array([-3, 9]),
    ),
    BrenemanExperimentResult(
        "Yellow",
        np.array([0.317, 0.550]),
        np.array([0.260, 0.547]),
        np.array([9, 5]),
        np.array([1, -3]),
        np.array([0, 3]),
    ),
    BrenemanExperimentResult(
        "Foliage",
        np.array([0.258, 0.543]),
        np.array([0.203, 0.520]),
        np.array([4, 6]),
        np.array([0, 8]),
        np.array([0, 9]),
    ),
    BrenemanExperimentResult(
        "Green",
        np.array([0.193, 0.543]),
        np.array([0.142, 0.516]),
        np.array([6, 9]),
        np.array([3, 8]),
        np.array([2, 6]),
    ),
    BrenemanExperimentResult(
        "Blue-green",
        np.array([0.180, 0.516]),
        np.array([0.140, 0.484]),
        np.array([9, 5]),
        np.array([-2, -1]),
        np.array([-1, -9]),
    ),
    BrenemanExperimentResult(
        "Blue",
        np.array([0.185, 0.445]),
        np.array([0.151, 0.394]),
        np.array([8, 10]),
        np.array([2, -8]),
        np.array([8, -24]),
    ),
    BrenemanExperimentResult(
        "Sky",
        np.array([0.225, 0.490]),
        np.array([0.180, 0.448]),
        np.array([4, 8]),
        np.array([1, -1]),
        np.array([3, -11]),
    ),
    BrenemanExperimentResult(
        "Purple",
        np.array([0.278, 0.455]),
        np.array([0.229, 0.388]),
        np.array([6, 14]),
        np.array([1, 12]),
        np.array([3, 0]),
    ),
)
"""
*Breneman (1987)* experiment 4 results.

Notes
-----
-   Illuminants : *A*, *D65*
-   White Luminance : 75 :math:`cd/m^2`
-   Observers Count : 7
"""


BRENEMAN_EXPERIMENT_5_RESULTS = (
    BrenemanExperimentResult(
        "Gray", np.array([0.028, 0.480]), np.array([0.212, 0.491]), np.array([2, 2])
    ),
    BrenemanExperimentResult(
        "Red", np.array([0.449, 0.512]), np.array([0.408, 0.514]), np.array([11, 5])
    ),
    BrenemanExperimentResult(
        "Skin", np.array([0.269, 0.505]), np.array([0.262, 0.511]), np.array([4, 2])
    ),
    BrenemanExperimentResult(
        "Orange", np.array([0.331, 0.548]), np.array([0.303, 0.545]), np.array([4, 3])
    ),
    BrenemanExperimentResult(
        "Brown", np.array([0.322, 0.541]), np.array([0.303, 0.538]), np.array([4, 4])
    ),
    BrenemanExperimentResult(
        "Yellow", np.array([0.268, 0.555]), np.array([0.264, 0.550]), np.array([3, 2])
    ),
    BrenemanExperimentResult(
        "Foliage", np.array([0.224, 0.538]), np.array([0.227, 0.535]), np.array([3, 3])
    ),
    BrenemanExperimentResult(
        "Green", np.array([0.134, 0.531]), np.array([0.159, 0.530]), np.array([9, 3])
    ),
    BrenemanExperimentResult(
        "Blue-green",
        np.array([0.145, 0.474]),
        np.array([0.165, 0.490]),
        np.array([8, 3]),
    ),
    BrenemanExperimentResult(
        "Blue", np.array([0.163, 0.329]), np.array([0.173, 0.378]), np.array([7, 12])
    ),
    BrenemanExperimentResult(
        "Sky", np.array([0.179, 0.438]), np.array([0.189, 0.462]), np.array([5, 4])
    ),
    BrenemanExperimentResult(
        "Purple", np.array([0.245, 0.364]), np.array([0.239, 0.401]), np.array([4, 16])
    ),
)
"""
*Breneman (1987)* experiment 5 results.

Notes
-----
-   Effective White Levels : 130 and 2120 :math:`cd/m^2`
-   Observers Count : 7
"""


BRENEMAN_EXPERIMENT_6_RESULTS = (
    BrenemanExperimentResult(
        "Illuminant", np.array([0.257, 0.525]), np.array([0.201, 0.482])
    ),
    BrenemanExperimentResult(
        "Gray",
        np.array([0.267, 0.521]),
        np.array([0.207, 0.485]),
        np.array([5, 3]),
        np.array([-1, 0]),
        np.array([0, 0]),
    ),
    BrenemanExperimentResult(
        "Red",
        np.array([0.457, 0.521]),
        np.array([0.398, 0.516]),
        np.array([9, 4]),
        np.array([-2, -5]),
        np.array([1, -9]),
    ),
    BrenemanExperimentResult(
        "Skin",
        np.array([0.316, 0.526]),
        np.array([0.253, 0.503]),
        np.array([5, 3]),
        np.array([-3, -2]),
        np.array([-1, -3]),
    ),
    BrenemanExperimentResult(
        "Orange",
        np.array([0.358, 0.545]),
        np.array([0.287, 0.550]),
        np.array([7, 3]),
        np.array([3, 0]),
        np.array([7, -6]),
    ),
    BrenemanExperimentResult(
        "Brown",
        np.array([0.350, 0.541]),
        np.array([0.282, 0.540]),
        np.array([6, 3]),
        np.array([-1, 0]),
        np.array([2, -5]),
    ),
    BrenemanExperimentResult(
        "Yellow",
        np.array([0.318, 0.551]),
        np.array([0.249, 0.556]),
        np.array([7, 2]),
        np.array([-1, 1]),
        np.array([2, -5]),
    ),
    BrenemanExperimentResult(
        "Foliage",
        np.array([0.256, 0.547]),
        np.array([0.188, 0.537]),
        np.array([5, 4]),
        np.array([3, 1]),
        np.array([4, -2]),
    ),
    BrenemanExperimentResult(
        "Green",
        np.array([0.193, 0.542]),
        np.array([0.133, 0.520]),
        np.array([13, 3]),
        np.array([5, -2]),
        np.array([5, -4]),
    ),
    BrenemanExperimentResult(
        "Blue-green",
        np.array([0.180, 0.516]),
        np.array([0.137, 0.466]),
        np.array([12, 10]),
        np.array([0, 0]),
        np.array([-2, 2]),
    ),
    BrenemanExperimentResult(
        "Blue",
        np.array([0.186, 0.445]),
        np.array([0.156, 0.353]),
        np.array([12, 45]),
        np.array([6, 1]),
        np.array([2, 6]),
    ),
    BrenemanExperimentResult(
        "Sky",
        np.array([0.225, 0.492]),
        np.array([0.178, 0.428]),
        np.array([6, 14]),
        np.array([1, -1]),
        np.array([-1, 3]),
    ),
    BrenemanExperimentResult(
        "Purple",
        np.array([0.276, 0.456]),
        np.array([0.227, 0.369]),
        np.array([6, 27]),
        np.array([-2, 4]),
        np.array([-3, 9]),
    ),
)
"""
*Breneman (1987)* experiment 6 results.

Notes
-----
-   Illuminants : *A*, *D55*
-   White Luminance : 11100 :math:`cd/m^2`
-   Observers Count : 8
"""


BRENEMAN_EXPERIMENT_7_RESULTS = (
    BrenemanExperimentResult(
        "Gray", np.array([0.208, 0.481]), np.array([0.211, 0.486]), np.array([2, 3])
    ),
    BrenemanExperimentResult(
        "Red", np.array([0.448, 0.512]), np.array([0.409, 0.516]), np.array([9, 2])
    ),
    BrenemanExperimentResult(
        "Skin", np.array([0.269, 0.505]), np.array([0.256, 0.506]), np.array([4, 3])
    ),
    BrenemanExperimentResult(
        "Orange", np.array([0.331, 0.549]), np.array([0.305, 0.547]), np.array([5, 4])
    ),
    BrenemanExperimentResult(
        "Brown", np.array([0.322, 0.541]), np.array([0.301, 0.539]), np.array([5, 2])
    ),
    BrenemanExperimentResult(
        "Yellow", np.array([0.268, 0.555]), np.array([0.257, 0.552]), np.array([3, 4])
    ),
    BrenemanExperimentResult(
        "Foliage", np.array([0.225, 0.538]), np.array([0.222, 0.536]), np.array([3, 2])
    ),
    BrenemanExperimentResult(
        "Green", np.array([0.135, 0.531]), np.array([0.153, 0.529]), np.array([8, 2])
    ),
    BrenemanExperimentResult(
        "Blue-green",
        np.array([0.145, 0.475]),
        np.array([0.160, 0.484]),
        np.array([3, 5]),
    ),
    BrenemanExperimentResult(
        "Blue", np.array([0.163, 0.331]), np.array([0.171, 0.379]), np.array([4, 11])
    ),
    BrenemanExperimentResult(
        "Sky", np.array([0.179, 0.438]), np.array([0.187, 0.452]), np.array([4, 7])
    ),
    BrenemanExperimentResult(
        "Purple", np.array([0.245, 0.365]), np.array([0.240, 0.398]), np.array([4, 10])
    ),
)
"""
*Breneman (1987)* experiment 7 results.

Notes
-----
-   Effective White Levels : 850 and 11100 :math:`cd/m^2`
-   Observers Count : 8
"""


BRENEMAN_EXPERIMENT_8_RESULTS = (
    BrenemanExperimentResult(
        "Illuminant", np.array([0.258, 0.524]), np.array([0.195, 0.469])
    ),
    BrenemanExperimentResult(
        "Gray",
        np.array([0.257, 0.525]),
        np.array([0.200, 0.494]),
        np.array([2, 3]),
        np.array([1, 2]),
        np.array([0, 0]),
    ),
    BrenemanExperimentResult(
        "Red",
        np.array([0.458, 0.522]),
        np.array([0.410, 0.508]),
        np.array([12, 4]),
        np.array([-3, 5]),
        np.array([-7, 2]),
    ),
    BrenemanExperimentResult(
        "Skin",
        np.array([0.308, 0.526]),
        np.array([0.249, 0.502]),
        np.array([6, 2]),
        np.array([-1, 1]),
        np.array([-3, -1]),
    ),
    BrenemanExperimentResult(
        "Orange",
        np.array([0.359, 0.545]),
        np.array([0.299, 0.545]),
        np.array([12, 4]),
        np.array([0, -2]),
        np.array([-3, 0]),
    ),
    BrenemanExperimentResult(
        "Brown",
        np.array([0.349, 0.540]),
        np.array([0.289, 0.532]),
        np.array([10, 4]),
        np.array([0, 1]),
        np.array([-2, 2]),
    ),
    BrenemanExperimentResult(
        "Yellow",
        np.array([0.317, 0.550]),
        np.array([0.256, 0.549]),
        np.array([9, 5]),
        np.array([0, -3]),
        np.array([-3, 1]),
    ),
    BrenemanExperimentResult(
        "Foliage",
        np.array([0.260, 0.545]),
        np.array([0.198, 0.529]),
        np.array([5, 5]),
        np.array([3, 1]),
        np.array([0, 3]),
    ),
    BrenemanExperimentResult(
        "Green",
        np.array([0.193, 0.543]),
        np.array([0.137, 0.520]),
        np.array([9, 5]),
        np.array([3, 0]),
        np.array([2, 1]),
    ),
    BrenemanExperimentResult(
        "Blue-green",
        np.array([0.182, 0.516]),
        np.array([0.139, 0.477]),
        np.array([9, 4]),
        np.array([-3, 0]),
        np.array([-2, -4]),
    ),
    BrenemanExperimentResult(
        "Blue",
        np.array([0.184, 0.444]),
        np.array([0.150, 0.387]),
        np.array([5, 11]),
        np.array([3, -10]),
        np.array([6, -22]),
    ),
    BrenemanExperimentResult(
        "Sky",
        np.array([0.224, 0.489]),
        np.array([0.177, 0.439]),
        np.array([5, 6]),
        np.array([1, 1]),
        np.array([1, -7]),
    ),
    BrenemanExperimentResult(
        "Purple",
        np.array([0.277, 0.454]),
        np.array([0.226, 0.389]),
        np.array([4, 10]),
        np.array([1, 4]),
        np.array([1, -8]),
    ),
)
"""
*Breneman (1987)* experiment 8 results.

Notes
-----
-   Illuminants : *A*, *D65*
-   White Luminance : 350 :math:`cd/m^2`
-   Observers Count : 8
"""


BRENEMAN_EXPERIMENT_9_RESULTS = (
    BrenemanExperimentResult(
        "Illuminant", np.array([0.254, 0.525]), np.array([0.195, 0.465])
    ),
    BrenemanExperimentResult(
        "Gray",
        np.array([0.256, 0.524]),
        np.array([0.207, 0.496]),
        np.array([4, 6]),
        np.array([3, 2]),
        np.array([0, 0]),
    ),
    BrenemanExperimentResult(
        "Red",
        np.array([0.459, 0.521]),
        np.array([0.415, 0.489]),
        np.array([20, 14]),
        np.array([2, 12]),
        np.array([-2, 21]),
    ),
    BrenemanExperimentResult(
        "Skin",
        np.array([0.307, 0.525]),
        np.array([0.261, 0.500]),
        np.array([7, 7]),
        np.array([0, 1]),
        np.array([-5, 2]),
    ),
    BrenemanExperimentResult(
        "Orange",
        np.array([0.359, 0.545]),
        np.array([0.313, 0.532]),
        np.array([7, 5]),
        np.array([-2, -3]),
        np.array([-6, 13]),
    ),
    BrenemanExperimentResult(
        "Brown",
        np.array([0.349, 0.540]),
        np.array([0.302, 0.510]),
        np.array([11, 15]),
        np.array([0, 12]),
        np.array([-5, 24]),
    ),
    BrenemanExperimentResult(
        "Yellow",
        np.array([0.317, 0.550]),
        np.array([0.268, 0.538]),
        np.array([7, 10]),
        np.array([1, -4]),
        np.array([-4, 12]),
    ),
    BrenemanExperimentResult(
        "Foliage",
        np.array([0.259, 0.544]),
        np.array([0.212, 0.510]),
        np.array([10, 11]),
        np.array([0, 14]),
        np.array([-4, 22]),
    ),
    BrenemanExperimentResult(
        "Green",
        np.array([0.193, 0.542]),
        np.array([0.150, 0.506]),
        np.array([6, 10]),
        np.array([-1, 13]),
        np.array([-2, 15]),
    ),
    BrenemanExperimentResult(
        "Blue-green",
        np.array([0.181, 0.517]),
        np.array([0.144, 0.487]),
        np.array([9, 6]),
        np.array([-3, 0]),
        np.array([-1, -9]),
    ),
    BrenemanExperimentResult(
        "Blue",
        np.array([0.184, 0.444]),
        np.array([0.155, 0.407]),
        np.array([4, 11]),
        np.array([-2, -6]),
        np.array([6, -36]),
    ),
    BrenemanExperimentResult(
        "Sky",
        np.array([0.225, 0.490]),
        np.array([0.183, 0.458]),
        np.array([5, 8]),
        np.array([1, -3]),
        np.array([2, -19]),
    ),
    BrenemanExperimentResult(
        "Purple",
        np.array([0.276, 0.454]),
        np.array([0.233, 0.404]),
        np.array([7, 12]),
        np.array([2, 9]),
        np.array([0, -16]),
    ),
    BrenemanExperimentResult(
        "(Gray)h", np.array([0.256, 0.525]), np.array([0.208, 0.498])
    ),
    BrenemanExperimentResult(
        "(Red)h",
        np.array([0.456, 0.521]),
        np.array([0.416, 0.501]),
        np.array([15, 7]),
        None,
        np.array([-6, -9]),
    ),
    BrenemanExperimentResult(
        "(Brown)h",
        np.array([0.349, 0.539]),
        np.array([0.306, 0.526]),
        np.array([11, 8]),
        None,
        np.array([-8, 7]),
    ),
    BrenemanExperimentResult(
        "(Foliage)h",
        np.array([0.260, 0.545]),
        np.array([0.213, 0.528]),
        np.array([7, 9]),
        None,
        np.array([-4, 5]),
    ),
    BrenemanExperimentResult(
        "(Green)h",
        np.array([0.193, 0.543]),
        np.array([0.149, 0.525]),
        np.array([10, 8]),
        None,
        np.array([-1, -1]),
    ),
    BrenemanExperimentResult(
        "(Blue)h",
        np.array([0.184, 0.444]),
        np.array([0.156, 0.419]),
        np.array([7, 8]),
        None,
        np.array([4, -45]),
    ),
    BrenemanExperimentResult(
        "(Purple)h",
        np.array([0.277, 0.456]),
        np.array([0.236, 0.422]),
        np.array([6, 11]),
        None,
        np.array([-2, -29]),
    ),
)
"""
*Breneman (1987)* experiment 9 results.

Notes
-----
-   Illuminants : *A*, *D65*
-   White Luminance : 15 :math:`cd/m^2`
-   Observers Count : 8
-   The colors indicated by (.)h are the darker colors presented at the higher
    luminescence level of the lighter colors.
"""


BRENEMAN_EXPERIMENT_10_RESULTS = (
    BrenemanExperimentResult(
        "Gray", np.array([0.208, 0.482]), np.array([0.213, 0.494]), np.array([3, 3])
    ),
    BrenemanExperimentResult(
        "Red", np.array([0.447, 0.512]), np.array([0.411, 0.506]), np.array([15, 7])
    ),
    BrenemanExperimentResult(
        "Skin", np.array([0.269, 0.505]), np.array([0.269, 0.511]), np.array([4, 3])
    ),
    BrenemanExperimentResult(
        "Orange", np.array([0.331, 0.549]), np.array([0.315, 0.536]), np.array([7, 8])
    ),
    BrenemanExperimentResult(
        "Brown", np.array([0.323, 0.542]), np.array([0.310, 0.526]), np.array([6, 8])
    ),
    BrenemanExperimentResult(
        "Yellow", np.array([0.268, 0.556]), np.array([0.268, 0.541]), np.array([3, 6])
    ),
    BrenemanExperimentResult(
        "Foliage", np.array([0.226, 0.538]), np.array([0.230, 0.525]), np.array([4, 8])
    ),
    BrenemanExperimentResult(
        "Green", np.array([0.135, 0.531]), np.array([0.158, 0.524]), np.array([6, 3])
    ),
    BrenemanExperimentResult(
        "Blue-green",
        np.array([0.145, 0.476]),
        np.array([0.161, 0.491]),
        np.array([4, 4]),
    ),
    BrenemanExperimentResult(
        "Blue", np.array([0.163, 0.330]), np.array([0.171, 0.377]), np.array([6, 19])
    ),
    BrenemanExperimentResult(
        "Sky", np.array([0.179, 0.439]), np.array([0.187, 0.465]), np.array([5, 5])
    ),
    BrenemanExperimentResult(
        "Purple", np.array([0.245, 0.366]), np.array([0.240, 0.402]), np.array([3, 12])
    ),
)
"""
*Breneman (1987)* experiment 10 results.

Notes
-----
-   Effective White Levels : 15 and 270 :math:`cd/m^2`
-   Observers Count : 7
"""


BRENEMAN_EXPERIMENT_11_RESULTS = (
    BrenemanExperimentResult(
        "Illuminant", np.array([0.208, 0.482]), np.array([0.174, 0.520])
    ),
    BrenemanExperimentResult(
        "Gray",
        np.array([0.209, 0.483]),
        np.array([0.176, 0.513]),
        np.array([3, 4]),
        np.array([2, 2]),
        np.array([0, 0]),
    ),
    BrenemanExperimentResult(
        "Red",
        np.array([0.450, 0.512]),
        np.array([0.419, 0.524]),
        np.array([10, 2]),
        np.array([3, 2]),
        np.array([8, -1]),
    ),
    BrenemanExperimentResult(
        "Skin",
        np.array([0.268, 0.506]),
        np.array([0.240, 0.528]),
        np.array([6, 2]),
        np.array([-4, 0]),
        np.array([-3, 0]),
    ),
    BrenemanExperimentResult(
        "Orange",
        np.array([0.331, 0.547]),
        np.array([0.293, 0.553]),
        np.array([6, 2]),
        np.array([3, -1]),
        np.array([5, 1]),
    ),
    BrenemanExperimentResult(
        "Brown",
        np.array([0.323, 0.542]),
        np.array([0.290, 0.552]),
        np.array([5, 2]),
        np.array([-1, -3]),
        np.array([0, -1]),
    ),
    BrenemanExperimentResult(
        "Yellow",
        np.array([0.266, 0.549]),
        np.array([0.236, 0.557]),
        np.array([4, 2]),
        np.array([-3, -2]),
        np.array([-4, 2]),
    ),
    BrenemanExperimentResult(
        "Foliage",
        np.array([0.227, 0.538]),
        np.array([0.194, 0.552]),
        np.array([4, 2]),
        np.array([2, -3]),
        np.array([-1, 1]),
    ),
    BrenemanExperimentResult(
        "Green",
        np.array([0.146, 0.534]),
        np.array([0.118, 0.551]),
        np.array([8, 3]),
        np.array([4, -2]),
        np.array([-6, 3]),
    ),
    BrenemanExperimentResult(
        "Blue-green",
        np.array([0.160, 0.475]),
        np.array([0.130, 0.513]),
        np.array([9, 4]),
        np.array([1, -1]),
        np.array([-4, -3]),
    ),
    BrenemanExperimentResult(
        "Blue",
        np.array([0.177, 0.340]),
        np.array([0.133, 0.427]),
        np.array([6, 14]),
        np.array([4, -17]),
        np.array([11, -29]),
    ),
    BrenemanExperimentResult(
        "Sky",
        np.array([0.179, 0.438]),
        np.array([0.146, 0.482]),
        np.array([6, 10]),
        np.array([1, 4]),
        np.array([0, -1]),
    ),
    BrenemanExperimentResult(
        "Purple",
        np.array([0.245, 0.366]),
        np.array([0.216, 0.419]),
        np.array([4, 13]),
        np.array([-3, 8]),
        np.array([4, -2]),
    ),
)
"""
*Breneman (1987)* experiment 11 results.

Notes
-----
-   Illuminants : *green*, *D65*
-   White Luminance : 1560 :math:`cd/m^2`
-   Observers Count : 7
"""


BRENEMAN_EXPERIMENT_12_RESULTS = (
    BrenemanExperimentResult(
        "Illuminant", np.array([0.205, 0.482]), np.array([0.174, 0.519])
    ),
    BrenemanExperimentResult(
        "Gray",
        np.array([0.208, 0.482]),
        np.array([0.181, 0.507]),
        np.array([4, 3]),
        np.array([0, 1]),
        np.array([0, 0]),
    ),
    BrenemanExperimentResult(
        "Red",
        np.array([0.451, 0.512]),
        np.array([0.422, 0.526]),
        np.array([20, 3]),
        np.array([0, -5]),
        np.array([10, -5]),
    ),
    BrenemanExperimentResult(
        "Skin",
        np.array([0.268, 0.506]),
        np.array([0.244, 0.525]),
        np.array([5, 2]),
        np.array([-6, 0]),
        np.array([-2, -1]),
    ),
    BrenemanExperimentResult(
        "Orange",
        np.array([0.331, 0.548]),
        np.array([0.292, 0.553]),
        np.array([10, 2]),
        np.array([5, 2]),
        np.array([11, 1]),
    ),
    BrenemanExperimentResult(
        "Brown",
        np.array([0.324, 0.542]),
        np.array([0.286, 0.554]),
        np.array([8, 1]),
        np.array([5, -3]),
        np.array([10, -4]),
    ),
    BrenemanExperimentResult(
        "Yellow",
        np.array([0.266, 0.548]),
        np.array([0.238, 0.558]),
        np.array([6, 2]),
        np.array([-3, -1]),
        np.array([-1, -2]),
    ),
    BrenemanExperimentResult(
        "Foliage",
        np.array([0.227, 0.538]),
        np.array([0.196, 0.555]),
        np.array([6, 3]),
        np.array([3, -4]),
        np.array([2, -5]),
    ),
    BrenemanExperimentResult(
        "Green",
        np.array([0.145, 0.534]),
        np.array([0.124, 0.551]),
        np.array([8, 6]),
        np.array([1, -1]),
        np.array([-8, -1]),
    ),
    BrenemanExperimentResult(
        "Blue-green",
        np.array([0.160, 0.474]),
        np.array([0.135, 0.505]),
        np.array([5, 2]),
        np.array([1, -1]),
        np.array([-4, -3]),
    ),
    BrenemanExperimentResult(
        "Blue",
        np.array([0.178, 0.339]),
        np.array([0.149, 0.392]),
        np.array([4, 20]),
        np.array([-1, -5]),
        np.array([3, -7]),
    ),
    BrenemanExperimentResult(
        "Sky",
        np.array([0.179, 0.440]),
        np.array([0.150, 0.473]),
        np.array([4, 8]),
        np.array([3, 2]),
        np.array([2, 0]),
    ),
    BrenemanExperimentResult(
        "Purple",
        np.array([0.246, 0.366]),
        np.array([0.222, 0.404]),
        np.array([5, 15]),
        np.array([-4, 2]),
        np.array([4, 2]),
    ),
)
"""
*Breneman (1987)* experiment 12 results.

Notes
-----
-   Illuminants : *D55*, *green*
-   White Luminance : 75 :math:`cd/m^2`
-   Observers Count : 7
"""


BRENEMAN_EXPERIMENT_PRIMARIES_CHROMATICITIES = {
    1: PrimariesChromaticityCoordinates(
        1,
        ("A", "D65"),
        1500,
        np.array([0.671, 0.519]),
        np.array([-0.586, 0.627]),
        np.array([0.253, 0.016]),
    ),
    2: PrimariesChromaticityCoordinates(
        2,
        ("Projector", "D55"),
        1500,
        np.array([0.675, 0.523]),
        np.array([-0.466, 0.617]),
        np.array([0.255, 0.018]),
    ),
    3: PrimariesChromaticityCoordinates(
        3,
        ("Projector", "D55"),
        75,
        np.array([0.664, 0.510]),
        np.array([-0.256, 0.729]),
        np.array([0.244, 0.003]),
    ),
    4: PrimariesChromaticityCoordinates(
        4,
        ("A", "D65"),
        75,
        np.array([0.674, 0.524]),
        np.array([-0.172, 0.628]),
        np.array([0.218, -0.026]),
    ),
    6: PrimariesChromaticityCoordinates(
        6,
        ("A", "D55"),
        11100,
        np.array([0.659, 0.506]),
        np.array([-0.141, 0.615]),
        np.array([0.249, 0.009]),
    ),
    8: PrimariesChromaticityCoordinates(
        8,
        ("A", "D65"),
        350,
        np.array([0.659, 0.505]),
        np.array([-0.246, 0.672]),
        np.array([0.235, -0.006]),
    ),
    9: PrimariesChromaticityCoordinates(
        9,
        ("A", "D65"),
        15,
        np.array([0.693, 0.546]),
        np.array([-0.446, 0.773]),
        np.array([0.221, -0.023]),
    ),
    11: PrimariesChromaticityCoordinates(
        11,
        ("D55", "green"),
        1560,
        np.array([0.680, 0.529]),
        np.array([0.018, 0.576]),
        np.array([0.307, 0.080]),
    ),
    12: PrimariesChromaticityCoordinates(
        12,
        ("D55", "green"),
        75,
        np.array([0.661, 0.505]),
        np.array([0.039, 0.598]),
        np.array([0.345, 0.127]),
    ),
}
if is_documentation_building():  # pragma: no cover
    BRENEMAN_EXPERIMENT_PRIMARIES_CHROMATICITIES = DocstringDict(
        BRENEMAN_EXPERIMENT_PRIMARIES_CHROMATICITIES
    )
    BRENEMAN_EXPERIMENT_PRIMARIES_CHROMATICITIES.__doc__ = """
*Breneman (1987)* experiments primaries chromaticities.

References
----------
:cite:`Breneman1987b`

BRENEMAN_EXPERIMENT_PRIMARIES_CHROMATICITIES : dict
"""

BRENEMAN_EXPERIMENTS = {
    1: BRENEMAN_EXPERIMENT_1_RESULTS,
    2: BRENEMAN_EXPERIMENT_2_RESULTS,
    3: BRENEMAN_EXPERIMENT_3_RESULTS,
    4: BRENEMAN_EXPERIMENT_4_RESULTS,
    5: BRENEMAN_EXPERIMENT_5_RESULTS,
    6: BRENEMAN_EXPERIMENT_6_RESULTS,
    7: BRENEMAN_EXPERIMENT_7_RESULTS,
    8: BRENEMAN_EXPERIMENT_8_RESULTS,
    9: BRENEMAN_EXPERIMENT_9_RESULTS,
    10: BRENEMAN_EXPERIMENT_10_RESULTS,
    11: BRENEMAN_EXPERIMENT_11_RESULTS,
    12: BRENEMAN_EXPERIMENT_12_RESULTS,
}
if is_documentation_building():  # pragma: no cover
    BRENEMAN_EXPERIMENTS = DocstringDict(BRENEMAN_EXPERIMENTS)
    BRENEMAN_EXPERIMENTS.__doc__ = """
*Breneman (1987)* experiments.

References
----------
:cite:`Breneman1987b`
"""
