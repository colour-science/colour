"""
Sony Colourspaces
=================

Defines the *Sony* colourspaces:

-   :attr:`colour.models.RGB_COLOURSPACE_S_GAMUT`.
-   :attr:`colour.models.RGB_COLOURSPACE_S_GAMUT3`.
-   :attr:`colour.models.RGB_COLOURSPACE_S_GAMUT3_CINE`.
-   :attr:`colour.models.RGB_COLOURSPACE_VENICE_S_GAMUT3`.
-   :attr:`colour.models.RGB_COLOURSPACE_VENICE_S_GAMUT3_CINE`.

Notes
-----
-   The *Venice S-Gamut3* and *Venice S-Gamut3.Cine* primaries and whitepoint
    were derived with the following `Google Colab Notebook \
<https://colab.research.google.com/drive/1ZGTij7jT8eZRMPUkyWlv_x5ix5Q5twMB>`__.

References
----------
-   :cite:`Gaggioni` : Gaggioni, H., Dhanendra, P., Yamashita, J., Kawada, N.,
    Endo, K., & Clark, C. (n.d.). S-Log: A new LUT for digital production
    mastering and interchange applications (Vol. 709, pp. 1-13).
    http://pro.sony.com/bbsccms/assets/files/mkt/cinema/solutions/slog_manual.pdf
-   :cite:`SonyCorporation` : Sony Corporation. (n.d.). S-Log Whitepaper (pp.
    1-17). http://www.theodoropoulos.info/attachments/076_on%20S-Log.pdf
-   :cite:`SonyCorporationd` : Sony Corporation. (n.d.). Technical Summary
    for S-Gamut3.Cine/S-Log3 and S-Gamut3/S-Log3 (pp. 1-7).
    http://community.sony.com/sony/attachments/sony/\
large-sensor-camera-F5-F55/12359/2/\
TechnicalSummary_for_S-Gamut3Cine_S-Gamut3_S-Log3_V1_00.pdf
-   :cite:`SonyCorporatione` : Sony Corporation. (n.d.).
    S-Gamut3_S-Gamut3Cine_Matrix.xlsx.
    https://community.sony.com/sony/attachments/sony/\
large-sensor-camera-F5-F55/12359/3/S-Gamut3_S-Gamut3Cine_Matrix.xlsx
-   :cite:`SonyElectronicsCorporation2020` : Sony Electronics Corporation.
    (2020). IDT.Sony.Venice_SLog3_SGamut3.ctl. https://github.com/ampas/\
aces-dev/blob/710ecbe52c87ce9f4a1e02c8ddf7ea0d6b611cc8/transforms/ctl/idt/\
vendorSupplied/sony/IDT.Sony.Venice_SLog3_SGamut3.ctl
-   :cite:`SonyElectronicsCorporation2020a` : Sony Electronics Corporation.
    (2020). IDT.Sony.Venice_SLog3_SGamut3Cine.ctl. https://github.com/ampas/\
aces-dev/blob/710ecbe52c87ce9f4a1e02c8ddf7ea0d6b611cc8/transforms/ctl/idt/\
vendorSupplied/sony/IDT.Sony.Venice_SLog3_SGamut3Cine.ctl
"""

from __future__ import annotations

import numpy as np

from colour.colorimetry import CCS_ILLUMINANTS
from colour.hints import NDArray
from colour.models.rgb import (
    RGB_Colourspace,
    log_encoding_SLog2,
    log_decoding_SLog2,
    log_encoding_SLog3,
    log_decoding_SLog3,
    normalised_primary_matrix,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "PRIMARIES_S_GAMUT",
    "WHITEPOINT_NAME_S_GAMUT",
    "CCS_WHITEPOINT_S_GAMUT",
    "MATRIX_S_GAMUT_TO_XYZ",
    "MATRIX_XYZ_TO_S_GAMUT",
    "RGB_COLOURSPACE_S_GAMUT",
    "PRIMARIES_S_GAMUT3",
    "WHITEPOINT_NAME_S_GAMUT3",
    "CCS_WHITEPOINT_S_GAMUT3",
    "MATRIX_S_GAMUT3_TO_XYZ",
    "MATRIX_XYZ_TO_S_GAMUT3",
    "RGB_COLOURSPACE_S_GAMUT3",
    "PRIMARIES_S_GAMUT3_CINE",
    "WHITEPOINT_NAME_S_GAMUT3_CINE",
    "CCS_WHITEPOINT_S_GAMUT3_CINE",
    "MATRIX_S_GAMUT3_CINE_TO_XYZ",
    "MATRIX_XYZ_TO_S_GAMUT3_CINE",
    "RGB_COLOURSPACE_S_GAMUT3_CINE",
    "PRIMARIES_VENICE_S_GAMUT3",
    "WHITEPOINT_NAME_VENICE_S_GAMUT3",
    "CCS_WHITEPOINT_VENICE_S_GAMUT3",
    "MATRIX_VENICE_S_GAMUT3_TO_XYZ",
    "MATRIX_XYZ_TO_VENICE_S_GAMUT3",
    "RGB_COLOURSPACE_VENICE_S_GAMUT3",
    "PRIMARIES_VENICE_S_GAMUT3_CINE",
    "WHITEPOINT_NAME_VENICE_S_GAMUT3_CINE",
    "CCS_WHITEPOINT_VENICE_S_GAMUT3_CINE",
    "MATRIX_VENICE_S_GAMUT3_CINE_TO_XYZ",
    "MATRIX_XYZ_TO_VENICE_S_GAMUT3_CINE",
    "RGB_COLOURSPACE_VENICE_S_GAMUT3_CINE",
]

PRIMARIES_S_GAMUT: NDArray = np.array(
    [
        [0.7300, 0.2800],
        [0.1400, 0.8550],
        [0.1000, -0.0500],
    ]
)
"""*S-Gamut* colourspace primaries."""

WHITEPOINT_NAME_S_GAMUT: str = "D65"
"""*S-Gamut* colourspace whitepoint name."""

CCS_WHITEPOINT_S_GAMUT: NDArray = CCS_ILLUMINANTS[
    "CIE 1931 2 Degree Standard Observer"
][WHITEPOINT_NAME_S_GAMUT]
"""*S-Gamut* colourspace whitepoint chromaticity coordinates."""

MATRIX_S_GAMUT_TO_XYZ: NDArray = np.array(
    [
        [0.7064827132, 0.1288010498, 0.1151721641],
        [0.2709796708, 0.7866064112, -0.0575860820],
        [-0.0096778454, 0.0046000375, 1.0941355587],
    ]
)
"""*S-Gamut* colourspace to *CIE XYZ* tristimulus values matrix."""

MATRIX_XYZ_TO_S_GAMUT: NDArray = np.array(
    [
        [1.5073998991, -0.2458221374, -0.1716116808],
        [-0.5181517271, 1.3553912409, 0.1258786682],
        [0.0155116982, -0.0078727714, 0.9119163656],
    ]
)
"""*CIE XYZ* tristimulus values to *S-Gamut* colourspace matrix."""

RGB_COLOURSPACE_S_GAMUT: RGB_Colourspace = RGB_Colourspace(
    "S-Gamut",
    PRIMARIES_S_GAMUT,
    CCS_WHITEPOINT_S_GAMUT,
    WHITEPOINT_NAME_S_GAMUT,
    MATRIX_S_GAMUT_TO_XYZ,
    MATRIX_XYZ_TO_S_GAMUT,
    log_encoding_SLog2,
    log_decoding_SLog2,
)
RGB_COLOURSPACE_S_GAMUT.__doc__ = """
*S-Gamut* colourspace.

References
----------
:cite:`Gaggioni`, :cite:`SonyCorporation`
"""

PRIMARIES_S_GAMUT3: NDArray = PRIMARIES_S_GAMUT
"""*S-Gamut3* colourspace primaries."""

WHITEPOINT_NAME_S_GAMUT3: str = WHITEPOINT_NAME_S_GAMUT
"""*S-Gamut3* colourspace whitepoint name."""

CCS_WHITEPOINT_S_GAMUT3: NDArray = CCS_WHITEPOINT_S_GAMUT
"""*S-Gamut3* colourspace whitepoint chromaticity coordinates."""

MATRIX_S_GAMUT3_TO_XYZ: NDArray = MATRIX_S_GAMUT_TO_XYZ
"""*S-Gamut3* colourspace to *CIE XYZ* tristimulus values matrix."""

MATRIX_XYZ_TO_S_GAMUT3: NDArray = MATRIX_XYZ_TO_S_GAMUT
"""*CIE XYZ* tristimulus values to *S-Gamut3* colourspace matrix."""

RGB_COLOURSPACE_S_GAMUT3: RGB_Colourspace = RGB_Colourspace(
    "S-Gamut3",
    PRIMARIES_S_GAMUT3,
    CCS_WHITEPOINT_S_GAMUT3,
    WHITEPOINT_NAME_S_GAMUT3,
    MATRIX_S_GAMUT3_TO_XYZ,
    MATRIX_XYZ_TO_S_GAMUT3,
    log_encoding_SLog3,
    log_decoding_SLog3,
)
RGB_COLOURSPACE_S_GAMUT3.__doc__ = """
*S-Gamut3* colourspace.

References
----------
:cite:`SonyCorporationd`
"""

PRIMARIES_S_GAMUT3_CINE: NDArray = np.array(
    [
        [0.76600, 0.27500],
        [0.22500, 0.80000],
        [0.08900, -0.08700],
    ]
)
"""*S-Gamut3.Cine* colourspace primaries."""

WHITEPOINT_NAME_S_GAMUT3_CINE: str = WHITEPOINT_NAME_S_GAMUT
"""*S-Gamut3.Cine* colourspace whitepoint name."""

CCS_WHITEPOINT_S_GAMUT3_CINE: NDArray = CCS_WHITEPOINT_S_GAMUT
"""*S-Gamut3.Cine* colourspace whitepoint chromaticity coordinates."""

MATRIX_S_GAMUT3_CINE_TO_XYZ: NDArray = np.array(
    [
        [0.5990839208, 0.2489255161, 0.1024464902],
        [0.2150758201, 0.8850685017, -0.1001443219],
        [-0.0320658495, -0.0276583907, 1.1487819910],
    ]
)
"""*S-Gamut3.Cine* colourspace to *CIE XYZ* tristimulus values matrix."""

MATRIX_XYZ_TO_S_GAMUT3_CINE: NDArray = np.array(
    [
        [1.8467789693, -0.5259861230, -0.2105452114],
        [-0.4441532629, 1.2594429028, 0.1493999729],
        [0.0408554212, 0.0156408893, 0.8682072487],
    ]
)
"""*CIE XYZ* tristimulus values to *S-Gamut3.Cine* colourspace matrix."""

RGB_COLOURSPACE_S_GAMUT3_CINE: RGB_Colourspace = RGB_Colourspace(
    "S-Gamut3.Cine",
    PRIMARIES_S_GAMUT3_CINE,
    CCS_WHITEPOINT_S_GAMUT3_CINE,
    WHITEPOINT_NAME_S_GAMUT3_CINE,
    MATRIX_S_GAMUT3_CINE_TO_XYZ,
    MATRIX_XYZ_TO_S_GAMUT3_CINE,
    log_encoding_SLog3,
    log_decoding_SLog3,
)
RGB_COLOURSPACE_S_GAMUT3_CINE.__doc__ = """
*S-Gamut3.Cine* colourspace.

References
----------
:cite:`SonyCorporatione`
"""

PRIMARIES_VENICE_S_GAMUT3: NDArray = np.array(
    [
        [0.740464264304292, 0.279364374750660],
        [0.089241145423286, 0.893809528608105],
        [0.110488236673827, -0.052579333080476],
    ]
)
"""*Venice S-Gamut3* colourspace primaries."""

WHITEPOINT_NAME_VENICE_S_GAMUT3: str = WHITEPOINT_NAME_S_GAMUT
"""*Venice S-Gamut3* colourspace whitepoint name."""

CCS_WHITEPOINT_VENICE_S_GAMUT3: NDArray = CCS_WHITEPOINT_S_GAMUT
"""*Venice S-Gamut3* colourspace whitepoint chromaticity coordinates."""

MATRIX_VENICE_S_GAMUT3_TO_XYZ: NDArray = normalised_primary_matrix(
    PRIMARIES_VENICE_S_GAMUT3, CCS_WHITEPOINT_VENICE_S_GAMUT3
)
"""*Venice S-Gamut3* colourspace to *CIE XYZ* tristimulus values matrix."""

MATRIX_XYZ_TO_VENICE_S_GAMUT3: NDArray = np.linalg.inv(
    MATRIX_VENICE_S_GAMUT3_TO_XYZ
)
"""*CIE XYZ* tristimulus values to *Venice S-Gamut3* colourspace matrix."""

RGB_COLOURSPACE_VENICE_S_GAMUT3: RGB_Colourspace = RGB_Colourspace(
    "Venice S-Gamut3",
    PRIMARIES_VENICE_S_GAMUT3,
    CCS_WHITEPOINT_VENICE_S_GAMUT3,
    WHITEPOINT_NAME_VENICE_S_GAMUT3,
    MATRIX_VENICE_S_GAMUT3_TO_XYZ,
    MATRIX_XYZ_TO_VENICE_S_GAMUT3,
    log_encoding_SLog3,
    log_decoding_SLog3,
)
RGB_COLOURSPACE_VENICE_S_GAMUT3.__doc__ = """
*Venice S-Gamut3* colourspace.

References
----------
:cite:`SonyElectronicsCorporation2020`
"""

PRIMARIES_VENICE_S_GAMUT3_CINE: NDArray = np.array(
    [
        [0.775901871567345, 0.274502392854799],
        [0.188682902773355, 0.828684937020288],
        [0.101337382499301, -0.089187517306263],
    ]
)
"""*Venice S-Gamut3.Cine* colourspace primaries."""

WHITEPOINT_NAME_VENICE_S_GAMUT3_CINE: str = WHITEPOINT_NAME_S_GAMUT
"""*Venice S-Gamut3.Cine* colourspace whitepoint name."""

CCS_WHITEPOINT_VENICE_S_GAMUT3_CINE: NDArray = CCS_WHITEPOINT_S_GAMUT
"""*Venice S-Gamut3.Cine* colourspace whitepoint chromaticity coordinates."""

MATRIX_VENICE_S_GAMUT3_CINE_TO_XYZ: NDArray = normalised_primary_matrix(
    PRIMARIES_VENICE_S_GAMUT3_CINE, CCS_WHITEPOINT_VENICE_S_GAMUT3_CINE
)
"""*Venice S-Gamut3.Cine* colourspace to *CIE XYZ* tristimulus values matrix."""

MATRIX_XYZ_TO_VENICE_S_GAMUT3_CINE: NDArray = np.linalg.inv(
    MATRIX_VENICE_S_GAMUT3_CINE_TO_XYZ
)
"""*CIE XYZ* tristimulus values to *Venice S-Gamut3.Cine* colourspace matrix."""

RGB_COLOURSPACE_VENICE_S_GAMUT3_CINE: RGB_Colourspace = RGB_Colourspace(
    "Venice S-Gamut3.Cine",
    PRIMARIES_VENICE_S_GAMUT3_CINE,
    CCS_WHITEPOINT_VENICE_S_GAMUT3_CINE,
    WHITEPOINT_NAME_VENICE_S_GAMUT3_CINE,
    MATRIX_VENICE_S_GAMUT3_CINE_TO_XYZ,
    MATRIX_XYZ_TO_VENICE_S_GAMUT3_CINE,
    log_encoding_SLog3,
    log_decoding_SLog3,
)
RGB_COLOURSPACE_VENICE_S_GAMUT3_CINE.__doc__ = """
*Venice S-Gamut3.Cine* colourspace.

References
----------
:cite:`SonyElectronicsCorporation2020a`
"""
