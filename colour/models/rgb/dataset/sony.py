# -*- coding: utf-8 -*-
"""
Sony Colourspaces
=================

Defines the *S-Gamut*, *S-Gamut3* and *S-Gamut3.Cine* colourspaces:

-   :attr:`colour.models.S_GAMUT_COLOURSPACE`.
-   :attr:`colour.models.S_GAMUT3_COLOURSPACE`.
-   :attr:`colour.models.S_GAMUT3_CINE_COLOURSPACE`.

See Also
--------
`RGB Colourspaces Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
-   :cite:`Gaggioni` : Gaggioni, H., Dhanendra, P., Yamashita, J., Kawada, N.,
    Endo, K., & Clark, C. (n.d.). S-Log: A new LUT for digital production
    mastering and interchange applications. Retrieved from
    http://pro.sony.com/bbsccms/assets/files/mkt/cinema/solutions/\
slog_manual.pdf
-   :cite:`SonyCorporation` : Sony Corporation. (n.d.). S-Log Whitepaper.
Retrieved from http://www.theodoropoulos.info/attachments/076_on S-Log.pdf
-   :cite:`SonyCorporationd` : Sony Corporation. (n.d.). Technical Summary for
S-Gamut3.Cine/S-Log3 and S-Gamut3/S-Log3. Retrieved from
    http://community.sony.com/sony/attachments/sony/
large-sensor-camera-F5-F55/12359/2/\
TechnicalSummary_for_S-Gamut3Cine_S-Gamut3_S-Log3_V1_00.pdf
-   :cite:`SonyCorporatione` : Sony Corporation. (n.d.).
    S-Gamut3_S-Gamut3Cine_Matrix.xlsx. Retrieved from
    https://community.sony.com/sony/attachments/sony/\
large-sensor-camera-F5-F55/12359/3/S-Gamut3_S-Gamut3Cine_Matrix.xlsx
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.colorimetry import ILLUMINANTS
from colour.models.rgb import (RGB_Colourspace, log_encoding_SLog2,
                               log_decoding_SLog2, log_encoding_SLog3,
                               log_decoding_SLog3)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'S_GAMUT_PRIMARIES', 'S_GAMUT_WHITEPOINT_NAME', 'S_GAMUT_WHITEPOINT',
    'S_GAMUT_TO_XYZ_MATRIX', 'XYZ_TO_S_GAMUT_MATRIX', 'S_GAMUT_COLOURSPACE',
    'S_GAMUT3_COLOURSPACE', 'S_GAMUT3_CINE_PRIMARIES',
    'S_GAMUT3_CINE_WHITEPOINT_NAME', 'S_GAMUT3_CINE_WHITEPOINT',
    'S_GAMUT3_CINE_TO_XYZ_MATRIX', 'XYZ_TO_S_GAMUT3_CINE_MATRIX',
    'S_GAMUT3_CINE_COLOURSPACE'
]

S_GAMUT_PRIMARIES = np.array([
    [0.7300, 0.2800],
    [0.1400, 0.8550],
    [0.1000, -0.0500],
])
"""
*S-Gamut* colourspace primaries.

S_GAMUT_PRIMARIES : ndarray, (3, 2)
"""

S_GAMUT_WHITEPOINT_NAME = 'D65'
"""
*S-Gamut* colourspace whitepoint name.

S_GAMUT_WHITEPOINT_NAME : unicode
"""

S_GAMUT_WHITEPOINT = (ILLUMINANTS['CIE 1931 2 Degree Standard Observer'][
    S_GAMUT_WHITEPOINT_NAME])
"""
*S-Gamut* colourspace whitepoint.

S_GAMUT_WHITEPOINT : ndarray
"""

S_GAMUT_TO_XYZ_MATRIX = np.array([
    [0.7064827132, 0.1288010498, 0.1151721641],
    [0.2709796708, 0.7866064112, -0.0575860820],
    [-0.0096778454, 0.0046000375, 1.0941355587],
])
"""
*S-Gamut* colourspace to *CIE XYZ* tristimulus values matrix.

S_GAMUT_TO_XYZ_MATRIX : array_like, (3, 3)
"""

XYZ_TO_S_GAMUT_MATRIX = np.array([
    [1.5073998991, -0.2458221374, -0.1716116808],
    [-0.5181517271, 1.3553912409, 0.1258786682],
    [0.0155116982, -0.0078727714, 0.9119163656],
])
"""
*CIE XYZ* tristimulus values to *S-Gamut* colourspace matrix.

XYZ_TO_S_GAMUT_MATRIX : array_like, (3, 3)
"""

S_GAMUT_COLOURSPACE = RGB_Colourspace(
    'S-Gamut',
    S_GAMUT_PRIMARIES,
    S_GAMUT_WHITEPOINT,
    S_GAMUT_WHITEPOINT_NAME,
    S_GAMUT_TO_XYZ_MATRIX,
    XYZ_TO_S_GAMUT_MATRIX,
    log_encoding_SLog2,
    log_decoding_SLog2,
)
S_GAMUT_COLOURSPACE.__doc__ = """
*S-Gamut* colourspace.

References
----------
:cite:`Gaggioni`, :cite:`SonyCorporation`

S_GAMUT_COLOURSPACE : RGB_Colourspace
"""

S_GAMUT3_COLOURSPACE = RGB_Colourspace(
    'S-Gamut3',
    S_GAMUT_PRIMARIES,
    S_GAMUT_WHITEPOINT,
    S_GAMUT_WHITEPOINT_NAME,
    S_GAMUT_TO_XYZ_MATRIX,
    XYZ_TO_S_GAMUT_MATRIX,
    log_encoding_SLog3,
    log_decoding_SLog3,
)
S_GAMUT3_COLOURSPACE.__doc__ = """
*S-Gamut3* colourspace.

References
----------
:cite:`SonyCorporationd`

S_GAMUT3_COLOURSPACE : RGB_Colourspace
"""

S_GAMUT3_CINE_PRIMARIES = np.array([
    [0.76600, 0.27500],
    [0.22500, 0.80000],
    [0.08900, -0.08700],
])
"""
*S-Gamut3.Cine* colourspace primaries.

S_GAMUT_PRIMARIES : ndarray, (3, 2)
"""

S_GAMUT3_CINE_WHITEPOINT_NAME = S_GAMUT_WHITEPOINT_NAME
"""
*S-Gamut3.Cine* colourspace whitepoint name.

S_GAMUT3_CINE_WHITEPOINT_NAME : unicode
"""

S_GAMUT3_CINE_WHITEPOINT = S_GAMUT_WHITEPOINT
"""
*S-Gamut3.Cine* colourspace whitepoint.

S_GAMUT3_CINE_WHITEPOINT : ndarray
"""

S_GAMUT3_CINE_TO_XYZ_MATRIX = np.array([
    [0.5990839208, 0.2489255161, 0.1024464902],
    [0.2150758201, 0.8850685017, -0.1001443219],
    [-0.0320658495, -0.0276583907, 1.1487819910],
])
"""
*S-Gamut3.Cine* colourspace to *CIE XYZ* tristimulus values matrix.

S_GAMUT3_CINE_TO_XYZ_MATRIX : array_like, (3, 3)
"""

XYZ_TO_S_GAMUT3_CINE_MATRIX = np.array([
    [1.8467789693, -0.5259861230, -0.2105452114],
    [-0.4441532629, 1.2594429028, 0.1493999729],
    [0.0408554212, 0.0156408893, 0.8682072487],
])
"""
*CIE XYZ* tristimulus values to *S-Gamut3.Cine* colourspace matrix.

XYZ_TO_S_GAMUT3_CINE_MATRIX : array_like, (3, 3)
"""

S_GAMUT3_CINE_COLOURSPACE = RGB_Colourspace(
    'S-Gamut3.Cine',
    S_GAMUT3_CINE_PRIMARIES,
    S_GAMUT3_CINE_WHITEPOINT,
    S_GAMUT3_CINE_WHITEPOINT_NAME,
    S_GAMUT3_CINE_TO_XYZ_MATRIX,
    XYZ_TO_S_GAMUT3_CINE_MATRIX,
    log_encoding_SLog3,
    log_decoding_SLog3,
)
S_GAMUT3_CINE_COLOURSPACE.__doc__ = """
*S-Gamut3.Cine* colourspace.

References
----------
:cite:`SonyCorporatione`

S_GAMUT3_CINE_COLOURSPACE : RGB_Colourspace
"""
