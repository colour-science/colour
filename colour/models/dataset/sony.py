#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sony Colourspaces
=================

Defines the *S-Gamut*, *S-Gamut3* and *S-Gamut3.Cine* colourspaces:

-   :attr:`S_GAMUT_COLOURSPACE`.
-   :attr:`S_GAMUT3_COLOURSPACE`.
-   :attr:`S_GAMUT3_CINE_COLOURSPACE`.

See Also
--------
`RGB Colourspaces IPython Notebook
<http://nbviewer.ipython.org/github/colour-science/colour-ipython/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
.. [1]  Gaggioni, H., Dhanendra, P., Yamashita, J., Kawada, N., Endo, K., &
        Clark, C. (n.d.). S-Log: A new LUT for digital production mastering
        and interchange applications. Retrieved from
        http://pro.sony.com/bbsccms/assets/files/mkt/cinema/solutions/\
slog_manual.pdf
.. [2]  Sony Corporation. (n.d.). S-Log Whitepaper. Retrieved from
        http://www.theodoropoulos.info/attachments/076_on S-Log.pdf
.. [3]  Sony Corporation. (n.d.). Technical Summary for
        S-Gamut3.Cine/S-Log3 and S-Gamut3/S-Log3. Retrieved from
        http://community.sony.com/sony/attachments/sony/\
large-sensor-camera-F5-F55/12359/2/\
TechnicalSummary_for_S-Gamut3Cine_S-Gamut3_S-Log3_V1_00.pdf
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.colorimetry import ILLUMINANTS
from colour.models import RGB_Colourspace, normalised_primary_matrix

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['S_GAMUT_PRIMARIES',
           'S_GAMUT_ILLUMINANT',
           'S_GAMUT_WHITEPOINT',
           'S_GAMUT_TO_XYZ_MATRIX',
           'XYZ_TO_S_GAMUT_MATRIX',
           'S_LOG_OECF',
           'S_LOG_EOCF',
           'S_LOG2_OECF',
           'S_LOG2_EOCF',
           'S_GAMUT_COLOURSPACE',
           'S_LOG3_OECF',
           'S_LOG3_EOCF',
           'S_GAMUT3_COLOURSPACE',
           'S_GAMUT3_CINE_PRIMARIES',
           'S_GAMUT3_CINE_ILLUMINANT',
           'S_GAMUT3_CINE_WHITEPOINT',
           'S_GAMUT3_CINE_TO_XYZ_MATRIX',
           'XYZ_TO_S_GAMUT3_CINE_MATRIX',
           'S_GAMUT3_CINE_COLOURSPACE']

S_GAMUT_PRIMARIES = np.array(
    [[0.730, 0.280],
     [0.140, 0.855],
     [0.100, -0.050]])
"""
*S-Gamut* colourspace primaries.

S_GAMUT_PRIMARIES : ndarray, (3, 2)
"""

S_GAMUT_ILLUMINANT = 'D65'
"""
*S-Gamut* colourspace whitepoint name as illuminant.

S_GAMUT_ILLUMINANT : unicode
"""

S_GAMUT_WHITEPOINT = ILLUMINANTS.get(
    'CIE 1931 2 Degree Standard Observer').get(S_GAMUT_ILLUMINANT)
"""
*S-Gamut* colourspace whitepoint.

S_GAMUT_WHITEPOINT : tuple
"""

S_GAMUT_TO_XYZ_MATRIX = normalised_primary_matrix(S_GAMUT_PRIMARIES,
                                                  S_GAMUT_WHITEPOINT)
"""
*S-Gamut* colourspace to *CIE XYZ* tristimulus values matrix.

S_GAMUT_TO_XYZ_MATRIX : array_like, (3, 3)
"""

XYZ_TO_S_GAMUT_MATRIX = np.linalg.inv(S_GAMUT_TO_XYZ_MATRIX)
"""
*CIE XYZ* tristimulus values to *S-Gamut* colourspace matrix.

XYZ_TO_S_GAMUT_MATRIX : array_like, (3, 3)
"""


def _linear_to_s_log(value):
    """
    Defines the *linear* to *S-Log* conversion function. [1]_

    Parameters
    ----------
    value : numeric or array_like
        Value.

    Returns
    -------
    numeric or ndarray
        Companded value.
    """

    value = np.asarray(value)

    return (0.432699 * np.log10(value + 0.037584) + 0.616596) + 0.03


def _s_log_to_linear(value):
    """
    Defines the *S-Log* to *linear* conversion function. [1]_

    Parameters
    ----------
    value : numeric or array_like
        Value.

    Returns
    -------
    numeric or ndarray
        Companded value.
    """

    value = np.asarray(value)

    return 10 ** ((value - 0.616596 - 0.03) / 0.432699) - 0.037584


S_LOG_OECF = _linear_to_s_log
"""
Opto-electronic conversion function of *S-Log*.

S_LOG_OECF : object
"""

S_LOG_EOCF = _s_log_to_linear
"""
Electro-optical conversion function of *S-Log* to linear.

S_LOG_EOCF : object
"""


def _linear_to_s_log2(value):
    """
    Defines the *linear* to *S-Log2* conversion function.

    Parameters
    ----------
    value : numeric or array_like
        Value.

    Returns
    -------
    numeric or ndarray
        Companded value.
    """

    value = np.asarray(value)

    return ((4 * (16 + 219 * (0.616596 + 0.03 + 0.432699 *
                              (np.log10(0.037584 + value / 0.9))))) / 1023)


def _s_log2_to_linear(value):
    """
    Defines the *S-Log2* to *linear* conversion function.

    Parameters
    ----------
    value : numeric or array_like
        Value.

    Returns
    -------
    numeric or ndarray
        Companded value.
    """

    value = np.asarray(value)

    return ((10 ** (((((value * 1023 / 4 - 16) / 219) - 0.616596 - 0.03)
                     / 0.432699)) - 0.037584) * 0.9)


S_LOG2_OECF = _linear_to_s_log2
"""
Opto-electronic conversion function of *S-Log2*.

S_LOG2_OECF : object
"""

S_LOG2_EOCF = _s_log2_to_linear
"""
Electro-optical conversion function of *S-Log2* to linear.

S_LOG2_EOCF : object
"""

S_GAMUT_COLOURSPACE = RGB_Colourspace(
    'S-Gamut',
    S_GAMUT_PRIMARIES,
    S_GAMUT_WHITEPOINT,
    S_GAMUT_ILLUMINANT,
    S_GAMUT_TO_XYZ_MATRIX,
    XYZ_TO_S_GAMUT_MATRIX,
    S_LOG2_OECF,
    S_LOG2_EOCF)
"""
*S-Gamut* colourspace.

S_GAMUT_COLOURSPACE : RGB_Colourspace
"""


def _linear_to_s_log3(value):
    """
    Defines the *linear* to *S-Log3* conversion function.

    Parameters
    ----------
    value : numeric or array_like
        Value.

    Returns
    -------
    numeric or ndarray
        Companded value.
    """

    value = np.asarray(value)

    return np.where(value >= 0.01125000,
                    (420 + np.log10((value + 0.01) /
                                    (0.18 + 0.01)) * 261.5) / 1023,
                    (value * (171.2102946929 - 95) / 0.01125000 + 95) / 1023)


def _s_log3_to_linear(value):
    """
    Defines the *S-Log3* to *linear* conversion function.

    Parameters
    ----------
    value : numeric or array_like
        Value.

    Returns
    -------
    numeric or ndarray
        Companded value.
    """

    value = np.asarray(value)

    return np.where(value >= 171.2102946929 / 1023,
                    ((10 ** ((value * 1023 - 420) / 261.5)) *
                     (0.18 + 0.01) - 0.01),
                    (value * 1023 - 95) * 0.01125000 / (171.2102946929 - 95))


S_LOG3_OECF = _linear_to_s_log3
"""
Opto-electronic conversion function of *S-Log3*.

S_LOG3_OECF : object
"""

S_LOG3_EOCF = _s_log3_to_linear
"""
Electro-optical conversion function of *S-Log3* to linear.

S_LOG3_EOCF : object
"""

S_GAMUT3_COLOURSPACE = RGB_Colourspace(
    'S-Gamut3',
    S_GAMUT_PRIMARIES,
    S_GAMUT_WHITEPOINT,
    S_GAMUT_ILLUMINANT,
    S_GAMUT_TO_XYZ_MATRIX,
    XYZ_TO_S_GAMUT_MATRIX,
    S_LOG3_OECF,
    S_LOG3_EOCF)
"""
*S-Gamut3* colourspace.

S_GAMUT3_COLOURSPACE : RGB_Colourspace
"""

S_GAMUT3_CINE_PRIMARIES = np.array(
    [[0.76600, 0.27500],
     [0.22500, 0.80000],
     [0.08900, -0.08700]])
"""
*S-Gamut3.Cine* colourspace primaries.

S_GAMUT_PRIMARIES : ndarray, (3, 2)
"""

S_GAMUT3_CINE_ILLUMINANT = S_GAMUT_ILLUMINANT
"""
*S-Gamut3.Cine* colourspace whitepoint name as illuminant.

S_GAMUT3_CINE_ILLUMINANT : unicode
"""

S_GAMUT3_CINE_WHITEPOINT = S_GAMUT_WHITEPOINT
"""
*S-Gamut3.Cine* colourspace whitepoint.

S_GAMUT3_CINE_WHITEPOINT : tuple
"""

S_GAMUT3_CINE_TO_XYZ_MATRIX = normalised_primary_matrix(
    S_GAMUT3_CINE_PRIMARIES,
    S_GAMUT3_CINE_WHITEPOINT)
"""
*S-Gamut3.Cine* colourspace to *CIE XYZ* tristimulus values matrix.

S_GAMUT3_CINE_TO_XYZ_MATRIX : array_like, (3, 3)
"""

XYZ_TO_S_GAMUT3_CINE_MATRIX = np.linalg.inv(S_GAMUT3_CINE_TO_XYZ_MATRIX)
"""
*CIE XYZ* tristimulus values to *S-Gamut3.Cine* colourspace matrix.

XYZ_TO_S_GAMUT3_CINE_MATRIX : array_like, (3, 3)
"""

S_GAMUT3_CINE_COLOURSPACE = RGB_Colourspace(
    'S-Gamut3.Cine',
    S_GAMUT3_CINE_PRIMARIES,
    S_GAMUT3_CINE_WHITEPOINT,
    S_GAMUT3_CINE_ILLUMINANT,
    S_GAMUT3_CINE_TO_XYZ_MATRIX,
    XYZ_TO_S_GAMUT3_CINE_MATRIX,
    S_LOG3_OECF,
    S_LOG3_EOCF)
"""
*S-Gamut3.Cine* colourspace.

S_GAMUT3_CINE_COLOURSPACE : RGB_Colourspace
"""
