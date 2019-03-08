# -*- coding: utf-8 -*-
"""
MacAdam (1942) Ellipses (Observer PGN)
======================================

Defines *MacAdam (1942) Ellipses (Observer PGN)* ellipses data.

See Also
--------
`MacAdam Ellipses Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/macadam_ellipses.ipynb>`_

References
----------
-   :cite:`Wyszecki2000` : Wyszecki, G., & Stiles, W. S. (2000). Table 2(5.4.1)
    MacAdam Ellipses (Observer PGN) Observed and Calculated on the Basis of a
    Normal Distribution of Color Matches about a Color Center
    (Silberstein and MacAdam, 1945). In Color Science: Concepts and Methods,
    Quantitative Data and Formulae (p. 309). Wiley. ISBN:978-0471399186
-   :cite:`Macadam1942` : Macadam, D. L. (1942). Visual Sensitivities to Color
    Differences in Daylight. Journal of the Optical Society of America, 32(5),
    28. doi:10.1364/JOSA.32.000247
"""

from __future__ import division, unicode_literals

import numpy as np

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['MACADAM_1942_ELLIPSES_DATA']

MACADAM_1942_ELLIPSES_DATA = np.array([
    [0.160, 0.057, 0.85, 0.35, 62.5, 0.94, 0.30, 62.3],
    [0.187, 0.118, 2.20, 0.55, 77.0, 2.31, 0.44, 74.8],
    [0.253, 0.125, 2.50, 0.50, 55.5, 2.49, 0.49, 54.8],
    [0.150, 0.680, 9.60, 2.30, 105.0, 9.09, 2.21, 102.9],
    [0.131, 0.521, 4.70, 2.00, 112.5, 4.67, 2.10, 110.5],
    [0.212, 0.550, 5.80, 2.30, 100.0, 5.63, 2.30, 100.0],
    [0.258, 0.450, 5.00, 2.00, 92.0, 4.54, 2.08, 88.5],
    [0.152, 0.365, 3.80, 1.90, 110.0, 3.81, 1.86, 111.0],
    [0.280, 0.385, 4.00, 1.50, 75.5, 4.26, 1.46, 74.6],
    [0.380, 0.498, 4.40, 1.20, 70.0, 4.23, 1.32, 69.4],
    [0.160, 0.200, 2.10, 0.95, 104.0, 2.08, 0.94, 95.4],
    [0.228, 0.250, 3.10, 0.90, 72.0, 3.09, 0.82, 70.9],
    [0.305, 0.323, 2.30, 0.90, 58.0, 2.55, 0.68, 57.2],
    [0.385, 0.393, 3.80, 1.60, 65.5, 3.70, 1.48, 65.5],
    [0.472, 0.399, 3.20, 1.40, 51.0, 3.21, 1.30, 54.0],
    [0.527, 0.350, 2.60, 1.30, 20.0, 2.56, 1.27, 22.8],
    [0.475, 0.300, 2.90, 1.10, 28.5, 2.89, 0.99, 29.1],
    [0.510, 0.236, 2.40, 1.20, 29.5, 2.40, 1.15, 30.7],
    [0.596, 0.283, 2.60, 1.30, 13.0, 2.49, 1.15, 11.1],
    [0.344, 0.284, 2.30, 0.90, 60.0, 2.24, 0.97, 65.7],
    [0.390, 0.237, 2.50, 1.00, 47.0, 2.43, 0.98, 44.2],
    [0.441, 0.198, 2.80, 0.95, 34.5, 2.73, 0.90, 33.7],
    [0.278, 0.223, 2.40, 0.55, 57.5, 2.34, 0.61, 60.3],
    [0.300, 0.163, 2.90, 0.60, 54.0, 3.01, 0.60, 53.4],
    [0.365, 0.153, 3.60, 0.95, 40.0, 4.12, 0.90, 38.6],
])
"""
*MacAdam (1942) Ellipses (Observer PGN)* ellipses data.

Table 2(5.4.1) data in *Wyszecki and Stiles (2000)* is as follows:

+--------------+---------------------------+---------------------------+
| Color Center | Observed                  | Calculated                |
+--------------+---------------------------+---------------------------+
| x_0   | y_0  | 10**3 a | 10**3 b | theta | 10**3 a | 10**3 b | theta |
+-------+------+---------+---------+-------+---------+---------+-------+

where :math:`x_0` and :math:`y_0` are the coordinates of the ellipse center,
:math:`a` is the semi-major axis length, :math:`b` is the semi-minor axis
length and :math:`\\theta` is the angle from the semi-major axis :math:`a`.

The *Calculated* column should be preferred to the *Observed* one as the later
is the result from measurements observed on *MacAdam (1942)* diagrams while the
former is fitted on his  observational data.

References
----------
:cite:`Wyszecki2000`, :cite:`Macadam1942`

MACADAM_1942_ELLIPSES_DATA : ndarray
"""
