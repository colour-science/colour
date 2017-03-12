#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Chromatic Adaptation Transforms
===============================

Defines various chromatic adaptation transforms (CAT):

-   :attr:`XYZ_SCALING_CAT`: *XYZ Scaling* chromatic adaptation transform [1]_
-   :attr:`VON_KRIES_CAT`: *Von Kries* chromatic adaptation transform [1]_
-   :attr:`BRADFORD_CAT`: *Bradford* chromatic adaptation transform [1]_
-   :attr:`SHARP_CAT`: *Sharp* chromatic adaptation transform [4]_
-   :attr:`FAIRCHILD_CAT`: *Fairchild* chromatic adaptation transform [2]_
-   :attr:`CMCCAT97_CAT`: *CMCCAT97* chromatic adaptation transform [5]_
-   :attr:`CMCCAT2000_CAT`: *CMCCAT2000* chromatic adaptation transform [5]_
-   :attr:`CAT02_CAT`: *CAT02* chromatic adaptation transform [3]_
-   :attr:`CAT02_BRILL_CAT`: *Brill and Süsstrunk (2008)* corrected CAT02
    chromatic adaptation transform [6]_ [7]_
-   :attr:`BS_CAT`: *Bianco and Schettini (2010)* chromatic adaptation
    transform [4]_
-   :attr:`BS_PC_CAT`: *Bianco and Schettini PC (2010)* chromatic adaptation
    transform [4]_

See Also
--------
`Chromatic Adaptation Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/adaptation/vonkries.ipynb>`_

References
----------
.. [1]  Lindbloom, B. (2009). Chromatic Adaptation. Retrieved February 24,
        2014, from http://brucelindbloom.com/Eqn_ChromAdapt.html
.. [2]  Fairchild, M. D. (n.d.). Fairchild YSh. Retrieved from
        http://rit-mcsl.org/fairchild//files/FairchildYSh.zip
.. [3]  Wikipedia. (n.d.). CAT02. Retrieved February 24, 2014,
        from http://en.wikipedia.org/wiki/CIECAM02#CAT02
.. [4]  Bianco, S., & Schettini, R. (2010). Two New von Kries Based Chromatic
        Adaptation Transforms Found by Numerical Optimization. Color Research
        & Application, 35(3), 184–192. doi:10.1002/col.20573
.. [5]  Westland, S., Ripamonti, C., & Cheung, V. (2012). CMCCAT97. In
        Computational Colour Science Using MATLAB (2nd ed., p. 80).
        ISBN:978-0-470-66569-5
.. [6]  Brill, M. H., & Süsstrunk, S. (2008). Repairing gamut problems in
        CIECAM02: A progress report. Color Research & Application, 33(5),
        424–426. doi:10.1002/col.20432
.. [7]  Li, C., Perales, E., Luo, M. R., & Martínez-verdú, F. (2007). The
        Problem with CAT02 and Its Correction, (July), 1–10.
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.utilities import CaseInsensitiveMapping

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['XYZ_SCALING_CAT',
           'VON_KRIES_CAT',
           'BRADFORD_CAT',
           'SHARP_CAT',
           'FAIRCHILD_CAT',
           'CMCCAT97_CAT',
           'CMCCAT2000_CAT',
           'CAT02_CAT',
           'CAT02_BRILL_CAT',
           'BS_CAT',
           'BS_PC_CAT',
           'CHROMATIC_ADAPTATION_TRANSFORMS']

XYZ_SCALING_CAT = np.array(np.identity(3)).reshape((3, 3))
"""
*XYZ Scaling* chromatic adaptation transform. [1]_

XYZ_SCALING_CAT : array_like, (3, 3)
"""

VON_KRIES_CAT = np.array(
    [[0.4002400, 0.7076000, -0.0808100],
     [-0.2263000, 1.1653200, 0.0457000],
     [0.0000000, 0.0000000, 0.9182200]])
"""
*Von Kries* chromatic adaptation transform. [1]_

VON_KRIES_CAT : array_like, (3, 3)
"""

BRADFORD_CAT = np.array(
    [[0.8951000, 0.2664000, -0.1614000],
     [-0.7502000, 1.7135000, 0.0367000],
     [0.0389000, -0.0685000, 1.0296000]])
"""
*Bradford* chromatic adaptation transform. [1]_

BRADFORD_CAT : array_like, (3, 3)
"""

SHARP_CAT = np.array(
    [[1.2694, -0.0988, -0.1706],
     [-0.8364, 1.8006, 0.0357],
     [0.0297, -0.0315, 1.0018]])
"""
*Sharp* chromatic adaptation transform. [4]_

SHARP_CAT : array_like, (3, 3)
"""

FAIRCHILD_CAT = np.array(
    [[0.8562, 0.3372, -0.1934],
     [-0.8360, 1.8327, 0.0033],
     [0.0357, -0.0469, 1.0112]])
"""
*Fairchild* chromatic adaptation transform. [2]_

FAIRCHILD_CAT : array_like, (3, 3)
"""

CMCCAT97_CAT = np.array(
    [[0.8951, -0.7502, 0.0389],
     [0.2664, 1.7135, 0.0685],
     [-0.1614, 0.0367, 1.0296]])
"""
*CMCCAT97* chromatic adaptation transform. [5]_

CMCCAT97_CAT : array_like, (3, 3)
"""

CMCCAT2000_CAT = np.array(
    [[0.7982, 0.3389, -0.1371],
     [-0.5918, 1.5512, 0.0406],
     [0.0008, 0.0239, 0.9753]])
"""
*CMCCAT2000* chromatic adaptation transform. [5]_

CMCCAT2000_CAT : array_like, (3, 3)
"""

CAT02_CAT = np.array(
    [[0.7328, 0.4296, -0.1624],
     [-0.7036, 1.6975, 0.0061],
     [0.0030, 0.0136, 0.9834]])
"""
*CAT02* chromatic adaptation transform. [3]_

CAT02_CAT : array_like, (3, 3)
"""

CAT02_BRILL_CAT = np.array(
    [[0.7328, 0.4296, -0.1624],
     [-0.7036, 1.6975, 0.0061],
     [0.0000, 0.0000, 1.0000]])
"""
*Brill and Süsstrunk (2008)* corrected CAT02 chromatic adaptation
transform. [6]_ [7]

CAT02_BRILL_CAT : array_like, (3, 3)
"""

BS_CAT = np.array(
    [[0.8752, 0.2787, -0.1539],
     [-0.8904, 1.8709, 0.0195],
     [-0.0061, 0.0162, 0.9899]])
"""
*Bianco and Schettini (2010)* chromatic adaptation transform. [4]_

BS_CAT : array_like, (3, 3)
"""

BS_PC_CAT = np.array(
    [[0.6489, 0.3915, -0.0404],
     [-0.3775, 1.3055, 0.0720],
     [-0.0271, 0.0888, 0.9383]])
"""
*Bianco and Schettini PC (2010)* chromatic adaptation transform. [4]_

BS_PC_CAT : array_like, (3, 3)

Notes
-----
-   This chromatic adaptation transform has no negative lobes.
"""

CHROMATIC_ADAPTATION_TRANSFORMS = CaseInsensitiveMapping(
    {'XYZ Scaling': XYZ_SCALING_CAT,
     'Von Kries': VON_KRIES_CAT,
     'Bradford': BRADFORD_CAT,
     'Sharp': SHARP_CAT,
     'Fairchild': FAIRCHILD_CAT,
     'CMCCAT97': CMCCAT97_CAT,
     'CMCCAT2000': CMCCAT2000_CAT,
     'CAT02': CAT02_CAT,
     'CAT02_BRILL_CAT': CAT02_BRILL_CAT,
     'Bianco': BS_CAT,
     'Bianco PC': BS_PC_CAT})
"""
Supported chromatic adaptation transforms.

CHROMATIC_ADAPTATION_TRANSFORMS : CaseInsensitiveMapping
    **{'CAT02', 'XYZ Scaling', 'Von Kries', 'Bradford', 'Sharp', 'Fairchild,
    'CMCCAT97', 'CMCCAT2000', 'CAT02_BRILL_CAT', 'Bianco', 'Bianco PC'}**
"""
