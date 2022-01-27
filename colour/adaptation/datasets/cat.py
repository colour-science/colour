# -*- coding: utf-8 -*-
"""
Chromatic Adaptation Transforms
===============================

Defines various chromatic adaptation transforms (CAT):

-   :attr:`colour.adaptation.CAT_XYZ_SCALING`: *XYZ Scaling* chromatic
    adaptation transform.
-   :attr:`colour.adaptation.CAT_VON_KRIES`: *Von Kries* chromatic adaptation
    transform.
-   :attr:`colour.adaptation.CAT_BRADFORD`: *Bradford* chromatic adaptation
    transform.
-   :attr:`colour.adaptation.CAT_SHARP`: *Sharp* chromatic adaptation
    transform.
-   :attr:`colour.adaptation.CAT_FAIRCHILD`: *Fairchild* chromatic adaptation
    transform.
-   :attr:`colour.adaptation.CAT_CMCCAT97`: *CMCCAT97* chromatic adaptation
    transform.
-   :attr:`colour.adaptation.CAT_CMCCAT2000`: *CMCCAT2000* chromatic adaptation
    transform.
-   :attr:`colour.adaptation.CAT_CAT02`: *CAT02* chromatic adaptation
    transform.
-   :attr:`colour.adaptation.CAT_CAT02_BRILL2008`: *Brill and Susstrunk (2008)*
    corrected CAT02 chromatic adaptation transform.
-   :attr:`colour.adaptation.CAT_CAT16`: *CAT16* chromatic adaptation
    transform.
-   :attr:`colour.adaptation.CAT_BIANCO2010`: *Bianco and Schettini (2010)*
    chromatic adaptation transform.
-   :attr:`colour.adaptation.CAT_PC_BIANCO2010`:
    *Bianco and Schettini PC (2010)* chromatic adaptation transform.

References
----------
-   :cite:`Bianco2010a` : Bianco, S., & Schettini, R. (2010). Two new von Kries
    based chromatic adaptation transforms found by numerical optimization.
    Color Research & Application, 35(3), 184-192. doi:10.1002/col.20573
-   :cite:`Brill2008a` : Brill, M. H., & Susstrunk, S. (2008). Repairing gamut
    problems in CIECAM02: A progress report. Color Research & Application,
    33(5), 424-426. doi:10.1002/col.20432
-   :cite:`CIETC1-321994b` : CIE TC 1-32. (1994). CIE 109-1994 A Method of
    Predicting Corresponding Colours under Different Chromatic and Illuminance
    Adaptations. Commission Internationale de l'Eclairage.
    ISBN:978-3-900734-51-0
-   :cite:`Fairchild2013ba` : Fairchild, M. D. (2013). The Nayatani et al.
    Model. In Color Appearance Models (3rd ed., pp. 4810-5085). Wiley.
    ISBN:B00DAYO8E2
-   :cite:`Fairchildb` : Fairchild, M. D. (n.d.). Fairchild YSh.
    http://rit-mcsl.org/fairchild//files/FairchildYSh.zip
-   :cite:`Li2007e` : Li, C., Perales, E., Luo, M. R., & Martinez-verdu, F.
    (2007). The Problem with CAT02 and Its Correction.
    https://pdfs.semanticscholar.org/b5a9/\
0215ad9a1fb6b01f310b3d64305f7c9feb3a.pdf
-   :cite:`Li2017` : Li, C., Li, Z., Wang, Z., Xu, Y., Luo, M. R., Cui, G.,
    Melgosa, M., Brill, M. H., & Pointer, M. (2017). Comprehensive color
    solutions: CAM16, CAT16, and CAM16-UCS. Color Research & Application,
    42(6), 703-718. doi:10.1002/col.22131
-   :cite:`Lindbloom2009g` : Fairchild, M. D. (2013). Chromatic Adaptation
    Models. In Color Appearance Models (3rd ed., pp. 4179-4252). Wiley.
    ISBN:B00DAYO8E2
-   :cite:`Nayatani1995a` : Nayatani, Y., Sobagaki, H., & Yano, K. H. T.
    (1995). Lightness dependency of chroma scales of a nonlinear
    color-appearance model and its latest formulation. Color Research &
    Application, 20(3), 156-167. doi:10.1002/col.5080200305
-   :cite:`Westland2012g` : Westland, S., Ripamonti, C., & Cheung, V. (2012).
    CMCCAT97. In Computational Colour Science Using MATLAB (2nd ed., p. 80).
    ISBN:978-0-470-66569-5
-   :cite:`Westland2012k` : Westland, S., Ripamonti, C., & Cheung, V. (2012).
    CMCCAT2000. In Computational Colour Science Using MATLAB (2nd ed., pp.
    83-86). ISBN:978-0-470-66569-5
-   :cite:`Wikipedia2007` : Wikipedia. (2007). CAT02. Retrieved February 24,
    2014, from http://en.wikipedia.org/wiki/CIECAM02#CAT02
"""

from __future__ import annotations

import numpy as np

from colour.hints import NDArray
from colour.utilities import CaseInsensitiveMapping

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'CAT_XYZ_SCALING',
    'CAT_VON_KRIES',
    'CAT_BRADFORD',
    'CAT_SHARP',
    'CAT_FAIRCHILD',
    'CAT_CMCCAT97',
    'CAT_CMCCAT2000',
    'CAT_CAT02',
    'CAT_CAT02_BRILL2008',
    'CAT_CAT16',
    'CAT_BIANCO2010',
    'CAT_PC_BIANCO2010',
    'CHROMATIC_ADAPTATION_TRANSFORMS',
]

CAT_XYZ_SCALING: NDArray = np.array(np.identity(3)).reshape([3, 3])
"""
*XYZ Scaling* chromatic adaptation transform.

References
----------
:cite:`Lindbloom2009g`
"""

CAT_VON_KRIES: NDArray = np.array([
    [0.4002400, 0.7076000, -0.0808100],
    [-0.2263000, 1.1653200, 0.0457000],
    [0.0000000, 0.0000000, 0.9182200],
])
"""
*Von Kries* chromatic adaptation transform.

References
----------
:cite:`CIETC1-321994b`, :cite:`Fairchild2013ba`, :cite:`Lindbloom2009g`,
:cite:`Nayatani1995a`
"""

CAT_BRADFORD: NDArray = np.array([
    [0.8951000, 0.2664000, -0.1614000],
    [-0.7502000, 1.7135000, 0.0367000],
    [0.0389000, -0.0685000, 1.0296000],
])
"""
*Bradford* chromatic adaptation transform.

References
----------
:cite:`Lindbloom2009g`
"""

CAT_SHARP: NDArray = np.array([
    [1.2694, -0.0988, -0.1706],
    [-0.8364, 1.8006, 0.0357],
    [0.0297, -0.0315, 1.0018],
])
"""
*Sharp* chromatic adaptation transform.

References
----------
:cite:`Bianco2010a`
"""

CAT_FAIRCHILD: NDArray = np.array([
    [0.8562, 0.3372, -0.1934],
    [-0.8360, 1.8327, 0.0033],
    [0.0357, -0.0469, 1.0112],
])
"""
*Fairchild* chromatic adaptation transform.

References
----------
:cite:`Fairchildb`
"""

CAT_CMCCAT97: NDArray = np.array([
    [0.8951, -0.7502, 0.0389],
    [0.2664, 1.7135, 0.0685],
    [-0.1614, 0.0367, 1.0296],
])
"""
*CMCCAT97* chromatic adaptation transform.

References
----------
:cite:`Westland2012g`
"""

CAT_CMCCAT2000: NDArray = np.array([
    [0.7982, 0.3389, -0.1371],
    [-0.5918, 1.5512, 0.0406],
    [0.0008, 0.0239, 0.9753],
])
"""
*CMCCAT2000* chromatic adaptation transform.

References
----------
:cite:`Westland2012k`
"""

CAT_CAT02: NDArray = np.array([
    [0.7328, 0.4296, -0.1624],
    [-0.7036, 1.6975, 0.0061],
    [0.0030, 0.0136, 0.9834],
])
"""
*CAT02* chromatic adaptation transform.

References
----------
:cite:`Wikipedia2007`
"""

CAT_CAT02_BRILL2008: NDArray = np.array([
    [0.7328, 0.4296, -0.1624],
    [-0.7036, 1.6975, 0.0061],
    [0.0000, 0.0000, 1.0000],
])
"""
*Brill and Susstrunk (2008)* corrected CAT02 chromatic adaptation
transform.

References
----------
:cite:`Brill2008a`, :cite:`Li2007e`
"""

CAT_CAT16: NDArray = np.array([
    [0.401288, 0.650173, -0.051461],
    [-0.250268, 1.204414, 0.045854],
    [-0.002079, 0.048952, 0.953127],
])
"""
*CAT16* chromatic adaptation transform.

References
----------
:cite:`Li2017`
"""

CAT_BIANCO2010: NDArray = np.array([
    [0.8752, 0.2787, -0.1539],
    [-0.8904, 1.8709, 0.0195],
    [-0.0061, 0.0162, 0.9899],
])
"""
*Bianco and Schettini (2010)* chromatic adaptation transform.

References
----------
:cite:`Bianco2010a`
"""

CAT_PC_BIANCO2010: NDArray = np.array([
    [0.6489, 0.3915, -0.0404],
    [-0.3775, 1.3055, 0.0720],
    [-0.0271, 0.0888, 0.9383],
])
"""
*Bianco and Schettini PC (2010)* chromatic adaptation transform.

References
----------
:cite:`Bianco2010a`

Notes
-----
-   This chromatic adaptation transform has no negative lobes.
"""

CHROMATIC_ADAPTATION_TRANSFORMS: CaseInsensitiveMapping = (
    CaseInsensitiveMapping({
        'XYZ Scaling': CAT_XYZ_SCALING,
        'Von Kries': CAT_VON_KRIES,
        'Bradford': CAT_BRADFORD,
        'Sharp': CAT_SHARP,
        'Fairchild': CAT_FAIRCHILD,
        'CMCCAT97': CAT_CMCCAT97,
        'CMCCAT2000': CAT_CMCCAT2000,
        'CAT02': CAT_CAT02,
        'CAT02 Brill 2008': CAT_CAT02_BRILL2008,
        'CAT16': CAT_CAT16,
        'Bianco 2010': CAT_BIANCO2010,
        'Bianco PC 2010': CAT_PC_BIANCO2010
    }))
CHROMATIC_ADAPTATION_TRANSFORMS.__doc__ = """
Chromatic adaptation transforms.

References
----------
:cite:`Bianco2010a`, :cite:`Brill2008a`, :cite:`Fairchildb`, :cite:`Li2007e`,
:cite:`Li2017`, :cite:`Lindbloom2009g`, :cite:`Westland2012g`,
:cite:`Westland2012k`, :cite:`Wikipedia2007`
"""
