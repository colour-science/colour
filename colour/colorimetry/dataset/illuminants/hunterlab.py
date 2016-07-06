#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
HunterLab Dataset
=================

Defines the *HunterLab* illuminants dataset for the
*CIE 1931 2 Degree Standard Observer* and
*CIE 1964 10 Degree Standard Observer*.

The currently implemented data has been extracted from [1]_, however you may
want to use different data accordingly to the tables given in [2]_.

See Also
--------
`Illuminants IPython Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/colorimetry/illuminants.ipynb>`_

References
----------
.. [1]  HunterLab. (2008). Hunter L,a,b Color Scale. Retrieved from
        http://www.hunterlab.se/wp-content/uploads/2012/11/Hunter-L-a-b.pdf
.. [2]  HunterLab. (2008). Illuminant Factors in Universal Software and
        EasyMatch Coatings. Retrieved from
        https://support.hunterlab.com/hc/en-us/article_attachments/\
201437785/an02_02.pdf
"""

from __future__ import division, unicode_literals

from collections import namedtuple

from colour.utilities import CaseInsensitiveMapping

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2016 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'HunterLab_Illuminant_Specification',
    'HUNTERLAB_ILLUMINANTS_CIE_1931_2_DEGREE_STANDARD_OBSERVER_DATA',
    'HUNTERLAB_ILLUMINANTS_CIE_1931_2_DEGREE_STANDARD_OBSERVER',
    'HUNTERLAB_ILLUMINANTS_CIE_1964_10_DEGREE_STANDARD_OBSERVER_DATA',
    'HUNTERLAB_ILLUMINANTS_CIE_1964_10_DEGREE_STANDARD_OBSERVER',
    'HUNTERLAB_ILLUMINANTS']

HunterLab_Illuminant_Specification = namedtuple(
    'HunterLab_Illuminant_Specification',
    ('name', 'XYZ_n', 'K_ab'))

HUNTERLAB_ILLUMINANTS_CIE_1931_2_DEGREE_STANDARD_OBSERVER_DATA = (
    ('A', (109.83, 100.00, 35.55), (185.20, 38.40)),
    ('C', (98.04, 100.00, 118.11), (175.00, 70.00)),
    ('D65', (95.02, 100.00, 108.82), (172.30, 67.20)),
    ('D50', (96.38, 100.00, 82.45), (173.51, 58.48)),
    ('D60', (95.23, 100.00, 100.86), (172.47, 64.72)),
    ('D75', (94.96, 100.00, 122.53), (172.22, 71.30)),
    ('F2', (98.09, 100.00, 67.53), (175.00, 52.90)),
    ('TL 4', (101.40, 100.00, 65.90), (178.00, 52.30)),
    ('UL 3000', (107.99, 100.00, 33.91), (183.70, 37.50)))

HUNTERLAB_ILLUMINANTS_CIE_1931_2_DEGREE_STANDARD_OBSERVER = (
    CaseInsensitiveMapping(dict(
        [(x[0], HunterLab_Illuminant_Specification(*x)) for x in
         HUNTERLAB_ILLUMINANTS_CIE_1931_2_DEGREE_STANDARD_OBSERVER_DATA])))
"""
*Hunter L,a,b* illuminant dataset for *CIE 1931 2 Degree Standard Observer*.

HUNTERLAB_ILLUMINANTS_CIE_1931_2_DEGREE_STANDARD_OBSERVER :
    CaseInsensitiveMapping
"""

HUNTERLAB_ILLUMINANTS_CIE_1964_10_DEGREE_STANDARD_OBSERVER_DATA = (
    ('A', (111.16, 100.00, 35.19), (186.30, 38.20)),
    ('C', (97.30, 100.00, 116.14), (174.30, 69.40)),
    ('D50', (96.72, 100.00, 81.45), (173.82, 58.13)),
    ('D60', (95.21, 100.00, 99.60), (172.45, 64.28)),
    ('D65', (94.83, 100.00, 107.38), (172.10, 66.70)),
    ('D75', (94.45, 100.00, 120.70), (171.76, 70.76)),
    ('F2', (102.13, 100.00, 69.37), (178.60, 53.60)),
    ('TL 4', (103.82, 100.00, 66.90), (180.10, 52.70)),
    ('UL 3000', (111.12, 100.00, 35.21), (186.30, 38.20)))

HUNTERLAB_ILLUMINANTS_CIE_1964_10_DEGREE_STANDARD_OBSERVER = (
    CaseInsensitiveMapping(dict(
        [(x[0], HunterLab_Illuminant_Specification(*x)) for x in
         HUNTERLAB_ILLUMINANTS_CIE_1964_10_DEGREE_STANDARD_OBSERVER_DATA])))
"""
*Hunter L,a,b* illuminant dataset for *CIE 1964 10 Degree Standard Observer*.

HUNTERLAB_ILLUMINANTS_CIE_1964_10_DEGREE_STANDARD_OBSERVER :
    CaseInsensitiveMapping
"""

HUNTERLAB_ILLUMINANTS = CaseInsensitiveMapping(
    {'CIE 1931 2 Degree Standard Observer':
        HUNTERLAB_ILLUMINANTS_CIE_1931_2_DEGREE_STANDARD_OBSERVER,
     'CIE 1964 10 Degree Standard Observer':
        HUNTERLAB_ILLUMINANTS_CIE_1964_10_DEGREE_STANDARD_OBSERVER})
"""
Aggregated *Hunter L,a,b* illuminant dataset.

HUNTERLAB_ILLUMINANTS : CaseInsensitiveMapping
    **{'CIE 1931 2 Degree Standard Observer',
    'CIE 1964 10 Degree Standard Observer'}**

Aliases:

-   'cie_2_1931': 'CIE 1931 2 Degree Standard Observer'
-   'cie_10_1964': 'CIE 1964 10 Degree Standard Observer'
"""
HUNTERLAB_ILLUMINANTS['cie_2_1931'] = (
    HUNTERLAB_ILLUMINANTS['CIE 1931 2 Degree Standard Observer'])
HUNTERLAB_ILLUMINANTS['cie_10_1964'] = (
    HUNTERLAB_ILLUMINANTS['CIE 1964 10 Degree Standard Observer'])
