"""
"colour.models" sub-package Benchmarks
======================================
"""


import colour

from benchmarks.factories import IJ_suites_factory, IJK_suites_factory

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"


IJK_suites_factory(
    [
        colour.models.XYZ_to_xyY,
        colour.models.xyY_to_XYZ,
        colour.models.xyY_to_xy,
        colour.models.XYZ_to_xy,
    ],
    __name__,
)


IJ_suites_factory(
    [
        colour.models.xy_to_XYZ,
        colour.models.xy_to_xyY,
    ],
    __name__,
)
