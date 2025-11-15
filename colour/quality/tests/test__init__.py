"""Define the unit tests for the :mod:`colour.quality` module."""

from __future__ import annotations

from colour.colorimetry import SpectralDistribution
from colour.quality import colour_fidelity_index, colour_fidelity_index_CIE2017

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestColourFidelityIndex",
]

SD_SAMPLE_5NM = SpectralDistribution(
    {
        380: 0.000000,
        385: 0.000000,
        390: 0.000000,
        395: 0.000000,
        400: 0.000642,
        405: 0.001794,
        410: 0.003869,
        415: 0.007260,
        420: 0.013684,
        425: 0.022404,
        430: 0.033320,
        435: 0.046180,
        440: 0.061634,
        445: 0.080328,
        450: 0.102237,
        455: 0.128970,
        460: 0.160470,
        465: 0.197828,
        470: 0.239503,
        475: 0.283953,
        480: 0.329005,
        485: 0.375990,
        490: 0.427058,
        495: 0.481182,
        500: 0.538935,
        505: 0.601326,
        510: 0.667810,
        515: 0.737560,
        520: 0.805584,
        525: 0.869760,
        530: 0.926894,
        535: 0.973047,
        540: 1.000000,
        545: 1.000000,
        550: 0.974168,
        555: 0.922383,
        560: 0.844381,
        565: 0.745335,
        570: 0.631498,
        575: 0.513094,
        580: 0.398057,
        585: 0.294135,
        590: 0.206677,
        595: 0.137232,
        600: 0.085690,
        605: 0.051055,
        610: 0.029677,
        615: 0.016702,
        620: 0.009289,
        625: 0.005141,
        630: 0.002809,
        635: 0.001529,
        640: 0.000831,
        645: 0.000452,
        650: 0.000246,
        655: 0.000134,
        660: 0.000073,
        665: 0.000040,
        670: 0.000022,
        675: 0.000012,
        680: 0.000006,
        685: 0.000004,
        690: 0.000002,
        695: 0.000001,
        700: 0.000001,
        705: 0.000000,
        710: 0.000000,
        715: 0.000000,
        720: 0.000000,
        725: 0.000000,
        730: 0.000000,
        735: 0.000000,
        740: 0.000000,
        745: 0.000000,
        750: 0.000000,
        755: 0.000000,
        760: 0.000000,
        765: 0.000000,
        770: 0.000000,
        775: 0.000000,
        780: 0.000000,
    }
)


class TestColourFidelityIndex:
    """
    Define :func:`colour.quality.colour_fidelity_index` definition
    unit tests methods.
    """

    def test_colour_fidelity_index(self) -> None:
        """Test :func:`colour.quality.colour_fidelity_index` definition."""

        sd = SD_SAMPLE_5NM

        # Test default method (CIE 2017)
        assert colour_fidelity_index(sd) == colour_fidelity_index_CIE2017(sd)

        # Test explicit CIE 2017 method
        assert colour_fidelity_index(
            sd, method="CIE 2017"
        ) == colour_fidelity_index_CIE2017(sd)
