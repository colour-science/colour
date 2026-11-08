"""
Define the unit tests for the
:mod:`colour.models.rgb.transfer_functions.common` module.
"""

from __future__ import annotations

import math
import typing
from functools import partial

import numpy as np
import pytest

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models.rgb import transfer_functions as transfer
from colour.models.rgb.transfer_functions import (
    CCTF_DECODINGS,
    CCTF_ENCODINGS,
    EOTF_INVERSES,
    EOTFS,
    LOG_DECODINGS,
    LOG_ENCODINGS,
    OETF_INVERSES,
    OETFS,
    OOTF_INVERSES,
    OOTFS,
    cctf_decoding,
    cctf_encoding,
)
from colour.utilities import ColourUsageWarning, xp_as_array, xp_assert_close

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Development"

__all__ = [
    "TestCctfEncoding",
    "TestCctfDecoding",
    "TestTransferFunctions",
]


class TestCctfEncoding:
    """
    Define :func:`colour.models.rgb.transfer_functions.cctf_encoding`
    definition unit tests methods.
    """

    def test_raise_exception_cctf_encoding(
        self, xp: ModuleType, request: pytest.FixtureRequest
    ) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.aces.\
log_encoding_ACESproxy` definition raised exception.
        """

        with pytest.warns(ColourUsageWarning):
            cctf_encoding(0.18, function="ITU-R BT.2100 HLG")
        with pytest.warns(ColourUsageWarning):
            cctf_encoding(0.18, function="ITU-R BT.2100 PQ")

        # Safeguards must preserve NaN propagation on selected branches.
        for name, kwargs in [
            ("log_encoding_ACEScc", {}),
            ("log_encoding_ACEScct", {}),
            ("log_encoding_NLog", {}),
            ("exponent_function_basic", {"exponent": 2.2, "style": "basicMirrorFwd"}),
            ("exponent_function_basic", {"exponent": 2.2, "style": "basicMirrorRev"}),
        ]:
            result = getattr(transfer, name)(
                xp_as_array([float("nan")], xp=xp), **kwargs
            )
            assert bool(xp.all(xp.isnan(result)))

        if xp.__name__ != "numpy":
            autodiff = request.getfixturevalue("autodiff")
            for name, value, slope, kwargs in [
                ("log_encoding_ARRILogC3", -0.052272 / 5.555556, 5.367655, {}),
                (
                    "log_encoding_ARRILogC4",
                    -64 / ((2**18 - 16) / 117.45),
                    8.803033210331753,
                    {},
                ),
                (
                    "oetf_BlackmagicFilmGeneration5",
                    -0.005494072432257808,
                    8.283605932402494,
                    {},
                ),
                ("oetf_DaVinciIntermediate", -0.0075, 10.44426855, {}),
                ("log_encoding_DJIDLog", -0.0108 / 0.9892, 6.025, {}),
                ("log_encoding_FLog", -0.009468 / 0.555556, 8.735631, {}),
                ("log_encoding_LLog", -0.008846153846153846, 8, {}),
                ("log_encoding_VLog", -0.00873, 5.6, {}),
                (
                    "log_encoding_SLog",
                    -0.037584,
                    5 * 876 / 1023,
                    {"in_reflection": False},
                ),
                (
                    "log_encoding_FilmLightTLog",
                    -0.0057048244042473785,
                    16.1843764896659,
                    {},
                ),
                (
                    "log_encoding_Log3G10",
                    -0.01,
                    0.224282 * 155.975327 / math.log(10),
                    {"method": "v2"},
                ),
            ] + [
                (
                    name,
                    sign / coefficient,
                    slope * coefficient / (2 * math.log(10)) * scale,
                    {"method": method, "in_reflection": False},
                )
                for name, coefficient, slopes in [
                    ("log_encoding_CanonLog", 10.1596, (0.529136, 0.45310179)),
                    ("log_encoding_CanonLog2", 87.09937546, (0.281863093, 0.24136077)),
                    ("log_encoding_CanonLog3", 14.98325, (0.42889912, 0.36726845)),
                ]
                for method, slope, scale in [
                    ("v1", slopes[0], 876 / 1023),
                    ("v1.2", slopes[1], 1),
                ]
                for sign in (-1, 1)
            ]:
                function = getattr(transfer, name)
                _result, (gradient,), _inputs = autodiff(
                    partial(function, **kwargs), [value]
                )
                xp_assert_close(gradient, [slope], atol=TOLERANCE_ABSOLUTE_TESTS)


class TestCctfDecoding:
    """
    Define :func:`colour.models.rgb.transfer_functions.cctf_decoding`
    definition unit tests methods.
    """

    def test_raise_exception_cctf_decoding(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.aces.\
log_encoding_ACESproxy` definition raised exception.
        """

        with pytest.warns(ColourUsageWarning):
            cctf_decoding(0.18, function="ITU-R BT.2100 HLG")
        with pytest.warns(ColourUsageWarning):
            cctf_decoding(0.18, function="ITU-R BT.2100 PQ")


class TestTransferFunctions:
    """Define the transfer functions unit tests methods."""

    def test_transfer_functions(self) -> None:
        """Test the transfer functions reciprocity."""

        ignored_transfer_functions = (
            "ACESproxy",
            "DICOM GSDF",
            "Filmic Pro 6",
        )

        tolerance = {"D-Log": 0.1, "F-Log": 5e-4, "L-Log": 5e-4, "N-Log": 5e-3}

        reciprocal_mappings = [
            (LOG_ENCODINGS, LOG_DECODINGS),
            (OETFS, OETF_INVERSES),
            (EOTFS, EOTF_INVERSES),
            (CCTF_ENCODINGS, CCTF_DECODINGS),
            (OOTFS, OOTF_INVERSES),
        ]

        samples = np.hstack(
            [np.linspace(0, 1, int(1e5)), np.linspace(0, 65504, 65504 * 10)]
        )

        for encoding_mapping, _decoding_mapping in reciprocal_mappings:
            for name in encoding_mapping:
                if name in ignored_transfer_functions:
                    continue

                samples_r = np.copy(samples)

                if name == "ITU-T H.273 Log":
                    samples_r = np.clip(samples_r, 0.1, np.inf)

                if name == "ITU-T H.273 Log Sqrt":
                    samples_r = np.clip(samples_r, np.sqrt(10) / 1000, np.inf)

                samples_e = CCTF_ENCODINGS[name](samples_r)
                samples_d = CCTF_DECODINGS[name](samples_e)

                xp_assert_close(
                    samples_r,
                    samples_d,
                    atol=tolerance.get(name, TOLERANCE_ABSOLUTE_TESTS),
                )
