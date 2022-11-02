# !/usr/bin/env python
"""Define the unit tests for the :mod:`colour.io.ctl` module."""

from __future__ import annotations

import numpy as np
import os
import shutil
import tempfile
import textwrap
import unittest

from colour.io import (
    ctl_render,
    process_image_ctl,
    template_ctl_transform_float,
    template_ctl_transform_float3,
)
from colour.io import read_image
from colour.utilities import full, is_ctlrender_installed

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "ROOT_RESOURCES",
    "TestCtlRender",
    "TestProcessImageCtl",
    "TestTemplateCtlTransformFloat",
    "TestTemplateCtlTransformFloat3",
]

ROOT_RESOURCES: str = os.path.join(os.path.dirname(__file__), "resources")

# TODO: Reinstate coverage when "ctlrender" is tivially available
# cross-platform.


class TestCtlRender(unittest.TestCase):
    """Define :func:`colour.io.ctl.ctl_render` definition unit tests methods."""

    def setUp(self):
        """Initialise the common tests attributes."""

        self._temporary_directory = tempfile.mkdtemp()

    def tearDown(self):
        """After tests actions."""

        shutil.rmtree(self._temporary_directory)

    def test_ctl_render(self):  # pragma: no cover
        """Test :func:`colour.io.ctl.ctl_render` definition."""

        if not is_ctlrender_installed():
            return

        ctl_adjust_gain_float = template_ctl_transform_float(
            "rIn * gain[0]",
            "gIn * gain[1]",
            "bIn * gain[2]",
            description="Adjust Gain",
            parameters=["input float gain[3] = {1.0, 1.0, 1.0}"],
        )

        ctl_adjust_exposure_float = template_ctl_transform_float(
            "rIn * pow(2, exposure)",
            "gIn * pow(2, exposure)",
            "bIn * pow(2, exposure)",
            description="Adjust Exposure",
            parameters=["input float exposure = 0.0"],
        )

        path_input = os.path.join(ROOT_RESOURCES, "CMS_Test_Pattern.exr")
        path_output = os.path.join(
            self._temporary_directory, "CMS_Test_Pattern_Float.exr"
        )

        ctl_render(
            path_input,
            path_output,
            {
                ctl_adjust_gain_float: ["-param3 gain 0.5 1.0 2.0"],
                ctl_adjust_exposure_float: ["-param1 exposure 1.0"],
            },
            "-verbose",
            "-force",
        )

        np.testing.assert_array_almost_equal(
            read_image(path_output)[..., 0:3],
            read_image(path_input) * [1, 2, 4],
            decimal=7,
        )

        ctl_render(
            path_input,
            path_output,
            {
                os.path.join(ROOT_RESOURCES, "Adjust_Exposure_Float3.ctl"): [
                    "-param1 exposure 1.0"
                ],
            },
            "-verbose",
            "-force",
            env=dict(os.environ, CTL_MODULE_PATH=ROOT_RESOURCES),
        )

        np.testing.assert_array_almost_equal(
            read_image(path_output)[..., 0:3],
            read_image(path_input) * 2,
            decimal=7,
        )


class TestProcessImageCtl(unittest.TestCase):
    """
    Define :func:`colour.io.ctl.process_image_ctl` definition unit tests
    methods.
    """

    def test_process_image_ctl(self):  # pragma: no cover
        """Test :func:`colour.io.ctl.process_image_ctl` definition."""

        if not is_ctlrender_installed():
            return

        ctl_adjust_gain_float = template_ctl_transform_float(
            "rIn * gain[0]",
            "gIn * gain[1]",
            "bIn * gain[2]",
            description="Adjust Gain",
            parameters=["input float gain[3] = {1.0, 1.0, 1.0}"],
        )

        np.testing.assert_allclose(
            process_image_ctl(
                0.18,
                {
                    ctl_adjust_gain_float: ["-param3 gain 0.5 1.0 2.0"],
                },
                "-verbose",
                "-force",
            ),
            0.18 / 2,
            rtol=0.0001,
            atol=0.0001,
        )

        np.testing.assert_allclose(
            process_image_ctl(
                np.array([0.18, 0.18, 0.18]),
                {
                    ctl_adjust_gain_float: ["-param3 gain 0.5 1.0 2.0"],
                },
            ),
            np.array([0.18 / 2, 0.18, 0.18 * 2]),
            rtol=0.0001,
            atol=0.0001,
        )

        np.testing.assert_allclose(
            process_image_ctl(
                np.array([[0.18, 0.18, 0.18]]),
                {
                    ctl_adjust_gain_float: ["-param3 gain 0.5 1.0 2.0"],
                },
            ),
            np.array([[0.18 / 2, 0.18, 0.18 * 2]]),
            rtol=0.0001,
            atol=0.0001,
        )

        np.testing.assert_allclose(
            process_image_ctl(
                np.array([[[0.18, 0.18, 0.18]]]),
                {
                    ctl_adjust_gain_float: ["-param3 gain 0.5 1.0 2.0"],
                },
            ),
            np.array([[[0.18 / 2, 0.18, 0.18 * 2]]]),
            rtol=0.0001,
            atol=0.0001,
        )

        np.testing.assert_allclose(
            process_image_ctl(
                full([4, 2, 3], 0.18),
                {
                    ctl_adjust_gain_float: ["-param3 gain 0.5 1.0 2.0"],
                },
            ),
            full([4, 2, 3], 0.18) * [0.5, 1.0, 2.0],
            rtol=0.0001,
            atol=0.0001,
        )


class TestTemplateCtlTransformFloat(unittest.TestCase):
    """
    Define :func:`colour.io.ctl.template_ctl_transform_float` definition unit
    tests methods.
    """

    def test_template_ctl_transform_float(self):
        """Test :func:`colour.io.ctl.template_ctl_transform_float` definition."""

        ctl_foo_bar_float = template_ctl_transform_float(
            "rIn + foo[0]",
            "gIn + foo[1]",
            "bIn + foo[2]",
            description="Foo & Bar",
            imports=['import "Foo.ctl";', 'import "Bar.ctl";'],
            parameters=[
                "input float foo[3] = {1.0, 1.0, 1.0}",
                "input float bar = 1.0",
            ],
            header="// Custom Header\n",
        )

        self.assertEqual(
            ctl_foo_bar_float,
            textwrap.dedent(
                """
                // Foo & Bar

                import "Foo.ctl";
                import "Bar.ctl";

                // Custom Header

                void main
                (
                    input varying float rIn,
                    input varying float gIn,
                    input varying float bIn,
                    input varying float aIn,
                    output varying float rOut,
                    output varying float gOut,
                    output varying float bOut,
                    output varying float aOut,
                    input float foo[3] = {1.0, 1.0, 1.0},
                    input float bar = 1.0
                )
                {
                    rOut = rIn + foo[0];
                    gOut = gIn + foo[1];
                    bOut = bIn + foo[2];
                    aOut = aIn;
                }"""[
                    1:
                ]
            ),
        )


class TestTemplateCtlTransformFloat3(unittest.TestCase):
    """
    Define :func:`colour.io.ctl.template_ctl_transform_float3` definition unit
    tests methods.
    """

    def test_template_ctl_transform_float3(self):
        """Test :func:`colour.io.ctl.template_ctl_transform_float3` definition."""

        ctl_foo_bar_float3 = template_ctl_transform_float3(
            "baz(rgbIn, foo, bar)",
            description="Foo, Bar & Baz",
            imports=[
                '// import "Foo.ctl";',
                '// import "Bar.ctl";',
                '// import "Baz.ctl";',
            ],
            parameters=[
                "input float foo[3] = {1.0, 1.0, 1.0}",
                "input float bar = 1.0",
            ],
            header=textwrap.dedent(
                """
                float[3] baz(float rgbIn[3], float foo[3], float qux)
                {
                    float rgbOut[3];

                    rgbOut[0] = rgbIn[0] * foo[0]* qux;
                    rgbOut[1] = rgbIn[1] * foo[1]* qux;
                    rgbOut[2] = rgbIn[2] * foo[2]* qux;

                    return rgbOut;
                }\n"""[
                    1:
                ]
            ),
        )

        self.assertEqual(
            ctl_foo_bar_float3,
            textwrap.dedent(
                """
                // Foo, Bar & Baz

                // import "Foo.ctl";
                // import "Bar.ctl";
                // import "Baz.ctl";

                float[3] baz(float rgbIn[3], float foo[3], float qux)
                {
                    float rgbOut[3];

                    rgbOut[0] = rgbIn[0] * foo[0]* qux;
                    rgbOut[1] = rgbIn[1] * foo[1]* qux;
                    rgbOut[2] = rgbIn[2] * foo[2]* qux;

                    return rgbOut;
                }

                void main
                (
                    input varying float rIn,
                    input varying float gIn,
                    input varying float bIn,
                    input varying float aIn,
                    output varying float rOut,
                    output varying float gOut,
                    output varying float bOut,
                    output varying float aOut,
                    input float foo[3] = {1.0, 1.0, 1.0},
                    input float bar = 1.0
                )
                {
                    float rgbIn[3] = {rIn, gIn, bIn};

                    float rgbOut[3] = baz(rgbIn, foo, bar);

                    rOut = rgbOut[0];
                    gOut = rgbOut[1];
                    bOut = rgbOut[2];
                    aOut = aIn;
                }"""[
                    1:
                ]
            ),
        )


if __name__ == "__main__":
    unittest.main()
