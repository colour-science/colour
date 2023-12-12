# !/usr/bin/env python
"""Define the unit tests for the :mod:`colour.io.image` module."""

from __future__ import annotations

import os
import platform
import shutil
import tempfile
import unittest

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.io import (
    ImageAttribute_Specification,
    as_3_channels_image,
    convert_bit_depth,
    read_image,
    read_image_Imageio,
    read_image_OpenImageIO,
    write_image,
    write_image_Imageio,
    write_image_OpenImageIO,
)
from colour.utilities import attest, full, is_openimageio_installed

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "ROOT_RESOURCES",
    "TestReadImageOpenImageIO",
    "TestWriteImageOpenImageIO",
    "TestReadImageImageio",
    "TestWriteImageImageio",
    "TestReadImage",
    "TestWriteImage",
]

ROOT_RESOURCES: str = os.path.join(os.path.dirname(__file__), "resources")


class TestConvertBitDepth(unittest.TestCase):
    """
    Define :func:`colour.io.image.convert_bit_depth` definition unit tests
    methods.
    """

    def test_convert_bit_depth(self):
        """Test :func:`colour.io.image.convert_bit_depth` definition."""

        a = np.around(np.linspace(0, 1, 10) * 255).astype("uint8")
        self.assertIs(convert_bit_depth(a, "uint8").dtype, np.dtype("uint8"))
        np.testing.assert_equal(convert_bit_depth(a, "uint8"), a)

        self.assertIs(convert_bit_depth(a, "uint16").dtype, np.dtype("uint16"))
        np.testing.assert_equal(
            convert_bit_depth(a, "uint16"),
            np.array(
                [
                    0,
                    7196,
                    14649,
                    21845,
                    29041,
                    36494,
                    43690,
                    50886,
                    58339,
                    65535,
                ]
            ),
        )

        self.assertIs(
            convert_bit_depth(a, "float16").dtype, np.dtype("float16")
        )
        np.testing.assert_allclose(
            convert_bit_depth(a, "float16"),
            np.array(
                [
                    0.0000,
                    0.1098,
                    0.2235,
                    0.3333,
                    0.443,
                    0.5566,
                    0.6665,
                    0.7764,
                    0.8900,
                    1.0000,
                ]
            ),
            atol=5e-4,
        )

        self.assertIs(
            convert_bit_depth(a, "float32").dtype, np.dtype("float32")
        )
        np.testing.assert_allclose(
            convert_bit_depth(a, "float32"),
            np.array(
                [
                    0.00000000,
                    0.10980392,
                    0.22352941,
                    0.33333334,
                    0.44313726,
                    0.55686277,
                    0.66666669,
                    0.77647060,
                    0.89019608,
                    1.00000000,
                ]
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        self.assertIs(
            convert_bit_depth(a, "float64").dtype, np.dtype("float64")
        )

        if hasattr(np, "float128"):  # pragma: no cover
            self.assertIs(
                convert_bit_depth(a, "float128").dtype, np.dtype("float128")
            )

        a = np.around(np.linspace(0, 1, 10) * 65535).astype("uint16")
        self.assertIs(convert_bit_depth(a, "uint8").dtype, np.dtype("uint8"))
        np.testing.assert_equal(
            convert_bit_depth(a, "uint8"),
            np.array([0, 28, 56, 85, 113, 141, 170, 198, 226, 255]),
        )

        self.assertIs(convert_bit_depth(a, "uint16").dtype, np.dtype("uint16"))
        np.testing.assert_equal(convert_bit_depth(a, "uint16"), a)

        self.assertIs(
            convert_bit_depth(a, "float16").dtype, np.dtype("float16")
        )
        np.testing.assert_allclose(
            convert_bit_depth(a, "float16"),
            np.array(
                [
                    0.0000,
                    0.1098,
                    0.2235,
                    0.3333,
                    0.443,
                    0.5566,
                    0.6665,
                    0.7764,
                    0.8900,
                    1.0000,
                ]
            ),
            atol=5e-2,
        )

        self.assertIs(
            convert_bit_depth(a, "float32").dtype, np.dtype("float32")
        )
        np.testing.assert_allclose(
            convert_bit_depth(a, "float32"),
            np.array(
                [
                    0.00000000,
                    0.11111620,
                    0.22221714,
                    0.33333334,
                    0.44444954,
                    0.55555046,
                    0.66666669,
                    0.77778286,
                    0.88888383,
                    1.00000000,
                ]
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        self.assertIs(
            convert_bit_depth(a, "float64").dtype, np.dtype("float64")
        )

        if hasattr(np, "float128"):  # pragma: no cover
            self.assertIs(
                convert_bit_depth(a, "float128").dtype, np.dtype("float128")
            )

        a = np.linspace(0, 1, 10, dtype=np.float64)
        self.assertIs(convert_bit_depth(a, "uint8").dtype, np.dtype("uint8"))
        np.testing.assert_equal(
            convert_bit_depth(a, "uint8"),
            np.array([0, 28, 57, 85, 113, 142, 170, 198, 227, 255]),
        )

        self.assertIs(convert_bit_depth(a, "uint16").dtype, np.dtype("uint16"))
        np.testing.assert_equal(
            convert_bit_depth(a, "uint16"),
            np.array(
                [
                    0,
                    7282,
                    14563,
                    21845,
                    29127,
                    36408,
                    43690,
                    50972,
                    58253,
                    65535,
                ]
            ),
        )

        self.assertIs(
            convert_bit_depth(a, "float16").dtype, np.dtype("float16")
        )
        np.testing.assert_allclose(
            convert_bit_depth(a, "float16"),
            np.array(
                [
                    0.0000,
                    0.1111,
                    0.2222,
                    0.3333,
                    0.4443,
                    0.5557,
                    0.6665,
                    0.7780,
                    0.8887,
                    1.0000,
                ]
            ),
            atol=5e-4,
        )

        self.assertIs(
            convert_bit_depth(a, "float32").dtype, np.dtype("float32")
        )
        np.testing.assert_allclose(
            convert_bit_depth(a, "float32"), a, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        self.assertIs(
            convert_bit_depth(a, "float64").dtype, np.dtype("float64")
        )

        if hasattr(np, "float128"):  # pragma: no cover
            self.assertIs(
                convert_bit_depth(a, "float128").dtype, np.dtype("float128")
            )


class TestReadImageOpenImageIO(unittest.TestCase):
    """
    Define :func:`colour.io.image.read_image_OpenImageIO` definition unit
    tests methods.
    """

    def test_read_image_OpenImageIO(self):  # pragma: no cover
        """Test :func:`colour.io.image.read_image_OpenImageIO` definition."""

        if not is_openimageio_installed():
            return

        image = read_image_OpenImageIO(
            os.path.join(ROOT_RESOURCES, "CMS_Test_Pattern.exr")
        )
        self.assertTupleEqual(image.shape, (1267, 1274, 3))
        self.assertIs(image.dtype, np.dtype("float32"))

        image = read_image_OpenImageIO(
            os.path.join(ROOT_RESOURCES, "CMS_Test_Pattern.exr"),
            "float16",
        )
        self.assertIs(image.dtype, np.dtype("float16"))

        image, attributes = read_image_OpenImageIO(
            os.path.join(ROOT_RESOURCES, "CMS_Test_Pattern.exr"),
            attributes=True,
        )
        self.assertTupleEqual(image.shape, (1267, 1274, 3))
        self.assertEqual(attributes[0].name, "oiio:ColorSpace")
        self.assertEqual(attributes[0].value, "Linear")

        image = read_image_OpenImageIO(
            os.path.join(ROOT_RESOURCES, "Single_Channel.exr")
        )
        self.assertTupleEqual(image.shape, (256, 256))

        image = read_image_OpenImageIO(
            os.path.join(ROOT_RESOURCES, "Colour_Logo.png"), "uint8"
        )
        self.assertTupleEqual(image.shape, (128, 256, 4))
        self.assertIs(image.dtype, np.dtype("uint8"))
        self.assertEqual(np.min(image), 0)
        self.assertEqual(np.max(image), 255)

        image = read_image_OpenImageIO(
            os.path.join(ROOT_RESOURCES, "Colour_Logo.png"), "uint16"
        )
        self.assertTupleEqual(image.shape, (128, 256, 4))
        self.assertIs(image.dtype, np.dtype("uint16"))
        self.assertEqual(np.min(image), 0)
        self.assertEqual(np.max(image), 65535)

        # TODO: Investigate "OIIO" behaviour here: 1.0 != 15360.0
        # image = read_image_OpenImageIO(
        #     os.path.join(ROOT_RESOURCES, 'Colour_Logo.png'), 'float16')
        # self.assertIs(image.dtype, np.dtype('float16'))
        # self.assertEqual(np.min(image), 0.0)
        # self.assertEqual(np.max(image), 1.0)

        image = read_image_OpenImageIO(
            os.path.join(ROOT_RESOURCES, "Colour_Logo.png"), "float32"
        )
        self.assertIs(image.dtype, np.dtype("float32"))
        self.assertEqual(np.min(image), 0.0)
        self.assertEqual(np.max(image), 1.0)


class TestWriteImageOpenImageIO(unittest.TestCase):
    """
    Define :func:`colour.io.image.write_image_OpenImageIO` definition unit
    tests methods.
    """

    def setUp(self):
        """Initialise the common tests attributes."""

        self._temporary_directory = tempfile.mkdtemp()

    def tearDown(self):
        """After tests actions."""

        shutil.rmtree(self._temporary_directory)

    def test_write_image_OpenImageIO(self):  # pragma: no cover
        """Test :func:`colour.io.image.write_image_OpenImageIO` definition."""

        if not is_openimageio_installed():
            return

        from OpenImageIO import TypeDesc

        image_path = os.path.join(self._temporary_directory, "8-bit.png")
        RGB = full((1, 1, 3), 255, np.uint8)
        write_image_OpenImageIO(RGB, image_path, bit_depth="uint8")
        image = read_image_OpenImageIO(image_path, bit_depth="uint8")
        np.testing.assert_equal(np.squeeze(RGB), image)

        image_path = os.path.join(self._temporary_directory, "16-bit.png")
        RGB = full((1, 1, 3), 65535, np.uint16)
        write_image_OpenImageIO(RGB, image_path, bit_depth="uint16")
        image = read_image_OpenImageIO(image_path, bit_depth="uint16")
        np.testing.assert_equal(np.squeeze(RGB), image)

        source_image_path = os.path.join(
            ROOT_RESOURCES, "Overflowing_Gradient.png"
        )
        target_image_path = os.path.join(
            self._temporary_directory, "Overflowing_Gradient.png"
        )
        RGB = np.arange(0, 256, 1, dtype=np.uint8)[None] * 2
        write_image_OpenImageIO(RGB, target_image_path, bit_depth="uint8")
        image = read_image_OpenImageIO(source_image_path, bit_depth="uint8")
        np.testing.assert_equal(np.squeeze(RGB), image)

        source_image_path = os.path.join(
            ROOT_RESOURCES, "CMS_Test_Pattern.exr"
        )
        target_image_path = os.path.join(
            self._temporary_directory, "CMS_Test_Pattern.exr"
        )
        image = read_image_OpenImageIO(source_image_path)
        write_image_OpenImageIO(image, target_image_path)
        image = read_image_OpenImageIO(target_image_path)
        self.assertTupleEqual(image.shape, (1267, 1274, 3))
        self.assertIs(image.dtype, np.dtype("float32"))

        chromaticities = (
            0.73470,
            0.26530,
            0.00000,
            1.00000,
            0.00010,
            -0.07700,
            0.32168,
            0.33767,
        )
        write_attributes = [
            ImageAttribute_Specification("acesImageContainerFlag", True),
            ImageAttribute_Specification(
                "chromaticities", chromaticities, TypeDesc("float[8]")
            ),
            ImageAttribute_Specification("compression", "none"),
        ]
        write_image_OpenImageIO(
            image, target_image_path, attributes=write_attributes
        )
        image, read_attributes = read_image_OpenImageIO(
            target_image_path, attributes=True
        )
        for write_attribute in write_attributes:
            attribute_exists = False
            for read_attribute in read_attributes:
                if write_attribute.name == read_attribute.name:
                    attribute_exists = True
                    if isinstance(write_attribute.value, tuple):
                        np.testing.assert_allclose(
                            write_attribute.value,
                            read_attribute.value,
                            atol=TOLERANCE_ABSOLUTE_TESTS,
                        )
                    else:
                        self.assertEqual(
                            write_attribute.value, read_attribute.value
                        )

            attest(
                attribute_exists,
                f'"{write_attribute.name}" attribute was not found on image!',
            )


class TestReadImageImageio(unittest.TestCase):
    """
    Define :func:`colour.io.image.read_image_Imageio` definition unit tests
    methods.
    """

    def test_read_image_Imageio(self):
        """Test :func:`colour.io.image.read_image_Imageio` definition."""

        image = read_image_Imageio(
            os.path.join(ROOT_RESOURCES, "CMS_Test_Pattern.exr")
        )
        self.assertTupleEqual(image.shape, (1267, 1274, 3))
        self.assertIs(image.dtype, np.dtype("float32"))

        image = read_image_Imageio(
            os.path.join(ROOT_RESOURCES, "CMS_Test_Pattern.exr"),
            "float16",
        )
        self.assertTupleEqual(image.shape, (1267, 1274, 3))
        self.assertIs(image.dtype, np.dtype("float16"))

        image = read_image_Imageio(
            os.path.join(ROOT_RESOURCES, "Single_Channel.exr")
        )
        self.assertTupleEqual(image.shape, (256, 256))

        image = read_image_Imageio(
            os.path.join(ROOT_RESOURCES, "Colour_Logo.png"), "uint8"
        )
        self.assertTupleEqual(image.shape, (128, 256, 4))
        self.assertIs(image.dtype, np.dtype("uint8"))
        self.assertEqual(np.min(image), 0)
        self.assertEqual(np.max(image), 255)

        image = read_image_Imageio(
            os.path.join(ROOT_RESOURCES, "Colour_Logo.png"), "uint16"
        )
        self.assertTupleEqual(image.shape, (128, 256, 4))
        self.assertIs(image.dtype, np.dtype("uint16"))
        self.assertEqual(np.min(image), 0)
        self.assertEqual(np.max(image), 65535)

        image = read_image_Imageio(
            os.path.join(ROOT_RESOURCES, "Colour_Logo.png"), "float16"
        )
        self.assertIs(image.dtype, np.dtype("float16"))
        self.assertEqual(np.min(image), 0.0)
        self.assertEqual(np.max(image), 1.0)

        image = read_image_Imageio(
            os.path.join(ROOT_RESOURCES, "Colour_Logo.png"), "float32"
        )
        self.assertIs(image.dtype, np.dtype("float32"))
        self.assertEqual(np.min(image), 0.0)
        self.assertEqual(np.max(image), 1.0)


class TestWriteImageImageio(unittest.TestCase):
    """
    Define :func:`colour.io.image.write_image_Imageio` definition unit
    tests methods.
    """

    def setUp(self):
        """Initialise the common tests attributes."""

        self._temporary_directory = tempfile.mkdtemp()

    def tearDown(self):
        """After tests actions."""

        shutil.rmtree(self._temporary_directory)

    def test_write_image_Imageio(self):
        """Test :func:`colour.io.image.write_image_Imageio` definition."""

        source_image_path = os.path.join(
            ROOT_RESOURCES, "Overflowing_Gradient.png"
        )
        target_image_path = os.path.join(
            self._temporary_directory, "Overflowing_Gradient.png"
        )
        RGB = np.arange(0, 256, 1, dtype=np.uint8)[None] * 2
        write_image_Imageio(RGB, target_image_path, bit_depth="uint8")
        image = read_image_Imageio(source_image_path, bit_depth="uint8")
        np.testing.assert_equal(np.squeeze(RGB), image)

        source_image_path = os.path.join(
            ROOT_RESOURCES, "CMS_Test_Pattern.exr"
        )
        target_image_path = os.path.join(
            self._temporary_directory, "CMS_Test_Pattern.exr"
        )
        image = read_image_Imageio(source_image_path)
        write_image_Imageio(image, target_image_path)
        image = read_image_Imageio(target_image_path)
        self.assertTupleEqual(image.shape, (1267, 1274, 3))
        self.assertIs(image.dtype, np.dtype("float32"))

        # NOTE: Those unit tests are breaking unpredictably on Linux, skipping
        # for now.
        if platform.system() != "Linux":  # pragma: no cover
            target_image_path = os.path.join(
                self._temporary_directory, "Full_White.exr"
            )
            image = full((32, 16, 3), 1e6, dtype=np.float16)
            write_image_Imageio(image, target_image_path)
            image = read_image_Imageio(target_image_path)
            self.assertEqual(np.max(image), np.inf)

            image = full((32, 16, 3), 1e6)
            write_image_Imageio(image, target_image_path)
            image = read_image_Imageio(target_image_path)
            self.assertEqual(np.max(image), 1e6)


class TestReadImage(unittest.TestCase):
    """
    Define :func:`colour.io.image.read_image` definition unit tests
    methods.
    """

    def test_read_image(self):
        """Test :func:`colour.io.image.read_image` definition."""

        image = read_image(
            os.path.join(ROOT_RESOURCES, "CMS_Test_Pattern.exr")
        )
        self.assertTupleEqual(image.shape, (1267, 1274, 3))
        self.assertIs(image.dtype, np.dtype("float32"))

        image = read_image(os.path.join(ROOT_RESOURCES, "Single_Channel.exr"))
        self.assertTupleEqual(image.shape, (256, 256))


class TestWriteImage(unittest.TestCase):
    """Define :func:`colour.io.image.write_image` definition unit tests methods."""

    def setUp(self):
        """Initialise the common tests attributes."""

        self._temporary_directory = tempfile.mkdtemp()

    def tearDown(self):
        """After tests actions."""

        shutil.rmtree(self._temporary_directory)

    def test_write_image(self):
        """Test :func:`colour.io.image.write_image` definition."""

        source_image_path = os.path.join(
            ROOT_RESOURCES, "CMS_Test_Pattern.exr"
        )
        target_image_path = os.path.join(
            self._temporary_directory, "CMS_Test_Pattern.exr"
        )
        image = read_image(source_image_path)
        write_image(image, target_image_path)
        image = read_image(target_image_path)
        self.assertTupleEqual(image.shape, (1267, 1274, 3))
        self.assertIs(image.dtype, np.dtype("float32"))


class TestAs3ChannelsImage(unittest.TestCase):
    """
    Define :func:`colour.io.image.as_3_channels_image` definition unit tests
    methods.
    """

    def test_as_3_channels_image(self):
        """Test :func:`colour.io.image.as_3_channels_image` definition."""

        a = 0.18
        b = np.array([[[0.18, 0.18, 0.18]]])
        np.testing.assert_equal(as_3_channels_image(a), b)
        a = np.array([0.18])
        np.testing.assert_equal(as_3_channels_image(a), b)
        a = np.array([0.18, 0.18, 0.18])
        np.testing.assert_equal(as_3_channels_image(a), b)
        a = np.array([[0.18, 0.18, 0.18]])
        np.testing.assert_equal(as_3_channels_image(a), b)
        a = np.array([[[0.18, 0.18, 0.18]]])
        np.testing.assert_equal(as_3_channels_image(a), b)


if __name__ == "__main__":
    unittest.main()
