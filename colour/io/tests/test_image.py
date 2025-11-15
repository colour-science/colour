"""Define the unit tests for the :mod:`colour.io.image` module."""

from __future__ import annotations

import os
import platform
import shutil
import tempfile

import numpy as np
import pytest

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.io import (
    Image_Specification_Attribute,
    as_3_channels_image,
    convert_bit_depth,
    image_specification_OpenImageIO,
    read_image,
    read_image_Imageio,
    read_image_OpenImageIO,
    write_image,
    write_image_Imageio,
    write_image_OpenImageIO,
)
from colour.utilities import attest, full

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "ROOT_RESOURCES",
    "TestImageSpecificationOpenImageIO",
    "TestConvertBitDepth",
    "TestReadImageOpenImageIO",
    "TestWriteImageOpenImageIO",
    "TestReadImageImageio",
    "TestWriteImageImageio",
    "TestReadImage",
    "TestWriteImage",
    "TestAs3ChannelsImage",
]

ROOT_RESOURCES: str = os.path.join(os.path.dirname(__file__), "resources")


@pytest.mark.skipif(
    platform.system() == "Windows",
    reason="OpenImageIO crashes on Windows due to thread-safety issues",
)
class TestImageSpecificationOpenImageIO:
    """
    Define :func:`colour.io.image.image_specification_OpenImageIO` definition
    unit tests methods.
    """

    def test_image_specification_OpenImageIO(self) -> None:  # pragma: no cover
        """
        Test :func:`colour.io.image.image_specification_OpenImageIO`
        definition.
        """

        from OpenImageIO import HALF  # noqa: PLC0415

        compression = Image_Specification_Attribute("Compression", "none")
        specification = image_specification_OpenImageIO(
            1920, 1080, 3, "float16", [compression]
        )

        assert specification.width == 1920  # pyright: ignore
        assert specification.height == 1080  # pyright: ignore
        assert specification.nchannels == 3  # pyright: ignore
        assert specification.format == HALF  # pyright: ignore
        assert specification.extra_attribs[0].name == "Compression"  # pyright: ignore


class TestConvertBitDepth:
    """
    Define :func:`colour.io.image.convert_bit_depth` definition unit tests
    methods.
    """

    def test_convert_bit_depth(self) -> None:
        """Test :func:`colour.io.image.convert_bit_depth` definition."""

        a = np.around(np.linspace(0, 1, 10) * 255).astype("uint8")
        assert convert_bit_depth(a, "uint8").dtype is np.dtype("uint8")
        np.testing.assert_equal(convert_bit_depth(a, "uint8"), a)

        assert convert_bit_depth(a, "uint16").dtype is np.dtype("uint16")
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

        assert convert_bit_depth(a, "float16").dtype is np.dtype("float16")
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

        assert convert_bit_depth(a, "float32").dtype is np.dtype("float32")
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

        assert convert_bit_depth(a, "float64").dtype is np.dtype("float64")

        if hasattr(np, "float128"):  # pragma: no cover
            assert convert_bit_depth(a, "float128").dtype is np.dtype("float128")

        a = np.around(np.linspace(0, 1, 10) * 65535).astype("uint16")
        assert convert_bit_depth(a, "uint8").dtype is np.dtype("uint8")
        np.testing.assert_equal(
            convert_bit_depth(a, "uint8"),
            np.array([0, 28, 56, 85, 113, 141, 170, 198, 226, 255]),
        )

        assert convert_bit_depth(a, "uint16").dtype is np.dtype("uint16")
        np.testing.assert_equal(convert_bit_depth(a, "uint16"), a)

        assert convert_bit_depth(a, "float16").dtype is np.dtype("float16")
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

        assert convert_bit_depth(a, "float32").dtype is np.dtype("float32")
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

        assert convert_bit_depth(a, "float64").dtype is np.dtype("float64")

        if hasattr(np, "float128"):  # pragma: no cover
            assert convert_bit_depth(a, "float128").dtype is np.dtype("float128")

        a = np.linspace(0, 1, 10, dtype=np.float64)
        assert convert_bit_depth(a, "uint8").dtype is np.dtype("uint8")
        np.testing.assert_equal(
            convert_bit_depth(a, "uint8"),
            np.array([0, 28, 57, 85, 113, 142, 170, 198, 227, 255]),
        )

        assert convert_bit_depth(a, "uint16").dtype is np.dtype("uint16")
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

        assert convert_bit_depth(a, "float16").dtype is np.dtype("float16")
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

        assert convert_bit_depth(a, "float32").dtype is np.dtype("float32")
        np.testing.assert_allclose(
            convert_bit_depth(a, "float32"), a, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        assert convert_bit_depth(a, "float64").dtype is np.dtype("float64")

        if hasattr(np, "float128"):  # pragma: no cover
            assert convert_bit_depth(a, "float128").dtype is np.dtype("float128")


@pytest.mark.skipif(
    platform.system() == "Windows",
    reason="OpenImageIO crashes on Windows due to thread-safety issues",
)
class TestReadImageOpenImageIO:
    """
    Define :func:`colour.io.image.read_image_OpenImageIO` definition unit
    tests methods.
    """

    def test_read_image_OpenImageIO(self) -> None:  # pragma: no cover
        """Test :func:`colour.io.image.read_image_OpenImageIO` definition."""

        image = read_image_OpenImageIO(
            os.path.join(ROOT_RESOURCES, "CMS_Test_Pattern.exr"),
            additional_data=False,
        )
        assert image.shape == (1267, 1274, 3)
        assert image.dtype is np.dtype("float32")

        image = read_image_OpenImageIO(
            os.path.join(ROOT_RESOURCES, "CMS_Test_Pattern.exr"),
            "float16",
            additional_data=False,
        )
        assert image.dtype is np.dtype("float16")

        image, attributes = read_image_OpenImageIO(
            os.path.join(ROOT_RESOURCES, "CMS_Test_Pattern.exr"),
            additional_data=True,
        )
        assert image.shape == (1267, 1274, 3)
        assert len(attributes) > 0
        compression_attribute = next(
            (attribute for attribute in attributes if attribute.name == "compression"),
            None,
        )
        assert compression_attribute is not None

        image = read_image_OpenImageIO(
            os.path.join(ROOT_RESOURCES, "Single_Channel.exr"),
            additional_data=False,
        )
        assert image.shape == (256, 256)

        image = read_image_OpenImageIO(
            os.path.join(ROOT_RESOURCES, "Colour_Logo.png"),
            "uint8",
            additional_data=False,
        )
        assert image.shape == (128, 256, 4)
        assert image.dtype is np.dtype("uint8")
        assert np.min(image) == 0
        assert np.max(image) == 255

        image = read_image_OpenImageIO(
            os.path.join(ROOT_RESOURCES, "Colour_Logo.png"),
            "uint16",
            additional_data=False,
        )
        assert image.shape == (128, 256, 4)
        assert image.dtype is np.dtype("uint16")
        assert np.min(image) == 0
        assert np.max(image) == 65535

        # TODO: Investigate "OIIO" behaviour here: 1.0 != 15360.0
        # image = read_image_OpenImageIO(
        #     os.path.join(ROOT_RESOURCES, 'Colour_Logo.png'), 'float16')
        # self.assertIs(image.dtype, np.dtype('float16'))
        # self.assertEqual(np.min(image), 0.0)
        # self.assertEqual(np.max(image), 1.0)

        image = read_image_OpenImageIO(
            os.path.join(ROOT_RESOURCES, "Colour_Logo.png"),
            "float32",
            additional_data=False,
        )
        assert image.dtype is np.dtype("float32")
        assert np.min(image) == 0.0
        assert np.max(image) == 1.0


@pytest.mark.skipif(
    platform.system() == "Windows",
    reason="OpenImageIO crashes on Windows due to thread-safety issues",
)
class TestWriteImageOpenImageIO:
    """
    Define :func:`colour.io.image.write_image_OpenImageIO` definition unit
    tests methods.
    """

    def setup_method(self) -> None:
        """Initialise the common tests attributes."""

        self._temporary_directory = tempfile.mkdtemp()

    def teardown_method(self) -> None:
        """After tests actions."""

        shutil.rmtree(self._temporary_directory)

    def test_write_image_OpenImageIO(self) -> None:  # pragma: no cover
        """Test :func:`colour.io.image.write_image_OpenImageIO` definition."""

        from OpenImageIO import TypeDesc  # noqa: PLC0415

        path = os.path.join(self._temporary_directory, "8-bit.png")
        RGB = full((1, 1, 3), 255, np.uint8)
        write_image_OpenImageIO(RGB, path, bit_depth="uint8")
        image = read_image_OpenImageIO(path, bit_depth="uint8")
        np.testing.assert_equal(np.squeeze(RGB), image)

        path = os.path.join(self._temporary_directory, "16-bit.png")
        RGB = full((1, 1, 3), 65535, np.uint16)
        write_image_OpenImageIO(RGB, path, bit_depth="uint16")
        image = read_image_OpenImageIO(path, bit_depth="uint16")
        np.testing.assert_equal(np.squeeze(RGB), image)

        source_path = os.path.join(ROOT_RESOURCES, "Overflowing_Gradient.png")
        source_image = read_image_OpenImageIO(source_path, bit_depth="uint8")
        target_path = os.path.join(
            self._temporary_directory, "Overflowing_Gradient.png"
        )
        RGB = np.arange(0, 256, 1, dtype=np.uint8)[None] * 2
        write_image_OpenImageIO(RGB, target_path, bit_depth="uint8")
        target_image = read_image_OpenImageIO(source_path, bit_depth="uint8")
        np.testing.assert_equal(source_image, target_image)
        np.testing.assert_equal(np.squeeze(RGB), target_image)

        source_path = os.path.join(ROOT_RESOURCES, "CMS_Test_Pattern.exr")
        source_image = read_image_OpenImageIO(
            source_path,
            additional_data=False,
        )
        target_path = os.path.join(self._temporary_directory, "CMS_Test_Pattern.exr")
        write_image_OpenImageIO(source_image, target_path)
        target_image = read_image_OpenImageIO(
            target_path,
            additional_data=False,
        )
        np.testing.assert_equal(source_image, target_image)
        assert target_image.shape == (1267, 1274, 3)
        assert target_image.dtype is np.dtype("float32")

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
            Image_Specification_Attribute("customBooleanFlag", True),
            Image_Specification_Attribute(
                "chromaticities", chromaticities, TypeDesc("float[8]")
            ),
            Image_Specification_Attribute("compression", "none"),
        ]
        write_image_OpenImageIO(target_image, target_path, attributes=write_attributes)
        target_image, read_attributes = read_image_OpenImageIO(
            target_path, additional_data=True
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
                        assert write_attribute.value == read_attribute.value

            attest(
                attribute_exists,
                f'"{write_attribute.name}" attribute was not found on image!',
            )


class TestReadImageImageio:
    """
    Define :func:`colour.io.image.read_image_Imageio` definition unit tests
    methods.
    """

    def test_read_image_Imageio(self) -> None:
        """Test :func:`colour.io.image.read_image_Imageio` definition."""

        image = read_image_Imageio(os.path.join(ROOT_RESOURCES, "CMS_Test_Pattern.exr"))
        assert image.shape == (1267, 1274, 3)
        assert image.dtype is np.dtype("float32")

        image = read_image_Imageio(
            os.path.join(ROOT_RESOURCES, "CMS_Test_Pattern.exr"),
            "float16",
        )
        assert image.shape == (1267, 1274, 3)
        assert image.dtype is np.dtype("float16")

        image = read_image_Imageio(os.path.join(ROOT_RESOURCES, "Single_Channel.exr"))
        assert image.shape == (256, 256)

        image = read_image_Imageio(
            os.path.join(ROOT_RESOURCES, "Colour_Logo.png"), "uint8"
        )
        assert image.shape == (128, 256, 4)
        assert image.dtype is np.dtype("uint8")
        assert np.min(image) == 0
        assert np.max(image) == 255

        image = read_image_Imageio(
            os.path.join(ROOT_RESOURCES, "Colour_Logo.png"), "uint16"
        )
        assert image.shape == (128, 256, 4)
        assert image.dtype is np.dtype("uint16")
        assert np.min(image) == 0
        assert np.max(image) == 65535

        image = read_image_Imageio(
            os.path.join(ROOT_RESOURCES, "Colour_Logo.png"), "float16"
        )
        assert image.dtype is np.dtype("float16")
        assert np.min(image) == 0.0
        assert np.max(image) == 1.0

        image = read_image_Imageio(
            os.path.join(ROOT_RESOURCES, "Colour_Logo.png"), "float32"
        )
        assert image.dtype is np.dtype("float32")
        assert np.min(image) == 0.0
        assert np.max(image) == 1.0


class TestWriteImageImageio:
    """
    Define :func:`colour.io.image.write_image_Imageio` definition unit
    tests methods.
    """

    def setup_method(self) -> None:
        """Initialise the common tests attributes."""

        self._temporary_directory = tempfile.mkdtemp()

    def teardown_method(self) -> None:
        """After tests actions."""

        shutil.rmtree(self._temporary_directory)

    def test_write_image_Imageio(self) -> None:
        """Test :func:`colour.io.image.write_image_Imageio` definition."""

        source_path = os.path.join(ROOT_RESOURCES, "Overflowing_Gradient.png")
        source_image = read_image_Imageio(source_path, bit_depth="uint8")
        target_path = os.path.join(
            self._temporary_directory, "Overflowing_Gradient.png"
        )
        RGB = np.arange(0, 256, 1, dtype=np.uint8)[None] * 2
        write_image_Imageio(RGB, target_path, bit_depth="uint8")
        target_image = read_image_Imageio(target_path, bit_depth="uint8")
        np.testing.assert_equal(np.squeeze(RGB), target_image)
        np.testing.assert_equal(source_image, target_image)

    @pytest.mark.skipif(
        platform.system() == "Linux",
        reason="EXR tests are breaking on Linux",
    )
    def test_write_image_Imageio_exr(self) -> None:
        """
        Test :func:`colour.io.image.write_image_Imageio` definition with EXR
        files.
        """

        source_path = os.path.join(ROOT_RESOURCES, "CMS_Test_Pattern.exr")
        source_image = read_image_Imageio(source_path)
        target_path = os.path.join(self._temporary_directory, "CMS_Test_Pattern.exr")
        write_image_Imageio(source_image, target_path)
        target_image = read_image_Imageio(target_path)
        np.testing.assert_allclose(
            source_image, target_image, atol=TOLERANCE_ABSOLUTE_TESTS
        )
        assert target_image.shape == (1267, 1274, 3)
        assert target_image.dtype is np.dtype("float32")

        target_path = os.path.join(self._temporary_directory, "Full_White.exr")
        target_image = full((32, 16, 3), 1e6, dtype=np.float16)
        write_image_Imageio(target_image, target_path)
        target_image = read_image_Imageio(target_path)
        assert np.max(target_image) == np.inf

        target_image = full((32, 16, 3), 1e6)
        write_image_Imageio(target_image, target_path)
        target_image = read_image_Imageio(target_path)
        assert np.max(target_image) == 1e6


class TestReadImage:
    """
    Define :func:`colour.io.image.read_image` definition unit tests
    methods.
    """

    def test_read_image(self) -> None:
        """Test :func:`colour.io.image.read_image` definition."""

        image = read_image(os.path.join(ROOT_RESOURCES, "CMS_Test_Pattern.exr"))
        assert image.shape == (1267, 1274, 3)
        assert image.dtype is np.dtype("float32")

        image = read_image(os.path.join(ROOT_RESOURCES, "Single_Channel.exr"))
        assert image.shape == (256, 256)


@pytest.mark.skipif(
    platform.system() == "Windows",
    reason="OpenImageIO crashes on Windows due to thread-safety issues",
)
class TestWriteImage:
    """Define :func:`colour.io.image.write_image` definition unit tests methods."""

    def setup_method(self) -> None:
        """Initialise the common tests attributes."""

        self._temporary_directory = tempfile.mkdtemp()

    def teardown_method(self) -> None:
        """After tests actions."""

        shutil.rmtree(self._temporary_directory)

    def test_write_image(self) -> None:
        """Test :func:`colour.io.image.write_image` definition."""

        source_path = os.path.join(ROOT_RESOURCES, "CMS_Test_Pattern.exr")
        source_image = read_image(source_path)
        target_path = os.path.join(self._temporary_directory, "CMS_Test_Pattern.exr")
        write_image(source_image, target_path)
        target_image = read_image(target_path)
        np.testing.assert_allclose(
            source_image, target_image, atol=TOLERANCE_ABSOLUTE_TESTS
        )
        assert target_image.shape == (1267, 1274, 3)
        assert target_image.dtype is np.dtype("float32")


class TestAs3ChannelsImage:
    """
    Define :func:`colour.io.image.as_3_channels_image` definition unit tests
    methods.
    """

    def test_as_3_channels_image(self) -> None:
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
        a = np.array([[[[0.18, 0.18, 0.18]]]])
        np.testing.assert_equal(as_3_channels_image(a), b)
        a = np.array([[0.18, 0.18, 0.18], [0.20, 0.20, 0.20]])
        result = as_3_channels_image(a)
        assert result.shape == (1, 2, 3)

    def test_raise_exception_as_3_channels_image(self) -> None:
        """
        Test :func:`colour.io.image.as_3_channels_image` definition raised
        exception.
        """

        pytest.raises(
            ValueError,
            as_3_channels_image,
            [
                [
                    [[0.18, 0.18, 0.18], [0.18, 0.18, 0.18]],
                    [[0.18, 0.18, 0.18], [0.18, 0.18, 0.18]],
                ],
                [
                    [[0.18, 0.18, 0.18], [0.18, 0.18, 0.18]],
                    [[0.18, 0.18, 0.18], [0.18, 0.18, 0.18]],
                ],
            ],
        )

        pytest.raises(
            ValueError,
            as_3_channels_image,
            [0.18, 0.18, 0.18, 0.18],
        )
