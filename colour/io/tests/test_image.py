# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.io.image` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import os
import platform
import shutil
import unittest
import tempfile

from colour.io import convert_bit_depth
from colour.io import read_image_OpenImageIO, write_image_OpenImageIO
from colour.io import read_image_Imageio, write_image_Imageio
from colour.io import read_image, write_image
from colour.io import ImageAttribute_Specification
from colour.utilities import is_openimageio_installed

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'RESOURCES_DIRECTORY', 'TestReadImageOpenImageIO',
    'TestWriteImageOpenImageIO', 'TestReadImageImageio',
    'TestWriteImageImageio', 'TestReadImage', 'TestWriteImage'
]

RESOURCES_DIRECTORY = os.path.join(os.path.dirname(__file__), 'resources')


class TestConvertBitDepth(unittest.TestCase):
    """
    Defines :func:`colour.io.image.convert_bit_depth` definition units tests
    methods.
    """

    def test_convert_bit_depth(self):
        """
        Tests :func:`colour.io.image.convert_bit_depth` definition.
        """

        a = np.around(np.linspace(0, 1, 10) * 255).astype('uint8')
        self.assertIs(convert_bit_depth(a, 'uint8').dtype, np.dtype('uint8'))
        np.testing.assert_equal(convert_bit_depth(a, 'uint8'), a)

        self.assertIs(convert_bit_depth(a, 'uint16').dtype, np.dtype('uint16'))
        np.testing.assert_equal(
            convert_bit_depth(a, 'uint16'),
            np.array([
                0, 7196, 14649, 21845, 29041, 36494, 43690, 50886, 58339, 65535
            ]))

        self.assertIs(
            convert_bit_depth(a, 'float16').dtype, np.dtype('float16'))
        np.testing.assert_almost_equal(
            convert_bit_depth(a, 'float16'),
            np.array([
                0.0000, 0.1098, 0.2235, 0.3333, 0.443, 0.5566, 0.6665, 0.7764,
                0.8900, 1.0000
            ]),
            decimal=3)

        self.assertIs(
            convert_bit_depth(a, 'float32').dtype, np.dtype('float32'))
        np.testing.assert_almost_equal(
            convert_bit_depth(a, 'float32'),
            np.array([
                0.00000000, 0.10980392, 0.22352941, 0.33333334, 0.44313726,
                0.55686277, 0.66666669, 0.77647060, 0.89019608, 1.00000000
            ]),
            decimal=7)

        self.assertIs(
            convert_bit_depth(a, 'float64').dtype, np.dtype('float64'))

        if platform.system() not in ('Windows',
                                     'Microsoft'):  # pragma: no cover
            self.assertIs(
                convert_bit_depth(a, 'float128').dtype, np.dtype('float128'))

        a = np.around(np.linspace(0, 1, 10) * 65535).astype('uint16')
        self.assertIs(convert_bit_depth(a, 'uint8').dtype, np.dtype('uint8'))
        np.testing.assert_equal(
            convert_bit_depth(a, 'uint8'),
            np.array([0, 28, 56, 85, 113, 141, 170, 198, 226, 255]))

        self.assertIs(convert_bit_depth(a, 'uint16').dtype, np.dtype('uint16'))
        np.testing.assert_equal(convert_bit_depth(a, 'uint16'), a)

        self.assertIs(
            convert_bit_depth(a, 'float16').dtype, np.dtype('float16'))
        np.testing.assert_almost_equal(
            convert_bit_depth(a, 'float16'),
            np.array([
                0.0000, 0.1098, 0.2235, 0.3333, 0.443, 0.5566, 0.6665, 0.7764,
                0.8900, 1.0000
            ]),
            decimal=3)

        self.assertIs(
            convert_bit_depth(a, 'float32').dtype, np.dtype('float32'))
        np.testing.assert_almost_equal(
            convert_bit_depth(a, 'float32'),
            np.array([
                0.00000000, 0.11111620, 0.22221714, 0.33333334, 0.44444954,
                0.55555046, 0.66666669, 0.77778286, 0.88888383, 1.00000000
            ]),
            decimal=7)

        self.assertIs(
            convert_bit_depth(a, 'float64').dtype, np.dtype('float64'))

        if platform.system() not in ('Windows',
                                     'Microsoft'):  # pragma: no cover
            self.assertIs(
                convert_bit_depth(a, 'float128').dtype, np.dtype('float128'))

        a = np.linspace(0, 1, 10, dtype=np.float64)
        self.assertIs(convert_bit_depth(a, 'uint8').dtype, np.dtype('uint8'))
        np.testing.assert_equal(
            convert_bit_depth(a, 'uint8'),
            np.array([0, 28, 57, 85, 113, 142, 170, 198, 227, 255]))

        self.assertIs(convert_bit_depth(a, 'uint16').dtype, np.dtype('uint16'))
        np.testing.assert_equal(
            convert_bit_depth(a, 'uint16'),
            np.array([
                0, 7282, 14563, 21845, 29127, 36408, 43690, 50972, 58253, 65535
            ]))

        self.assertIs(
            convert_bit_depth(a, 'float16').dtype, np.dtype('float16'))
        np.testing.assert_almost_equal(
            convert_bit_depth(a, 'float16'),
            np.array([
                0.0000, 0.1111, 0.2222, 0.3333, 0.4443, 0.5557, 0.6665, 0.7780,
                0.8887, 1.0000
            ]),
            decimal=3)

        self.assertIs(
            convert_bit_depth(a, 'float32').dtype, np.dtype('float32'))
        np.testing.assert_almost_equal(
            convert_bit_depth(a, 'float32'), a, decimal=7)

        self.assertIs(
            convert_bit_depth(a, 'float64').dtype, np.dtype('float64'))

        if platform.system() not in ('Windows',
                                     'Microsoft'):  # pragma: no cover
            self.assertIs(
                convert_bit_depth(a, 'float128').dtype, np.dtype('float128'))


class TestReadImageOpenImageIO(unittest.TestCase):
    """
    Defines :func:`colour.io.image.read_image_OpenImageIO` definition units
    tests methods.
    """

    def test_read_image_OpenImageIO(self):  # pragma: no cover
        """
        Tests :func:`colour.io.image.read_image_OpenImageIO` definition.
        """

        if not is_openimageio_installed():
            return

        image = read_image_OpenImageIO(
            os.path.join(RESOURCES_DIRECTORY, 'CMS_Test_Pattern.exr'))
        self.assertTupleEqual(image.shape, (1267, 1274, 3))
        self.assertIs(image.dtype, np.dtype('float32'))

        image = read_image_OpenImageIO(
            os.path.join(RESOURCES_DIRECTORY, 'CMS_Test_Pattern.exr'),
            'float16')
        self.assertIs(image.dtype, np.dtype('float16'))

        image, attributes = read_image_OpenImageIO(
            os.path.join(RESOURCES_DIRECTORY, 'CMS_Test_Pattern.exr'),
            attributes=True)
        self.assertTupleEqual(image.shape, (1267, 1274, 3))
        self.assertEqual(attributes[0].name, 'oiio:ColorSpace')
        self.assertEqual(attributes[0].value, 'Linear')

        image = read_image_OpenImageIO(
            os.path.join(RESOURCES_DIRECTORY, 'Single_Channel.exr'))
        self.assertTupleEqual(image.shape, (256, 256))

        image = read_image_OpenImageIO(
            os.path.join(RESOURCES_DIRECTORY, 'Colour_Logo.png'), 'uint8')
        self.assertTupleEqual(image.shape, (128, 256, 4))
        self.assertIs(image.dtype, np.dtype('uint8'))
        self.assertEqual(np.min(image), 0)
        self.assertEqual(np.max(image), 255)

        image = read_image_OpenImageIO(
            os.path.join(RESOURCES_DIRECTORY, 'Colour_Logo.png'), 'uint16')
        self.assertTupleEqual(image.shape, (128, 256, 4))
        self.assertIs(image.dtype, np.dtype('uint16'))
        self.assertEqual(np.min(image), 0)
        self.assertEqual(np.max(image), 65535)

        # TODO: Investigate "OIIO" behaviour here: 1.0 != 15360.0
        # image = read_image_OpenImageIO(
        #     os.path.join(RESOURCES_DIRECTORY, 'Colour_Logo.png'), 'float16')
        # self.assertIs(image.dtype, np.dtype('float16'))
        # self.assertEqual(np.min(image), 0.0)
        # self.assertEqual(np.max(image), 1.0)

        image = read_image_OpenImageIO(
            os.path.join(RESOURCES_DIRECTORY, 'Colour_Logo.png'), 'float32')
        self.assertIs(image.dtype, np.dtype('float32'))
        self.assertEqual(np.min(image), 0.0)
        self.assertEqual(np.max(image), 1.0)


class TestWriteImageOpenImageIO(unittest.TestCase):
    """
    Defines :func:`colour.io.image.write_image_OpenImageIO` definition units
    tests methods.
    """

    def setUp(self):
        """
        Initialises common tests attributes.
        """

        self._temporary_directory = tempfile.mkdtemp()

    def tearDown(self):
        """
        After tests actions.
        """

        shutil.rmtree(self._temporary_directory)

    def test_write_image_OpenImageIO(self):  # pragma: no cover
        """
        Tests :func:`colour.io.image.write_image_OpenImageIO` definition.
        """

        if not is_openimageio_installed():
            return

        source_image_path = os.path.join(RESOURCES_DIRECTORY,
                                         'CMS_Test_Pattern.exr')
        target_image_path = os.path.join(self._temporary_directory,
                                         'CMS_Test_Pattern.exr')
        image = read_image_OpenImageIO(source_image_path)
        write_image_OpenImageIO(image, target_image_path)
        image = read_image_OpenImageIO(target_image_path)
        self.assertTupleEqual(image.shape, (1267, 1274, 3))
        self.assertIs(image.dtype, np.dtype('float32'))

        write_image_OpenImageIO(
            image,
            target_image_path,
            attributes=[ImageAttribute_Specification('John', 'Doe')])
        image, attributes = read_image_OpenImageIO(
            target_image_path, attributes=True)
        for attribute in attributes:
            if attribute.name == 'John':
                self.assertEqual(attribute.value, 'Doe')


class TestReadImageImageio(unittest.TestCase):
    """
    Defines :func:`colour.io.image.read_image_Imageio` definition units tests
    methods.
    """

    def test_read_image_Imageio(self):
        """
        Tests :func:`colour.io.image.read_image_Imageio` definition.
        """

        image = read_image_Imageio(
            os.path.join(RESOURCES_DIRECTORY, 'CMS_Test_Pattern.exr'))
        self.assertTupleEqual(image.shape, (1267, 1274, 3))
        self.assertIs(image.dtype, np.dtype('float32'))

        image = read_image_Imageio(
            os.path.join(RESOURCES_DIRECTORY, 'CMS_Test_Pattern.exr'),
            'float16')
        self.assertTupleEqual(image.shape, (1267, 1274, 3))
        self.assertIs(image.dtype, np.dtype('float16'))

        image = read_image_Imageio(
            os.path.join(RESOURCES_DIRECTORY, 'Single_Channel.exr'))
        self.assertTupleEqual(image.shape, (256, 256))

        image = read_image_Imageio(
            os.path.join(RESOURCES_DIRECTORY, 'Colour_Logo.png'), 'uint8')
        self.assertTupleEqual(image.shape, (128, 256, 4))
        self.assertIs(image.dtype, np.dtype('uint8'))
        self.assertEqual(np.min(image), 0)
        self.assertEqual(np.max(image), 255)

        image = read_image_Imageio(
            os.path.join(RESOURCES_DIRECTORY, 'Colour_Logo.png'), 'uint16')
        self.assertTupleEqual(image.shape, (128, 256, 4))
        self.assertIs(image.dtype, np.dtype('uint16'))
        self.assertEqual(np.min(image), 0)
        self.assertEqual(np.max(image), 65535)

        image = read_image_Imageio(
            os.path.join(RESOURCES_DIRECTORY, 'Colour_Logo.png'), 'float16')
        self.assertIs(image.dtype, np.dtype('float16'))
        self.assertEqual(np.min(image), 0.0)
        self.assertEqual(np.max(image), 1.0)

        image = read_image_Imageio(
            os.path.join(RESOURCES_DIRECTORY, 'Colour_Logo.png'), 'float32')
        self.assertIs(image.dtype, np.dtype('float32'))
        self.assertEqual(np.min(image), 0.0)
        self.assertEqual(np.max(image), 1.0)


class TestWriteImageImageio(unittest.TestCase):
    """
    Defines :func:`colour.io.image.write_image_Imageio` definition units
    tests methods.
    """

    def setUp(self):
        """
        Initialises common tests attributes.
        """

        self._temporary_directory = tempfile.mkdtemp()

    def tearDown(self):
        """
        After tests actions.
        """

        shutil.rmtree(self._temporary_directory)

    def test_write_image_Imageio(self):
        """
        Tests :func:`colour.io.image.write_image_Imageio` definition.
        """

        source_image_path = os.path.join(RESOURCES_DIRECTORY,
                                         'CMS_Test_Pattern.exr')
        target_image_path = os.path.join(self._temporary_directory,
                                         'CMS_Test_Pattern.exr')
        image = read_image_Imageio(source_image_path)
        write_image_Imageio(image, target_image_path)
        image = read_image_Imageio(target_image_path)
        self.assertTupleEqual(image.shape, (1267, 1274, 3))
        self.assertIs(image.dtype, np.dtype('float32'))


class TestReadImage(unittest.TestCase):
    """
    Defines :func:`colour.io.image.read_image` definition units tests
    methods.
    """

    def test_read_image(self):
        """
        Tests :func:`colour.io.image.read_image` definition.
        """

        image = read_image(
            os.path.join(RESOURCES_DIRECTORY, 'CMS_Test_Pattern.exr'))
        self.assertTupleEqual(image.shape, (1267, 1274, 3))
        self.assertIs(image.dtype, np.dtype('float32'))

        image = read_image(
            os.path.join(RESOURCES_DIRECTORY, 'Single_Channel.exr'))
        self.assertTupleEqual(image.shape, (256, 256))


class TestWriteImage(unittest.TestCase):
    """
    Defines :func:`colour.io.image.write_image` definition units tests methods.
    """

    def setUp(self):
        """
        Initialises common tests attributes.
        """

        self._temporary_directory = tempfile.mkdtemp()

    def tearDown(self):
        """
        After tests actions.
        """

        shutil.rmtree(self._temporary_directory)

    def test_write_image(self):
        """
        Tests :func:`colour.io.image.write_image` definition.
        """

        source_image_path = os.path.join(RESOURCES_DIRECTORY,
                                         'CMS_Test_Pattern.exr')
        target_image_path = os.path.join(self._temporary_directory,
                                         'CMS_Test_Pattern.exr')
        image = read_image(source_image_path)
        write_image(image, target_image_path)
        image = read_image(target_image_path)
        self.assertTupleEqual(image.shape, (1267, 1274, 3))
        self.assertIs(image.dtype, np.dtype('float32'))


if __name__ == '__main__':
    unittest.main()
