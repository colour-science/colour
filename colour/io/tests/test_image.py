# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.io.image` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import os
import shutil
import unittest
import tempfile

from colour.io import read_image_OpenImageIO, write_image_OpenImageIO
from colour.io import read_image_Imageio, write_image_Imageio
from colour.io import read_image, write_image
from colour.io import ImageAttribute_Specification
from colour.utilities import is_openimageio_installed

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'RESOURCES_DIRECTORY', 'TestReadImageOpenImageIO',
    'TestWriteImageOpenImageIO', 'TestReadImageImageio',
    'TestWriteImageImageio', 'TestReadImage', 'TestWriteImage'
]

RESOURCES_DIRECTORY = os.path.join(os.path.dirname(__file__), 'resources')


class TestReadImageOpenImageIO(unittest.TestCase):
    """
    Defines :func:`colour.io.image.read_image_OpenImageIO` definition units
    tests methods.
    """

    def test_read_image_OpenImageIO(self):
        """
        Tests :func:`colour.io.image.read_image_OpenImageIO` definition.
        """

        if not is_openimageio_installed():  # pragma: no cover
            return

        image = read_image_OpenImageIO(
            os.path.join(RESOURCES_DIRECTORY, 'CMSTestPattern.exr'))
        self.assertTupleEqual(image.shape, (1267, 1274, 3))
        self.assertIs(image.dtype, np.dtype('float32'))

        image = read_image_OpenImageIO(
            os.path.join(RESOURCES_DIRECTORY, 'CMSTestPattern.exr'),
            bit_depth='float16')
        self.assertIs(image.dtype, np.dtype('float16'))

        image, attributes = read_image_OpenImageIO(
            os.path.join(RESOURCES_DIRECTORY, 'CMSTestPattern.exr'),
            attributes=True)
        self.assertTupleEqual(image.shape, (1267, 1274, 3))
        self.assertEqual(attributes[0].name, 'oiio:ColorSpace')
        self.assertEqual(attributes[0].value, 'Linear')

        image = read_image_OpenImageIO(
            os.path.join(RESOURCES_DIRECTORY, 'SingleChannel.exr'))
        self.assertTupleEqual(image.shape, (256, 256))


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

    def test_write_image_OpenImageIO(self):
        """
        Tests :func:`colour.io.image.write_image_OpenImageIO` definition.
        """

        if not is_openimageio_installed():  # pragma: no cover
            return

        source_image_path = os.path.join(RESOURCES_DIRECTORY,
                                         'CMSTestPattern.exr')
        target_image_path = os.path.join(self._temporary_directory,
                                         'CMSTestPattern.exr')
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
            os.path.join(RESOURCES_DIRECTORY, 'CMSTestPattern.exr'))
        self.assertTupleEqual(image.shape, (1267, 1274, 3))
        self.assertIs(image.dtype, np.dtype('float32'))

        image = read_image_Imageio(
            os.path.join(RESOURCES_DIRECTORY, 'SingleChannel.exr'))
        self.assertTupleEqual(image.shape, (256, 256))


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
                                         'CMSTestPattern.exr')
        target_image_path = os.path.join(self._temporary_directory,
                                         'CMSTestPattern.exr')
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
            os.path.join(RESOURCES_DIRECTORY, 'CMSTestPattern.exr'))
        self.assertTupleEqual(image.shape, (1267, 1274, 3))
        self.assertIs(image.dtype, np.dtype('float32'))

        image = read_image(
            os.path.join(RESOURCES_DIRECTORY, 'SingleChannel.exr'))
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
                                         'CMSTestPattern.exr')
        target_image_path = os.path.join(self._temporary_directory,
                                         'CMSTestPattern.exr')
        image = read_image(source_image_path)
        write_image(image, target_image_path)
        image = read_image(target_image_path)
        self.assertTupleEqual(image.shape, (1267, 1274, 3))
        self.assertIs(image.dtype, np.dtype('float32'))


if __name__ == '__main__':
    unittest.main()
