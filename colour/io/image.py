#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Image Input / Output Utilities
==============================

Defines image related input / output utilities objects.
"""

from __future__ import division, unicode_literals

import numpy as np
from collections import namedtuple

from colour.utilities import CaseInsensitiveMapping, is_openimageio_installed

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['BitDepth_Specification',
           'BIT_DEPTH_MAPPING',
           'read_image',
           'write_image']

BitDepth_Specification = namedtuple(
    'BitDepth_Specification',
    ('name', 'numpy', 'openimageio', 'domain', 'clip'))

if is_openimageio_installed():
    from OpenImageIO import UINT8, UINT16, HALF, FLOAT

    BIT_DEPTH_MAPPING = CaseInsensitiveMapping(
        {'uint8': BitDepth_Specification(
            'uint8', np.uint8, UINT8, 255, True),
         'uint16': BitDepth_Specification(
             'uint16', np.uint16, UINT16, 65535, True),
         'float16': BitDepth_Specification(
             'float16', np.float16, HALF, 1, False),
         'float32': BitDepth_Specification(
             'float32', np.float32, FLOAT, 1, False)})
else:
    BIT_DEPTH_MAPPING = CaseInsensitiveMapping(
        {'uint8': BitDepth_Specification(
            'uint8', np.uint8, None, 255, True),
         'uint16': BitDepth_Specification(
             'uint16', np.uint16, None, 65535, True),
         'float16': BitDepth_Specification(
             'float16', np.float16, None, 1, False),
         'float32': BitDepth_Specification(
             'float32', np.float32, None, 1, False)})


def read_image(path, bit_depth='float32'):
    """
    Reads given image using *OpenImageIO*.

    Parameters
    ----------
    path : unicode
        Image path.
    bit_depth : unicode, optional
        **{'float32', 'uint8', 'uint16', 'float16'}**,
        Image bit_depth.

    Returns
    -------
    ndarray
        Image as a ndarray.

    Notes
    -----
    -   For convenience, single channel images are squeezed to 2d arrays.

    Examples
    --------
    >>> import os
    >>> path = os.path.join('tests', 'resources', 'CMSTestPattern.exr')
    >>> image = read_image(path)  # doctest: +SKIP
    """

    if is_openimageio_installed(raise_exception=True):
        from OpenImageIO import ImageInput

        bit_depth = BIT_DEPTH_MAPPING[bit_depth].openimageio

        image = ImageInput.open(path)
        specification = image.spec()

        shape = (specification.height,
                 specification.width,
                 specification.nchannels)

        return np.squeeze(np.array(image.read_image(bit_depth)).reshape(shape))


def write_image(image, path, bit_depth='float32'):
    """
    Writes given image using *OpenImageIO*.

    Parameters
    ----------
    image : array_like
        Image data.
    path : unicode
        Image path.
    bit_depth : unicode, optional
        **{'float32', 'uint8', 'uint16', 'float16'}**,
        Image bit_depth.

    Returns
    -------
    bool
        Definition success.

    Examples
    --------
    >>> import os
    >>> path = os.path.join('tests', 'resources', 'CMSTestPattern.exr')
    >>> image = read_image(path)  # doctest: +SKIP
    >>> path = os.path.join('tests', 'resources', 'CMSTestPattern.png')
    >>> write_image(image, path, 'uint8')  # doctest: +SKIP
    True
    """

    if is_openimageio_installed(raise_exception=True):
        from OpenImageIO import ImageOutput, ImageOutputOpenMode, ImageSpec

        bit_depth_specification = BIT_DEPTH_MAPPING[bit_depth]
        bit_depth = bit_depth_specification.openimageio

        image = np.asarray(image)
        image *= bit_depth_specification.domain
        if bit_depth_specification.clip:
            image = np.clip(image, 0, bit_depth_specification.domain)
        image = image.astype(bit_depth_specification.numpy)

        if image.ndim == 2:
            height, width = image.shape
            channels = 1
        else:
            height, width, channels = image.shape
        specification = ImageSpec(width, height, channels, bit_depth)

        image_output = ImageOutput.create(path)
        image_output.open(path, specification, ImageOutputOpenMode.Create)
        image_output.write_image(bit_depth, image.tostring())
        image_output.close()

        return True
