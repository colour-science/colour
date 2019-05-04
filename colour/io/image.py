# -*- coding: utf-8 -*-
"""
Image Input / Output Utilities
==============================

Defines image related input / output utilities objects.
"""

from __future__ import division, unicode_literals

import numpy as np
from collections import namedtuple
from six import string_types

from colour.utilities import (CaseInsensitiveMapping, as_float_array,
                              is_openimageio_installed, filter_kwargs,
                              usage_warning)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'BitDepth_Specification', 'ImageAttribute_Specification',
    'read_image_OpenImageIO', 'read_image_Imageio', 'READ_IMAGE_METHODS',
    'read_image', 'write_image_OpenImageIO', 'write_image_Imageio',
    'WRITE_IMAGE_METHODS', 'write_image'
]

BitDepth_Specification = namedtuple(
    'BitDepth_Specification',
    ('name', 'numpy', 'openimageio', 'domain', 'clip'))


class ImageAttribute_Specification(
        namedtuple('ImageAttribute_Specification',
                   ('name', 'value', 'type_'))):
    """
    Defines the an image specification attribute.

    Parameters
    ----------
    name : unicode
        Attribute name.
    value : object
        Attribute value.
    type_ : TypeDesc, optional
        Attribute type as an *OpenImageIO* :class:`TypeDesc` class instance.
    """

    def __new__(cls, name, value, type_=None):
        """
        Returns a new instance of the
        :class:`colour.ImageAttribute_Specification` class.
        """

        return super(ImageAttribute_Specification, cls).__new__(
            cls, name, value, type_)


if is_openimageio_installed():  # pragma: no cover
    from OpenImageIO import UINT8, UINT16, HALF, FLOAT

    BIT_DEPTH_MAPPING = CaseInsensitiveMapping({
        'uint8':
            BitDepth_Specification('uint8', np.uint8, UINT8, 255, True),
        'uint16':
            BitDepth_Specification('uint16', np.uint16, UINT16, 65535, True),
        'float16':
            BitDepth_Specification('float16', np.float16, HALF, 1, False),
        'float32':
            BitDepth_Specification('float32', np.float32, FLOAT, 1, False)
    })
else:
    BIT_DEPTH_MAPPING = CaseInsensitiveMapping({
        'uint8':
            BitDepth_Specification('uint8', np.uint8, None, 255, True),
        'uint16':
            BitDepth_Specification('uint16', np.uint16, None, 65535, True),
        'float16':
            BitDepth_Specification('float16', np.float16, None, 1, False),
        'float32':
            BitDepth_Specification('float32', np.float32, None, 1, False)
    })


def read_image_OpenImageIO(path, bit_depth='float32', attributes=False):
    """
    Reads the image at given path using *OpenImageIO*.

    Parameters
    ----------
    path : unicode
        Image path.
    bit_depth : unicode, optional
        **{'float32', 'uint8', 'uint16', 'float16'}**,
        Image bit_depth.
    attributes : bool, optional
        Whether to return the image attributes.

    Returns
    -------
    ndarray or tuple
        Image as a ndarray or tuple of image as ndarray and list of attributes

    Notes
    -----
    -   For convenience, single channel images are squeezed to 2d arrays.

    Examples
    --------
    >>> import os
    >>> import colour
    >>> path = os.path.join(colour.__path__[0], 'io', 'tests', 'resources',
    ...                     'CMSTestPattern.exr')
    >>> image = read_image(path)  # doctest: +SKIP
    """

    if is_openimageio_installed(raise_exception=True):  # pragma: no cover
        from OpenImageIO import ImageInput

        path = str(path)

        bit_depth = BIT_DEPTH_MAPPING[bit_depth]

        image = ImageInput.open(path)
        specification = image.spec()

        shape = (specification.height, specification.width,
                 specification.nchannels)

        image_data = image.read_image(bit_depth.openimageio)
        image.close()
        image = np.squeeze(
            np.array(image_data, dtype=bit_depth.numpy).reshape(shape))

        if attributes:
            extra_attributes = []
            for i in range(len(specification.extra_attribs)):
                attribute = specification.extra_attribs[i]
                extra_attributes.append(
                    ImageAttribute_Specification(
                        attribute.name, attribute.value, attribute.type))

            return image, extra_attributes
        else:
            return image


def read_image_Imageio(path, **kwargs):
    """
    Reads the image at given path using *Imageio*.

    Parameters
    ----------
    path : unicode
        Image path.

    Other Parameters
    ----------------
    \\**kwargs : dict, optional
        Keywords arguments.

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
    >>> import colour
    >>> path = os.path.join(colour.__path__[0], 'io', 'tests', 'resources',
    ...                     'CMSTestPattern.exr')
    >>> image = read_image_Imageio(path)
    >>> image.shape  # doctest: +SKIP
    (1267, 1274, 3)
    >>> image.dtype
    dtype('float32')
    """

    from imageio import imread

    return imread(path, **kwargs)


READ_IMAGE_METHODS = CaseInsensitiveMapping({
    'Imageio': read_image_Imageio,
    'OpenImageIO': read_image_OpenImageIO,
})
READ_IMAGE_METHODS.__doc__ = """
Supported read image methods.

READ_IMAGE_METHODS : CaseInsensitiveMapping
    **{'Imageio', 'OpenImageIO'}**
"""


def read_image(path, method='OpenImageIO', **kwargs):
    """
    Reads the image at given path using given method.

    Parameters
    ----------
    path : unicode
        Image path.
    method : unicode, optional
        **{'OpenImageIO', 'Imageio'}**,
        Read method, i.e. the image library used for reading images.

    Other Parameters
    ----------------
    bit_depth : unicode, optional
        **{'float32', 'uint8', 'uint16', 'float16'}**,
        {:func:`colour.io.read_image_OpenImageIO`},
        Image bit_depth.
    attributes : bool, optional
        {:func:`colour.io.read_image_OpenImageIO`},
        Whether to return the image attributes.

    Returns
    -------
    ndarray
        Image as a ndarray.

    Notes
    -----
    -   If the given method is *OpenImageIO* but the library is not available
        writing will be performed by *Imageio*.
    -   If the given method is *Imageio*, ``kwargs`` is passed directly to the
        wrapped definition.
    -   For convenience, single channel images are squeezed to 2d arrays.

    Examples
    --------
    >>> import os
    >>> import colour
    >>> path = os.path.join(colour.__path__[0], 'io', 'tests', 'resources',
    ...                     'CMSTestPattern.exr')
    >>> image = read_image(path)
    >>> image.shape  # doctest: +SKIP
    (1267, 1274, 3)
    >>> image.dtype
    dtype('float32')
    """

    if method.lower() == 'openimageio':
        if not is_openimageio_installed():
            usage_warning(
                '"OpenImageIO" related API features are not available, '
                'switching to "Imageio"!')
            method = 'Imageio'

    function = READ_IMAGE_METHODS[method]

    if method.lower() == 'openimageio':  # pragma: no cover
        kwargs = filter_kwargs(function, **kwargs)

    return function(path, **kwargs)


def write_image_OpenImageIO(image, path, bit_depth='float32', attributes=None):
    """
    Writes given image at given path using *OpenImageIO*.

    Parameters
    ----------
    image : array_like
        Image data.
    path : unicode
        Image path.
    bit_depth : unicode, optional
        **{'float32', 'uint8', 'uint16', 'float16'}**,
        Image bit_depth.
    attributes : array_like, optional
        An array of :class:`colour.io.ImageAttribute_Specification` class
        instances used to set attributes of the image.

    Returns
    -------
    bool
        Definition success.

    Examples
    --------
    Basic image writing:

    >>> import os
    >>> import colour
    >>> path = os.path.join(colour.__path__[0], 'io', 'tests', 'resources',
    ...                     'CMSTestPattern.exr')
    >>> image = read_image(path)  # doctest: +SKIP
    >>> path = os.path.join(colour.__path__[0], 'io', 'tests', 'resources',
    ...                     'CMSTestPattern.tif')
    >>> write_image(image, path)  # doctest: +SKIP
    True

    Advanced image writing while setting attributes:

    >>> compression = ImageAttribute_Specification('Compression', 'none')
    >>> write_image(image, path, 'uint8', [compression])  # doctest: +SKIP
    True
    """

    if is_openimageio_installed(raise_exception=True):  # pragma: no cover
        from OpenImageIO import ImageOutput, ImageOutputOpenMode, ImageSpec

        path = str(path)

        if attributes is None:
            attributes = []

        bit_depth_specification = BIT_DEPTH_MAPPING[bit_depth]
        bit_depth = bit_depth_specification.openimageio

        image = as_float_array(image)
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
        for attribute in attributes:
            name = str(attribute.name)
            value = (str(attribute.value) if isinstance(
                attribute.value, string_types) else attribute.value)
            type_ = attribute.type_
            if attribute.type_ is None:
                specification.attribute(name, value)
            else:
                specification.attribute(name, type_, value)

        image_output = ImageOutput.create(path)
        image_output.open(path, specification, ImageOutputOpenMode.Create)
        image_output.write_image(bit_depth, image.tostring())
        image_output.close()

        return True


def write_image_Imageio(image, path, **kwargs):
    """
    Writes given image at given path using *Imageio*.

    Parameters
    ----------
    image : array_like
        Image data.
    path : unicode
        Image path.

    Other Parameters
    ----------------
    \\**kwargs : dict, optional
        Keywords arguments.

    Returns
    -------
    bool
        Definition success.

    Examples
    --------
    >>> import os
    >>> import colour
    >>> path = os.path.join(colour.__path__[0], 'io', 'tests', 'resources',
    ...                     'CMSTestPattern.exr')
    >>> image = read_image(path)  # doctest: +SKIP
    >>> path = os.path.join(colour.__path__[0], 'io', 'tests', 'resources',
    ...                     'CMSTestPattern.tif')
    >>> write_image(image, path)  # doctest: +SKIP
    True
    """

    from imageio import imwrite

    return imwrite(path, image, **kwargs)


WRITE_IMAGE_METHODS = CaseInsensitiveMapping({
    'Imageio': write_image_Imageio,
    'OpenImageIO': write_image_OpenImageIO,
})
WRITE_IMAGE_METHODS.__doc__ = """
Supported write image methods.

WRITE_IMAGE_METHODS : CaseInsensitiveMapping
    **{'Imageio', 'OpenImageIO'}**
"""


def write_image(image, path, method='OpenImageIO', **kwargs):
    """
    Writes given image at given path using given method.

    Parameters
    ----------
    image : array_like
        Image data.
    path : unicode
        Image path.
    method : unicode, optional
        **{'OpenImageIO', 'Imageio'}**,
        Write method, i.e. the image library used for writing images.

    Other Parameters
    ----------------
    bit_depth : unicode, optional
        {:func:`colour.io.write_image_OpenImageIO`},
        **{'float32', 'uint8', 'uint16', 'float16'}**,
        Image bit_depth.
    attributes : array_like, optional
        {:func:`colour.io.write_image_OpenImageIO`},
        An array of :class:`colour.io.ImageAttribute_Specification` class
        instances used to set attributes of the image.

    Returns
    -------
    bool
        Definition success.

    Notes
    -----
    -   If the given method is *OpenImageIO* but the library is not available
        writing will be performed by *Imageio*.
    -   If the given method is *Imageio*, ``kwargs`` is passed directly to the
        wrapped definition.

    Examples
    --------
    Basic image writing:

    >>> import os
    >>> import colour
    >>> path = os.path.join(colour.__path__[0], 'io', 'tests', 'resources',
    ...                     'CMSTestPattern.exr')
    >>> image = read_image(path)  # doctest: +SKIP
    >>> path = os.path.join(colour.__path__[0], 'io', 'tests', 'resources',
    ...                     'CMSTestPattern.tif')
    >>> write_image(image, path)  # doctest: +SKIP
    True

    Advanced image writing while setting attributes using *OpenImageIO*:

    >>> compression = ImageAttribute_Specification('Compression', 'none')
    >>> write_image(image, path, bit_depth='uint8', attributes=[compression])
    ... # doctest: +SKIP
    True
    """

    if method.lower() == 'openimageio':
        if not is_openimageio_installed():
            usage_warning(
                '"OpenImageIO" related API features are not available, '
                'switching to "Imageio"!')
            method = 'Imageio'

    function = WRITE_IMAGE_METHODS[method]

    if method.lower() == 'openimageio':  # pragma: no cover
        kwargs = filter_kwargs(function, **kwargs)

    return function(image, path, **kwargs)
