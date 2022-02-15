"""
Image Input / Output Utilities
==============================

Defines the image related input / output utilities objects.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

from colour.hints import (
    Any,
    ArrayLike,
    Boolean,
    DTypeNumber,
    List,
    Literal,
    NDArray,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    cast,
)
from colour.utilities import (
    CaseInsensitiveMapping,
    as_float_array,
    as_int_array,
    attest,
    is_openimageio_installed,
    filter_kwargs,
    optional,
    required,
    usage_warning,
    validate_method,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "BitDepth_Specification",
    "ImageAttribute_Specification",
    "convert_bit_depth",
    "read_image_OpenImageIO",
    "read_image_Imageio",
    "READ_IMAGE_METHODS",
    "read_image",
    "write_image_OpenImageIO",
    "write_image_Imageio",
    "WRITE_IMAGE_METHODS",
    "write_image",
]


@dataclass(frozen=True)
class BitDepth_Specification:
    """
    Define a bit depth specification.

    Parameters
    ----------
    name
        Attribute name.
    numpy
        Object representing the *Numpy* bit depth.
    openimageio
        Object representing the *OpenImageIO* bit depth.
    """

    name: str
    numpy: Type[DTypeNumber]
    openimageio: Any


@dataclass
class ImageAttribute_Specification:
    """
    Define an image specification attribute.

    Parameters
    ----------
    name
        Attribute name.
    value
        Attribute value.
    type_
        Attribute type as an *OpenImageIO* :class:`TypeDesc` class instance.
    """

    name: str
    value: Any
    type_: Optional[OpenImageIO.TypeDesc] = field(  # type: ignore[name-defined] # noqa
        default_factory=lambda: None
    )


if is_openimageio_installed():  # pragma: no cover
    from OpenImageIO import UINT8, UINT16, HALF, FLOAT, DOUBLE

    MAPPING_BIT_DEPTH: CaseInsensitiveMapping = CaseInsensitiveMapping(
        {
            "uint8": BitDepth_Specification("uint8", np.uint8, UINT8),
            "uint16": BitDepth_Specification("uint16", np.uint16, UINT16),
            "float16": BitDepth_Specification("float16", np.float16, HALF),
            "float32": BitDepth_Specification("float32", np.float32, FLOAT),
            "float64": BitDepth_Specification("float64", np.float64, DOUBLE),
        }
    )
    if hasattr(np, "float128"):  # pragma: no cover
        MAPPING_BIT_DEPTH["float128"] = BitDepth_Specification(
            "float128", np.float128, DOUBLE  # type: ignore[arg-type]
        )
else:  # pragma: no cover
    MAPPING_BIT_DEPTH: CaseInsensitiveMapping = (  # type: ignore[no-redef]
        CaseInsensitiveMapping(
            {
                "uint8": BitDepth_Specification("uint8", np.uint8, None),
                "uint16": BitDepth_Specification("uint16", np.uint16, None),
                "float16": BitDepth_Specification("float16", np.float16, None),
                "float32": BitDepth_Specification("float32", np.float32, None),
                "float64": BitDepth_Specification("float64", np.float64, None),
            }
        )
    )
    if hasattr(np, "float128"):  # pragma: no cover
        MAPPING_BIT_DEPTH["float128"] = BitDepth_Specification(
            "float128", np.float128, None  # type: ignore[arg-type]
        )


def convert_bit_depth(
    a: ArrayLike,
    bit_depth: Literal[
        "uint8", "uint16", "float16", "float32", "float64", "float128"
    ] = "float32",
) -> NDArray:
    """
    Convert given array to given bit depth, the current bit depth of the array
    is used to determine the appropriate conversion path.

    Parameters
    ----------
    a
        Array to convert to given bit depth.
    bit_depth
        Bit depth.

    Returns
    -------
    :class`numpy.ndarray`
        Converted array.

    Examples
    --------
    >>> a = np.array([0.0, 0.5, 1.0])
    >>> convert_bit_depth(a, 'uint8')
    array([  0, 128, 255], dtype=uint8)
    >>> convert_bit_depth(a, 'uint16')
    array([    0, 32768, 65535], dtype=uint16)
    >>> convert_bit_depth(a, 'float16')
    array([ 0. ,  0.5,  1. ], dtype=float16)
    >>> a = np.array([0, 128, 255], dtype=np.uint8)
    >>> convert_bit_depth(a, 'uint16')
    array([    0, 32896, 65535], dtype=uint16)
    >>> convert_bit_depth(a, 'float32')  # doctest: +ELLIPSIS
    array([ 0.       ,  0.501960...,  1.       ], dtype=float32)
    """

    a = np.asarray(a)

    bit_depths = ", ".join(sorted(MAPPING_BIT_DEPTH.keys()))

    attest(
        bit_depth in bit_depths,
        f'Incorrect bit depth was specified, it must be one of: "{bit_depths}"!',
    )

    attest(
        str(a.dtype) in bit_depths,
        f'Image bit depth must be one of: "{bit_depths}"!',
    )

    source_dtype = str(a.dtype)
    target_dtype = MAPPING_BIT_DEPTH[bit_depth].numpy

    if source_dtype == "uint8":
        if bit_depth == "uint16":
            a = (a * 257).astype(target_dtype)
        elif bit_depth in ("float16", "float32", "float64", "float128"):
            a = (a / 255).astype(target_dtype)
    elif source_dtype == "uint16":
        if bit_depth == "uint8":
            a = (a / 257).astype(target_dtype)
        elif bit_depth in ("float16", "float32", "float64", "float128"):
            a = (a / 65535).astype(target_dtype)
    elif source_dtype in ("float16", "float32", "float64", "float128"):
        if bit_depth == "uint8":
            a = np.around(a * 255).astype(target_dtype)
        elif bit_depth == "uint16":
            a = np.around(a * 65535).astype(target_dtype)
        elif bit_depth in ("float16", "float32", "float64", "float128"):
            a = a.astype(target_dtype)

    return a  # type: ignore[return-value]


@required("OpenImageIO")
def read_image_OpenImageIO(
    path: str,
    bit_depth: Literal[
        "uint8", "uint16", "float16", "float32", "float64", "float128"
    ] = "float32",
    attributes: Boolean = False,
) -> Union[NDArray, Tuple[NDArray, List]]:  # noqa: D405,D410,D407,D411
    """
    Read the image at given path using *OpenImageIO*.

    Parameters
    ----------
    path
        Image path.
    bit_depth
        Returned image bit depth, the bit depth conversion behaviour is driven
        directly by *OpenImageIO*, this definition only converts to the
        relevant data type after reading.
    attributes
        Whether to return the image attributes.

    Returns
    -------
    :class`numpy.ndarray` or :class:`tuple`
        Image data or tuple of image data and list of
        :class:`colour.io.ImageAttribute_Specification` class instances.

    Notes
    -----
    -   For convenience, single channel images are squeezed to 2D arrays.

    Examples
    --------
    >>> import os
    >>> import colour
    >>> path = os.path.join(colour.__path__[0], 'io', 'tests', 'resources',
    ...                     'CMS_Test_Pattern.exr')
    >>> image = read_image_OpenImageIO(path)  # doctest: +SKIP
    """

    from OpenImageIO import ImageInput

    path = str(path)

    bit_depth_specification = MAPPING_BIT_DEPTH[bit_depth]

    image = ImageInput.open(path)
    specification = image.spec()

    shape = (
        specification.height,
        specification.width,
        specification.nchannels,
    )

    image_data = image.read_image(bit_depth_specification.openimageio)
    image.close()
    image = np.squeeze(
        np.array(image_data, dtype=bit_depth_specification.numpy).reshape(
            shape
        )
    )

    if attributes:
        extra_attributes = []
        for i in range(len(specification.extra_attribs)):
            attribute = specification.extra_attribs[i]
            extra_attributes.append(
                ImageAttribute_Specification(
                    attribute.name, attribute.value, attribute.type
                )
            )

        return image, extra_attributes
    else:
        return image


def read_image_Imageio(
    path: str,
    bit_depth: Literal[
        "uint8", "uint16", "float16", "float32", "float64", "float128"
    ] = "float32",
    **kwargs: Any,
) -> NDArray:
    """
    Read the image at given path using *Imageio*.

    Parameters
    ----------
    path
        Image path.
    bit_depth
        Returned image bit depth, the image data is converted with
        :func:`colour.io.convert_bit_depth` definition after reading the
        image.

    Other Parameters
    ----------------
    kwargs
        Keywords arguments.

    Returns
    -------
    :class`numpy.ndarray`
        Image data.

    Notes
    -----
    -   For convenience, single channel images are squeezed to 2D arrays.

    Examples
    --------
    >>> import os
    >>> import colour
    >>> path = os.path.join(colour.__path__[0], 'io', 'tests', 'resources',
    ...                     'CMS_Test_Pattern.exr')
    >>> image = read_image_Imageio(path)
    >>> image.shape  # doctest: +SKIP
    (1267, 1274, 3)
    >>> image.dtype
    dtype('float32')
    """

    from imageio import imread

    image = np.squeeze(imread(path, **kwargs))

    return convert_bit_depth(image, bit_depth)


READ_IMAGE_METHODS: CaseInsensitiveMapping = CaseInsensitiveMapping(
    {
        "Imageio": read_image_Imageio,
        "OpenImageIO": read_image_OpenImageIO,
    }
)
READ_IMAGE_METHODS.__doc__ = """
Supported image read methods.
"""


def read_image(
    path: str,
    bit_depth: Literal[
        "uint8", "uint16", "float16", "float32", "float64", "float128"
    ] = "float32",
    method: Union[Literal["Imageio", "OpenImageIO"], str] = "OpenImageIO",
    **kwargs: Any,
) -> NDArray:  # noqa: D405,D407,D410,D411,D414
    """
    Read the image at given path using given method.

    Parameters
    ----------
    path
        Image path.
    bit_depth
        Returned image bit depth, for the *Imageio* method, the image data is
        converted with :func:`colour.io.convert_bit_depth` definition after
        reading the image, for the *OpenImageIO* method, the bit depth
        conversion behaviour is driven directly by the library, this definition
        only converts to the relevant data type after reading.
    method
        Read method, i.e. the image library used for reading images.

    Other Parameters
    ----------------
    attributes
        {:func:`colour.io.read_image_OpenImageIO`},
        Whether to return the image attributes.

    Returns
    -------
    :class`numpy.ndarray`
        Image data.

    Notes
    -----
    -   If the given method is *OpenImageIO* but the library is not available
        writing will be performed by *Imageio*.
    -   If the given method is *Imageio*, ``kwargs`` is passed directly to the
        wrapped definition.
    -   For convenience, single channel images are squeezed to 2D arrays.

    Examples
    --------
    >>> import os
    >>> import colour
    >>> path = os.path.join(colour.__path__[0], 'io', 'tests', 'resources',
    ...                     'CMS_Test_Pattern.exr')
    >>> image = read_image(path)
    >>> image.shape  # doctest: +SKIP
    (1267, 1274, 3)
    >>> image.dtype
    dtype('float32')
    """

    method = validate_method(method, READ_IMAGE_METHODS)

    if method == "openimageio":  # pragma: no cover
        if not is_openimageio_installed():
            usage_warning(
                '"OpenImageIO" related API features are not available, '
                'switching to "Imageio"!'
            )
            method = "Imageio"

    function = READ_IMAGE_METHODS[method]

    if method == "openimageio":  # pragma: no cover
        kwargs = filter_kwargs(function, **kwargs)

    return function(path, bit_depth, **kwargs)


@required("OpenImageIO")
def write_image_OpenImageIO(
    image: ArrayLike,
    path: str,
    bit_depth: Literal[
        "uint8", "uint16", "float16", "float32", "float64", "float128"
    ] = "float32",
    attributes: Optional[Sequence] = None,
) -> Boolean:  # noqa: D405,D407,D410,D411
    """
    Write given image at given path using *OpenImageIO*.

    Parameters
    ----------
    image
        Image data.
    path
        Image path.
    bit_depth
        Bit depth to write the image at, the bit depth conversion behaviour is
        ruled directly by *OpenImageIO*.
    attributes
        An array of :class:`colour.io.ImageAttribute_Specification` class
        instances used to set attributes of the image.

    Returns
    -------
    :class:`bool`
        Definition success.

    Examples
    --------
    Basic image writing:

    >>> import os
    >>> import colour
    >>> path = os.path.join(colour.__path__[0], 'io', 'tests', 'resources',
    ...                     'CMS_Test_Pattern.exr')
    >>> image = read_image(path)  # doctest: +SKIP
    >>> path = os.path.join(colour.__path__[0], 'io', 'tests', 'resources',
    ...                     'CMSTestPattern.tif')
    >>> write_image_OpenImageIO(image, path)  # doctest: +SKIP
    True

    Advanced image writing while setting attributes:

    >>> compression = ImageAttribute_Specification('Compression', 'none')
    >>> write_image_OpenImageIO(image, path, 'uint8', [compression])
    ... # doctest: +SKIP
    True

    Writing an "ACES" compliant "EXR" file:

    >>> if is_openimageio_installed():  # doctest: +SKIP
    ...     from OpenImageIO import TypeDesc
    ...     chromaticities = (
    ...         0.7347, 0.2653, 0.0, 1.0, 0.0001, -0.077, 0.32168, 0.33767)
    ...     attributes = [
    ...         ImageAttribute_Specification('acesImageContainerFlag', True),
    ...         ImageAttribute_Specification(
    ...             'chromaticities', chromaticities, TypeDesc('float[8]')),
    ...         ImageAttribute_Specification('compression', 'none')]
    ...     write_image_OpenImageIO(image, path, attributes=attributes)
    """

    from OpenImageIO import ImageOutput, ImageSpec

    image = as_float_array(image)
    path = str(path)

    attributes = cast(List, optional(attributes, []))

    bit_depth_specification = MAPPING_BIT_DEPTH[bit_depth]

    if bit_depth_specification.numpy in [np.uint8, np.uint16]:
        mininum, maximum = np.iinfo(np.uint8).min, np.iinfo(np.uint8).max
        image = np.clip(image * maximum, mininum, maximum)

        image = as_int_array(image, bit_depth_specification.numpy)

    image = image.astype(bit_depth_specification.numpy)

    if image.ndim == 2:
        height, width = image.shape
        channels = 1
    else:
        height, width, channels = image.shape

    specification = ImageSpec(
        width, height, channels, bit_depth_specification.openimageio
    )
    for attribute in attributes:
        name = str(attribute.name)
        value = (
            str(attribute.value)
            if isinstance(attribute.value, str)
            else attribute.value
        )
        type_ = attribute.type_
        if attribute.type_ is None:
            specification.attribute(name, value)
        else:
            specification.attribute(name, type_, value)

    image_output = ImageOutput.create(path)

    image_output.open(path, specification)
    image_output.write_image(image)

    image_output.close()

    return True


def write_image_Imageio(
    image: ArrayLike,
    path: str,
    bit_depth: Literal[
        "uint8", "uint16", "float16", "float32", "float64", "float128"
    ] = "float32",
    **kwargs: Any,
) -> Boolean:
    """
    Write given image at given path using *Imageio*.

    Parameters
    ----------
    image
        Image data.
    path
        Image path.
    bit_depth
        Bit depth to write the image at, the image data is converted with
        :func:`colour.io.convert_bit_depth` definition prior to writing the
        image.

    Other Parameters
    ----------------
    kwargs
        Keywords arguments.

    Returns
    -------
    :class:`bool`
        Definition success.

    Examples
    --------
    >>> import os
    >>> import colour
    >>> path = os.path.join(colour.__path__[0], 'io', 'tests', 'resources',
    ...                     'CMS_Test_Pattern.exr')
    >>> image = read_image(path)  # doctest: +SKIP
    >>> path = os.path.join(colour.__path__[0], 'io', 'tests', 'resources',
    ...                     'CMSTestPattern.tif')
    >>> write_image_Imageio(image, path)  # doctest: +SKIP
    True
    """

    from imageio import imwrite

    image = convert_bit_depth(image, bit_depth)

    return imwrite(path, image, **kwargs)


WRITE_IMAGE_METHODS: CaseInsensitiveMapping = CaseInsensitiveMapping(
    {
        "Imageio": write_image_Imageio,
        "OpenImageIO": write_image_OpenImageIO,
    }
)
WRITE_IMAGE_METHODS.__doc__ = """
Supported image write methods.
"""


def write_image(
    image: ArrayLike,
    path: str,
    bit_depth: Literal[
        "uint8", "uint16", "float16", "float32", "float64", "float128"
    ] = "float32",
    method: Union[Literal["Imageio", "OpenImageIO"], str] = "OpenImageIO",
    **kwargs: Any,
) -> Boolean:  # noqa: D405,D407,D410,D411,D414
    """
    Write given image at given path using given method.

    Parameters
    ----------
    image
        Image data.
    path
        Image path.
    bit_depth
        Bit depth to write the image at, for the *Imageio* method, the image
        data is converted with :func:`colour.io.convert_bit_depth` definition
        prior to writing the image.
    method
        Write method, i.e. the image library used for writing images.

    Other Parameters
    ----------------
    attributes
        {:func:`colour.io.write_image_OpenImageIO`},
        An array of :class:`colour.io.ImageAttribute_Specification` class
        instances used to set attributes of the image.

    Returns
    -------
    :class:`bool`
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
    ...                     'CMS_Test_Pattern.exr')
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

    method = validate_method(method, WRITE_IMAGE_METHODS)

    if method == "openimageio":  # pragma: no cover
        if not is_openimageio_installed():
            usage_warning(
                '"OpenImageIO" related API features are not available, '
                'switching to "Imageio"!'
            )
            method = "Imageio"

    function = WRITE_IMAGE_METHODS[method]

    if method == "openimageio":  # pragma: no cover
        kwargs = filter_kwargs(function, **kwargs)

    return function(image, path, bit_depth, **kwargs)
