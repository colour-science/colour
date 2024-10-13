"""
Image Input / Output Utilities
==============================

Define the image related input / output utilities objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from colour.hints import (
    TYPE_CHECKING,
    Any,
    ArrayLike,
    DTypeReal,
    Literal,
    NDArrayFloat,
    NDArrayReal,
    Sequence,
    Tuple,
    Type,
    cast,
)
from colour.utilities import (
    CanonicalMapping,
    as_float_array,
    as_int_array,
    attest,
    filter_kwargs,
    is_openimageio_installed,
    optional,
    required,
    tstack,
    usage_warning,
    validate_method,
)
from colour.utilities.deprecation import handle_arguments_deprecation

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "Image_Specification_BitDepth",
    "Image_Specification_Attribute",
    "MAPPING_BIT_DEPTH",
    "add_attributes_to_image_specification_OpenImageIO",
    "image_specification_OpenImageIO",
    "convert_bit_depth",
    "read_image_OpenImageIO",
    "read_image_Imageio",
    "READ_IMAGE_METHODS",
    "read_image",
    "write_image_OpenImageIO",
    "write_image_Imageio",
    "WRITE_IMAGE_METHODS",
    "write_image",
    "as_3_channels_image",
]


@dataclass(frozen=True)
class Image_Specification_BitDepth:
    """
    Define a bit-depth specification.

    Parameters
    ----------
    name
        Attribute name.
    numpy
        Object representing the *Numpy* bit-depth.
    openimageio
        Object representing the *OpenImageIO* bit-depth.
    """

    name: str
    numpy: Type[DTypeReal]
    openimageio: Any


@dataclass
class Image_Specification_Attribute:
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
    type_: OpenImageIO.TypeDesc | None = field(  # noqa: F821, RUF100 # pyright: ignore # noqa: F821
        default_factory=lambda: None
    )


if is_openimageio_installed():  # pragma: no cover
    from OpenImageIO import ImageSpec  # pyright: ignore
    from OpenImageIO import DOUBLE, FLOAT, HALF, UINT8, UINT16  # pyright: ignore

    MAPPING_BIT_DEPTH: CanonicalMapping = CanonicalMapping(
        {
            "uint8": Image_Specification_BitDepth("uint8", np.uint8, UINT8),
            "uint16": Image_Specification_BitDepth("uint16", np.uint16, UINT16),
            "float16": Image_Specification_BitDepth("float16", np.float16, HALF),
            "float32": Image_Specification_BitDepth("float32", np.float32, FLOAT),
            "float64": Image_Specification_BitDepth("float64", np.float64, DOUBLE),
        }
    )
    if not TYPE_CHECKING and hasattr(np, "float128"):  # pragma: no cover
        MAPPING_BIT_DEPTH["float128"] = Image_Specification_BitDepth(
            "float128", np.float128, DOUBLE
        )
else:  # pragma: no cover
    #
    class ImageSpec:
        attribute: Any

    MAPPING_BIT_DEPTH: CanonicalMapping = CanonicalMapping(
        {
            "uint8": Image_Specification_BitDepth("uint8", np.uint8, None),
            "uint16": Image_Specification_BitDepth("uint16", np.uint16, None),
            "float16": Image_Specification_BitDepth("float16", np.float16, None),
            "float32": Image_Specification_BitDepth("float32", np.float32, None),
            "float64": Image_Specification_BitDepth("float64", np.float64, None),
        }
    )
    if not TYPE_CHECKING and hasattr(np, "float128"):  # pragma: no cover
        MAPPING_BIT_DEPTH["float128"] = Image_Specification_BitDepth(
            "float128", np.float128, None
        )


def add_attributes_to_image_specification_OpenImageIO(
    image_specification: ImageSpec, attributes: Sequence
):
    """
    Add given attributes to given *OpenImageIO* image specification.

    Parameters
    ----------
    image_specification
        *OpenImageIO* image specification.
    attributes
        An array of :class:`colour.io.Image_Specification_Attribute` class
        instances used to set attributes of the image.

    Returns
    -------
    :class:`ImageSpec`
        *OpenImageIO*. image specification.

    Examples
    --------
    >>> image_specification = image_specification_OpenImageIO(
    ...     1920, 1080, 3, "float16"
    ... )  # doctest: +SKIP
    >>> compression = Image_Specification_Attribute("Compression", "none")
    >>> image_specification = add_attributes_to_image_specification_OpenImageIO(
    ...     image_specification, [compression]
    ... )  # doctest: +SKIP
    >>> image_specification.extra_attribs[0].value  # doctest: +SKIP
    'none'
    """  # noqa: D405, D407, D410, D411

    for attribute in attributes:
        name = str(attribute.name)
        value = (
            str(attribute.value)
            if isinstance(attribute.value, str)
            else attribute.value
        )
        type_ = attribute.type_
        if attribute.type_ is None:
            image_specification.attribute(name, value)
        else:
            image_specification.attribute(name, type_, value)

    return image_specification


@required("OpenImageIO")
def image_specification_OpenImageIO(
    width: int,
    height: int,
    channels: int,
    bit_depth: Literal[
        "uint8", "uint16", "float16", "float32", "float64", "float128"
    ] = "float32",
    attributes: Sequence | None = None,
) -> ImageSpec:
    """
    Create an *OpenImageIO* image specification.

    Parameters
    ----------
    width
        Image width.
    height
        Image height.
    channels
        Image channel count.
    bit_depth
        Bit-depth to create the image with, the bit-depth conversion behaviour is
        ruled directly by *OpenImageIO*.
    attributes
        An array of :class:`colour.io.Image_Specification_Attribute` class
        instances used to set attributes of the image.

    Returns
    -------
    :class:`ImageSpec`
        *OpenImageIO*. image specification.

    Examples
    --------
    >>> compression = Image_Specification_Attribute("Compression", "none")
    >>> image_specification_OpenImageIO(
    ...     1920, 1080, 3, "float16", [compression]
    ... )  # doctest: +SKIP
    <OpenImageIO.ImageSpec object at 0x...>
    """  # noqa: D405, D407, D410, D411

    from OpenImageIO import ImageSpec  # pyright: ignore

    attributes = cast(list, optional(attributes, []))

    bit_depth_specification = MAPPING_BIT_DEPTH[bit_depth]

    image_specification = ImageSpec(
        width, height, channels, bit_depth_specification.openimageio
    )

    add_attributes_to_image_specification_OpenImageIO(
        image_specification, attributes or []
    )

    return image_specification


def convert_bit_depth(
    a: ArrayLike,
    bit_depth: Literal[
        "uint8", "uint16", "float16", "float32", "float64", "float128"
    ] = "float32",
) -> NDArrayReal:
    """
    Convert given array to given bit-depth, the current bit-depth of the array
    is used to determine the appropriate conversion path.

    Parameters
    ----------
    a
        Array to convert to given bit-depth.
    bit_depth
        Bit-depth.

    Returns
    -------
    :class`numpy.ndarray`
        Converted array.

    Examples
    --------
    >>> a = np.array([0.0, 0.5, 1.0])
    >>> convert_bit_depth(a, "uint8")
    array([  0, 128, 255], dtype=uint8)
    >>> convert_bit_depth(a, "uint16")
    array([    0, 32768, 65535], dtype=uint16)
    >>> convert_bit_depth(a, "float16")
    array([ 0. ,  0.5,  1. ], dtype=float16)
    >>> a = np.array([0, 128, 255], dtype=np.uint8)
    >>> convert_bit_depth(a, "uint16")
    array([    0, 32896, 65535], dtype=uint16)
    >>> convert_bit_depth(a, "float32")  # doctest: +ELLIPSIS
    array([ 0.       ,  0.501960...,  1.       ], dtype=float32)
    """

    a = np.asarray(a)

    bit_depths = ", ".join(sorted(MAPPING_BIT_DEPTH.keys()))

    attest(
        bit_depth in bit_depths,
        f'Incorrect bit-depth was specified, it must be one of: "{bit_depths}"!',
    )

    attest(
        str(a.dtype) in bit_depths,
        f'Image bit-depth must be one of: "{bit_depths}"!',
    )

    source_dtype = str(a.dtype)
    target_dtype = MAPPING_BIT_DEPTH[bit_depth].numpy

    if source_dtype == "uint8":
        if bit_depth == "uint16":
            a = a.astype(target_dtype) * 257
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

    return a


@required("OpenImageIO")
def read_image_OpenImageIO(
    path: str | Path,
    bit_depth: Literal[
        "uint8", "uint16", "float16", "float32", "float64", "float128"
    ] = "float32",
    additional_data: bool = False,
    **kwargs: Any,
) -> NDArrayReal | Tuple[NDArrayReal, list]:
    """
    Read the image data at given path using *OpenImageIO*.

    Parameters
    ----------
    path
        Image path.
    bit_depth
        Returned image bit-depth, the bit-depth conversion behaviour is driven
        directly by *OpenImageIO*, this definition only converts to the
        relevant data type after reading.
    additional_data
        Whether to return additional data.

    Returns
    -------
    :class`numpy.ndarray` or :class:`tuple`
        Image data or tuple of image data and list of
        :class:`colour.io.Image_Specification_Attribute` class instances.

    Notes
    -----
    -   For convenience, single channel images are squeezed to 2D arrays.

    Examples
    --------
    >>> import os
    >>> import colour
    >>> path = os.path.join(
    ...     colour.__path__[0],
    ...     "io",
    ...     "tests",
    ...     "resources",
    ...     "CMS_Test_Pattern.exr",
    ... )
    >>> image = read_image_OpenImageIO(path)  # doctest: +SKIP
    """

    from OpenImageIO import ImageInput  # pyright: ignore

    path = str(path)

    kwargs = handle_arguments_deprecation(
        {
            "ArgumentRenamed": [["attributes", "additional_data"]],
        },
        **kwargs,
    )

    additional_data = kwargs.get("additional_data", additional_data)

    bit_depth_specification = MAPPING_BIT_DEPTH[bit_depth]

    image_input = ImageInput.open(path)
    image_specification = image_input.spec()

    shape = (
        image_specification.height,
        image_specification.width,
        image_specification.nchannels,
    )

    image = image_input.read_image(bit_depth_specification.openimageio)
    image_input.close()

    image = np.reshape(np.array(image, dtype=bit_depth_specification.numpy), shape)
    image = cast(NDArrayReal, np.squeeze(image))

    if additional_data:
        extra_attributes = []
        for attribute in image_specification.extra_attribs:
            extra_attributes.append(
                Image_Specification_Attribute(
                    attribute.name, attribute.value, attribute.type
                )
            )

        return image, extra_attributes
    else:
        return image


def read_image_Imageio(
    path: str | Path,
    bit_depth: Literal[
        "uint8", "uint16", "float16", "float32", "float64", "float128"
    ] = "float32",
    **kwargs: Any,
) -> NDArrayReal:
    """
    Read the image data at given path using *Imageio*.

    Parameters
    ----------
    path
        Image path.
    bit_depth
        Returned image bit-depth, the image data is converted with
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
    >>> path = os.path.join(
    ...     colour.__path__[0],
    ...     "io",
    ...     "tests",
    ...     "resources",
    ...     "CMS_Test_Pattern.exr",
    ... )
    >>> image = read_image_Imageio(path)
    >>> image.shape  # doctest: +SKIP
    (1267, 1274, 3)
    >>> image.dtype
    dtype('float32')
    """

    from imageio.v2 import imread

    path = str(path)

    image = np.squeeze(imread(path, **kwargs))

    return convert_bit_depth(image, bit_depth)


READ_IMAGE_METHODS: CanonicalMapping = CanonicalMapping(
    {
        "Imageio": read_image_Imageio,
        "OpenImageIO": read_image_OpenImageIO,
    }
)
READ_IMAGE_METHODS.__doc__ = """
Supported image read methods.
"""


def read_image(
    path: str | Path,
    bit_depth: Literal[
        "uint8", "uint16", "float16", "float32", "float64", "float128"
    ] = "float32",
    method: Literal["Imageio", "OpenImageIO"] | str = "OpenImageIO",
    **kwargs: Any,
) -> NDArrayReal:
    """
    Read the image data at given path using given method.

    Parameters
    ----------
    path
        Image path.
    bit_depth
        Returned image bit-depth, for the *Imageio* method, the image data is
        converted with :func:`colour.io.convert_bit_depth` definition after
        reading the image, for the *OpenImageIO* method, the bit-depth
        conversion behaviour is driven directly by the library, this definition
        only converts to the relevant data type after reading.
    method
        Read method, i.e., the image library used for reading images.

    Other Parameters
    ----------------
    additional_data
        {:func:`colour.io.read_image_OpenImageIO`},
        Whether to return additional data.

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
    >>> path = os.path.join(
    ...     colour.__path__[0],
    ...     "io",
    ...     "tests",
    ...     "resources",
    ...     "CMS_Test_Pattern.exr",
    ... )
    >>> image = read_image(path)
    >>> image.shape  # doctest: +SKIP
    (1267, 1274, 3)
    >>> image.dtype
    dtype('float32')
    """

    method = validate_method(method, tuple(READ_IMAGE_METHODS))

    if method == "openimageio" and not is_openimageio_installed():  # pragma: no cover
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
    path: str | Path,
    bit_depth: Literal[
        "uint8", "uint16", "float16", "float32", "float64", "float128"
    ] = "float32",
    attributes: Sequence | None = None,
) -> bool:
    """
    Write given image data at given path using *OpenImageIO*.

    Parameters
    ----------
    image
        Image data.
    path
        Image path.
    bit_depth
        Bit-depth to write the image at, the bit-depth conversion behaviour is
        ruled directly by *OpenImageIO*.
    attributes
        An array of :class:`colour.io.Image_Specification_Attribute` class
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
    >>> path = os.path.join(
    ...     colour.__path__[0],
    ...     "io",
    ...     "tests",
    ...     "resources",
    ...     "CMS_Test_Pattern.exr",
    ... )
    >>> image = read_image(path)  # doctest: +SKIP
    >>> path = os.path.join(
    ...     colour.__path__[0],
    ...     "io",
    ...     "tests",
    ...     "resources",
    ...     "CMSTestPattern.tif",
    ... )
    >>> write_image_OpenImageIO(image, path)  # doctest: +SKIP
    True

    Advanced image writing while setting attributes:

    >>> compression = Image_Specification_Attribute("Compression", "none")
    >>> write_image_OpenImageIO(image, path, "uint8", [compression])
    ... # doctest: +SKIP
    True

    Writing an "ACES" compliant "EXR" file:

    >>> if is_openimageio_installed():  # doctest: +SKIP
    ...     from OpenImageIO import TypeDesc
    ...
    ...     chromaticities = (
    ...         0.7347,
    ...         0.2653,
    ...         0.0,
    ...         1.0,
    ...         0.0001,
    ...         -0.077,
    ...         0.32168,
    ...         0.33767,
    ...     )
    ...     attributes = [
    ...         Image_Specification_Attribute("acesImageContainerFlag", True),
    ...         Image_Specification_Attribute(
    ...             "chromaticities", chromaticities, TypeDesc("float[8]")
    ...         ),
    ...         Image_Specification_Attribute("compression", "none"),
    ...     ]
    ...     write_image_OpenImageIO(image, path, attributes=attributes)
    """  # noqa: D405, D407, D410, D411

    from OpenImageIO import ImageOutput  # pyright: ignore

    image = as_float_array(image)
    path = str(path)

    attributes = cast(list, optional(attributes, []))

    bit_depth_specification = MAPPING_BIT_DEPTH[bit_depth]

    if bit_depth_specification.numpy in [np.uint8, np.uint16]:
        minimum, maximum = (
            np.iinfo(bit_depth_specification.numpy).min,
            np.iinfo(bit_depth_specification.numpy).max,
        )
        image = np.clip(image * maximum, minimum, maximum)

        image = as_int_array(image, bit_depth_specification.numpy)

    image = image.astype(bit_depth_specification.numpy)

    if image.ndim == 2:
        height, width = image.shape
        channels = 1
    else:
        height, width, channels = image.shape

    image_specification = image_specification_OpenImageIO(
        width, height, channels, bit_depth, attributes
    )

    image_output = ImageOutput.create(path)

    image_output.open(path, image_specification)
    image_output.write_image(image)

    image_output.close()

    return True


def write_image_Imageio(
    image: ArrayLike,
    path: str | Path,
    bit_depth: Literal[
        "uint8", "uint16", "float16", "float32", "float64", "float128"
    ] = "float32",
    **kwargs: Any,
) -> bytes | None:
    """
    Write given image data at given path using *Imageio*.

    Parameters
    ----------
    image
        Image data.
    path
        Image path.
    bit_depth
        Bit-depth to write the image at, the image data is converted with
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

    Notes
    -----
    -   It is possible to control how the image are saved by the *Freeimage*
        backend by using the ``flags`` keyword argument and passing a desired
        value. See the *Load / Save flag constants* section in
        https://sourceforge.net/p/freeimage/svn/HEAD/tree/FreeImage/trunk/\
Source/FreeImage.h

    Examples
    --------
    >>> import os
    >>> import colour
    >>> path = os.path.join(
    ...     colour.__path__[0],
    ...     "io",
    ...     "tests",
    ...     "resources",
    ...     "CMS_Test_Pattern.exr",
    ... )
    >>> image = read_image(path)  # doctest: +SKIP
    >>> path = os.path.join(
    ...     colour.__path__[0],
    ...     "io",
    ...     "tests",
    ...     "resources",
    ...     "CMSTestPattern.tif",
    ... )
    >>> write_image_Imageio(image, path)  # doctest: +SKIP
    True
    """

    from imageio.v2 import imwrite

    path = str(path)

    if all(
        [
            path.lower().endswith(".exr"),
            bit_depth in ("float32", "float64", "float128"),
        ]
    ):
        # Ensures that "OpenEXR" images are saved as "Float32" according to the
        # image bit-depth.
        kwargs["flags"] = 0x0001

    image = convert_bit_depth(image, bit_depth)

    return imwrite(path, image, **kwargs)


WRITE_IMAGE_METHODS: CanonicalMapping = CanonicalMapping(
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
    path: str | Path,
    bit_depth: Literal[
        "uint8", "uint16", "float16", "float32", "float64", "float128"
    ] = "float32",
    method: Literal["Imageio", "OpenImageIO"] | str = "OpenImageIO",
    **kwargs: Any,
) -> bool:
    """
    Write given image data at given path using given method.

    Parameters
    ----------
    image
        Image data.
    path
        Image path.
    bit_depth
        Bit-depth to write the image at, for the *Imageio* method, the image
        data is converted with :func:`colour.io.convert_bit_depth` definition
        prior to writing the image.
    method
        Write method, i.e., the image library used for writing images.

    Other Parameters
    ----------------
    attributes
        {:func:`colour.io.write_image_OpenImageIO`},
        An array of :class:`colour.io.Image_Specification_Attribute` class
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
    -   It is possible to control how the image are saved by the *Freeimage*
        backend by using the ``flags`` keyword argument and passing a desired
        value. See the *Load / Save flag constants* section in
        https://sourceforge.net/p/freeimage/svn/HEAD/tree/FreeImage/trunk/\
Source/FreeImage.h

    Examples
    --------
    Basic image writing:

    >>> import os
    >>> import colour
    >>> path = os.path.join(
    ...     colour.__path__[0],
    ...     "io",
    ...     "tests",
    ...     "resources",
    ...     "CMS_Test_Pattern.exr",
    ... )
    >>> image = read_image(path)  # doctest: +SKIP
    >>> path = os.path.join(
    ...     colour.__path__[0],
    ...     "io",
    ...     "tests",
    ...     "resources",
    ...     "CMSTestPattern.tif",
    ... )
    >>> write_image(image, path)  # doctest: +SKIP
    True

    Advanced image writing while setting attributes using *OpenImageIO*:

    >>> compression = Image_Specification_Attribute("Compression", "none")
    >>> write_image(image, path, bit_depth="uint8", attributes=[compression])
    ... # doctest: +SKIP
    True
    """  # noqa: D405, D407, D410, D411, D414

    method = validate_method(method, tuple(WRITE_IMAGE_METHODS))

    if method == "openimageio" and not is_openimageio_installed():  # pragma: no cover
        usage_warning(
            '"OpenImageIO" related API features are not available, '
            'switching to "Imageio"!'
        )
        method = "Imageio"

    function = WRITE_IMAGE_METHODS[method]

    if method == "openimageio":  # pragma: no cover
        kwargs = filter_kwargs(function, **kwargs)

    return function(image, path, bit_depth, **kwargs)


def as_3_channels_image(a: ArrayLike) -> NDArrayFloat:
    """
    Convert given array :math:`a` to a 3-channels image-like representation.

    Parameters
    ----------
    a
         Array :math:`a` to convert to a 3-channels image-like representation.

    Returns
    -------
    :class`numpy.ndarray`
        3-channels image-like representation of array :math:`a`.

    Examples
    --------
    >>> as_3_channels_image(0.18)
    array([[[ 0.18,  0.18,  0.18]]])
    >>> as_3_channels_image([0.18])
    array([[[ 0.18,  0.18,  0.18]]])
    >>> as_3_channels_image([0.18, 0.18, 0.18])
    array([[[ 0.18,  0.18,  0.18]]])
    >>> as_3_channels_image([[0.18, 0.18, 0.18]])
    array([[[ 0.18,  0.18,  0.18]]])
    >>> as_3_channels_image([[[0.18, 0.18, 0.18]]])
    array([[[ 0.18,  0.18,  0.18]]])
    >>> as_3_channels_image([[[[0.18, 0.18, 0.18]]]])
    array([[[ 0.18,  0.18,  0.18]]])
    """

    a = np.squeeze(as_float_array(a))

    if len(a.shape) > 3:
        raise ValueError(
            "Array has more than 3-dimensions and cannot be converted to a "
            "3-channels image-like representation!"
        )

    if len(a.shape) > 0 and a.shape[-1] not in (1, 3):
        raise ValueError(
            "Array has more than 1 or 3 channels and cannot be converted to a "
            "3-channels image-like representation!"
        )

    if len(a.shape) == 0 or a.shape[-1] == 1:
        a = tstack([a, a, a])

    if len(a.shape) == 1:
        a = a[None, None, ...]
    elif len(a.shape) == 2:
        a = a[None, ...]

    return a
