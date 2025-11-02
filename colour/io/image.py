"""
Image Input / Output Utilities
==============================

Define image-related input/output utility objects for colour science
applications.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass, field

import numpy as np

if typing.TYPE_CHECKING:
    from colour.hints import (
        Any,
        ArrayLike,
        DTypeReal,
        Literal,
        NDArrayFloat,
        PathLike,
        Sequence,
        Tuple,
        Type,
    )

from colour.hints import NDArrayReal, cast
from colour.utilities import (
    CanonicalMapping,
    as_float_array,
    as_int_array,
    attest,
    filter_kwargs,
    is_imageio_installed,
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
    Define a bit-depth specification for image processing operations.

    Parameters
    ----------
    name
        Attribute name identifying the bit-depth specification.
    numpy
        Object representing the *NumPy* bit-depth data type.
    openimageio
        Object representing the *OpenImageIO* bit-depth specification.
    """

    name: str
    numpy: Type[DTypeReal]
    openimageio: Any


@dataclass
class Image_Specification_Attribute:
    """
    Define an image specification attribute for OpenImageIO operations.

    Parameters
    ----------
    name
        Attribute name identifying the metadata field.
    value
        Attribute value containing the metadata content.
    type_
        Attribute type as an *OpenImageIO* :class:`TypeDesc` class instance
        specifying the data type of the value.
    """

    name: str
    value: Any
    type_: OpenImageIO.TypeDesc | None = field(  # noqa: F821, RUF100 # pyright: ignore # noqa: F821
        default_factory=lambda: None
    )


if is_openimageio_installed():  # pragma: no cover
    from OpenImageIO import ImageSpec  # pyright: ignore
    from OpenImageIO import DOUBLE, FLOAT, HALF, UINT8, UINT16

    MAPPING_BIT_DEPTH: CanonicalMapping = CanonicalMapping(
        {
            "uint8": Image_Specification_BitDepth("uint8", np.uint8, UINT8),
            "uint16": Image_Specification_BitDepth("uint16", np.uint16, UINT16),
            "float16": Image_Specification_BitDepth("float16", np.float16, HALF),
            "float32": Image_Specification_BitDepth("float32", np.float32, FLOAT),
            "float64": Image_Specification_BitDepth("float64", np.float64, DOUBLE),
        }
    )
    if not typing.TYPE_CHECKING and hasattr(np, "float128"):  # pragma: no cover
        MAPPING_BIT_DEPTH["float128"] = Image_Specification_BitDepth(
            "float128", np.float128, DOUBLE
        )
else:  # pragma: no cover

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
    if not typing.TYPE_CHECKING and hasattr(np, "float128"):  # pragma: no cover
        MAPPING_BIT_DEPTH["float128"] = Image_Specification_BitDepth(
            "float128", np.float128, None
        )


def add_attributes_to_image_specification_OpenImageIO(
    image_specification: ImageSpec, attributes: Sequence
) -> ImageSpec:
    """
    Add the specified attributes to the specified *OpenImageIO* image
    specification.

    Apply metadata attributes to an existing image specification object,
    enabling customization of image properties such as compression,
    colour space information, or other format-specific metadata.

    Parameters
    ----------
    image_specification
        *OpenImageIO* image specification to modify.
    attributes
        Sequence of :class:`colour.io.Image_Specification_Attribute`
        instances containing metadata to apply to the image
        specification.

    Returns
    -------
    :class:`ImageSpec`
        Modified *OpenImageIO* image specification with applied
        attributes.

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
    """

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
        Bit-depth to create the image with. The bit-depth conversion
        behaviour is ruled directly by *OpenImageIO*.
    attributes
        An array of :class:`colour.io.Image_Specification_Attribute`
        class instances used to set attributes of the image.

    Returns
    -------
    :class:`ImageSpec`
        *OpenImageIO* image specification.

    Examples
    --------
    >>> compression = Image_Specification_Attribute("Compression", "none")
    >>> image_specification_OpenImageIO(
    ...     1920, 1080, 3, "float16", [compression]
    ... )  # doctest: +SKIP
    <OpenImageIO.ImageSpec object at 0x...>
    """

    from OpenImageIO import ImageSpec  # noqa: PLC0415

    attributes = cast("list", optional(attributes, []))

    bit_depth_specification = MAPPING_BIT_DEPTH[bit_depth]

    image_specification = ImageSpec(
        width, height, channels, bit_depth_specification.openimageio
    )

    add_attributes_to_image_specification_OpenImageIO(
        image_specification,  # pyright: ignore
        attributes or [],
    )

    return image_specification  # pyright: ignore


def convert_bit_depth(
    a: ArrayLike,
    bit_depth: Literal[
        "uint8", "uint16", "float16", "float32", "float64", "float128"
    ] = "float32",
) -> NDArrayReal:
    """
    Convert the specified array to the specified bit-depth.

    The conversion path is determined by the current bit-depth of the input
    array and the target bit-depth. Supports conversions between unsigned
    integers, floating-point types, and mixed type conversions with
    appropriate scaling.

    Parameters
    ----------
    a
        Array to convert to the specified bit-depth.
    bit_depth
        Target bit-depth. Supported types include unsigned integers
        ("uint8", "uint16") and floating-point ("float16", "float32",
        "float64", "float128").

    Returns
    -------
    :class:`numpy.ndarray`
        Array converted to the specified bit-depth.

    Raises
    ------
    AssertionError
        If the source or target bit-depth is not supported.

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


@typing.overload
@required("OpenImageIO")
def read_image_OpenImageIO(
    path: str | PathLike,
    bit_depth: Literal[
        "uint8", "uint16", "float16", "float32", "float64", "float128"
    ] = ...,
    additional_data: Literal[True] = True,
    **kwargs: Any,
) -> Tuple[NDArrayReal, Tuple[Image_Specification_Attribute, ...]]: ...


@typing.overload
@required("OpenImageIO")
def read_image_OpenImageIO(
    path: str | PathLike,
    bit_depth: Literal[
        "uint8", "uint16", "float16", "float32", "float64", "float128"
    ] = ...,
    *,
    additional_data: Literal[False],
    **kwargs: Any,
) -> NDArrayReal: ...


@typing.overload
@required("OpenImageIO")
def read_image_OpenImageIO(
    path: str | PathLike,
    bit_depth: Literal["uint8", "uint16", "float16", "float32", "float64", "float128"],
    additional_data: Literal[False],
    **kwargs: Any,
) -> NDArrayReal: ...


@required("OpenImageIO")
def read_image_OpenImageIO(
    path: str | PathLike,
    bit_depth: Literal[
        "uint8", "uint16", "float16", "float32", "float64", "float128"
    ] = "float32",
    additional_data: bool = False,
    **kwargs: Any,
) -> NDArrayReal | Tuple[NDArrayReal, Tuple[Image_Specification_Attribute, ...]]:
    """
    Read image data from the specified path using *OpenImageIO*.

    Load image data from the file system with support for various bit-depth
    formats. The bit-depth conversion behaviour is controlled by
    *OpenImageIO*, with this function performing only the final type
    conversion after reading.

    Parameters
    ----------
    path
        Path to the image file.
    bit_depth
        Target bit-depth for the returned image data. The bit-depth
        conversion is handled by *OpenImageIO* during the read operation,
        with this function converting to the appropriate *NumPy* data type
        afterwards.
    additional_data
        Whether to return additional metadata from the image file.

    Returns
    -------
    :class:`numpy.ndarray` or :class:`tuple`
        Image data as an array when ``additional_data`` is ``False``, or a
        tuple containing the image data and a tuple of
        :class:`colour.io.Image_Specification_Attribute` instances when
        ``additional_data`` is ``True``.

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

    from OpenImageIO import ImageInput  # noqa: PLC0415

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
    image = cast("NDArrayReal", np.squeeze(image))

    if additional_data:
        extra_attributes = [
            Image_Specification_Attribute(
                attribute.name, attribute.value, attribute.type
            )
            for attribute in image_specification.extra_attribs
        ]

        return image, tuple(extra_attributes)

    return image


@required("Imageio")
def read_image_Imageio(
    path: str | PathLike,
    bit_depth: Literal[
        "uint8", "uint16", "float16", "float32", "float64", "float128"
    ] = "float32",
    **kwargs: Any,
) -> NDArrayReal:
    """
    Read image data from the specified path using *Imageio*.

    Parameters
    ----------
    path
        Path to the image file.
    bit_depth
        Target bit-depth for the returned image data. The image data is
        converted with :func:`colour.io.convert_bit_depth` definition after
        reading the image.

    Other Parameters
    ----------------
    kwargs
        Keywords arguments.

    Returns
    -------
    :class:`numpy.ndarray`
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

    from imageio.v2 import imread  # noqa: PLC0415

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
Supported image reading methods.
"""


def read_image(
    path: str | PathLike,
    bit_depth: Literal[
        "uint8", "uint16", "float16", "float32", "float64", "float128"
    ] = "float32",
    method: Literal["Imageio", "OpenImageIO"] | str = "OpenImageIO",
    **kwargs: Any,
) -> NDArrayReal:
    """
    Read image data from the specified path.

    Load and optionally convert image data from various formats,
    supporting multiple bit-depth conversions and backend libraries for
    flexible image I/O operations in colour science workflows.

    Parameters
    ----------
    path
        Path to the image file.
    bit_depth
        Target bit-depth for the returned image data. For the *Imageio*
        method, image data is converted using
        :func:`colour.io.convert_bit_depth` after reading. For the
        *OpenImageIO* method, bit-depth conversion is handled by the
        library with this parameter controlling only the final data type.
    method
        Image reading backend library. Defaults to *OpenImageIO* with
        automatic fallback to *Imageio* if unavailable.

    Other Parameters
    ----------------
    additional_data
        {:func:`colour.io.read_image_OpenImageIO`},
        Whether to return additional metadata with the image data.

    Returns
    -------
    :class:`numpy.ndarray`
        Image data as a NumPy array with the specified bit-depth.

    Notes
    -----
    -   If the specified method is *OpenImageIO* but the library is not
        available, reading will be performed by *Imageio*.
    -   If the specified method is *Imageio*, ``kwargs`` is passed
        directly to the wrapped definition.
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

    if method.lower() == "imageio" and not is_imageio_installed():  # pragma: no cover
        usage_warning(
            '"Imageio" related API features are not available, '
            'switching to "OpenImageIO"!'
        )
        method = "openimageio"

    method = validate_method(method, tuple(READ_IMAGE_METHODS))

    function = READ_IMAGE_METHODS[method]

    if method == "openimageio":  # pragma: no cover
        kwargs = filter_kwargs(function, **kwargs)

    return function(path, bit_depth, **kwargs)


@required("OpenImageIO")
def write_image_OpenImageIO(
    image: ArrayLike,
    path: str | PathLike,
    bit_depth: Literal[
        "uint8", "uint16", "float16", "float32", "float64", "float128"
    ] = "float32",
    attributes: Sequence | None = None,
) -> bool:
    """
    Write image data to the specified path using *OpenImageIO*.

    Parameters
    ----------
    image
        Image data to write.
    path
        Path to the image file.
    bit_depth
        Bit-depth to write the image at. The bit-depth conversion behaviour
        is ruled directly by *OpenImageIO*.
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

    >>> from OpenImageIO import TypeDesc
    >>> chromaticities = (
    ...     0.7347,
    ...     0.2653,
    ...     0.0,
    ...     1.0,
    ...     0.0001,
    ...     -0.077,
    ...     0.32168,
    ...     0.33767,
    ... )
    >>> attributes = [
    ...     Image_Specification_Attribute("openexr:ACESContainerPolicy", "relaxed"),
    ...     Image_Specification_Attribute(
    ...         "chromaticities", chromaticities, TypeDesc("float[8]")
    ...     ),
    ...     Image_Specification_Attribute("compression", "none"),
    ... ]
    >>> write_image_OpenImageIO(image, path, attributes=attributes)  # doctest: +SKIP
    True

    Notes
    -----
    -   When using ``openexr:ACESContainerPolicy`` with ``relaxed`` mode,
        *OpenImageIO* automatically sets the ``colorInteropId`` attribute to
        ``lin_ap0_scene`` for ACES-compliant files.
    -   The ``acesImageContainerFlag`` attribute should not be set manually
        in *OpenImageIO* 3.1.7.0+, as it triggers strict ACES validation.
        Use ``openexr:ACESContainerPolicy`` instead.
    """

    from OpenImageIO import ImageOutput  # noqa: PLC0415

    image = as_float_array(image)
    path = str(path)

    attributes = cast("list", optional(attributes, []))

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

    image_output.open(path, image_specification)  # pyright: ignore
    success = image_output.write_image(image)

    image_output.close()

    return success


@required("Imageio")
def write_image_Imageio(
    image: ArrayLike,
    path: str | PathLike,
    bit_depth: Literal[
        "uint8", "uint16", "float16", "float32", "float64", "float128"
    ] = "float32",
    **kwargs: Any,
) -> bytes | None:
    """
    Write image data to the specified path using *Imageio*.

    Parameters
    ----------
    image
        Image data to write.
    path
        Path to the image file.
    bit_depth
        Bit-depth to write the image at. The image data is converted with
        :func:`colour.io.convert_bit_depth` definition prior to writing.

    Other Parameters
    ----------------
    kwargs
        Keywords arguments passed to the underlying *Imageio* ``imwrite``
        function.

    Returns
    -------
    :class:`bytes` or :py:data:`None`
        Image data written as bytes if successful, :py:data:`None`
        otherwise.

    Notes
    -----
    -   Control how images are saved by the *Freeimage* backend using the
        ``flags`` keyword argument with desired values. See the *Load /
        Save flag constants* section in
        https://sourceforge.net/p/freeimage/svn/HEAD/tree/FreeImage/trunk/Source/FreeImage.h

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

    from imageio.v2 import imwrite  # noqa: PLC0415

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
Supported image writing methods.
"""


def write_image(
    image: ArrayLike,
    path: str | PathLike,
    bit_depth: Literal[
        "uint8", "uint16", "float16", "float32", "float64", "float128"
    ] = "float32",
    method: Literal["Imageio", "OpenImageIO"] | str = "OpenImageIO",
    **kwargs: Any,
) -> bool:
    """
    Write image data to the specified path.

    Parameters
    ----------
    image
        Image data to write.
    path
        Path to the image file.
    bit_depth
        Bit-depth to write the image at. For the *Imageio* method, the
        image data is converted with :func:`colour.io.convert_bit_depth`
        definition prior to writing the image.
    method
        Image writing backend library.

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
    -   If the specified method is *OpenImageIO* but the library is not
        available writing will be performed by *Imageio*.
    -   If the specified method is *Imageio*, ``kwargs`` is passed directly
        to the wrapped definition.
    -   It is possible to control how the images are saved by the
        *Freeimage* backend by using the ``flags`` keyword argument and
        passing a desired value. See the *Load / Save flag constants*
        section in
        https://sourceforge.net/p/freeimage/svn/HEAD/tree/FreeImage/trunk/Source/FreeImage.h

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
    """

    if method.lower() == "imageio" and not is_imageio_installed():  # pragma: no cover
        usage_warning(
            '"Imageio" related API features are not available, '
            'switching to "OpenImageIO"!'
        )
        method = "openimageio"

    method = validate_method(method, tuple(WRITE_IMAGE_METHODS))

    function = WRITE_IMAGE_METHODS[method]

    if method == "openimageio":  # pragma: no cover
        kwargs = filter_kwargs(function, **kwargs)

    return function(image, path, bit_depth, **kwargs)


def as_3_channels_image(a: ArrayLike) -> NDArrayFloat:
    """
    Convert the specified array :math:`a` to a 3-channel image-like
    representation.

    Parameters
    ----------
    a
        Array :math:`a` to convert to a 3-channel image-like representation.

    Returns
    -------
    :class:`numpy.ndarray`
        3-channel image-like representation of array :math:`a`.

    Raises
    ------
    ValueError
        If the array has more than 3 dimensions or more than 1 or 3 channels.

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
        error = (
            "Array has more than 3-dimensions and cannot be converted to a "
            "3-channels image-like representation!"
        )

        raise ValueError(error)

    if len(a.shape) > 0 and a.shape[-1] not in (1, 3):
        error = (
            "Array has more than 1 or 3 channels and cannot be converted to a "
            "3-channels image-like representation!"
        )

        raise ValueError(error)

    if len(a.shape) == 0 or a.shape[-1] == 1:
        a = tstack([a, a, a])

    if len(a.shape) == 1:
        a = a[None, None, ...]
    elif len(a.shape) == 2:
        a = a[None, ...]

    return a
