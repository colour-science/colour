"""
OpenColorIO Processing
======================

Define the object for *OpenColorIO* processing:

-   :func:`colour.io.process_image_OpenColorIO`
"""

from __future__ import annotations

import numpy as np

from colour.hints import Any, ArrayLike, NDArrayFloat
from colour.io import as_3_channels_image
from colour.utilities import as_float, as_float_array, required

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "process_image_OpenColorIO",
]


@required("OpenColorIO")
def process_image_OpenColorIO(a: ArrayLike, *args: Any, **kwargs: Any) -> NDArrayFloat:
    """
    Process given image data with *OpenColorIO*.

    Parameters
    ----------
    a
        Image data to process with *OpenColorIO*.

    Other Parameters
    ----------------
    config
        *OpenColorIO* config to use for processing. If not defined, the
        *OpenColorIO* set defined by the ``$OCIO`` environment variable is
        used.
    args
        Arguments for `Config.getProcessor` method.
        See https://opencolorio.readthedocs.io/en/latest/api/config.html for
        more information.

    Returns
    -------
    :class:`numpy.ndarray`
        Processed image data.

    Examples
    --------
    >>> import os
    >>> import PyOpenColorIO as ocio  # doctest: +SKIP
    >>> from colour.utilities import full
    >>> config = os.path.join(
    ...     os.path.dirname(__file__),
    ...     "tests",
    ...     "resources",
    ...     "config-aces-reference.ocio.yaml",
    ... )
    >>> a = 0.18
    >>> process_image_OpenColorIO(  # doctest: +SKIP
    ...     a, "ACES - ACES2065-1", "ACES - ACEScct", config=config
    ... )
    0.4135884...
    >>> a = np.array([0.18])
    >>> process_image_OpenColorIO(  # doctest: +SKIP
    ...     a, "ACES - ACES2065-1", "ACES - ACEScct", config=config
    ... )
    array([ 0.4135884...])
    >>> a = np.array([0.18, 0.18, 0.18])
    >>> process_image_OpenColorIO(  # doctest: +SKIP
    ...     a, "ACES - ACES2065-1", "ACES - ACEScct", config=config
    ... )
    array([ 0.4135884...,  0.4135884...,  0.4135884...])
    >>> a = np.array([[0.18, 0.18, 0.18]])
    >>> process_image_OpenColorIO(  # doctest: +SKIP
    ...     a, "ACES - ACES2065-1", "ACES - ACEScct", config=config
    ... )
    array([[ 0.4135884...,  0.4135884...,  0.4135884...]])
    >>> a = np.array([[[0.18, 0.18, 0.18]]])
    >>> process_image_OpenColorIO(  # doctest: +SKIP
    ...     a, "ACES - ACES2065-1", "ACES - ACEScct", config=config
    ... )
    array([[[ 0.4135884...,  0.4135884...,  0.4135884...]]])
    >>> a = full([4, 2, 3], 0.18)
    >>> process_image_OpenColorIO(  # doctest: +SKIP
    ...     a, "ACES - ACES2065-1", "ACES - ACEScct", config=config
    ... )
    array([[[ 0.4135884...,  0.4135884...,  0.4135884...],
            [ 0.4135884...,  0.4135884...,  0.4135884...]],
    <BLANKLINE>
           [[ 0.4135884...,  0.4135884...,  0.4135884...],
            [ 0.4135884...,  0.4135884...,  0.4135884...]],
    <BLANKLINE>
           [[ 0.4135884...,  0.4135884...,  0.4135884...],
            [ 0.4135884...,  0.4135884...,  0.4135884...]],
    <BLANKLINE>
           [[ 0.4135884...,  0.4135884...,  0.4135884...],
            [ 0.4135884...,  0.4135884...,  0.4135884...]]])
    >>> process_image_OpenColorIO(  # doctest: +SKIP
    ...     a,
    ...     "ACES - ACES2065-1",
    ...     "Display - sRGB",
    ...     "Output - SDR Video - ACES 1.0",
    ...     ocio.TRANSFORM_DIR_FORWARD,
    ...     config=config,
    ... )
    array([[[ 0.3559542...,  0.3559542...,  0.3559542...],
            [ 0.3559542...,  0.3559542...,  0.3559542...]],
    <BLANKLINE>
           [[ 0.3559542...,  0.3559542...,  0.3559542...],
            [ 0.3559542...,  0.3559542...,  0.3559542...]],
    <BLANKLINE>
           [[ 0.3559542...,  0.3559542...,  0.3559542...],
            [ 0.3559542...,  0.3559542...,  0.3559542...]],
    <BLANKLINE>
           [[ 0.3559542...,  0.3559542...,  0.3559542...],
            [ 0.3559542...,  0.3559542...,  0.3559542...]]])
    """

    import PyOpenColorIO as ocio

    config = kwargs.get("config")
    config = (
        ocio.Config.CreateFromEnv()  # pyright: ignore
        if config is None
        else ocio.Config.CreateFromFile(config)  # pyright: ignore
    )

    a = as_float_array(a)
    shape, dtype = a.shape, a.dtype
    a = as_3_channels_image(a).astype(np.float32)

    height, width, channels = a.shape

    processor = config.getProcessor(*args).getDefaultCPUProcessor()

    image_desc = ocio.PackedImageDesc(  # pyright: ignore
        a, width, height, channels
    )

    processor.apply(image_desc)

    b = np.reshape(image_desc.getData(), (height, width, channels)).astype(dtype)

    if len(shape) == 0:
        return as_float(np.squeeze(b)[0])
    elif shape[-1] == 1:
        return np.reshape(b[..., 0], shape)
    else:
        return np.reshape(b, shape)
