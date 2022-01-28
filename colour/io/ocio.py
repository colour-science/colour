# -*- coding: utf-8 -*-
"""
OpenColorIO Processing
======================

Defines the object for *OpenColorIO* processing:

-   :func:`colour.io.process_image_OpenColorIO`
"""

from __future__ import annotations

import numpy as np

from colour.hints import Any, ArrayLike, NDArray
from colour.utilities import as_float_array, required

__author__ = "Colour Developers"
__copyright__ = "Copyright (C) 2013-2021 - Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "process_image_OpenColorIO",
]


@required("OpenColorIO")
def process_image_OpenColorIO(a: ArrayLike, *args: Any, **kwargs: Any) -> NDArray:
    """
    Processes given image with *OpenColorIO*.

    Parameters
    ----------
    a
        Image to process with *OpenColorIO*.

    Other Parameters
    ----------------
    args
        Arguments for `Config.getProcessor` method.
        See https://opencolorio.readthedocs.io/en/latest/api/config.html for
        more information.
    config
        *OpenColorIO* config to use for processing. If not defined, the
        *OpenColorIO* set defined by the ``$OCIO`` environment variable is
        used.

    Returns
    -------
    :class:`numpy.ndarray`
        Processed image.

    Examples
    --------
    >>> import os
    >>> import PyOpenColorIO as ocio
    >>> from colour.utilities import full
    >>> config = os.path.join(
    ...     os.path.dirname(__file__), 'tests', 'resources',
    ...     'config-aces-reference.ocio.yaml')
    >>> a = full([4, 2, 3], 0.18)
    >>> process_image_OpenColorIO(  # doctest: +ELLIPSIS
    ...     a, 'ACES - ACES2065-1', 'ACES - ACEScct', config=config)
    array([[[ 0.4135878...,  0.4135878...,  0.4135878...],
            [ 0.4135878...,  0.4135878...,  0.4135878...]],
    <BLANKLINE>
           [[ 0.4135878...,  0.4135878...,  0.4135878...],
            [ 0.4135878...,  0.4135878...,  0.4135878...]],
    <BLANKLINE>
           [[ 0.4135878...,  0.4135878...,  0.4135878...],
            [ 0.4135878...,  0.4135878...,  0.4135878...]],
    <BLANKLINE>
           [[ 0.4135878...,  0.4135878...,  0.4135878...],
            [ 0.4135878...,  0.4135878...,  0.4135878...]]], dtype=float32)
    >>> process_image_OpenColorIO(  # doctest: +ELLIPSIS
    ...     a, 'ACES - ACES2065-1', 'Display - sRGB',
    ...     'Output - SDR Video - ACES 1.0', ocio.TRANSFORM_DIR_FORWARD,
    ...     config=config)
    array([[[ 0.3559523...,  0.3559525...,  0.3559525...],
            [ 0.3559523...,  0.3559525...,  0.3559525...]],
    <BLANKLINE>
           [[ 0.3559523...,  0.3559525...,  0.3559525...],
            [ 0.3559523...,  0.3559525...,  0.3559525...]],
    <BLANKLINE>
           [[ 0.3559523...,  0.3559525...,  0.3559525...],
            [ 0.3559523...,  0.3559525...,  0.3559525...]],
    <BLANKLINE>
           [[ 0.3559523...,  0.3559525...,  0.3559525...],
            [ 0.3559523...,  0.3559525...,  0.3559525...]]], dtype=float32)
    """

    import PyOpenColorIO as ocio

    config = kwargs.get("config")
    config = (
        ocio.Config.CreateFromEnv()
        if config is None
        else ocio.Config.CreateFromFile(config)
    )

    a = as_float_array(np.atleast_3d(a), dtype=np.float32)
    height, width, channels = a.shape

    processor = config.getProcessor(*args).getDefaultCPUProcessor()

    image_desc = ocio.PackedImageDesc(a, width, height, channels)

    processor.apply(image_desc)

    return image_desc.getData().reshape([height, width, channels])
