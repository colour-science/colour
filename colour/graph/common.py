"""
Graph Common Utilities
======================

Defines various common utilities for the graph module.
"""

from __future__ import annotations

import typing

import numpy as np

from colour.graph.conversion import conversion_path
from colour.utilities import as_float_array, get_domain_range_scale_metadata

if typing.TYPE_CHECKING:
    from colour.hints import ArrayLike, NDArrayFloat

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "colourspace_model_to_reference",
]


def colourspace_model_to_reference(
    a: ArrayLike,
    model: str,
) -> NDArrayFloat:
    """
    Scale given colourspace model array from normalized [0, 1] to the model's
    reference scale by extracting scale metadata from the conversion function.

    This function multiplies the input array by the model's reference scale
    (e.g., [0, 100] for CIE Lab) extracted from the XYZ_to_model conversion
    function's range annotation.

    Parameters
    ----------
    a
        Colourspace model array in normalized [0, 1] scale.
    model
        Colourspace model name (e.g., "CIE Lab", "CIE XYZ").

    Returns
    -------
    :class:`numpy.ndarray`
        Colourspace model array scaled to reference scale.

    Examples
    --------
    >>> import numpy as np
    >>> Lab = np.array([0.41527875, 0.52638583, 0.26923179])
    >>> colourspace_model_to_reference(Lab, "CIE Lab")  # doctest: +ELLIPSIS
    array([ 41.527875...,  52.638583...,  26.923179...])
    """

    import networkx as nx  # noqa: PLC0415

    a = as_float_array(a)

    try:
        # Get conversion path from XYZ to model (lowercase for graph)
        path_functions = conversion_path("cie xyz", model.lower())

        # Get the last function in the path (final conversion to target model)
        if path_functions:
            last_function = path_functions[-1]
            metadata = get_domain_range_scale_metadata(last_function)
            range_scale = metadata.get("range")

            if range_scale is not None:
                # Handle tuple scales (e.g., (100, 100, 360))
                if isinstance(range_scale, tuple):
                    scale_factor = np.array(range_scale)
                else:
                    # Scalar scale applied uniformly
                    scale_factor = range_scale

                return a * scale_factor
    except (nx.NodeNotFound, nx.NetworkXNoPath):
        # Model not in graph or no conversion path exists
        pass

    # Fallback: return unchanged if no scale metadata found
    return a
