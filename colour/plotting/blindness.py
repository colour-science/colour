"""
Colour Blindness Plotting
=========================

Defines the colour blindness plotting objects:

-   :func:`colour.plotting.plot_cvd_simulation_Machado2009`
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from colour.algebra import vector_dot
from colour.blindness import matrix_cvd_Machado2009
from colour.hints import (
    Any,
    ArrayLike,
    Dict,
    Floating,
    Literal,
    Optional,
    Tuple,
    Union,
    cast,
)
from colour.plotting import CONSTANTS_COLOUR_STYLE, plot_image, override_style
from colour.utilities import optional

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "plot_cvd_simulation_Machado2009",
]


@override_style()
def plot_cvd_simulation_Machado2009(
    RGB: ArrayLike,
    deficiency: Union[
        Literal["Deuteranomaly", "Protanomaly", "Tritanomaly"], str
    ] = "Protanomaly",
    severity: Floating = 0.5,
    M_a: Optional[ArrayLike] = None,
    **kwargs: Any,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Perform colour vision deficiency simulation on given *RGB* colourspace
    array using *Machado et al. (2009)* model.

    Parameters
    ----------
    RGB
        *RGB* colourspace array.
    deficiency
        Colour blindness / vision deficiency type.
    severity
        Severity of the colour vision deficiency in domain [0, 1].
    M_a
        Anomalous trichromacy matrix to use instead of Machado (2010)
        pre-computed matrix.

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.plotting.artist`, :func:`colour.plotting.plot_image`,
        :func:`colour.plotting.render`},
        See the documentation of the previously listed definitions.

    Notes
    -----
    -  Input *RGB* array is expected to be linearly encoded.

    Returns
    -------
    :class:`tuple`
        Current figure and axes.

    Examples
    --------
    >>> import numpy as np
    >>> RGB = np.random.rand(32, 32, 3)
    >>> plot_cvd_simulation_Machado2009(RGB)  # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...AxesSubplot...>)

    .. image:: ../_static/Plotting_Plot_CVD_Simulation_Machado2009.png
        :align: center
        :alt: plot_cvd_simulation_Machado2009
    """

    M_a = cast(
        ArrayLike, optional(M_a, matrix_cvd_Machado2009(deficiency, severity))
    )

    settings: Dict[str, Any] = {
        "text_kwargs": {
            "text": f"Deficiency: {deficiency} - Severity: {severity}"
        }
    }
    settings.update(kwargs)

    return plot_image(
        CONSTANTS_COLOUR_STYLE.colour.colourspace.cctf_encoding(
            vector_dot(M_a, RGB)
        ),
        **settings,
    )
