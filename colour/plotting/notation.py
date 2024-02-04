"""
Colour Notation Systems Plotting
================================

Defines the colour notation systems plotting objects:

-   :func:`colour.plotting.plot_single_munsell_value_function`
-   :func:`colour.plotting.plot_multi_munsell_value_functions`
"""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from colour.hints import Any, Callable, Dict, Sequence, Tuple
from colour.notation import MUNSELL_VALUE_METHODS
from colour.plotting import (
    filter_passthrough,
    override_style,
    plot_multi_functions,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "plot_single_munsell_value_function",
    "plot_multi_munsell_value_functions",
]


@override_style()
def plot_single_munsell_value_function(
    function: Callable | str, **kwargs: Any
) -> Tuple[Figure, Axes]:
    """
    Plot given *Lightness* function.

    Parameters
    ----------
    function
        *Munsell* value function to plot. ``function`` can be of any type or
        form supported by the :func:`colour.plotting.common.filter_passthrough`
        definition.

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.plot_multi_functions`,
        :func:`colour.plotting.render`},
        See the documentation of the previously listed definitions.

    Returns
    -------
    :class:`tuple`
        Current figure and axes.

    Examples
    --------
    >>> plot_single_munsell_value_function("ASTM D1535")  # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...Axes...>)

    .. image:: ../_static/Plotting_Plot_Single_Munsell_Value_Function.png
        :align: center
        :alt: plot_single_munsell_value_function
    """

    settings: Dict[str, Any] = {"title": f"{function} - Munsell Value Function"}
    settings.update(kwargs)

    return plot_multi_munsell_value_functions((function,), **settings)


@override_style()
def plot_multi_munsell_value_functions(
    functions: Callable | str | Sequence[Callable | str],
    **kwargs: Any,
) -> Tuple[Figure, Axes]:
    """
    Plot given *Munsell* value functions.

    Parameters
    ----------
    functions
        *Munsell* value functions to plot. ``functions`` elements can be of any
        type or form supported by the
        :func:`colour.plotting.common.filter_passthrough` definition.

    Other Parameters
    ----------------
    kwargs
        {:func:`colour.plotting.artist`,
        :func:`colour.plotting.plot_multi_functions`,
        :func:`colour.plotting.render`},
        See the documentation of the previously listed definitions.

    Returns
    -------
    :class:`tuple`
        Current figure and axes.

    Examples
    --------
    >>> plot_multi_munsell_value_functions(["ASTM D1535", "McCamy 1987"])
    ... # doctest: +ELLIPSIS
    (<Figure size ... with 1 Axes>, <...Axes...>)

    .. image:: ../_static/Plotting_Plot_Multi_Munsell_Value_Functions.png
        :align: center
        :alt: plot_multi_munsell_value_functions
    """

    functions_filtered = filter_passthrough(MUNSELL_VALUE_METHODS, functions)

    settings: Dict[str, Any] = {
        "bounding_box": (0, 100, 0, 10),
        "legend": True,
        "title": f"{', '.join(functions_filtered)} - Munsell Functions",
        "x_label": "Luminance Y",
        "y_label": "Munsell Value V",
    }
    settings.update(kwargs)

    return plot_multi_functions(
        functions_filtered, samples=np.linspace(0, 100, 1000), **settings
    )
