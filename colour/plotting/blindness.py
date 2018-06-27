# -*- coding: utf-8 -*-
"""
Colour Blindness Plotting
=========================

Defines the colour blindness plotting objects:

-   :func:`cvd_simulation_Machado2009_plot`
"""

from __future__ import division

from colour.blindness import cvd_matrix_Machado2009
from colour.plotting import DEFAULT_PLOTTING_SETTINGS, image_plot
from colour.utilities import dot_vector

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2018 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['cvd_simulation_Machado2009_plot']


def cvd_simulation_Machado2009_plot(RGB,
                                    deficiency='Protanomaly',
                                    severity=0.5,
                                    M_a=None,
                                    **kwargs):
    """
    Performs colour vision deficiency simulation on given *RGB* colourspace
    array using *Machado et alii (2009)* model.

    Parameters
    ----------
    RGB : array_like
        *RGB* colourspace array.
    deficiency : unicode, optional
        {'Protanomaly', 'Deuteranomaly', 'Tritanomaly'}
        Colour blindness / vision deficiency type.
    severity : numeric, optional
        Severity of the colour vision deficiency in domain [0, 1].
    M_a : array_like, optional
        Anomalous trichromacy matrix to use instead of Machado (2010)
        pre-computed matrix.

    Other Parameters
    ----------------
    \**kwargs : dict, optional
        {:func:`colour.plotting.render`},
        Please refer to the documentation of the previously listed definition.

    Notes
    -----
    -  Input *RGB* array is expected to be linearly encoded.

    Returns
    -------
    Figure
        Current figure or None.

    Examples
    --------
    >>> import numpy as np
    >>> RGB = np.random.rand(32, 32, 3)
    >>> cvd_simulation_Machado2009_plot(RGB)  # doctest: +SKIP

    .. image:: ../_static/Plotting_CVD_Simulation_Machado2009_Plot.png
        :align: center
        :alt: cvd_simulation_Machado2009_plot
    """

    if M_a is None:
        M_a = cvd_matrix_Machado2009(deficiency, severity)

    label = 'Deficiency: {0} - Severity: {1}'.format(deficiency, severity)

    settings = {'label': None if M_a is None else label}
    settings.update(kwargs)

    return image_plot(
        DEFAULT_PLOTTING_SETTINGS.colourspace.encoding_cctf(
            dot_vector(M_a, RGB)), **settings)
