#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Cameras Spectral Sensitivities
==============================

Defines spectral power distributions classes for the dataset from
:mod:`colour.characterisation.dataset.cameras` module:

-   :class:`RGB_SpectralSensitivities`: Implements support for a camera
    *RGB* spectral sensitivities.

See Also
--------
`Cameras Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/characterisation/cameras.ipynb>`_
"""

from __future__ import division, unicode_literals

from colour.colorimetry import TriSpectralPowerDistribution

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['RGB_SpectralSensitivities']


class RGB_SpectralSensitivities(TriSpectralPowerDistribution):
    """
    Implements support for a camera *RGB* spectral sensitivities.

    Parameters
    ----------
    name : unicode
        Camera name.
    data : dict
        Camera *RGB* spectral sensitivities data.

    Attributes
    ----------
    red
    green
    blue
    """

    def __init__(self, name, data):
        TriSpectralPowerDistribution.__init__(
            self,
            name,
            data,
            mapping={'x': 'red', 'y': 'green', 'z': 'blue'},
            labels={'x': 'R', 'y': 'G', 'z': 'B'},
            title=name)

    @property
    def red(self):
        """
        Property for **self.x** attribute.

        Returns
        -------
        SpectralPowerDistribution
            self.x

        Warning
        -------
        :attr:`RGB_SpectralSensitivities.red` is read only.
        """

        return self.x

    @red.setter
    def red(self, value):
        """
        Setter for **self.x** attribute.

        Parameters
        ----------
        value : object
            Attribute value.
        """

        raise AttributeError('"{0}" attribute is read only!'.format('red'))

    @property
    def green(self):
        """
        Property for **self.y** attribute.

        Returns
        -------
        SpectralPowerDistribution
            self.y

        Warning
        -------
        :attr:`RGB_SpectralSensitivities.green` is read only.
        """

        return self.y

    @green.setter
    def green(self, value):
        """
        Setter for **self.y** attribute.

        Parameters
        ----------
        value : object
            Attribute value.
        """

        raise AttributeError('"{0}" attribute is read only!'.format('green'))

    @property
    def blue(self):
        """
        Property for **self.z** attribute.

        Returns
        -------
        SpectralPowerDistribution
            self.z

        Warning
        -------
        :attr:`RGB_SpectralSensitivities.blue` is read only.
        """

        return self.z

    @blue.setter
    def blue(self, value):
        """
        Setter for **self.z** attribute.

        Parameters
        ----------

        value : object
            Attribute value.
        """

        raise AttributeError('"{0}" attribute is read only!'.format('blue'))
