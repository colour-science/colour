#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Colour Matching Functions
=========================

Defines colour matching functions classes for the dataset from
:mod:`colour.colorimetry.dataset.cmfs` module:

-   :class:`LMS_ConeFundamentals`: Implements support for the
    Stockman and Sharpe *LMS* cone fundamentals colour matching functions.
-   :class:`RGB_ColourMatchingFunctions`: Implements support for the *CIE RGB*
    colour matching functions.
-   :class:`XYZ_ColourMatchingFunctions`: Implements support for the *CIE*
    Standard Observers *XYZ* colour matching functions.

See Also
--------
`Colour Matching Functions Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/colorimetry/cmfs.ipynb>`_
colour.colorimetry.dataset.cmfs,
colour.colorimetry.spectrum.TriSpectralPowerDistribution
"""

from __future__ import division, unicode_literals

import pprint

from colour.colorimetry import TriSpectralPowerDistribution

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['LMS_ConeFundamentals',
           'RGB_ColourMatchingFunctions',
           'XYZ_ColourMatchingFunctions']


def _format_cmfs(self):
    """
    Helper definition returning the formatted string representation of the
    tri-spectral power distribution instance.

    Parameters
    ----------
    self : object
        Tri-spectral power distribution instance.

    Returns
    -------
    unicode
        Formatted string representation.
    """

    mapping = self.mapping
    data = {mapping['x']: dict(self.x.data),
            mapping['y']: dict(self.y.data),
            mapping['z']: dict(self.z.data)}

    return '{0}(\n    \'{1}\',\n    {2},\n    {3})'.format(
        self.__class__.__name__,
        self.name,
        pprint.pformat(data).replace('\n', '\n    '),
        ('\'{0}\''.format(self.title)
         if self.title is not None else
         self.title))


class LMS_ConeFundamentals(TriSpectralPowerDistribution):
    """
    Implements support for the Stockman and Sharpe *LMS* cone fundamentals
    colour matching functions.

    Parameters
    ----------
    name : unicode
        *LMS* colour matching functions name.
    data : dict
        *LMS* colour matching functions.
    title : unicode, optional
        *LMS* colour matching functions title for figures.

    Attributes
    ----------
    l_bar
    m_bar
    s_bar

    """

    def __init__(self, name, data, title=None):
        TriSpectralPowerDistribution.__init__(self,
                                              name,
                                              data,
                                              mapping={'x': 'l_bar',
                                                       'y': 'm_bar',
                                                       'z': 's_bar'},
                                              labels={'x': '$\\bar{l}$',
                                                      'y': '$\\bar{m}$',
                                                      'z': '$\\bar{s}$'},
                                              title=title)

    @property
    def l_bar(self):
        """
        Property for **self.x** attribute.

        Returns
        -------
        SpectralPowerDistribution
            self.x

        Warning
        -------
        :attr:`LMS_ConeFundamentals.l_bar` is read only.
        """

        return self.x

    @l_bar.setter
    def l_bar(self, value):
        """
        Setter for **self.x** attribute.

        Parameters
        ----------
        value : object
            Attribute value.
        """

        raise AttributeError('"{0}" attribute is read only!'.format('l_bar'))

    @property
    def m_bar(self):
        """
        Property for **self.y** attribute.

        Returns
        -------
        SpectralPowerDistribution
            self.y

        Warning
        -------
        :attr:`LMS_ConeFundamentals.m_bar` is read only.
        """

        return self.y

    @m_bar.setter
    def m_bar(self, value):
        """
        Setter for **self.y** attribute.

        Parameters
        ----------
        value : object
            Attribute value.
        """

        raise AttributeError('"{0}" attribute is read only!'.format('m_bar'))

    @property
    def s_bar(self):
        """
        Property for **self.z** attribute.

        Returns
        -------
        SpectralPowerDistribution
            self.z

        Warning
        -------
        :attr:`LMS_ConeFundamentals.s_bar` is read only.
        """

        return self.z

    @s_bar.setter
    def s_bar(self, value):
        """
        Setter for **self.z** attribute.

        Parameters
        ----------

        value : object
            Attribute value.
        """

        raise AttributeError('"{0}" attribute is read only!'.format('s_bar'))

    def __repr__(self):
        """
        Returns a formatted string representation of the *LMS* cone
        fundamentals colour matching functions.

        Returns
        -------
        unicode
            Formatted string representation.

        Notes
        -----
        -   Reimplements the :meth:`object.__repr__` method.

        Examples
        --------
        >>> l_bar = {510: 49.67, 520: 69.59, 530: 81.73, 540: 88.19}
        >>> m_bar = {510: 90.56, 520: 87.34, 530: 45.76, 540: 23.45}
        >>> s_bar = {510: 12.43, 520: 23.15, 530: 67.98, 540: 90.28}
        >>> data = {'l_bar': l_bar, 'm_bar': m_bar, 's_bar': s_bar}
        >>> LMS_ConeFundamentals(  # doctest: +ELLIPSIS
        ...     'LMS CMFS', data, '*LMS* Cone Fundamentals')
        LMS_ConeFundamentals(
            'LMS CMFS',
            {...'l_bar': \
{510...: 49.67, 520...: 69.59, 530...: 81.73, 540...: 88.19},
             ...'m_bar': \
{510...: 90.56, 520...: 87.34, 530...: 45.76, 540...: 23.45},
             ...'s_bar': \
{510...: 12.43, 520...: 23.15, 530...: 67.98, 540...: 90.28}},
            '*LMS* Cone Fundamentals')
        """

        return _format_cmfs(self)


class RGB_ColourMatchingFunctions(TriSpectralPowerDistribution):
    """
    Implements support for the *CIE RGB* colour matching functions.

    Parameters
    ----------
    name : unicode
        *CIE RGB* colour matching functions name.
    data : dict
        *CIE RGB* colour matching functions.
    title : unicode, optional
        *CIE RGB* colour matching functions title for figures.

    Attributes
    ----------
    r_bar
    g_bar
    b_bar
    """

    def __init__(self, name, data, title=None):
        TriSpectralPowerDistribution.__init__(self,
                                              name,
                                              data,
                                              mapping={'x': 'r_bar',
                                                       'y': 'g_bar',
                                                       'z': 'b_bar'},
                                              labels={'x': '$\\bar{r}$',
                                                      'y': '$\\bar{g}$',
                                                      'z': '$\\bar{b}$'},
                                              title=title)

    @property
    def r_bar(self):
        """
        Property for **self.x** attribute.

        Returns
        -------
        SpectralPowerDistribution
            self.x

        Warning
        -------
        :attr:`RGB_ColourMatchingFunctions.r_bar` is read only.
        """

        return self.x

    @r_bar.setter
    def r_bar(self, value):
        """
        Setter for **self.x** attribute.

        Parameters
        ----------
        value : object
            Attribute value.
        """

        raise AttributeError('"{0}" attribute is read only!'.format('r_bar'))

    @property
    def g_bar(self):
        """
        Property for **self.y** attribute.

        Returns
        -------
        SpectralPowerDistribution
            self.y

        Warning
        -------
        :attr:`RGB_ColourMatchingFunctions.g_bar` is read only.
        """

        return self.y

    @g_bar.setter
    def g_bar(self, value):
        """
        Setter for **self.y** attribute.

        Parameters
        ----------
        value : object
            Attribute value.

        """

        raise AttributeError('"{0}" attribute is read only!'.format('g_bar'))

    @property
    def b_bar(self):
        """
        Property for **self.z** attribute.

        Returns
        -------
        SpectralPowerDistribution
            self.z

        Warning
        -------
        :attr:`RGB_ColourMatchingFunctions.b_bar` is read only.
        """

        return self.z

    @b_bar.setter
    def b_bar(self, value):
        """
        Setter for **self.z** attribute.

        Parameters
        ----------
        value : object
            Attribute value.

        """

        raise AttributeError('"{0}" attribute is read only!'.format('b_bar'))

    def __repr__(self):
        """
        Returns a formatted string representation of the *CIE RGB* colour
        matching functions.

        Returns
        -------
        unicode
            Formatted string representation.

        Notes
        -----
        -   Reimplements the :meth:`object.__repr__` method.

        Examples
        --------
        >>> r_bar = {510: 49.67, 520: 69.59, 530: 81.73, 540: 88.19}
        >>> g_bar = {510: 90.56, 520: 87.34, 530: 45.76, 540: 23.45}
        >>> b_bar = {510: 12.43, 520: 23.15, 530: 67.98, 540: 90.28}
        >>> data = {'r_bar': r_bar, 'g_bar': g_bar, 'b_bar': b_bar}
        >>> RGB_ColourMatchingFunctions(  # doctest: +ELLIPSIS
        ...     'RGB CMFS', data, '*RGB* CMFS')
        RGB_ColourMatchingFunctions(
            'RGB CMFS',
            {...'b_bar': \
{510...: 12.43, 520...: 23.15, 530...: 67.98, 540...: 90.28},
             ...'g_bar': \
{510...: 90.56, 520...: 87.34, 530...: 45.76, 540...: 23.45},
             ...'r_bar': \
{510...: 49.67, 520...: 69.59, 530...: 81.73, 540...: 88.19}},
            '*RGB* CMFS')
        """

        return _format_cmfs(self)


class XYZ_ColourMatchingFunctions(TriSpectralPowerDistribution):
    """
    Implements support for the *CIE* Standard Observers *XYZ* colour matching
    functions.

    Parameters
    ----------
    name : unicode
        *CIE* Standard Observer *XYZ* colour matching functions name.
    data : dict
        *CIE* Standard Observer *XYZ* colour matching functions.
    title : unicode, optional
        *CIE* Standard Observer *XYZ* colour matching functions title for
        figures.

    Attributes
    ----------
    x_bar
    y_bar
    z_bar
    """

    def __init__(self, name, data, title=None):
        TriSpectralPowerDistribution.__init__(self,
                                              name,
                                              data,
                                              mapping={'x': 'x_bar',
                                                       'y': 'y_bar',
                                                       'z': 'z_bar'},
                                              labels={'x': '$\\bar{x}$',
                                                      'y': '$\\bar{y}$',
                                                      'z': '$\\bar{z}$'},
                                              title=title)

    @property
    def x_bar(self):
        """
        Property for **self.x** attribute.

        Returns
        -------
        SpectralPowerDistribution
            self.x

        Warning
        -------
        :attr:`XYZ_ColourMatchingFunctions.x_bar` is read only.
        """

        return self.x

    @x_bar.setter
    def x_bar(self, value):
        """
        Setter for **self.x** attribute.

        Parameters
        ----------
        value : object
            Attribute value.
        """

        raise AttributeError('"{0}" attribute is read only!'.format('x_bar'))

    @property
    def y_bar(self):
        """
        Property for **self.y** attribute.

        Returns
        -------
        SpectralPowerDistribution
            self.y

        Warning
        -------
        :attr:`XYZ_ColourMatchingFunctions.y_bar` is read only.
        """

        return self.y

    @y_bar.setter
    def y_bar(self, value):
        """
        Setter for **self.y** attribute.

        Parameters
        ----------
        value : object
            Attribute value.
        """

        raise AttributeError('"{0}" attribute is read only!'.format('y_bar'))

    @property
    def z_bar(self):
        """
        Property for **self.z** attribute.

        Returns
        -------
        SpectralPowerDistribution
            self.z

        Warning
        -------
        :attr:`XYZ_ColourMatchingFunctions.z_bar` is read only.
        """

        return self.z

    @z_bar.setter
    def z_bar(self, value):
        """
        Setter for **self.z** attribute.

        Parameters
        ----------
        value : object
            Attribute value.
        """

        raise AttributeError('"{0}" attribute is read only!'.format('z_bar'))

    def __repr__(self):
        """
        Returns a formatted string representation of the *XYZ* colour matching
        functions.

        Returns
        -------
        unicode
            Formatted string representation.

        Notes
        -----
        -   Reimplements the :meth:`object.__repr__` method.

        Examples
        --------
        >>> x_bar = {510: 49.67, 520: 69.59, 530: 81.73, 540: 88.19}
        >>> y_bar = {510: 90.56, 520: 87.34, 530: 45.76, 540: 23.45}
        >>> z_bar = {510: 12.43, 520: 23.15, 530: 67.98, 540: 90.28}
        >>> data = {'x_bar': x_bar, 'y_bar': y_bar, 'z_bar': z_bar}
        >>> XYZ_ColourMatchingFunctions(  # doctest: +ELLIPSIS
        ...     'XYZ CMFS', data, '*XYZ* CMFS')
        XYZ_ColourMatchingFunctions(
            'XYZ CMFS',
            {...'x_bar': \
{510...: 49.67, 520...: 69.59, 530...: 81.73, 540...: 88.19},
             ...'y_bar': \
{510...: 90.56, 520...: 87.34, 530...: 45.76, 540...: 23.45},
             ...'z_bar': \
{510...: 12.43, 520...: 23.15, 530...: 67.98, 540...: 90.28}},
            '*XYZ* CMFS')
        """

        return _format_cmfs(self)
