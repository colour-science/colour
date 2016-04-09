#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Colourspace Model
=================

Defines the :class:`ColourspaceModel` class providing support for colourspace
models.
"""

from __future__ import division, unicode_literals

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['ColourspaceModel']


class ColourspaceModel(object):
    """
    Implements support for colourspace models.

    Parameters
    ----------
    name : unicode
        Colourspace model name.
    mapping : dict
        Colourspace model axis labels mapping.
    encoding_function : callable
        Colourspace model encoding function converting from *CIE XYZ*
        tristimulus values to colourspace model.
    decoding_function : callable
        Colourspace model encoding function converting colourspace model to
        *CIE XYZ* tristimulus values.
    title : unicode, optional
        Colourspace model title for figures.
    labels : dict, optional
        Colourspace model axis labels mapping for figures.

    Attributes
    ----------
    name
    mapping
    encoding_function
    decoding_function
    title
    labels
    """

    def __init__(self,
                 name,
                 mapping,
                 encoding_function,
                 decoding_function,
                 title=None,
                 labels=None):
        self.__name = None
        self.name = name
        self.__encoding_function = None
        self.encoding_function = encoding_function
        self.__decoding_function = None
        self.decoding_function = decoding_function
        self.__mapping = None
        self.mapping = mapping
        self.__title = None
        self.title = title
        self.__labels = None
        self.labels = labels

    @property
    def name(self):
        """
        Property for **self.__name** private attribute.

        Returns
        -------
        unicode
            self.__name.
        """

        return self.__name

    @name.setter
    def name(self, value):
        """
        Setter for **self.__name** private attribute.

        Parameters
        ----------
        value : unicode
            Attribute value.
        """

        if value is not None:
            assert isinstance(value, basestring), (  # noqa
                ('"{0}" attribute: "{1}" is not a '
                 '"basestring" instance!').format('name', value))
        self.__name = value

    @property
    def mapping(self):
        """
        Property for **self.__mapping** private attribute.

        Returns
        -------
        dict
            self.__mapping.
        """

        return self.__mapping

    @mapping.setter
    def mapping(self, value):
        """
        Setter for **self.__mapping** private attribute.

        Parameters
        ----------
        value : dict
            Attribute value.
        """

        if value is not None:
            assert isinstance(value, dict), (
                '"{0}" attribute: "{1}" is not a "dict" instance!'.format(
                    'mapping', value))
            for axis in (0, 1, 2):
                assert axis in value.keys(), (
                    '"{0}" attribute: "{1}" axis label is missing!'.format(
                        'mapping', axis))
        self.__mapping = value

    @property
    def encoding_function(self):
        """
        Property for **self.__encoding_function** private attribute.

        Returns
        -------
        callable
            self.__encoding_function.
        """

        return self.__encoding_function

    @encoding_function.setter
    def encoding_function(self, value):
        """
        Setter for **self.__encoding_function** private attribute.

        Parameters
        ----------
        value : callable
            Attribute value.
        """

        if value is not None:
            assert hasattr(value, '__call__'), (
                '"{0}" attribute: "{1}" is not a "callable"!'.format(
                    'name', value))
        self.__encoding_function = value

    @property
    def decoding_function(self):
        """
        Property for **self.__decoding_function** private attribute.

        Returns
        -------
        callable
            self.__decoding_function.
        """

        return self.__decoding_function

    @decoding_function.setter
    def decoding_function(self, value):
        """
        Setter for **self.__decoding_function** private attribute.

        Parameters
        ----------
        value : callable
            Attribute value.
        """

        if value is not None:
            assert hasattr(value, '__call__'), (
                '"{0}" attribute: "{1}" is not a "callable"!'.format(
                    'name', value))
        self.__decoding_function = value

    @property
    def title(self):
        """
        Property for **self.__title** private attribute.

        Returns
        -------
        unicode
            self.__title.
        """

        if self.__title is not None:
            return self.__title
        else:
            return self.__name

    @title.setter
    def title(self, value):
        """
        Setter for **self.__title** private attribute.

        Parameters
        ----------
        value : unicode
            Attribute value.
        """

        if value is not None:
            assert isinstance(value, basestring), (  # noqa
                ('"{0}" attribute: "{1}" is not a '
                 '"basestring" instance!').format('title', value))
        self.__title = value

    @property
    def labels(self):
        """
        Property for **self.__labels** private attribute.

        Returns
        -------
        dict
            self.__labels.
        """

        if self.__labels is not None:
            return self.__labels
        else:
            return self.__mapping

    @labels.setter
    def labels(self, value):
        """
        Setter for **self.__labels** private attribute.

        Parameters
        ----------
        value : dict
            Attribute value.
        """

        if value is not None:
            assert isinstance(value, dict), (
                '"{0}" attribute: "{1}" is not a "dict" instance!'.format(
                    'labels', value))
            for axis in (0, 1, 2):
                assert axis in value.keys(), (
                    '"{0}" attribute: "{1}" axis label is missing!'.format(
                        'labels', axis))
        self.__labels = value
