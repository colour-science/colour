#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Colour Models Metadata
======================

Defines the objects implementing the colour models metadata system support:

-   :class:``
-   :func:`set_metadata`
"""

from __future__ import division, unicode_literals

from colour.metadata import EntityMetadata, FunctionMetadata
from colour.utilities import is_iterable

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2016 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['ColourModelMetadata',
           'ColourModelFunctionMetadata',
           'EncodingCCTFMetadata',
           'DecodingCCTFMetadata',
           'OETFMetadata',
           'EOTFMetadata']


class ColourModelMetadata(EntityMetadata):
    """
    Defines the metadata class for colour models.

    Parameters
    ----------
    name : unicode
        Colour model name.
    axes : array_like
        Colour model axes names.
    strict_name : unicode, optional
        Colour model strict name, the scientific name for use in diagrams,
        figures, etc...
    strict_axes : unicode, optional
        Colour model strict axes name, the scientific name for use in diagrams,
        figures, etc...

    Attributes
    ----------
    axes
    strict_axes

    Methods
    -------
    __init__
    __str__
    __repr__

    Examples
    --------
    >>> ColourModelMetadata(
    ... 'CIE Lab',
    ... ('L', 'a', 'b'),
    ... '$CIE L^* a^* b^*$',
    ... ('$L^*$', '$a^*$', '$b^*$'))
    ColourModelMetadata('CIE Lab', ('L', 'a', 'b'), \
'$CIE L^* a^* b^*$', ('$L^*$', '$a^*$', '$b^*$'))
    """

    _FAMILY = 'Colour Model'
    """
    Metadata class family.

    _FAMILY : unicode
    """

    def __init__(self, name, axes, strict_name=None, strict_axes=None):
        super(ColourModelMetadata, self).__init__(name, strict_name)

        self._axes = None
        self.axes = axes
        self._strict_axes = None
        self.strict_axes = strict_axes

    @property
    def axes(self):
        """
        Property for **self._axes** private attribute.

        Returns
        -------
        dict
            self._axes.
        """

        return self._axes

    @axes.setter
    def axes(self, value):
        """
        Setter for **self._axes** private attribute.

        Parameters
        ----------
        value : dict
            Attribute value.
        """

        if value is not None:
            assert is_iterable(value), (
                '"{0}" attribute: "{1}" is not an "iterable"!'.format(
                    'axes', value))

        self._axes = value

    @property
    def strict_axes(self):
        """
        Property for **self._strict_axes** private attribute.

        Returns
        -------
        dict
            self._strict_axes.
        """

        if self._strict_axes is not None:
            return self._strict_axes
        else:
            return self._axes

    @strict_axes.setter
    def strict_axes(self, value):
        """
        Setter for **self._strict_axes** private attribute.

        Parameters
        ----------
        value : dict
            Attribute value.
        """

        if value is not None:
            assert is_iterable(value), (
                '"{0}" attribute: "{1}" is not an "iterable"!'.format(
                    'strict_axes', value))

        self._strict_axes = value

    def __str__(self):
        """
        Returns a pretty formatted string representation of the metadata.

        Returns
        -------
        unicode
            Pretty formatted string representation.

        See Also
        --------
        ColourModelMetadata.__repr__

        Notes
        -----
        -   Reimplements the :meth:`Metadata.__str__` method.

        Examples
        --------
        >>> print(ColourModelMetadata(
        ...     'CIE Lab',
        ...     ('L', 'a', 'b'),
        ...     '$CIE L^*a^*b^*$',
        ...     ('$L^*$', '$a^*$', '$b^*$')))
        Colour Model
            Name          : CIE Lab
            Strict name   : $CIE L^*a^*b^*$
            Axes          : ('L', 'a', 'b')
            Strict axes   : ('$L^*$', '$a^*$', '$b^*$')
        """

        text = self.family
        text += '\n    Name          : {0}'.format(self.name)
        text += '\n    Strict name   : {0}'.format(self.strict_name)
        text += '\n    Axes          : {0}'.format(self.axes)
        text += '\n    Strict axes   : {0}'.format(self.strict_axes)

        return text

    def __repr__(self):
        """
        Returns a formatted string representation of the metadata.

        Returns
        -------
        unicode
            Formatted string representation.

        See Also
        --------
        ColourModelMetadata.__repr__

        Notes
        -----
        -   Reimplements the :meth:`object.__repr__` method.

        Examples
        --------
        >>> ColourModelMetadata(
        ...     'CIE Lab',
        ...     ('L', 'a', 'b'),
        ...     '$CIE L^*a^*b^*$',
        ...     ('$L^*$', '$a^*$', '$b^*$'))
        ColourModelMetadata('CIE Lab', ('L', 'a', 'b'), '$CIE L^*a^*b^*$', \
('$L^*$', '$a^*$', '$b^*$'))
        """

        text = '{0}({1}, {2}, {3}, {4})'.format(
            self.__class__.__name__,
            repr(self._name),
            repr(self._axes),
            repr(self._strict_name),
            repr(self._strict_axes))

        return text


class ColourModelFunctionMetadata(FunctionMetadata):
    """
    Defines the metadata class for function converting an input entity into
    an output entity using a given method.

    Parameters
    ----------
    input_colour_model : ColourModelMetadata
        Input colour model metadata.
    output_colour_model : ColourModelMetadata
        Output colour model metadata.
    input_domain : array_like
        Input domain.
    output_range : array_like
        Output range.
    method : unicode, optional
        Method used by the function.
    strict_method : unicode, optional
        Strict method name, the scientific name for use in diagrams,
        figures, etc...
    callable_ : callable, optional
        Callable to store within the metadata.

    Attributes
    ----------
    input_colour_model
    output_colour_model

    Methods
    -------

    Examples
    --------
    >>> ColourModelFunctionMetadata(
    ...     ColourModelMetadata(
    ...         'CIE XYX',
    ...         ('X', 'Y', 'Z'),
    ...         '$CIE XYZ$',
    ...         ('$X$', '$Y$', '$Z')),
    ...     ColourModelMetadata(
    ...         'CIE Lab',
    ...         ('L', 'a', 'b'),
    ...         '$CIE L^*a^*b^*$',
    ...         ('$L^*$', '$a^*$', '$b^*$')),
    ...     (0, 1),
    ...     (0, 1))
    ColourModelFunctionMetadata(\
ColourModelMetadata('CIE XYX', ('X', 'Y', 'Z'), \
'$CIE XYZ$', ('$X$', '$Y$', '$Z')), \
ColourModelMetadata('CIE Lab', ('L', 'a', 'b'), \
'$CIE L^*a^*b^*$', ('$L^*$', '$a^*$', '$b^*$')), \
(0, 1), (0, 1), '', '')
    """

    _FAMILY = 'Colour Model Function'
    """
    Metadata class family.

    _FAMILY : unicode
    """

    def __init__(self,
                 input_colour_model,
                 output_colour_model,
                 input_domain,
                 output_range,
                 method=None,
                 strict_method=None,
                 callable_=None):
        super(ColourModelFunctionMetadata, self).__init__(
            input_colour_model,
            output_colour_model,
            input_domain,
            output_range,
            method,
            strict_method,
            callable_)

    @property
    def input_colour_model(self):
        """
        Property for **self.input_entity** attribute.

        Returns
        -------
        ColourModelMetadata
            self.input_entity.
        """

        return self.input_entity

    @input_colour_model.setter
    def input_colour_model(self, value):
        """
        Setter for **self.input_entity** attribute.

        Parameters
        ----------
        value : ColourModelMetadata
            Attribute value.
        """

        self.input_entity = value

    @property
    def output_colour_model(self):
        """
        Property for **self.output_entity** attribute.

        Returns
        -------
        ColourModelMetadata
            self.output_entity.
        """

        return self.output_entity

    @output_colour_model.setter
    def output_colour_model(self, value):
        """
        Setter for **self.output_entity** attribute.

        Parameters
        ----------
        value : ColourModelMetadata
            Attribute value.
        """

        self.output_entity = value


class EncodingCCTFMetadata(FunctionMetadata):
    """
    Defines the metadata class for encoding colour component transfer
    functions (CCTF).
    """

    _FAMILY = 'Encoding Colour Component Transfer Function'


class DecodingCCTFMetadata(FunctionMetadata):
    """
    Defines the metadata class for decoding colour component transfer
    functions (CCTF).
    """

    _FAMILY = 'Decoding Colour Component Transfer Function'


class OETFMetadata(EncodingCCTFMetadata):
    """
    Defines the metadata class for opto-electrical transfer functions
    (OETF / OECF).
    """

    _FAMILY = 'Opto-Electrical Transfer Function'


class EOTFMetadata(DecodingCCTFMetadata):
    """
    Defines the metadata class for electro-optical transfer functions
    (EOTF / EOCF).
    """

    _FAMILY = 'Electro-Optical Transfer Function'
