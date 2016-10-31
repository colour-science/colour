#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.metadata.common` module.
"""

from __future__ import division, unicode_literals

import unittest

import colour
from colour.metadata.common import (
    parse_parameters_field_metadata, parse_returns_field_metadata,
    parse_notes_field_metadata, set_metadata, filter_metadata_registry)
from colour.metadata.common import NotesMetadata, ParameterMetadata

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'TestParseParametersFieldMetadata', 'TestParseReturnsFieldMetadata',
    'TestParseNotesFieldMetadata', 'TestSetMetadata',
    'TestFilterMetadataRegistry'
]


class TestParseParametersFieldMetadata(unittest.TestCase):
    """
    Defines :func:`colour.metadata.common.parse_parameters_field_metadata`
    definition units tests methods.
    """

    def test_parse_parameters_field_metadata(self):
        """
        Tests :func:`colour.metadata.common.parse_parameters_field_metadata`
        definition.
        """

        field = (['Lstar', u'numeric or array_like'],
                 "metadata : {'type': 'Lightness', 'symbol': 'L^\\star', \
'extent': 100} *Lightness* :math:`L^\\star`")
        self.assertTupleEqual(
            tuple(parse_parameters_field_metadata(field)), ('Lightness',
                                                            'L^\\star', 100))


class TestParseReturnsFieldMetadata(unittest.TestCase):
    """
    Defines :func:`colour.metadata.common.parse_returns_field_metadata`
    definition units tests methods.
    """

    def test_parse_returns_field_metadata(self):
        """
        Tests :func:`colour.metadata.common.parse_returns_field_metadata`
        definition.
        """

        field = (['Lstar', u'numeric or array_like'],
                 "metadata : {'type': 'Lightness', 'symbol': 'L^\\star', \
'extent': 100} *Lightness* :math:`L^\\star`")
        self.assertTupleEqual(
            tuple(parse_returns_field_metadata(field)), ('Lightness',
                                                         'L^\\star', 100))


class TestParseNotesFieldMetadata(unittest.TestCase):
    """
    Defines :func:`colour.metadata.common.parse_notes_field_metadata`
    definition units tests methods.
    """

    def test_parse_notes_field_metadata(self):
        """
        Tests :func:`colour.metadata.common.parse_notes_field_metadata`
        definition.
        """

        field = ("metadata : {'classifier': 'Lightness Conversion Function', "
                 "'method_name': 'Wyszecki 1963', 'method_strict_name':",
                 "'Wyszecki (1963)'}")
        self.assertTupleEqual(
            tuple(parse_notes_field_metadata(field)),
            ('Lightness Conversion Function', 'Wyszecki 1963',
             'Wyszecki (1963)'))


class TestSetMetadata(unittest.TestCase):
    """
    Defines :func:`colour.metadata.common.set_metadata` definition units
    tests methods.
    """

    def test_set_metadata(self):
        """
        Tests :func:`colour.metadata.common.set_metadata` definition.
        """

        def fn_a(argument_1):
            """
            Summary of docstring.

            Description of docstring.

            Parameters
            ----------
            argument_1 : object
                metadata : {'type': 'type', 'symbol': 'symbol',
                    'extent': 'extent'}
                Description of `argument_1`.

            Returns
            -------
            object
                metadata : {'type': 'type', 'symbol': 'symbol',
                    'extent': 'extent'}
                Description of `object`.

            Notes
            -----
            -   metadata : {'classifier': 'classifier', 'method_name':
                'method_name', 'method_strict_name': 'method_strict_name'}
            """

            return argument_1

        set_metadata(fn_a)

        self.assertTrue(hasattr(fn_a, '__metadata__'))
        self.assertDictEqual(
            dict(fn_a.__metadata__), {
                'returns': [
                    ParameterMetadata(
                        type='type', symbol='symbol', extent='extent')
                ],
                'notes': [
                    NotesMetadata(
                        classifier='classifier',
                        method_name='method_name',
                        method_strict_name='method_strict_name')
                ],
                'parameters': [
                    ParameterMetadata(
                        type='type', symbol='symbol', extent='extent')
                ]
            })


class TestFilterMetadataRegistry(unittest.TestCase):
    """
    Defines :func:`colour.metadata.common.filter_metadata_registry`
    definition units tests methods.
    """

    def test_filter_metadata_registry(self):
        """
        Tests :func:`colour.metadata.common.filter_metadata_registry`
        definition.
        """

        self.assertSetEqual(
            set(
                filter_metadata_registry(
                    'Luminance', categories='parameters', attributes='type')),
            set([
                colour.colorimetry.lightness_Glasser1958,
                colour.colorimetry.lightness_Wyszecki1963,
                colour.colorimetry.lightness_CIE1976,
                colour.colorimetry.lightness_Fairchild2010,
                colour.colorimetry.lightness_Fairchild2011
            ]))

        self.assertSetEqual(
            set(
                filter_metadata_registry(
                    'Luminance',
                    categories='parameters',
                    attributes='type',
                    any_parameter=True)),
            set([
                colour.colorimetry.lightness_Glasser1958,
                colour.colorimetry.lightness_Wyszecki1963,
                colour.colorimetry.lightness_CIE1976,
                colour.colorimetry.lightness_Fairchild2010,
                colour.colorimetry.lightness_Fairchild2011,
                colour.colorimetry.luminance_CIE1976
            ]))

        self.assertSetEqual(
            set(
                filter_metadata_registry(
                    'Luminance', categories='returns', attributes='type')),
            set([
                colour.colorimetry.luminance_ASTMD153508,
                colour.colorimetry.luminance_Newhall1943,
                colour.colorimetry.luminance_CIE1976,
                colour.colorimetry.luminance_Fairchild2010,
                colour.colorimetry.luminance_Fairchild2011
            ]))

        self.assertSetEqual(
            set(
                filter_metadata_registry(
                    'Luminance',
                    categories=('parameters', 'returns'),
                    attributes='type')),
            set([
                colour.colorimetry.lightness_Glasser1958,
                colour.colorimetry.lightness_Wyszecki1963,
                colour.colorimetry.lightness_CIE1976,
                colour.colorimetry.lightness_Fairchild2010,
                colour.colorimetry.lightness_Fairchild2011,
                colour.colorimetry.luminance_ASTMD153508,
                colour.colorimetry.luminance_Newhall1943,
                colour.colorimetry.luminance_CIE1976,
                colour.colorimetry.luminance_Fairchild2010,
                colour.colorimetry.luminance_Fairchild2011
            ]))

        self.assertSetEqual(
            set(
                filter_metadata_registry(
                    'Luminance',
                    categories=('parameters', 'returns'),
                    attributes='type',
                    any_parameter=True)),
            set([
                colour.colorimetry.lightness_Glasser1958,
                colour.colorimetry.lightness_Wyszecki1963,
                colour.colorimetry.lightness_CIE1976,
                colour.colorimetry.lightness_Fairchild2010,
                colour.colorimetry.lightness_Fairchild2011,
                colour.colorimetry.luminance_CIE1976,
                colour.colorimetry.luminance_ASTMD153508,
                colour.colorimetry.luminance_Newhall1943,
                colour.colorimetry.luminance_CIE1976,
                colour.colorimetry.luminance_Fairchild2010,
                colour.colorimetry.luminance_Fairchild2011
            ]))

        self.assertSetEqual(
            set(
                filter_metadata_registry(
                    'CIE 1976', categories='notes', attributes='method_name')),
            set([
                colour.colorimetry.lightness_CIE1976,
                colour.colorimetry.luminance_CIE1976,
                colour.models.Lab_to_LCHab,
                colour.models.Lab_to_XYZ,
                colour.models.LCHab_to_Lab,
                colour.models.LCHuv_to_Luv,
                colour.models.Luv_to_LCHuv,
                colour.models.Luv_to_uv,
                colour.models.Luv_to_XYZ,
                colour.models.Luv_uv_to_xy,
                colour.models.XYZ_to_Lab,
                colour.models.XYZ_to_Luv,
            ]))


if __name__ == '__main__':
    unittest.main()
