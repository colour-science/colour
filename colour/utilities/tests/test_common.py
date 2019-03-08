# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.utilities.common` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest
from collections import OrderedDict

from colour.utilities import (
    batch, is_iterable, is_string, is_numeric, is_integer, is_sibling,
    filter_kwargs, filter_mapping, first_item, get_domain_range_scale,
    set_domain_range_scale, domain_range_scale, to_domain_1, to_domain_10,
    to_domain_100, to_domain_int, to_domain_degrees, from_range_1,
    from_range_10, from_range_100, from_range_int, from_range_degrees)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'TestBatch', 'TestIsIterable', 'TestIsString', 'TestIsNumeric',
    'TestIsInteger', 'TestIsSibling', 'TestFilterKwargs', 'TestFilterMapping',
    'TestFirstItem'
]


class TestBatch(unittest.TestCase):
    """
    Defines :func:`colour.utilities.common.batch` definition unit tests
    methods.
    """

    def test_batch(self):
        """
        Tests :func:`colour.utilities.common.batch` definition.
        """

        self.assertListEqual(
            list(batch(tuple(range(10)))),
            [(0, 1, 2), (3, 4, 5), (6, 7, 8), (9,)])  # yapf: disable

        self.assertListEqual(
            list(batch(tuple(range(10)), 5)),
            [(0, 1, 2, 3, 4), (5, 6, 7, 8, 9)])  # yapf: disable

        self.assertListEqual(
            list(batch(tuple(range(10)), 1)),
            [(0,), (1,), (2,), (3,), (4,),
             (5,), (6,), (7,), (8,), (9,)])  # yapf: disable


class TestIsIterable(unittest.TestCase):
    """
    Defines :func:`colour.utilities.common.is_iterable` definition unit tests
    methods.
    """

    def test_is_iterable(self):
        """
        Tests :func:`colour.utilities.common.is_iterable` definition.
        """

        self.assertTrue(is_iterable(''))

        self.assertTrue(is_iterable(()))

        self.assertTrue(is_iterable([]))

        self.assertTrue(is_iterable(dict()))

        self.assertTrue(is_iterable(set()))

        self.assertTrue(is_iterable(np.array([])))

        self.assertFalse(is_iterable(1))

        self.assertFalse(is_iterable(2))

        generator = (a for a in range(10))
        self.assertTrue(is_iterable(generator))
        self.assertEqual(len(list(generator)), 10)


class TestIsString(unittest.TestCase):
    """
    Defines :func:`colour.utilities.common.is_string` definition unit tests
    methods.
    """

    def test_is_string(self):
        """
        Tests :func:`colour.utilities.common.is_string` definition.
        """

        self.assertTrue(is_string(str('Hello World!')))

        self.assertTrue(is_string('Hello World!'))

        self.assertTrue(is_string(r'Hello World!'))

        self.assertFalse(is_string(1))

        self.assertFalse(is_string([1]))

        self.assertFalse(is_string({1: None}))


class TestIsNumeric(unittest.TestCase):
    """
    Defines :func:`colour.utilities.common.is_numeric` definition unit tests
    methods.
    """

    def test_is_numeric(self):
        """
        Tests :func:`colour.utilities.common.is_numeric` definition.
        """

        self.assertTrue(is_numeric(1))

        self.assertTrue(is_numeric(1))

        self.assertTrue(is_numeric(complex(1)))

        self.assertFalse(is_numeric((1, )))

        self.assertFalse(is_numeric([1]))

        self.assertFalse(is_numeric('1'))


class TestIsInteger(unittest.TestCase):
    """
    Defines :func:`colour.utilities.common.is_integer` definition units
    tests methods.
    """

    def test_is_integer(self):
        """
        Tests :func:`colour.utilities.common.is_integer` definition.
        """

        self.assertTrue(is_integer(1))

        self.assertTrue(is_integer(1.001))

        self.assertFalse(is_integer(1.01))


class TestIsSibling(unittest.TestCase):
    """
    Defines :func:`colour.utilities.common.is_sibling` definition units tests
    methods.
    """

    def test_is_sibling(self):
        """
        Tests :func:`colour.utilities.common.is_sibling` definition.
        """

        class Element(object):
            """
            :func:`is_sibling` unit tests :class:`Element` class.
            """

            def __init__(self, name):
                self.name = name

        class NotElement(object):
            """
            :func:`is_sibling` unit tests :class:`NotElement` class.
            """

            def __init__(self, name):
                self.name = name

        mapping = {
            'Element A': Element('A'),
            'Element B': Element('B'),
            'Element C': Element('C'),
        }

        self.assertTrue(is_sibling(Element('D'), mapping))

        self.assertFalse(is_sibling(NotElement('Not D'), mapping))


class TestFilterKwargs(unittest.TestCase):
    """
    Defines :func:`colour.utilities.common.filter_kwargs` definition units
    tests methods.
    """

    def test_filter_kwargs(self):
        """
        Tests :func:`colour.utilities.common.filter_kwargs` definition.
        """

        def fn_a(a):
            """
            :func:`filter_kwargs` unit tests :func:`fn_a` definition.
            """
            return a

        def fn_b(a, b=0):
            """
            :func:`filter_kwargs` unit tests :func:`fn_b` definition.
            """

            return a, b

        def fn_c(a, b=0, c=0):
            """
            :func:`filter_kwargs` unit tests :func:`fn_c` definition.
            """

            return a, b, c

        self.assertEqual(1, fn_a(1, **filter_kwargs(fn_a, b=2, c=3)))

        self.assertTupleEqual((1, 2), fn_b(1, **filter_kwargs(fn_b, b=2, c=3)))

        self.assertTupleEqual((1, 2, 3),
                              fn_c(1, **filter_kwargs(fn_c, b=2, c=3)))


class TestFilterMapping(unittest.TestCase):
    """
    Defines :func:`colour.utilities.common.filter_mapping` definition units
    tests methods.
    """

    def test_filter_mapping(self):
        """
        Tests :func:`colour.utilities.common.filter_mapping` definition.
        """

        class Element(object):
            """
            :func:`filter_mapping` unit tests :class:`Element` class.
            """

            def __init__(self, name):
                self.name = name

        mapping = {
            'Element A': Element('A'),
            'Element B': Element('B'),
            'Element C': Element('C'),
            'Not Element C': Element('Not C'),
        }

        self.assertListEqual(
            sorted(filter_mapping(mapping, '\\w+\\s+A')), ['Element A'])

        self.assertListEqual(
            sorted(filter_mapping(mapping, 'Element.*')), [
                'Element A',
                'Element B',
                'Element C',
            ])

        self.assertListEqual(
            sorted(filter_mapping(mapping, '^Element.*')), [
                'Element A',
                'Element B',
                'Element C',
            ])

        self.assertListEqual(
            sorted(filter_mapping(mapping, '^Element.*', False)), [
                'Element A',
                'Element B',
                'Element C',
            ])

        self.assertListEqual(
            sorted(filter_mapping(mapping, ['.*A', '.*B'])), [
                'Element A',
                'Element B',
            ])

        self.assertIsInstance(
            filter_mapping(mapping, '^Element.*', False), type(mapping))

        self.assertIsInstance(
            filter_mapping(OrderedDict(mapping), '^Element.*', False),
            OrderedDict)


class TestFirstItem(unittest.TestCase):
    """
    Defines :func:`colour.utilities.common.first_item` definition units
    tests methods.
    """

    def test_first_item(self):
        """
        Tests :func:`colour.utilities.common.first_item` definition.
        """

        self.assertEqual(first_item(range(10)), 0)

        dictionary = OrderedDict([(0, 'a'), (1, 'b'), (2, 'c')])
        self.assertEqual(first_item(dictionary.items()), (0, 'a'))

        self.assertEqual(first_item(dictionary.values()), 'a')


class TestGetDomainRangeScale(unittest.TestCase):
    """
    Defines :func:`colour.utilities.common.get_domain_range_scale` definition
    units tests methods.
    """

    def test_get_domain_range_scale(self):
        """
        Tests :func:`colour.utilities.common.get_domain_range_scale`
        definition.
        """

        with domain_range_scale('Reference'):
            self.assertEqual(get_domain_range_scale(), 'reference')

        with domain_range_scale('1'):
            self.assertEqual(get_domain_range_scale(), '1')

        with domain_range_scale('100'):
            self.assertEqual(get_domain_range_scale(), '100')


class TestSetDomainRangeScale(unittest.TestCase):
    """
    Defines :func:`colour.utilities.common.set_domain_range_scale` definition
    units tests methods.
    """

    def test_set_domain_range_scale(self):
        """
        Tests :func:`colour.utilities.common.set_domain_range_scale`
        definition.
        """

        with domain_range_scale('Reference'):
            set_domain_range_scale('1')
            self.assertEqual(get_domain_range_scale(), '1')

        with domain_range_scale('Reference'):
            set_domain_range_scale('100')
            self.assertEqual(get_domain_range_scale(), '100')

        with domain_range_scale('1'):
            set_domain_range_scale('Reference')
            self.assertEqual(get_domain_range_scale(), 'reference')

        self.assertRaises(AssertionError,
                          lambda: set_domain_range_scale('Invalid'))


class TestDomainRangeScale(unittest.TestCase):
    """
    Defines :func:`colour.utilities.common.domain_range_scale` definition
    units tests methods.
    """

    def test_domain_range_scale(self):
        """
        Tests :func:`colour.utilities.common.domain_range_scale`
        definition.
        """

        self.assertEqual(get_domain_range_scale(), 'reference')

        with domain_range_scale('Reference'):
            self.assertEqual(get_domain_range_scale(), 'reference')

        self.assertEqual(get_domain_range_scale(), 'reference')

        with domain_range_scale('1'):
            self.assertEqual(get_domain_range_scale(), '1')

        self.assertEqual(get_domain_range_scale(), 'reference')

        with domain_range_scale('100'):
            self.assertEqual(get_domain_range_scale(), '100')

        self.assertEqual(get_domain_range_scale(), 'reference')


class TestToDomain1(unittest.TestCase):
    """
    Defines :func:`colour.utilities.common.to_domain_1` definition units
    tests methods.
    """

    def test_to_domain_1(self):
        """
        Tests :func:`colour.utilities.common.to_domain_1` definition.
        """

        with domain_range_scale('Reference'):
            self.assertEqual(to_domain_1(1), 1)

        with domain_range_scale('1'):
            self.assertEqual(to_domain_1(1), 1)

        with domain_range_scale('100'):
            self.assertEqual(to_domain_1(1), 0.01)

        with domain_range_scale('100'):
            self.assertEqual(to_domain_1(1, np.pi), 1 / np.pi)

        with domain_range_scale('100'):
            self.assertEqual(
                to_domain_1(1, dtype=np.float16).dtype, np.float16)


class TestToDomain10(unittest.TestCase):
    """
    Defines :func:`colour.utilities.common.to_domain_10` definition units
    tests methods.
    """

    def test_to_domain_10(self):
        """
        Tests :func:`colour.utilities.common.to_domain_10` definition.
        """

        with domain_range_scale('Reference'):
            self.assertEqual(to_domain_10(1), 1)

        with domain_range_scale('1'):
            self.assertEqual(to_domain_10(1), 10)

        with domain_range_scale('100'):
            self.assertEqual(to_domain_10(1), 0.1)

        with domain_range_scale('100'):
            self.assertEqual(to_domain_10(1, np.pi), 1 / np.pi)

        with domain_range_scale('100'):
            self.assertEqual(
                to_domain_10(1, dtype=np.float16).dtype, np.float16)


class TestToDomain100(unittest.TestCase):
    """
    Defines :func:`colour.utilities.common.to_domain_100` definition units
    tests methods.
    """

    def test_to_domain_100(self):
        """
        Tests :func:`colour.utilities.common.to_domain_100` definition.
        """

        with domain_range_scale('Reference'):
            self.assertEqual(to_domain_100(1), 1)

        with domain_range_scale('1'):
            self.assertEqual(to_domain_100(1), 100)

        with domain_range_scale('100'):
            self.assertEqual(to_domain_100(1), 1)

        with domain_range_scale('1'):
            self.assertEqual(to_domain_100(1, np.pi), np.pi)

        with domain_range_scale('100'):
            self.assertEqual(
                to_domain_100(1, dtype=np.float16).dtype, np.float16)


class TestToDomainDegrees(unittest.TestCase):
    """
    Defines :func:`colour.utilities.common.to_domain_degrees` definition units
    tests methods.
    """

    def test_to_domain_degrees(self):
        """
        Tests :func:`colour.utilities.common.to_domain_degrees` definition.
        """

        with domain_range_scale('Reference'):
            self.assertEqual(to_domain_degrees(1), 1)

        with domain_range_scale('1'):
            self.assertEqual(to_domain_degrees(1), 360)

        with domain_range_scale('100'):
            self.assertEqual(to_domain_degrees(1), 3.6)

        with domain_range_scale('100'):
            self.assertEqual(to_domain_degrees(1, np.pi), np.pi / 100)

        with domain_range_scale('100'):
            self.assertEqual(
                to_domain_degrees(1, dtype=np.float16).dtype, np.float16)


class TestToDomainInt(unittest.TestCase):
    """
    Defines :func:`colour.utilities.common.to_domain_int` definition units
    tests methods.
    """

    def test_to_domain_int(self):
        """
        Tests :func:`colour.utilities.common.to_domain_int` definition.
        """

        with domain_range_scale('Reference'):
            self.assertEqual(to_domain_int(1), 1)

        with domain_range_scale('1'):
            self.assertEqual(to_domain_int(1), 255)

        with domain_range_scale('100'):
            self.assertEqual(to_domain_int(1), 2.55)

        with domain_range_scale('100'):
            self.assertEqual(to_domain_int(1, 10), 10.23)

        with domain_range_scale('100'):
            self.assertEqual(
                to_domain_int(1, dtype=np.float16).dtype, np.float16)


class TestFromRange1(unittest.TestCase):
    """
    Defines :func:`colour.utilities.common.from_range_1` definition units
    tests methods.
    """

    def test_from_range_1(self):
        """
        Tests :func:`colour.utilities.common.from_range_1` definition.
        """

        with domain_range_scale('Reference'):
            self.assertEqual(from_range_1(1), 1)

        with domain_range_scale('1'):
            self.assertEqual(from_range_1(1), 1)

        with domain_range_scale('100'):
            self.assertEqual(from_range_1(1), 100)

        with domain_range_scale('100'):
            self.assertEqual(from_range_1(1, np.pi), 1 * np.pi)


class TestFromRange10(unittest.TestCase):
    """
    Defines :func:`colour.utilities.common.from_range_10` definition units
    tests methods.
    """

    def test_from_range_10(self):
        """
        Tests :func:`colour.utilities.common.from_range_10` definition.
        """

        with domain_range_scale('Reference'):
            self.assertEqual(from_range_10(1), 1)

        with domain_range_scale('1'):
            self.assertEqual(from_range_10(1), 0.1)

        with domain_range_scale('100'):
            self.assertEqual(from_range_10(1), 10)

        with domain_range_scale('100'):
            self.assertEqual(from_range_10(1, np.pi), 1 * np.pi)


class TestFromRange100(unittest.TestCase):
    """
    Defines :func:`colour.utilities.common.from_range_100` definition units
    tests methods.
    """

    def test_from_range_100(self):
        """
        Tests :func:`colour.utilities.common.from_range_100` definition.
        """

        with domain_range_scale('Reference'):
            self.assertEqual(from_range_100(1), 1)

        with domain_range_scale('1'):
            self.assertEqual(from_range_100(1), 0.01)

        with domain_range_scale('100'):
            self.assertEqual(from_range_100(1), 1)

        with domain_range_scale('1'):
            self.assertEqual(from_range_100(1, np.pi), 1 / np.pi)


class TestFromRangeDegrees(unittest.TestCase):
    """
    Defines :func:`colour.utilities.common.from_range_degrees` definition units
    tests methods.
    """

    def test_from_range_degrees(self):
        """
        Tests :func:`colour.utilities.common.from_range_degrees` definition.
        """

        with domain_range_scale('Reference'):
            self.assertEqual(from_range_degrees(1), 1)

        with domain_range_scale('1'):
            self.assertEqual(from_range_degrees(1), 1 / 360)

        with domain_range_scale('100'):
            self.assertEqual(from_range_degrees(1), 1 / 3.6)

        with domain_range_scale('100'):
            self.assertEqual(from_range_degrees(1, np.pi), 1 / (np.pi / 100))


class TestFromRangeInt(unittest.TestCase):
    """
    Defines :func:`colour.utilities.common.from_range_int` definition units
    tests methods.
    """

    def test_from_range_int(self):
        """
        Tests :func:`colour.utilities.common.from_range_int` definition.
        """

        with domain_range_scale('Reference'):
            self.assertEqual(from_range_int(1), 1)

        with domain_range_scale('1'):
            self.assertEqual(from_range_int(1), 1 / 255)

        with domain_range_scale('100'):
            self.assertEqual(from_range_int(1), 1 / 2.55)

        with domain_range_scale('100'):
            self.assertEqual(from_range_int(1, 10), 1 / (1023 / 100))

        with domain_range_scale('100'):
            self.assertEqual(
                from_range_int(1, dtype=np.float16).dtype, np.float16)


if __name__ == '__main__':
    unittest.main()
