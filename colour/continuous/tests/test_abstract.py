"""Defines the unit tests for the :mod:`colour.continuous.abstract` module."""

import unittest

from colour.continuous import AbstractContinuousFunction

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestAbstractContinuousFunction",
]


class TestAbstractContinuousFunction(unittest.TestCase):
    """
    Define :class:`colour.continuous.abstract.AbstractContinuousFunction`
    class unit tests methods.
    """

    def test_required_attributes(self):
        """Test the presence of required attributes."""

        required_attributes = (
            "name",
            "domain",
            "range",
            "interpolator",
            "interpolator_kwargs",
            "extrapolator",
            "extrapolator_kwargs",
            "function",
        )

        for attribute in required_attributes:
            self.assertIn(attribute, dir(AbstractContinuousFunction))

    def test_required_methods(self):
        """Test the presence of required methods."""

        required_methods = (
            "__init__",
            "__str__",
            "__repr__",
            "__hash__",
            "__getitem__",
            "__setitem__",
            "__contains__",
            "__len__",
            "__eq__",
            "__ne__",
            "__iadd__",
            "__add__",
            "__isub__",
            "__sub__",
            "__imul__",
            "__mul__",
            "__idiv__",
            "__div__",
            "__ipow__",
            "__pow__",
            "arithmetical_operation",
            "fill_nan",
            "domain_distance",
            "is_uniform",
            "copy",
        )

        for method in required_methods:
            self.assertIn(method, dir(AbstractContinuousFunction))


if __name__ == "__main__":
    unittest.main()
