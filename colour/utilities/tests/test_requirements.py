"""Define the unit tests for the :mod:`colour.utilities.requirements` module."""

from __future__ import annotations

import sys
from unittest import mock

import pytest

from colour.utilities import (
    is_array_api_compat_installed,
    is_array_api_extra_installed,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestIsArrayApiCompatInstalled",
    "TestIsArrayApiExtraInstalled",
]


class TestIsArrayApiCompatInstalled:
    """
    Define :func:`colour.utilities.is_array_api_compat_installed` definition
    unit tests methods.
    """

    def test_is_array_api_compat_installed(self) -> None:
        """
        Test :func:`colour.utilities.is_array_api_compat_installed`
        definition.
        """

        assert is_array_api_compat_installed()

        with mock.patch.dict(sys.modules, {"array_api_compat": None}):
            assert not is_array_api_compat_installed()

            with pytest.raises(ImportError):
                is_array_api_compat_installed(raise_exception=True)


class TestIsArrayApiExtraInstalled:
    """
    Define :func:`colour.utilities.is_array_api_extra_installed` definition
    unit tests methods.
    """

    def test_is_array_api_extra_installed(self) -> None:
        """
        Test :func:`colour.utilities.is_array_api_extra_installed`
        definition.
        """

        assert is_array_api_extra_installed()

        with mock.patch.dict(sys.modules, {"array_api_extra": None}):
            assert not is_array_api_extra_installed()

            with pytest.raises(ImportError):
                is_array_api_extra_installed(raise_exception=True)
