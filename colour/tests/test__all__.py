"""Define the unit tests for package public API lists."""

from __future__ import annotations

from importlib import import_module

import pytest

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestPackageAll",
]


class TestPackageAll:
    """Define the package public API list unit tests methods."""

    @pytest.mark.parametrize(
        ("package_name", "submodule_name"),
        [
            ("colour.blindness", "colour.blindness.datasets"),
            ("colour.corresponding", "colour.corresponding.datasets"),
            ("colour.io", "colour.io.luts"),
            ("colour.plotting", "colour.plotting.datasets"),
            ("colour.quality", "colour.quality.datasets"),
            ("colour.recovery", "colour.recovery.datasets"),
            ("colour.volume", "colour.volume.datasets"),
        ],
    )
    def test_package_all_does_not_mutate_submodule_all(
        self,
        package_name: str,
        submodule_name: str,
    ) -> None:
        """Test that package ``__all__`` does not mutate submodule ``__all__``."""

        package = import_module(package_name)
        submodule = import_module(submodule_name)

        assert package.__all__ is not submodule.__all__
        assert set(submodule.__all__).issubset(package.__all__)
        assert all(hasattr(submodule, name) for name in submodule.__all__)
