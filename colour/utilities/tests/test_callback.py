# !/usr/bin/env python
"""Define the unit tests for the :mod:`colour.utilities.callback` module."""

from __future__ import annotations

from colour.utilities import MixinCallback

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestMixinCallback",
]


class TestMixinCallback:
    """
    Define :class:`colour.utilities.callback.MixinCallback` class unit
    tests methods.
    """

    def setup_method(self):
        """Initialise the common tests attributes."""

        class WithCallback(MixinCallback):
            """Test :class:`MixinCallback` class."""

            def __init__(self):
                super().__init__()

                self.attribute_a = "a"

        self._with_callback = WithCallback()

        def _on_attribute_a_changed(self, name: str, value: str) -> str:
            """Transform *self._attribute_a* to uppercase."""

            value = value.upper()

            if getattr(self, name) != "a":
                raise RuntimeError(  # pragma: no cover
                    '"self" was not able to retrieve class instance value!'
                )

            return value

        self._on_attribute_a_changed = _on_attribute_a_changed

    def test_required_attributes(self):
        """Test the presence of required attributes."""

        required_attributes = ("callbacks",)

        for attribute in required_attributes:
            assert attribute in dir(MixinCallback)

    def test_required_methods(self):
        """Test the presence of required methods."""

        required_methods = (
            "__init__",
            "register_callback",
            "unregister_callback",
        )

        for method in required_methods:
            assert method in dir(MixinCallback)

    def test_register_callback(self):
        """
        Test :class:`colour.utilities.callback.MixinCallback.register_callback`
        method.
        """

        self._with_callback.register_callback(
            "attribute_a",
            "on_attribute_a_changed",
            self._on_attribute_a_changed,
        )

        self._with_callback.attribute_a = "a"
        assert self._with_callback.attribute_a == "A"
        assert len(self._with_callback.callbacks) == 1

    def test_unregister_callback(self):
        """
        Test :class:`colour.utilities.callback.MixinCallback.unregister_callback`
        method.
        """

        if len(self._with_callback.callbacks) == 0:
            self._with_callback.register_callback(
                "attribute_a",
                "on_attribute_a_changed",
                self._on_attribute_a_changed,
            )

        assert len(self._with_callback.callbacks) == 1
        self._with_callback.unregister_callback("attribute_a", "on_attribute_a_changed")
        assert len(self._with_callback.callbacks) == 0
        self._with_callback.attribute_a = "a"
        assert self._with_callback.attribute_a == "a"
