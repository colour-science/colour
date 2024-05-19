# !/usr/bin/env python
"""Define the unit tests for the :mod:`colour.utilities.deprecation` module."""

import sys

import pytest

from colour.utilities import ColourUsageWarning
from colour.utilities.deprecation import (
    ArgumentFutureRemove,
    ArgumentFutureRename,
    ArgumentRemoved,
    ArgumentRenamed,
    ModuleAPI,
    ObjectFutureAccessChange,
    ObjectFutureAccessRemove,
    ObjectFutureRemove,
    ObjectFutureRename,
    ObjectRemoved,
    ObjectRenamed,
    build_API_changes,
    get_attribute,
    handle_arguments_deprecation,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestObjectRenamed",
    "TestObjectRemoved",
    "TestObjectFutureRename",
    "TestObjectFutureRemove",
    "TestObjectFutureAccessChange",
    "TestObjectFutureAccessRemove",
    "TestArgumentRenamed",
    "TestArgumentRemoved",
    "TestArgumentFutureRename",
    "TestArgumentFutureRemove",
    "TestModuleAPI",
    "TestGetAttribute",
    "TestBuildAPIChanges",
    "TestHandleArgumentsDeprecation",
]


class TestObjectRenamed:
    """
    Define :class:`colour.utilities.deprecation.ObjectRenamed` class unit
    tests methods.
    """

    def test_required_methods(self):
        """Test the presence of required methods."""

        required_methods = ("__str__",)

        for method in required_methods:
            assert method in dir(ObjectRenamed)

    def test__str__(self):
        """
        Test :meth:`colour.utilities.deprecation.ObjectRenamed.__str__`
        method.
        """

        assert "name" in str(ObjectRenamed("name", "new_name"))
        assert "new_name" in str(ObjectRenamed("name", "new_name"))


class TestObjectRemoved:
    """
    Define :class:`colour.utilities.deprecation.ObjectRemoved` class unit
    tests methods.
    """

    def test_required_methods(self):
        """Test the presence of required methods."""

        required_methods = ("__str__",)

        for method in required_methods:
            assert method in dir(ObjectRemoved)

    def test__str__(self):
        """
        Test :meth:`colour.utilities.deprecation.ObjectRemoved.__str__`
        method.
        """

        assert "name" in str(ObjectRemoved("name"))


class TestObjectFutureRename:
    """
    Define :class:`colour.utilities.deprecation.ObjectFutureRename` class unit
    tests methods.
    """

    def test_required_methods(self):
        """Test the presence of required methods."""

        required_methods = ("__str__",)

        for method in required_methods:
            assert method in dir(ObjectFutureRename)

    def test__str__(self):
        """
        Test :meth:`colour.utilities.deprecation.ObjectFutureRename.__str__`
        method.
        """

        assert "name" in str(ObjectFutureRename("name", "new_name"))
        assert "new_name" in str(ObjectFutureRename("name", "new_name"))


class TestObjectFutureRemove:
    """
    Define :class:`colour.utilities.deprecation.ObjectFutureRemove` class unit
    tests methods.
    """

    def test_required_methods(self):
        """Test the presence of required methods."""

        required_methods = ("__str__",)

        for method in required_methods:
            assert method in dir(ObjectFutureRemove)

    def test__str__(self):
        """
        Test :meth:`colour.utilities.deprecation.ObjectFutureRemove.__str__`
        method.
        """

        assert "name" in str(
            ObjectFutureRemove(
                "name",
            )
        )


class TestObjectFutureAccessChange:
    """
    Define :class:`colour.utilities.deprecation.ObjectFutureAccessChange`
    class unit tests methods.
    """

    def test_required_methods(self):
        """Test the presence of required methods."""

        required_methods = ("__str__",)

        for method in required_methods:
            assert method in dir(ObjectFutureAccessChange)

    def test__str__(self):
        """
        Test :meth:`colour.utilities.deprecation.\
ObjectFutureAccessChange.__str__` method.
        """

        assert "name" in str(ObjectFutureAccessChange("name", "new_access"))
        assert "new_access" in str(ObjectFutureAccessChange("name", "new_access"))


class TestObjectFutureAccessRemove:
    """
    Define :class:`colour.utilities.deprecation.ObjectFutureAccessRemove`
    class unit tests methods.
    """

    def test_required_methods(self):
        """Test the presence of required methods."""

        required_methods = ("__str__",)

        for method in required_methods:
            assert method in dir(ObjectFutureAccessRemove)

    def test__str__(self):
        """
        Test :meth:`colour.utilities.deprecation.\
ObjectFutureAccessRemove.__str__` method.
        """

        assert "name" in str(
            ObjectFutureAccessRemove(
                "name",
            )
        )


class TestArgumentRenamed:
    """
    Define :class:`colour.utilities.deprecation.ArgumentRenamed` class unit
    tests methods.
    """

    def test_required_methods(self):
        """Test the presence of required methods."""

        required_methods = ("__str__",)

        for method in required_methods:
            assert method in dir(ArgumentRenamed)

    def test__str__(self):
        """
        Test :meth:`colour.utilities.deprecation.ArgumentRenamed.__str__`
        method.
        """

        assert "name" in str(ArgumentRenamed("name", "new_name"))
        assert "new_name" in str(ArgumentRenamed("name", "new_name"))


class TestArgumentRemoved:
    """
    Define :class:`colour.utilities.deprecation.ArgumentRemoved` class unit
    tests methods.
    """

    def test_required_methods(self):
        """Test the presence of required methods."""

        required_methods = ("__str__",)

        for method in required_methods:
            assert method in dir(ArgumentRemoved)

    def test__str__(self):
        """
        Test :meth:`colour.utilities.deprecation.ArgumentRemoved.__str__`
        method.
        """

        assert "name" in str(ArgumentRemoved("name"))


class TestArgumentFutureRename:
    """
    Define :class:`colour.utilities.deprecation.ArgumentFutureRename` class
    unit tests methods.
    """

    def test_required_methods(self):
        """Test the presence of required methods."""

        required_methods = ("__str__",)

        for method in required_methods:
            assert method in dir(ArgumentFutureRename)

    def test__str__(self):
        """
        Test :meth:`colour.utilities.deprecation.\
ArgumentFutureRename.__str__` method.
        """

        assert "name" in str(ArgumentFutureRename("name", "new_name"))
        assert "new_name" in str(ArgumentFutureRename("name", "new_name"))


class TestArgumentFutureRemove:
    """
    Define :class:`colour.utilities.deprecation.ArgumentFutureRemove` class
    unit tests methods.
    """

    def test_required_methods(self):
        """Test the presence of required methods."""

        required_methods = ("__str__",)

        for method in required_methods:
            assert method in dir(ArgumentFutureRemove)

    def test__str__(self):
        """
        Test :meth:`colour.utilities.deprecation.\
ArgumentFutureRemove.__str__` method.
        """

        assert "name" in str(
            ArgumentFutureRemove(
                "name",
            )
        )


class TestModuleAPI:
    """
    Define :class:`colour.utilities.deprecation.ModuleAPI` class unit tests
    methods.
    """

    def test_required_methods(self):
        """Test the presence of required methods."""

        required_methods = ("__init__", "__getattr__", "__dir__")

        for method in required_methods:
            assert method in dir(ModuleAPI)

    def test__getattr__(self):
        """
        Test :meth:`colour.utilities.deprecation.ModuleAPI.__getattr__`
        method.
        """

        import colour.utilities.tests.test_deprecated

        assert colour.utilities.tests.test_deprecated.NAME is None

        def assert_warns():
            """Help to test the runtime warning."""

            colour.utilities.tests.test_deprecated.OLD_NAME  # noqa: B018

        pytest.warns(ColourUsageWarning, assert_warns)

        del sys.modules["colour.utilities.tests.test_deprecated"]

    def test_raise_exception__getattr__(self):
        """
        Test :func:`colour.utilities.deprecation.ModuleAPI.__getattr__`
        method raised exception.
        """

        import colour.utilities.tests.test_deprecated

        pytest.raises(
            AttributeError,
            getattr,
            colour.utilities.tests.test_deprecated,
            "REMOVED",
        )

        del sys.modules["colour.utilities.tests.test_deprecated"]


class TestGetAttribute:
    """
    Define :func:`colour.utilities.deprecation.get_attribute` definition unit
    tests methods.
    """

    def test_get_attribute(self):
        """Test :func:`colour.utilities.deprecation.get_attribute` definition."""

        from colour import adaptation

        assert get_attribute("colour.adaptation") is adaptation

        from colour.models import eotf_inverse_sRGB

        assert get_attribute("colour.models.eotf_inverse_sRGB") is eotf_inverse_sRGB

        from colour.utilities.array import as_float

        assert get_attribute("colour.utilities.array.as_float") is as_float

        if "colour.utilities.tests.test_deprecated" in sys.modules:  # pragma: no cover
            del sys.modules["colour.utilities.tests.test_deprecated"]

        attribute = get_attribute("colour.utilities.tests.test_deprecated.NEW_NAME")

        import colour.utilities.tests.test_deprecated

        assert attribute is colour.utilities.tests.test_deprecated.NEW_NAME
        del sys.modules["colour.utilities.tests.test_deprecated"]


class TestBuildAPIChanges:
    """
    Define :func:`colour.utilities.deprecation.build_API_changes` definition
    unit tests methods.
    """

    def test_build_API_changes(self):
        """
        Test :func:`colour.utilities.deprecation.build_API_changes`
        definition.
        """

        changes = build_API_changes(
            {
                "ObjectRenamed": [
                    [
                        "module.object_1_name",
                        "module.object_1_new_name",
                    ]
                ],
                "ObjectFutureRename": [
                    [
                        "module.object_2_name",
                        "module.object_2_new_name",
                    ]
                ],
                "ObjectFutureAccessChange": [
                    [
                        "module.object_3_access",
                        "module.sub_module.object_3_new_access",
                    ]
                ],
                "ObjectRemoved": ["module.object_4_name"],
                "ObjectFutureRemove": ["module.object_5_name"],
                "ObjectFutureAccessRemove": ["module.object_6_access"],
                "ArgumentRenamed": [
                    [
                        "argument_1_name",
                        "argument_1_new_name",
                    ]
                ],
                "ArgumentFutureRename": [
                    [
                        "argument_2_name",
                        "argument_2_new_name",
                    ]
                ],
                "ArgumentRemoved": ["argument_3_name"],
                "ArgumentFutureRemove": ["argument_4_name"],
            }
        )
        for name, change_type in (
            ("object_1_name", ObjectRenamed),
            ("object_2_name", ObjectFutureRename),
            ("object_3_access", ObjectFutureAccessChange),
            ("object_4_name", ObjectRemoved),
            ("object_5_name", ObjectFutureRemove),
            ("object_6_access", ObjectFutureAccessRemove),
            ("argument_1_name", ArgumentRenamed),
            ("argument_2_name", ArgumentFutureRename),
            ("argument_3_name", ArgumentRemoved),
            ("argument_4_name", ArgumentFutureRemove),
        ):
            assert isinstance(changes[name], change_type)


class TestHandleArgumentsDeprecation:
    """
    Define :func:`colour.utilities.deprecation.handle_arguments_deprecation`
    definition unit tests methods.
    """

    def test_handle_arguments_deprecation(self):
        """
        Test :func:`colour.utilities.deprecation.handle_arguments_deprecation`
        definition.
        """

        changes = {
            "ArgumentRenamed": [
                [
                    "argument_1_name",
                    "argument_1_new_name",
                ]
            ],
            "ArgumentFutureRename": [
                [
                    "argument_2_name",
                    "argument_2_new_name",
                ]
            ],
            "ArgumentRemoved": ["argument_3_name"],
            "ArgumentFutureRemove": ["argument_4_name"],
        }

        assert handle_arguments_deprecation(
            changes,
            argument_1_name=True,
            argument_2_name=True,
            argument_3_name=True,
            argument_4_name=True,
            argument_5_name=True,
        ) == {
            "argument_1_new_name": True,
            "argument_2_new_name": True,
            "argument_4_name": True,
            "argument_5_name": True,
        }
