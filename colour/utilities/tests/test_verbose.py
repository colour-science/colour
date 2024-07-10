"""Define the unit tests for the :mod:`colour.utilities.verbose` module."""

import os
import sys
import textwrap

from colour.hints import Optional
from colour.utilities import (
    MixinLogging,
    as_bool,
    describe_environment,
    multiline_repr,
    multiline_str,
    show_warning,
    suppress_stdout,
    suppress_warnings,
    warning,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestMixinLogging",
    "TestShowWarning",
    "TestAsBool",
    "TestSuppressWarnings",
    "TestSuppressStdout",
    "TestDescribeEnvironment",
    "TestMultilineStr",
    "TestMultilineRepr",
]


class TestMixinLogging:
    """
    Define :class:`colour.utilities.verbose.MixinLogging` class unit tests
    methods.
    """

    def test_required_methods(self):
        """Test the presence of required methods."""

        required_methods = ("log",)

        for method in required_methods:
            assert method in dir(MixinLogging)


class TestShowWarning:
    """
    Define :func:`colour.utilities.verbose.show_warning` definition unit tests
    methods.
    """

    def test_show_warning(self):
        """Test :func:`colour.utilities.verbose.show_warning` definition."""

        show_warning("This is a unit test warning!", Warning, None, None)

        with open(os.devnull) as dev_null:
            show_warning("This is a unit test warning!", Warning, None, None, dev_null)

        stderr = sys.stderr
        try:
            sys.stderr = None
            show_warning("This is a unit test warning!", Warning, None, None)
        finally:
            sys.stderr = stderr


class TestAsBool:
    """
    Define :func:`colour.utilities.common.as_bool` definition unit tests
    methods.
    """

    def test_as_bool(self):
        """Test :func:`colour.utilities.common.as_bool` definition."""

        assert as_bool("1")

        assert as_bool("On")

        assert as_bool("True")

        assert not as_bool("0")

        assert not as_bool("Off")

        assert not as_bool("False")

        assert not as_bool("")


class TestSuppressWarnings:
    """
    Define :func:`colour.utilities.verbose.suppress_warnings` definition unit
    tests methods.
    """

    def test_suppress_warnings(self):
        """Test :func:`colour.utilities.verbose.suppress_warnings` definition."""

        with suppress_warnings():
            warning("This is a suppressed unit test warning!")


class TestSuppressStdout:
    """
    Define :func:`colour.utilities.verbose.suppress_stdout` definition unit
    tests methods.
    """

    def test_suppress_stdout(self):
        """Test :func:`colour.utilities.verbose.suppress_stdout` definition."""

        with suppress_stdout():
            print("This is a suppressed message!")  # noqa: T201


class TestDescribeEnvironment:
    """
    Define :func:`colour.utilities.verbose.describe_environment` definition
    unit tests methods.
    """

    def test_describe_environment(self):
        """Test :func:`colour.utilities.verbose.describe_environment` definition."""

        environment = describe_environment()
        assert isinstance(environment, dict)
        assert sorted(environment.keys()) == [
            "Interpreter",
            "Runtime",
            "colour-science.org",
        ]

        environment = describe_environment(development_packages=True)
        assert sorted(environment.keys()) == [
            "Development",
            "Interpreter",
            "Runtime",
            "colour-science.org",
        ]

        environment = describe_environment(
            development_packages=True, extras_packages=True
        )
        assert sorted(environment.keys()) == [
            "Development",
            "Extras",
            "Interpreter",
            "Runtime",
            "colour-science.org",
        ]


class TestMultilineStr:
    """
    Define :func:`colour.utilities.verbose.multiline_str` definition unit
    tests methods.
    """

    def test_multiline_str(self):
        """Test :func:`colour.utilities.verbose.multiline_str` definition."""

        class Data:
            def __init__(self, a: str, b: int, c: list) -> None:
                self._a = a
                self._b = b
                self._c = c

            def __str__(self) -> str:
                return multiline_str(
                    self,
                    [
                        {
                            "formatter": lambda x: (  # noqa: ARG005
                                f"Object - {self.__class__.__name__}"
                            ),
                            "header": True,
                        },
                        {"line_break": True},
                        {"label": "Data", "section": True},
                        {"line_break": True},
                        {"label": "String", "section": True},
                        {"name": "_a", "label": 'String "a"'},
                        {"line_break": True},
                        {"label": "Integer", "section": True},
                        {"name": "_b", "label": 'Integer "b"'},
                        {"line_break": True},
                        {"label": "List", "section": True},
                        {
                            "name": "_c",
                            "label": 'List "c"',
                            "formatter": lambda x: "; ".join(x),
                        },
                    ],
                )

        assert str(Data("Foo", 1, ["John", "Doe"])) == (
            textwrap.dedent(
                """
                Object - Data
                =============

                Data
                ----

                String
                ------
                String "a"  : Foo

                Integer
                -------
                Integer "b" : 1

                List
                ----
                List "c"    : John; Doe
                """
            ).strip()
        )


class TestMultilineRepr:
    """
    Define :func:`colour.utilities.verbose.multiline_repr` definition unit
    tests methods.
    """

    def test_multiline_repr(self):
        """Test :func:`colour.utilities.verbose.multiline_repr` definition."""

        class Data:
            def __init__(
                self, a: str, b: int, c: list, d: Optional[str] = None
            ) -> None:
                self._a = a
                self._b = b
                self._c = c
                self._d = d

            def __repr__(self) -> str:
                return multiline_repr(
                    self,
                    [
                        {"name": "_a"},
                        {"name": "_b"},
                        {
                            "name": "_c",
                            "formatter": lambda x: repr(x)
                            .replace("[", "(")
                            .replace("]", ")"),
                        },
                        {
                            "name": "_d",
                            "formatter": lambda x: None,  # noqa: ARG005
                        },
                    ],
                )

        assert repr(Data("Foo", 1, ["John", "Doe"])) == (
            textwrap.dedent(
                """
                Data('Foo',
                     1,
                     ('John', 'Doe'),
                     None)
                """
            ).strip()
        )
