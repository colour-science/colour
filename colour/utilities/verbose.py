"""
Verbose
=======

Defines the verbose related objects.
"""

from __future__ import annotations

import numpy as np
import os
import sys
import traceback
import warnings
from collections import defaultdict
from contextlib import contextmanager
from itertools import chain
from textwrap import TextWrapper
from warnings import filterwarnings, formatwarning, warn

from colour.utilities import is_string, optional
from colour.hints import (
    Any,
    Boolean,
    Callable,
    Dict,
    Integer,
    LiteralWarning,
    Mapping,
    Generator,
    Optional,
    TextIO,
    Type,
    Union,
    cast,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "ColourWarning",
    "ColourUsageWarning",
    "ColourRuntimeWarning",
    "message_box",
    "show_warning",
    "warning",
    "runtime_warning",
    "usage_warning",
    "filter_warnings",
    "suppress_warnings",
    "numpy_print_options",
    "ANCILLARY_COLOUR_SCIENCE_PACKAGES",
    "ANCILLARY_RUNTIME_PACKAGES",
    "ANCILLARY_DEVELOPMENT_PACKAGES",
    "ANCILLARY_EXTRAS_PACKAGES",
    "describe_environment",
]


class ColourWarning(Warning):
    """
    Define the base class of *Colour* warnings.

    It is a subclass of the :class:`Warning` class.
    """


class ColourUsageWarning(Warning):
    """
    Define the base class of *Colour* usage warnings.

    It is a subclass of the :class:`colour.utilities.ColourWarning` class.
    """


class ColourRuntimeWarning(Warning):
    """
    Define the base class of *Colour* runtime warnings.

    It is a subclass of the :class:`colour.utilities.ColourWarning` class.
    """


def message_box(
    message: str,
    width: Integer = 79,
    padding: Integer = 3,
    print_callable: Callable = print,
):
    """
    Print a message inside a box.

    Parameters
    ----------
    message
        Message to print.
    width
        Message box width.
    padding
        Padding on each side of the message.
    print_callable
        Callable used to print the message box.

    Examples
    --------
    >>> message = (
    ...     'Lorem ipsum dolor sit amet, consectetur adipiscing elit, '
    ...     'sed do eiusmod tempor incididunt ut labore et dolore magna '
    ...     'aliqua.')
    >>> message_box(message, width=75)
    ===========================================================================
    *                                                                         *
    *   Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do       *
    *   eiusmod tempor incididunt ut labore et dolore magna aliqua.           *
    *                                                                         *
    ===========================================================================
    >>> message_box(message, width=60)
    ============================================================
    *                                                          *
    *   Lorem ipsum dolor sit amet, consectetur adipiscing     *
    *   elit, sed do eiusmod tempor incididunt ut labore et    *
    *   dolore magna aliqua.                                   *
    *                                                          *
    ============================================================
    >>> message_box(message, width=75, padding=16)
    ===========================================================================
    *                                                                         *
    *                Lorem ipsum dolor sit amet, consectetur                  *
    *                adipiscing elit, sed do eiusmod tempor                   *
    *                incididunt ut labore et dolore magna                     *
    *                aliqua.                                                  *
    *                                                                         *
    ===========================================================================
    """

    ideal_width = width - padding * 2 - 2

    def inner(text):
        """Format and pads inner text for the message box."""

        return (
            f'*{" " * padding}'
            f'{text}{" " * (width - len(text) - padding * 2 - 2)}'
            f'{" " * padding}*'
        )

    print_callable("=" * width)
    print_callable(inner(""))

    wrapper = TextWrapper(
        width=ideal_width, break_long_words=False, replace_whitespace=False
    )

    lines = [wrapper.wrap(line) for line in message.split("\n")]
    for line in chain(*[" " if len(line) == 0 else line for line in lines]):
        print_callable(inner(line.expandtabs()))

    print_callable(inner(""))
    print_callable("=" * width)


def show_warning(
    message: Union[Warning, str],
    category: Type[Warning],
    filename: str,
    lineno: Integer,
    file: Optional[TextIO] = None,
    line: Optional[str] = None,
) -> None:
    """
    Alternative :func:`warnings.showwarning` definition that allows traceback
    printing.

    This definition is expected to be used by setting the
    *COLOUR_SCIENCE__COLOUR__SHOW_WARNINGS_WITH_TRACEBACK* environment variable
    prior to importing *colour*.

    Parameters
    ----------
    message
        Warning message.
    category
        :class:`Warning` sub-class.
    filename
        File path to read the line at ``lineno`` from if ``line`` is None.
    lineno
        Line number to read the line at in ``filename`` if ``line`` is None.
    file
        :class:`file` object to write the warning to, defaults to
        :attr:`sys.stderr` attribute.
    line
        Source code to be included in the warning message.

    Notes
    -----
    -   Setting the *COLOUR_SCIENCE__COLOUR__SHOW_WARNINGS_WITH_TRACEBACK*
        environment variable will result in the :func:`warnings.showwarning`
        definition to be replaced with the
        :func:`colour.utilities.show_warning` definition and thus providing
        complete traceback from the point where the warning occurred.
    """

    frame_range = (1, None)

    file = optional(file, sys.stderr)
    if file is None:
        return

    try:
        # Generating a traceback to print useful warning origin.
        frame_in, frame_out = frame_range

        try:
            raise ZeroDivisionError
        except ZeroDivisionError:
            exception_traceback = sys.exc_info()[2]
            frame = (
                exception_traceback.tb_frame.f_back
                if exception_traceback is not None
                else None
            )
            while frame_in and frame is not None:
                frame = frame.f_back
                frame_in -= 1

        traceback.print_stack(frame, frame_out, file)

        file.write(formatwarning(message, category, filename, lineno, line))
    except (OSError, UnicodeError):
        pass


if os.environ.get(  # pragma: no cover
    "COLOUR_SCIENCE__COLOUR__SHOW_WARNINGS_WITH_TRACEBACK"
):
    warnings.showwarning = show_warning  # pragma: no cover


def warning(*args: Any, **kwargs: Any):
    """
    Issue a warning.

    Other Parameters
    ----------------
    args
        Arguments.
    kwargs
        Keywords arguments.

    Examples
    --------
    >>> warning('This is a warning!')  # doctest: +SKIP
    """

    kwargs["category"] = kwargs.get("category", ColourWarning)

    warn(*args, **kwargs)


def runtime_warning(*args: Any, **kwargs: Any):
    """
    Issue a runtime warning.

    Other Parameters
    ----------------
    args
        Arguments.
    kwargs
        Keywords arguments.

    Examples
    --------
    >>> usage_warning('This is a runtime warning!')  # doctest: +SKIP
    """

    kwargs["category"] = ColourRuntimeWarning

    warning(*args, **kwargs)


def usage_warning(*args: Any, **kwargs: Any):
    """
    Issue a usage warning.

    Other Parameters
    ----------------
    args
        Arguments.
    kwargs
        Keywords arguments.

    Examples
    --------
    >>> usage_warning('This is an usage warning!')  # doctest: +SKIP
    """

    kwargs["category"] = ColourUsageWarning

    warning(*args, **kwargs)


def filter_warnings(
    colour_runtime_warnings: Optional[Union[bool, LiteralWarning]] = None,
    colour_usage_warnings: Optional[Union[bool, LiteralWarning]] = None,
    colour_warnings: Optional[Union[bool, LiteralWarning]] = None,
    python_warnings: Optional[Union[bool, LiteralWarning]] = None,
):
    """
    Filter *Colour* and also optionally overall Python warnings.

    The possible values for all the actions, i.e. each argument, are as
    follows:

    - *None* (No action is taken)
    - *True* (*ignore*)
    - *False* (*default*)
    - *error*
    - *ignore*
    - *always*
    - *default*
    - *module*
    - *once*

    Parameters
    ----------
    colour_runtime_warnings
        Whether to filter *Colour* runtime warnings according to the action
        value.
    colour_usage_warnings
        Whether to filter *Colour* usage warnings according to the action
        value.
    colour_warnings
        Whether to filter *Colour* warnings, this also filters *Colour* usage
        and runtime warnings according to the action value.
    python_warnings
        Whether to filter *Python* warnings  according to the action value.

    Examples
    --------
    Filtering *Colour* runtime warnings:

    >>> filter_warnings(colour_runtime_warnings=True)

    Filtering *Colour* usage warnings:

    >>> filter_warnings(colour_usage_warnings=True)

    Filtering *Colour* warnings:

    >>> filter_warnings(colour_warnings=True)

    Filtering all the *Colour* and also Python warnings:

    >>> filter_warnings(python_warnings=True)

    Enabling all the *Colour* and Python warnings:

    >>> filter_warnings(*[False] * 4)

    Enabling all the *Colour* and Python warnings using the *default* action:

    >>> filter_warnings(*['default'] * 4)

    Setting back the default state:

    >>> filter_warnings(colour_runtime_warnings=True)
    """

    for action, category in [
        (colour_warnings, ColourWarning),
        (colour_runtime_warnings, ColourRuntimeWarning),
        (colour_usage_warnings, ColourUsageWarning),
        (python_warnings, Warning),
    ]:
        if action is None:
            continue

        if is_string(action):
            action = cast(LiteralWarning, str(action))
        else:
            action = "ignore" if action else "default"

        filterwarnings(action, category=category)


# Defaulting to filter *Colour* runtime warnings.
filter_warnings(colour_runtime_warnings=True)


@contextmanager
def suppress_warnings(
    colour_runtime_warnings: Optional[Union[bool, LiteralWarning]] = None,
    colour_usage_warnings: Optional[Union[bool, LiteralWarning]] = None,
    colour_warnings: Optional[Union[bool, LiteralWarning]] = None,
    python_warnings: Optional[Union[bool, LiteralWarning]] = None,
) -> Generator:
    """
    Define a context manager filtering *Colour* and also optionally overall
    Python warnings.

    The possible values for all the actions, i.e. each argument, are as
    follows:

    - *None* (No action is taken)
    - *True* (*ignore*)
    - *False* (*default*)
    - *error*
    - *ignore*
    - *always*
    - *default*
    - *module*
    - *once*

    Parameters
    ----------
    colour_runtime_warnings
        Whether to filter *Colour* runtime warnings according to the action
        value.
    colour_usage_warnings
        Whether to filter *Colour* usage warnings according to the action
        value.
    colour_warnings
        Whether to filter *Colour* warnings, this also filters *Colour* usage
        and runtime warnings according to the action value.
    python_warnings
        Whether to filter *Python* warnings  according to the action value.
    """

    filters = warnings.filters
    show_warnings = warnings.showwarning

    filter_warnings(
        colour_warnings=colour_warnings,
        colour_runtime_warnings=colour_runtime_warnings,
        colour_usage_warnings=colour_usage_warnings,
        python_warnings=python_warnings,
    )

    try:
        yield
    finally:
        warnings.filters = filters
        warnings.showwarning = show_warnings


@contextmanager
def numpy_print_options(*args: Any, **kwargs: Any) -> Generator:
    """
    Define a context manager implementing context changes to *Numpy* print
    behaviour.

    Other Parameters
    ----------------
    args
        Arguments.
    kwargs
        Keywords arguments.

    Examples
    --------
    >>> np.array([np.pi])  # doctest: +ELLIPSIS
    array([ 3.1415926...])
    >>> with numpy_print_options(formatter={'float': '{:0.1f}'.format}):
    ...      np.array([np.pi])
    array([3.1])
    """

    options = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**options)


ANCILLARY_COLOUR_SCIENCE_PACKAGES: Dict[str, str] = {}
"""
Ancillary *colour-science.org* packages to describe.

ANCILLARY_COLOUR_SCIENCE_PACKAGES
"""

ANCILLARY_RUNTIME_PACKAGES: Dict[str, str] = {}
"""
Ancillary runtime packages to describe.

ANCILLARY_RUNTIME_PACKAGES
"""

ANCILLARY_DEVELOPMENT_PACKAGES: Dict[str, str] = {}
"""
Ancillary development packages to describe.

ANCILLARY_DEVELOPMENT_PACKAGES
"""

ANCILLARY_EXTRAS_PACKAGES: Dict[str, str] = {}
"""
Ancillary extras packages to describe.

ANCILLARY_EXTRAS_PACKAGES
"""


def describe_environment(
    runtime_packages: Boolean = True,
    development_packages: Boolean = False,
    extras_packages: Boolean = False,
    print_environment: Boolean = True,
    **kwargs: Any,
) -> defaultdict:
    """
    Describe *Colour* running environment, i.e. interpreter, runtime and
    development packages.

    Parameters
    ----------
    runtime_packages
        Whether to return the runtime packages versions.
    development_packages
        Whether to return the development packages versions.
    extras_packages
        Whether to return the extras packages versions.
    print_environment
        Whether to print the environment.

    Other Parameters
    ----------------
    padding
        {:func:`colour.utilities.message_box`},
        Padding on each side of the message.
    print_callable
        {:func:`colour.utilities.message_box`},
        Callable used to print the message box.
    width
        {:func:`colour.utilities.message_box`},
        Message box width.

    Returns
    -------
    :class:`collections.defaultdict`
        Environment.

    Examples
    --------
    >>> environment = describe_environment(width=75)  # doctest: +SKIP
    ===========================================================================
    *                                                                         *
    *   Interpreter :                                                         *
    *       python : 3.8.6 (default, Nov 20 2020, 18:29:40)                   *
    *                [Clang 12.0.0 (clang-1200.0.32.27)]                      *
    *                                                                         *
    *   colour-science.org :                                                  *
    *       colour : v0.3.16-3-gd8bac475                                      *
    *                                                                         *
    *   Runtime :                                                             *
    *       imageio : 2.9.0                                                   *
    *       matplotlib : 3.3.3                                                *
    *       networkx : 2.5                                                    *
    *       numpy : 1.19.4                                                    *
    *       pandas : 0.25.3                                                   *
    *       pygraphviz : 1.6                                                  *
    *       scipy : 1.5.4                                                     *
    *       tqdm : 4.54.0                                                     *
    *                                                                         *
    ===========================================================================
    >>> environment = describe_environment(True, True, True, width=75)
    ... # doctest: +SKIP
    ===========================================================================
    *                                                                         *
    *   Interpreter :                                                         *
    *       python : 3.8.6 (default, Nov 20 2020, 18:29:40)                   *
    *                [Clang 12.0.0 (clang-1200.0.32.27)]                      *
    *                                                                         *
    *   colour-science.org :                                                  *
    *       colour : v0.3.16-3-gd8bac475                                      *
    *                                                                         *
    *   Runtime :                                                             *
    *       imageio : 2.9.0                                                   *
    *       matplotlib : 3.3.3                                                *
    *       networkx : 2.5                                                    *
    *       numpy : 1.19.4                                                    *
    *       pandas : 0.25.3                                                   *
    *       pygraphviz : 1.6                                                  *
    *       scipy : 1.5.4                                                     *
    *       tqdm : 4.54.0                                                     *
    *                                                                         *
    *   Development :                                                         *
    *       biblib-simple : 0.1.1                                             *
    *       coverage : 5.3                                                    *
    *       coveralls : 2.2.0                                                 *
    *       flake8 : 3.8.4                                                    *
    *       invoke : 1.4.1                                                    *
    *       jupyter : 1.0.0                                                   *
    *       mock : 4.0.2                                                      *
    *       nose : 1.3.7                                                      *
    *       pre-commit : 2.1.1                                                *
    *       pytest : 6.1.2                                                    *
    *       restructuredtext-lint : 1.3.2                                     *
    *       sphinx : 3.1.2                                                    *
    *       sphinx_rtd_theme : 0.5.0                                          *
    *       sphinxcontrib-bibtex : 1.0.0                                      *
    *       toml : 0.10.2                                                     *
    *       twine : 3.2.0                                                     *
    *       yapf : 0.23.0                                                     *
    *                                                                         *
    *   Extras :                                                              *
    *       ipywidgets : 7.5.1                                                *
    *       notebook : 6.1.5                                                  *
    *                                                                         *
    ===========================================================================
    """

    environment: defaultdict = defaultdict(dict)

    environment["Interpreter"]["python"] = sys.version

    import subprocess  # nosec

    import colour

    # TODO: Implement support for "pyproject.toml" file whenever "TOML" is
    # supported in the standard library.

    # NOTE: A few clauses are not reached and a few packages are not available
    # during continuous integration and are thus ignored for coverage.
    try:  # pragma: no cover
        output = subprocess.check_output(  # nosec
            ["git", "describe"],
            cwd=colour.__path__[0],
            stderr=subprocess.STDOUT,
        ).strip()
        version = output.decode("utf-8")
    except Exception:  # pragma: no cover
        version = colour.__version__

    environment["colour-science.org"]["colour"] = version
    environment["colour-science.org"].update(ANCILLARY_COLOUR_SCIENCE_PACKAGES)

    if runtime_packages:
        for package in [
            "imageio",
            "matplotlib",
            "networkx",
            "numpy",
            "pandas",
            "pygraphviz",
            "PyOpenColorIO",
            "scipy",
            "sklearn",
            "tqdm",
            "trimesh",
        ]:
            try:
                namespace = __import__(package)
                environment["Runtime"][package] = namespace.__version__
            except ImportError:
                continue

        # OpenImageIO
        try:  # pragma: no cover
            namespace = __import__("OpenImageIO")
            environment["Runtime"]["OpenImageIO"] = namespace.VERSION_STRING
        except ImportError:  # pragma: no cover
            pass

        environment["Runtime"].update(ANCILLARY_RUNTIME_PACKAGES)

    def _get_package_version(package: str, mapping: Mapping) -> str:
        """Return given package version."""

        namespace = __import__(package)

        if package in mapping:
            import pkg_resources

            distributions = [
                distribution for distribution in pkg_resources.working_set
            ]

            for distribution in distributions:
                if distribution.project_name == mapping[package]:
                    return distribution.version

        return namespace.__version__

    if development_packages:
        mapping = {
            "biblib.bib": "biblib-simple",
            "pre_commit": "pre-commit",
            "restructuredtext_lint": "restructuredtext-lint",
            "sphinxcontrib.bibtex": "sphinxcontrib-bibtex",
        }
        for package in [
            "biblib.bib",
            "coverage",
            "coveralls",
            "flake8",
            "invoke",
            "jupyter",
            "mock",
            "nose",
            "pre_commit",
            "pytest",
            "restructuredtext_lint",
            "sphinx",
            "sphinx_rtd_theme",
            "sphinxcontrib.bibtex",
            "toml",
            "twine",
            "yapf",
        ]:
            try:
                version = _get_package_version(package, mapping)
                package = mapping.get(package, package)

                environment["Development"][package] = version
            except Exception:  # pragma: no cover
                # pylint: disable=B112
                continue

        environment["Development"].update(ANCILLARY_DEVELOPMENT_PACKAGES)

    if extras_packages:
        mapping = {}
        for package in ["ipywidgets", "notebook"]:
            try:
                version = _get_package_version(package, mapping)
                package = mapping.get(package, package)

                environment["Extras"][package] = version
            except Exception:  # pragma: no cover
                # pylint: disable=B112
                continue

        environment["Extras"].update(ANCILLARY_EXTRAS_PACKAGES)

    if print_environment:
        message = ""
        for category in (
            "Interpreter",
            "colour-science.org",
            "Runtime",
            "Development",
            "Extras",
        ):
            elements = environment.get(category)
            if not elements:
                continue

            message += f"{category} :\n"
            for key, value in elements.items():
                lines = value.split("\n")
                message += f"    {key} : {lines.pop(0)}\n"
                indentation = len(f"    {key} : ")
                for line in lines:
                    message += f"{' ' * indentation}{line}\n"

            message += "\n"

        message_box(message.strip(), **kwargs)

    return environment
