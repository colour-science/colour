# -*- coding: utf-8 -*-
"""
Verbose
=======

Defines verbose related objects.
"""

from __future__ import division, print_function, unicode_literals

import numpy as np
import os
import sys
import traceback
import warnings
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from itertools import chain
from textwrap import TextWrapper
from warnings import filterwarnings, formatwarning, warn

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'ColourWarning', 'ColourUsageWarning', 'ColourRuntimeWarning',
    'message_box', 'show_warning', 'warning', 'runtime_warning',
    'usage_warning', 'filter_warnings', 'suppress_warnings',
    'numpy_print_options', 'ANCILLARY_COLOUR_SCIENCE_PACKAGES',
    'ANCILLARY_RUNTIME_PACKAGES', 'ANCILLARY_DEVELOPMENT_PACKAGES',
    'ANCILLARY_EXTRAS_PACKAGES', 'describe_environment'
]


class ColourWarning(Warning):
    """
    This is the base class of *Colour* warnings. It is a subclass of
    :class:`Warning` class.
    """


class ColourUsageWarning(ColourWarning):
    """
    This is the base class of *Colour* usage warnings. It is a subclass
    of :class:`colour.utilities.ColourWarning` class.
    """


class ColourRuntimeWarning(ColourWarning):
    """
    This is the base class of *Colour* runtime warnings. It is a subclass
    of :class:`colour.utilities.ColourWarning` class.
    """


def message_box(message, width=79, padding=3, print_callable=print):
    """
    Prints a message inside a box.

    Parameters
    ----------
    message : unicode
        Message to print.
    width : int, optional
        Message box width.
    padding : unicode, optional
        Padding on each sides of the message.
    print_callable : callable, optional
        Callable used to print the message box.

    Returns
    -------
    bool
        Definition success.

    Examples
    --------
    >>> message = ('Lorem ipsum dolor sit amet, consectetur adipiscing elit, '
    ...     'sed do eiusmod tempor incididunt ut labore et dolore magna '
    ...     'aliqua.')
    >>> message_box(message, width=75)
    ===========================================================================
    *                                                                         *
    *   Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do       *
    *   eiusmod tempor incididunt ut labore et dolore magna aliqua.           *
    *                                                                         *
    ===========================================================================
    True
    >>> message_box(message, width=60)
    ============================================================
    *                                                          *
    *   Lorem ipsum dolor sit amet, consectetur adipiscing     *
    *   elit, sed do eiusmod tempor incididunt ut labore et    *
    *   dolore magna aliqua.                                   *
    *                                                          *
    ============================================================
    True
    >>> message_box(message, width=75, padding=16)
    ===========================================================================
    *                                                                         *
    *                Lorem ipsum dolor sit amet, consectetur                  *
    *                adipiscing elit, sed do eiusmod tempor                   *
    *                incididunt ut labore et dolore magna                     *
    *                aliqua.                                                  *
    *                                                                         *
    ===========================================================================
    True
    """

    ideal_width = width - padding * 2 - 2

    def inner(text):
        """
        Formats and pads inner text for the message box.
        """

        return '*{0}{1}{2}{0}*'.format(
            ' ' * padding, text, (' ' * (width - len(text) - padding * 2 - 2)))

    print_callable('=' * width)
    print_callable(inner(''))

    wrapper = TextWrapper(
        width=ideal_width, break_long_words=False, replace_whitespace=False)

    lines = [wrapper.wrap(line) for line in message.split("\n")]
    lines = [' ' if len(line) == 0 else line for line in lines]
    for line in chain(*lines):
        print_callable(inner(line.expandtabs()))

    print_callable(inner(''))
    print_callable('=' * width)

    return True


def show_warning(message,
                 category,
                 path,
                 line,
                 file_=None,
                 code=None,
                 frame_range=(1, None)):
    """
    Alternative :func:`warnings.showwarning` definition that allows traceback
    printing.

    This definition is expected to be used by setting the
    *COLOUR_SCIENCE__COLOUR__SHOW_WARNINGS_WITH_TRACEBACK* environment variable
    prior to importing *colour*.

    Parameters
    ----------
    message : unicode
        Warning message.
    category : Warning
        :class:`Warning` sub-class.
    path : unicode
        File path to read the line at ``lineno`` from if ``line`` is None.
    line : int
        Line number to read the line at in ``filename`` if ``line`` is None.
    file_ : file, optional
        :class:`file` object to write the warning to, defaults to
        :attr:`sys.stderr` attribute.
    code : unicode, optional
        Source code to be included in the warning message.
    frame_range : array_like, optional
        Traceback frame range, i.e first frame and numbers of frame above it.

    Notes
    -----
    -   Setting the *COLOUR_SCIENCE__COLOUR__SHOW_WARNINGS_WITH_TRACEBACK*
        environment variable will result in the :func:`warnings.showwarning`
        definition to be replaced with the
        :func:`colour.utilities.show_warning` definition and thus providing
        complete traceback from the point where the warning occurred.
    """

    if file_ is None:
        file_ = sys.stderr
        if file_ is None:
            return

    try:
        # Generating a traceback to print useful warning origin.
        frame_in, frame_out = frame_range

        try:
            raise ZeroDivisionError
        except ZeroDivisionError:
            frame = sys.exc_info()[2].tb_frame.f_back
            while frame_in:
                frame = frame.f_back
                frame_in -= 1

        traceback.print_stack(frame, frame_out, file_)

        file_.write(formatwarning(message, category, path, line, code))
    except (IOError, UnicodeError):
        pass


if os.environ.get(  # pragma: no cover
        'COLOUR_SCIENCE__COLOUR__SHOW_WARNINGS_WITH_TRACEBACK'):
    warnings.showwarning = show_warning


def warning(*args, **kwargs):
    """
    Issues a warning.

    Other Parameters
    ----------------
    \\*args : list, optional
        Arguments.
    \\**kwargs : dict, optional
        Keywords arguments.

    Returns
    -------
    bool
        Definition success.

    Examples
    --------
    >>> warning('This is a warning!')  # doctest: +SKIP
    """

    kwargs['category'] = kwargs.get('category', ColourWarning)

    warn(*args, **kwargs)

    return True


def runtime_warning(*args, **kwargs):
    """
    Issues a runtime warning.

    Other Parameters
    ----------------
    \\*args : list, optional
        Arguments.
    \\**kwargs : dict, optional
        Keywords arguments.

    Returns
    -------
    bool
        Definition success.

    Examples
    --------
    >>> usage_warning('This is a runtime warning!')  # doctest: +SKIP
    """

    kwargs['category'] = ColourRuntimeWarning

    warning(*args, **kwargs)

    return True


def usage_warning(*args, **kwargs):
    """
    Issues an usage warning.

    Other Parameters
    ----------------
    \\*args : list, optional
        Arguments.
    \\**kwargs : dict, optional
        Keywords arguments.

    Returns
    -------
    bool
        Definition success.

    Examples
    --------
    >>> usage_warning('This is an usage warning!')  # doctest: +SKIP
    """

    kwargs['category'] = ColourUsageWarning

    warning(*args, **kwargs)

    return True


def filter_warnings(state=True,
                    colour_warnings=True,
                    colour_runtime_warnings=False,
                    colour_usage_warnings=False,
                    python_warnings=False):
    """
    Filters *Colour* and also optionally overall Python warnings.

    Parameters
    ----------
    state : bool, optional
        Warnings filter state.
    colour_warnings : bool, optional
        Whether to filter *Colour* warnings, this also filters *Colour* usage
        and runtime warnings.
    colour_runtime_warnings : bool, optional
        Whether to filter *Colour* runtime warnings.
    colour_usage_warnings : bool, optional
        Whether to filter *Colour* usage warnings.
    python_warnings : bool, optional
        Whether to filter *Python* warnings.

    Returns
    -------
    bool
        Definition success.

    Examples
    --------
    # Filtering *Colour* warnings:
    >>> filter_warnings()
    True

    # Filtering *Colour* runtime warnings:
    >>> filter_warnings(colour_warnings=False, colour_runtime_warnings=True)
    True

    # Filtering *Colour* usage warnings:
    >>> filter_warnings(colour_warnings=False, colour_usage_warnings=True)
    True

    # Filtering *Colour* and also Python warnings:
    >>> filter_warnings(python_warnings=True)
    True
    """

    action = 'ignore' if state else 'default'

    if colour_warnings:
        filterwarnings(action, category=ColourWarning)

    if colour_runtime_warnings:
        filterwarnings(action, category=ColourRuntimeWarning)

    if colour_usage_warnings:
        filterwarnings(action, category=ColourUsageWarning)

    if python_warnings:
        filterwarnings(action, category=Warning)

    return True


# Defaulting to filter *Colour* runtime warnings.
filter_warnings(colour_warnings=False, colour_runtime_warnings=True)


@contextmanager
def suppress_warnings(colour_warnings=True,
                      colour_runtime_warnings=False,
                      colour_usage_warnings=False,
                      python_warnings=False):
    """
    A context manager filtering *Colour* and also optionally overall Python
    warnings.

    Parameters
    ----------
    colour_warnings : bool, optional
        Whether to filter *Colour* warnings, this also filters *Colour* usage
        and runtime warnings.
    colour_runtime_warnings : bool, optional
        Whether to filter *Colour* runtime warnings.
    colour_usage_warnings : bool, optional
        Whether to filter *Colour* usage warnings.
    python_warnings : bool, optional
        Whether to filter *Python* warnings.
    """

    filters = warnings.filters
    show_warnings = warnings.showwarning

    filter_warnings(
        colour_warnings=colour_warnings,
        colour_runtime_warnings=colour_runtime_warnings,
        colour_usage_warnings=colour_usage_warnings,
        python_warnings=python_warnings)
    try:
        yield
    finally:
        warnings.filters = filters
        warnings.showwarning = show_warnings


@contextmanager
def numpy_print_options(*args, **kwargs):
    """
    A context manager implementing context changes to *Numpy* print behaviour.

    Other Parameters
    ----------------
    \\*args : list, optional
        Arguments.
    \\**kwargs : dict, optional
        Keywords arguments.

    Examples
    -------
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


ANCILLARY_COLOUR_SCIENCE_PACKAGES = OrderedDict()
"""
Ancillary *colour-science.org* packages to describe.

ANCILLARY_COLOUR_SCIENCE_PACKAGES : OrderedDict
"""

ANCILLARY_RUNTIME_PACKAGES = OrderedDict()
"""
Ancillary runtime packages to describe.

ANCILLARY_RUNTIME_PACKAGES : OrderedDict
"""

ANCILLARY_DEVELOPMENT_PACKAGES = OrderedDict()
"""
Ancillary development packages to describe.

ANCILLARY_DEVELOPMENT_PACKAGES : OrderedDict
"""

ANCILLARY_EXTRAS_PACKAGES = OrderedDict()
"""
Ancillary extras packages to describe.

ANCILLARY_EXTRAS_PACKAGES : OrderedDict
"""


def describe_environment(runtime_packages=True,
                         development_packages=False,
                         extras_packages=False,
                         print_environment=True,
                         **kwargs):
    """
    Describes *Colour* running environment, i.e. interpreter, runtime and
    development packages.

    Parameters
    ----------
    runtime_packages : bool, optional
        Whether to return the runtime packages versions.
    development_packages : bool, optional
        Whether to return the development packages versions.
    extras_packages : bool, optional
        Whether to return the extras packages versions.
    print_environment : bool, optional
        Whether to print the environment.

    Other Parameters
    ----------------
    padding : unicode, optional
        {:func:`colour.utilities.message_box`},
        Padding on each sides of the message.
    print_callable : callable, optional
        {:func:`colour.utilities.message_box`},
        Callable used to print the message box.
    width : int, optional
        {:func:`colour.utilities.message_box`},
        Message box width.

    Returns
    -------
    defaultdict
        Environment.

    Examples
    --------
    >>> environment = describe_environment(width=75)  # doctest: +SKIP
    ===========================================================================
    *                                                                         *
    *   Interpreter :                                                         *
    *       python : 3.7.4 (default, Sep  7 2019, 18:27:02)                   *
    *                [Clang 10.0.1 (clang-1001.0.46.4)]                       *
    *                                                                         *
    *   colour-science.org :                                                  *
    *       colour : v0.3.13-293-gecf1dc8a                                    *
    *                                                                         *
    *   Runtime :                                                             *
    *       imageio : 2.6.1                                                   *
    *       numpy : 1.17.2                                                    *
    *       scipy : 1.3.1                                                     *
    *       six : 1.12.0                                                      *
    *       pandas : 0.24.2                                                   *
    *       matplotlib : 3.0.3                                                *
    *       networkx : 2.3                                                    *
    *       pygraphviz : 1.5                                                  *
    *                                                                         *
    ===========================================================================
    >>> environment = describe_environment(True, True, True, width=75)
    ... # doctest: +SKIP
    ===========================================================================
    *                                                                         *
    *   Interpreter :                                                         *
    *       python : 3.7.4 (default, Sep  7 2019, 18:27:02)                   *
    *                [Clang 10.0.1 (clang-1001.0.46.4)]                       *
    *                                                                         *
    *   colour-science.org :                                                  *
    *       colour : v0.3.13-293-gecf1dc8a                                    *
    *                                                                         *
    *   Runtime :                                                             *
    *       imageio : 2.6.1                                                   *
    *       numpy : 1.17.2                                                    *
    *       scipy : 1.3.1                                                     *
    *       six : 1.12.0                                                      *
    *       pandas : 0.24.2                                                   *
    *       matplotlib : 3.0.3                                                *
    *       networkx : 2.3                                                    *
    *       pygraphviz : 1.5                                                  *
    *                                                                         *
    *   Development :                                                         *
    *       biblib-simple : 0.1.1                                             *
    *       coverage : 4.5.4                                                  *
    *       coveralls : 1.8.2                                                 *
    *       flake8 : 3.7.8                                                    *
    *       invoke : 1.3.0                                                    *
    *       jupyter : 1.0.0                                                   *
    *       mock : 3.0.5                                                      *
    *       nose : 1.3.7                                                      *
    *       pre-commit : 1.18.3                                               *
    *       pytest : 5.2.1                                                    *
    *       restructuredtext-lint : 1.3.0                                     *
    *       sphinx : 2.2.0                                                    *
    *       sphinx_rtd_theme : 0.4.3                                          *
    *       sphinxcontrib-bibtex : 1.0.0                                      *
    *       toml : 0.10.0                                                     *
    *       twine : 1.15.0                                                    *
    *       yapf : 0.23.0                                                     *
    *                                                                         *
    *   Extras :                                                              *
    *       ipywidgets : 7.5.1                                                *
    *       notebook : 6.0.1                                                  *
    *                                                                         *
    ===========================================================================
    """

    environment = defaultdict(OrderedDict)

    environment['Interpreter']['python'] = sys.version

    import subprocess  # nosec

    import colour

    # TODO: Implement support for "pyproject.toml" file whenever "TOML" is
    # supported in the standard library.

    # NOTE: A few clauses are not reached and a few packages are not available
    # during continuous integration and are thus ignored for coverage.
    try:  # pragma: no cover
        version = subprocess.check_output(  # nosec
            ['git', 'describe'],
            cwd=colour.__path__[0],
            stderr=subprocess.STDOUT).strip()
        version = version.decode('utf-8')
    except Exception:  # pragma: no cover
        version = colour.__version__

    environment['colour-science.org']['colour'] = version
    environment['colour-science.org'].update(ANCILLARY_COLOUR_SCIENCE_PACKAGES)

    if runtime_packages:
        for package in [
                'imageio', 'matplotlib', 'networkx', 'numpy', 'pandas',
                'pygraphviz', 'scipy', 'six'
        ]:
            try:
                namespace = __import__(package)
                environment['Runtime'][package] = namespace.__version__
            except ImportError:
                continue

        # OpenImageIO
        try:  # pragma: no cover
            namespace = __import__('OpenImageIO')
            environment['Runtime']['OpenImageIO'] = namespace.VERSION_STRING
        except ImportError:  # pragma: no cover
            pass

        environment['Runtime'].update(ANCILLARY_RUNTIME_PACKAGES)

    def _get_package_version(package, mapping):
        """
        Returns given package version.
        """

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
            'biblib.bib': 'biblib-simple',
            'pre_commit': 'pre-commit',
            'restructuredtext_lint': 'restructuredtext-lint',
            'sphinxcontrib.bibtex': 'sphinxcontrib-bibtex'
        }
        for package in [
                'biblib.bib', 'coverage', 'coveralls', 'flake8', 'invoke',
                'jupyter', 'mock', 'nose', 'pre_commit', 'pytest',
                'restructuredtext_lint', 'sphinx', 'sphinx_rtd_theme',
                'sphinxcontrib.bibtex', 'toml', 'twine', 'yapf'
        ]:
            try:
                version = _get_package_version(package, mapping)
                package = mapping.get(package, package)

                environment['Development'][package] = version
            except Exception:
                continue

        environment['Development'].update(ANCILLARY_DEVELOPMENT_PACKAGES)

    if extras_packages:
        mapping = {}
        for package in ['ipywidgets', 'notebook']:
            try:
                version = _get_package_version(package, mapping)
                package = mapping.get(package, package)

                environment['Extras'][package] = version
            except Exception:
                continue

        environment['Extras'].update(ANCILLARY_EXTRAS_PACKAGES)

    if print_environment:
        message = str()
        for category in ('Interpreter', 'colour-science.org', 'Runtime',
                         'Development', 'Extras'):
            elements = environment.get(category)
            if not elements:
                continue

            message += '{0} :\n'.format(category)
            for key, value in elements.items():
                lines = value.split('\n')
                message += '    {0} : {1}\n'.format(key, lines.pop(0))
                indentation = len('    {0} : '.format(key))
                for line in lines:
                    message += '{0}{1}\n'.format(' ' * indentation, line)

            message += '\n'

        message_box(message.strip(), **kwargs)

    return environment
