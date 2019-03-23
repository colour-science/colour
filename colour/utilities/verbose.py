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
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'ColourWarning', 'ColourUsageWarning', 'ColourRuntimeWarning',
    'message_box', 'show_warning', 'warning', 'runtime_warning',
    'usage_warning', 'filter_warnings', 'suppress_warnings',
    'numpy_print_options', 'ANCILLARY_COLOUR_SCIENCE_PACKAGES',
    'ANCILLARY_RUNTIME_PACKAGES', 'ANCILLARY_DEVELOPMENT_PACKAGES',
    'describe_environment'
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
                 frame_range=(1, 2)):
    """
    Replaces :func:`warnings.showwarning` definition to allow traceback
    printing.

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


def describe_environment(runtime_packages=True,
                         development_packages=False,
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
    *       python : 2.7.14 | packaged by conda-forge | (default, Dec 25      *
    *   2017, 01:18:54)                                                       *
    *                [GCC 4.2.1 Compatible Apple LLVM 6.1.0                   *
    *   (clang-602.0.53)]                                                     *
    *                                                                         *
    *   colour-science.org :                                                  *
    *       colour : v0.3.11-323-g380c1838                                    *
    *                                                                         *
    *   Runtime :                                                             *
    *       numpy : 1.14.3                                                    *
    *       scipy : 1.0.0                                                     *
    *       pandas : 0.22.0                                                   *
    *       matplotlib : 2.2.2                                                *
    *       notebook : 5.4.0                                                  *
    *       ipywidgets : 7.2.1                                                *
    *                                                                         *
    ===========================================================================
    >>> environment = describe_environment(True, True, width=75)
    ... # doctest: +SKIP
    ===========================================================================
    *                                                                         *
    *   Interpreter :                                                         *
    *       python : 2.7.14 | packaged by conda-forge | (default, Dec 25      *
    *   2017, 01:18:54)                                                       *
    *                [GCC 4.2.1 Compatible Apple LLVM 6.1.0                   *
    *   (clang-602.0.53)]                                                     *
    *                                                                         *
    *   colour-science.org :                                                  *
    *       colour : v0.3.11-323-g380c1838                                    *
    *                                                                         *
    *   Runtime :                                                             *
    *       numpy : 1.14.3                                                    *
    *       scipy : 1.0.0                                                     *
    *       pandas : 0.22.0                                                   *
    *       matplotlib : 2.2.2                                                *
    *       notebook : 5.4.0                                                  *
    *       ipywidgets : 7.2.1                                                *
    *                                                                         *
    *   Development :                                                         *
    *       coverage : 4.5.1                                                  *
    *       flake8 : 3.5.0                                                    *
    *       invoke : 0.22.1                                                   *
    *       mock : 2.0.0                                                      *
    *       nose : 1.3.7                                                      *
    *       restructuredtext_lint : 1.1.3                                     *
    *       six : 1.11.0                                                      *
    *       sphinx : 1.7.5                                                    *
    *       sphinx_rtd_theme : 0.2.4                                          *
    *       twine : 1.10.0                                                    *
    *       yapf : 0.20.2                                                     *
    *                                                                         *
    ===========================================================================
    """

    environment = defaultdict(OrderedDict)

    environment['Interpreter']['python'] = sys.version

    import subprocess  # nosec

    import colour

    try:
        version = subprocess.check_output(  # nosec
            ['git', 'describe'], cwd=colour.__path__[0]).strip()
        version = version.decode('utf-8')
    except Exception:
        version = colour.__version__

    environment['colour-science.org']['colour'] = version
    environment['colour-science.org'].update(ANCILLARY_COLOUR_SCIENCE_PACKAGES)

    if runtime_packages:
        for package in ('numpy', 'scipy', 'pandas', 'matplotlib', 'notebook',
                        'ipywidgets'):
            try:
                namespace = __import__(package)
                environment['Runtime'][package] = namespace.__version__
            except ImportError:
                continue

        # OpenImageIO
        try:
            namespace = __import__('OpenImageIO')
            environment['Runtime']['OpenImageIO'] = namespace.VERSION_STRING
        except ImportError:
            pass

        environment['Runtime'].update(ANCILLARY_RUNTIME_PACKAGES)

    if development_packages:
        for package in ('coverage', 'flake8', 'invoke', 'mock', 'nose',
                        'restructuredtext_lint', 'six', 'sphinx',
                        'sphinxcontrib.bibtex', 'sphinx_rtd_theme', 'twine',
                        'yapf'):
            try:
                namespace = __import__(package)
                if package == 'restructuredtext_lint':
                    with open(
                            os.path.join(
                                os.path.dirname(namespace.__file__),
                                'VERSION'), 'r') as version_file:
                        version = version_file.read().strip()
                elif package == 'sphinxcontrib.bibtex':
                    import pip

                    for distribution in pip.get_installed_distributions():
                        if distribution.name == package:
                            version = distribution.version
                            break
                else:
                    version = namespace.__version__

                environment['Development'][package] = version
            except (AttributeError, ImportError):
                continue

        environment['Development'].update(ANCILLARY_DEVELOPMENT_PACKAGES)

    if print_environment:
        message = str()
        for category in ('Interpreter', 'colour-science.org', 'Runtime',
                         'Development'):
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
