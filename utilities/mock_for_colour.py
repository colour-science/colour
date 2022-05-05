"""
Mock for Colour
===============

Defines various mock objects to use with
`Colour <https://github.com/colour-science/colour>`__.

References
----------
-   :cite:`SphinxTeam` : Sphinx Team. (n.d.). sphinx.ext.autodoc.mock.
    Retrieved May 2, 2019, from https://github.com/sphinx-doc/sphinx/blob/\
master/sphinx/ext/autodoc/mock.py
"""

import os

from types import FunctionType, MethodType, ModuleType

__author__ = "Sphinx Team, Colour Developers"
__copyright__ = "Copyright 2007-2019 - Sphinx Team"
__copyright__ += ", "
__copyright__ += "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "MockObject",
    "MockModule",
    "mock_scipy_for_colour",
]


class MockObject:
    """
    Mock an object to handle *Colour* requirements such as *Scipy*.

    Other Parameters
    ----------------
    args
        Arguments.
    kwargs
        Keywords arguments.

    References
    ----------
    :cite:`SphinxTeam`
    """

    __display_name__ = "MockObject"

    def __new__(cls, *args, **kwargs):
        """
        Return a new instance of the :class:`MockObject` class.

        Other Parameters
        ----------------
        args
            Arguments.
        kwargs
            Keywords arguments.
        """

        if len(args) == 3 and isinstance(args[1], tuple):
            superclass = args[1][-1].__class__
            if superclass is cls:
                return _make_subclass(
                    args[0],
                    superclass.__display_name__,
                    superclass=superclass,
                    attributes=args[2],
                )

        return super().__new__(cls)

    def __init__(self, *args, **kwargs):
        pass

    def __len__(self):
        """Return the length of the :class:`MockObject` class instance, i.e. 0."""

        return 0

    def __contains__(self, key):
        """
        Return whether the :class:`MockObject` class instance contains given
        key.

        Parameters
        ----------
        key
            Key to check whether is is contained in the :class:`MockObject`
            class instance.
        """

        return False

    def __iter__(self):
        """Iterate over the :class:`MockObject` class instance."""

        return iter([])

    def __mro_entries__(self, bases):
        """
        If an object that is not a class object appears in the tuple of bases
        of a class definition, then method __mro_entries__ is searched on it.
        """

        return (self.__class__,)

    def __getitem__(self, key):
        """
        Return the value at given key from the :class:`MockObject` class
        instance.

        Parameters
        ----------
        key
            Key to return the value at.
        """

        return _make_subclass(key, self.__display_name__, self.__class__)()

    def __getattr__(self, key):
        """
        Return the attribute at given key from the :class:`MockObject` class
        instance.

        Parameters
        ----------
        key
            Key to return the attribute at.
        """

        return _make_subclass(key, self.__display_name__, self.__class__)()

    def __call__(self, *args, **kwargs):
        """
        Call the :class:`MockObject` class instance.

        Other Parameters
        ----------------
        args
            Arguments.
        kwargs
            Keywords arguments.
        """

        if args and type(args[0]) in [FunctionType, MethodType]:
            return args[0]

        return self

    def __repr__(self):
        """
        Return an evaluable string representation of the :class:`MockObject`
        class instance.
        """

        return self.__display_name__


def _make_subclass(
    name, module, superclass=MockObject, attributes=None
):  # noqa: D405,D407,D410,D411
    """
    Produce sub-classes of given super-class type.

    Parameters
    ----------
    name
        Name of the sub-class.
    module
        Name of the sub-class module.
    superclass
        Super-class type.
    attributes
        Attributes to set the sub-class with.
    """

    attrs = {"__module__": module, "__display_name__": module + "." + name}
    attrs.update(attributes or {})

    return type(name, (superclass,), attrs)


class MockModule(ModuleType):
    """
    A mock object used to mock modules.

    Parameters
    ----------
    name
        Name of the mocked module.

    References
    ----------
    :cite:`SphinxTeam`
    """

    __file__ = os.devnull

    def __init__(self, name):
        super().__init__(name)
        self.__all__ = []
        self.__path__ = []

    def __getattr__(self, name):
        """
        Return the attribute at given name from the :class:`MockModule` class
        instance.

        Parameters
        ----------
        name
            Name to return the attribute at.
        """

        return _make_subclass(name, self.__name__)()

    def __repr__(self):
        """
        Return an evaluable string representation of the :class:`MockModule`
        class instance.
        """

        return self.__name__


def mock_scipy_for_colour():
    """Mock *Scipy* for *Colour*."""

    import sys

    for module in (
        "scipy",
        "scipy.interpolate",
        "scipy.linalg",
        "scipy.ndimage",
        "scipy.spatial",
        "scipy.spatial.distance",
        "scipy.optimize",
    ):
        sys.modules[str(module)] = MockModule(str(module))


if __name__ == "__main__":
    import sys

    for module in (
        "scipy",
        "scipy.interpolate",
        "scipy.linalg",
        "scipy.ndimage",
        "scipy.spatial",
        "scipy.spatial.distance",
        "scipy.optimize",
    ):
        sys.modules[str(module)] = MockModule(str(module))

    import colour

    xyY = (0.4316, 0.3777, 0.1008)
    print(colour.xyY_to_XYZ(xyY))
