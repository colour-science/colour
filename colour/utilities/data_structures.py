"""
Data Structures
===============

Defines various data structures classes:

-   :class:`colour.utilities.Structure`: An object similar to C/C++ structured
    type.
-   :class:`colour.utilities.Lookup`: A :class:`dict` sub-class acting as a
    lookup to retrieve keys by values.
-   :class:`CaseInsensitiveMapping`: A case insensitive
    :class:`dict`-like object allowing values retrieving from keys while
    ignoring the key case.
-   :class:`colour.utilities.LazyCaseInsensitiveMapping`: Another case
    insensitive mapping allowing lazy values retrieving from keys while
    ignoring the key case.
-   :class:`colour.utilities.Node`: A basic node object supporting creation of
    basic node trees.

References
----------
-   :cite:`Mansencalc` : Mansencal, T. (n.d.). Lookup.
    https://github.com/KelSolaar/Foundations/blob/develop/foundations/\
data_structures.py
-   :cite:`Rakotoarison2017` : Rakotoarison, H. (2017). Bunch. Retrieved
    December 4, 2021, from https://github.com/scikit-learn/scikit-learn/blob/\
0d378913b/sklearn/utils/__init__.py#L83
-   :cite:`Reitza` : Reitz, K. (n.d.). CaseInsensitiveDict.
    https://github.com/kennethreitz/requests/blob/v1.2.3/requests/\
structures.py#L37
"""

from __future__ import annotations

from collections.abc import MutableMapping

from colour.hints import (
    Any,
    Boolean,
    Dict,
    Generator,
    Integer,
    Iterable,
    List,
    Mapping,
    Optional,
    Union,
)
from colour.utilities.documentation import is_documentation_building

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "attest",
    "Structure",
    "Lookup",
    "CaseInsensitiveMapping",
    "LazyCaseInsensitiveMapping",
    "Node",
]


def attest(condition: Boolean, message: str = ""):
    """
    Provide the `assert` statement functionality without being disabled by
    optimised Python execution.

    See :func:`colour.utilities.assert` for more information.

    Notes
    -----
    -   This definition is duplicated to avoid import circular dependency.
    """

    # Avoiding circular dependency.
    import colour.utilities

    colour.utilities.attest(condition, message)


class Structure(dict):
    """
    Define a :class:`dict`-like object allowing to access key values using dot
    syntax.

    Other Parameters
    ----------------
    args
        Arguments.
    kwargs
        Key / value pairs.

    Methods
    -------
    -   :meth:`~colour.utilities.Structure.__init__`
    -   :meth:`~colour.utilities.Structure.__setattr__`
    -   :meth:`~colour.utilities.Structure.__delattr__`
    -   :meth:`~colour.utilities.Structure.__dir__`
    -   :meth:`~colour.utilities.Structure.__getattr__`
    -   :meth:`~colour.utilities.Structure.__setstate__`

    References
    ----------
    :cite:`Rakotoarison2017`

    Examples
    --------
    >>> person = Structure(first_name='John', last_name='Doe', gender='male')
    >>> person.first_name
    'John'
    >>> sorted(person.keys())
    ['first_name', 'gender', 'last_name']
    >>> person['gender']
    'male'
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    def __setattr__(self, name: str, value: Any):
        """
        Assign given value to the attribute with given name.

        Parameters
        ----------
        name
            Name of the attribute to assign the ``value`` to.
        value
            Value to assign to the attribute.
        """

        self[name] = value

    def __delattr__(self, name: str):
        """
        Delete the attribute with given name.

        Parameters
        ----------
        name
            Name of the attribute to delete.
        """

        del self[name]

    def __dir__(self) -> Iterable:
        """
        Return a list of valid attributes for the :class:`dict`-like object.

        Returns
        -------
        :class:`list`
            List of valid attributes for the :class:`dict`-like object.
        """

        return self.keys()

    def __getattr__(self, name: str) -> Any:
        """
        Return the value from the attribute with given name.

        Parameters
        ----------
        name
            Name of the attribute to get the value from.

        Returns
        -------
        :class:`object`

        Raises
        ------
        AttributeError
            If the attribute is not defined.
        """

        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setstate__(self, state):
        """Set the object state when unpickling."""
        # See https://github.com/scikit-learn/scikit-learn/issues/6196 for more
        # information.

        pass


class Lookup(dict):
    """
    Extend :class:`dict` type to provide a lookup by value(s).

    Methods
    -------
    -   :meth:`~colour.utilities.Lookup.keys_from_value`
    -   :meth:`~colour.utilities.Lookup.first_key_from_value`

    References
    ----------
    :cite:`Mansencalc`

    Examples
    --------
    >>> person = Lookup(first_name='John', last_name='Doe', gender='male')
    >>> person.first_key_from_value('John')
    'first_name'
    >>> persons = Lookup(John='Doe', Jane='Doe', Luke='Skywalker')
    >>> sorted(persons.keys_from_value('Doe'))
    ['Jane', 'John']
    """

    def keys_from_value(self, value: Any) -> List:
        """
        Get the keys associated with given value.

        Parameters
        ----------
        value
            Value to find the associated keys.

        Returns
        -------
        :class:`list`
            Keys associated with given value.
        """

        keys = []
        for key, data in self.items():
            matching = data == value
            try:
                matching = all(matching)

            except TypeError:
                matching = all((matching,))

            if matching:
                keys.append(key)

        return keys

    def first_key_from_value(self, value: Any) -> Any:
        """
        Get the first key associated with given value.

        Parameters
        ----------
        value
            Value to find the associated first key.

        Returns
        -------
        :class:`object`
            First key associated with given value.
        """

        return self.keys_from_value(value)[0]


class CaseInsensitiveMapping(MutableMapping):
    """
    Implement a case-insensitive :class:`dict`-like object.

    Allows values retrieving from keys while ignoring the key case.
    The keys are expected to be str or :class:`str`-like objects supporting the
    :meth:`str.lower` method.

    Parameters
    ----------
    data
        Data to store into the case-insensitive :class:`dict`-like object at
        initialisation.

    Other Parameters
    ----------------
    kwargs
        Key / value pairs to store into the mapping at initialisation.

    Attributes
    ----------
    -   :attr:`~colour.utilities.CaseInsensitiveMapping.data`

    Methods
    -------
    -   :meth:`~colour.utilities.CaseInsensitiveMapping.__init__`
    -   :meth:`~colour.utilities.CaseInsensitiveMapping.__repr__`
    -   :meth:`~colour.utilities.CaseInsensitiveMapping.__setitem__`
    -   :meth:`~colour.utilities.CaseInsensitiveMapping.__getitem__`
    -   :meth:`~colour.utilities.CaseInsensitiveMapping.__delitem__`
    -   :meth:`~colour.utilities.CaseInsensitiveMapping.__contains__`
    -   :meth:`~colour.utilities.CaseInsensitiveMapping.__iter__`
    -   :meth:`~colour.utilities.CaseInsensitiveMapping.__len__`
    -   :meth:`~colour.utilities.CaseInsensitiveMapping.__eq__`
    -   :meth:`~colour.utilities.CaseInsensitiveMapping.__ne__`
    -   :meth:`~colour.utilities.CaseInsensitiveMapping.copy`
    -   :meth:`~colour.utilities.CaseInsensitiveMapping.lower_items`

    References
    ----------
    :cite:`Reitza`

    Examples
    --------
    >>> methods = CaseInsensitiveMapping({'McCamy': 1, 'Hernandez': 2})
    >>> methods['mccamy']
    1
    """

    def __init__(
        self, data: Optional[Union[Generator, Mapping]] = None, **kwargs: Any
    ):
        self._data: Dict = dict()

        self.update({} if data is None else data, **kwargs)

    @property
    def data(self) -> Dict:
        """
        Getter property for the case-insensitive :class:`dict`-like object
        data.

        Returns
        -------
        :class:`dict`
            Data.
        """

        return self._data

    def __repr__(self) -> str:
        """
        Return an evaluable string representation of the case-insensitive
        :class:`dict`-like object.

        Returns
        -------
        :class:`str`
            Evaluable string representation.
        """

        if is_documentation_building():  # pragma: no cover
            representation = repr(
                dict(zip(self.keys(), ["..."] * len(self)))
            ).replace("'...'", "...")
            return f"{self.__class__.__name__}({representation})"
        else:
            return f"{self.__class__.__name__}({dict(self.items())})"

    def __setitem__(self, item: Union[str, Any], value: Any):
        """
        Set given item with given value in the case-insensitive
        :class:`dict`-like object.

        Parameters
        ----------
        item
            Item to set in the case-insensitive :class:`dict`-like object.
        value
            Value to store in the case-insensitive :class:`dict`-like object.

        Notes
        -----
        -   The item is stored as lower-case while the original name and its
            value are stored together as the value in a *tuple*::

            {"item.lower()": ("item", value)}
        """

        self._data[self._lower_key(item)] = (item, value)

    def __getitem__(self, item: Union[str, Any]) -> Any:
        """
        Return the value of given item from the case-insensitive
        :class:`dict`-like object.

        Parameters
        ----------
        item
            Item to retrieve the value of from the case-insensitive
            :class:`dict`-like object.

        Returns
        -------
        :class:`object`
            Item value.

        Notes
        -----
        -   The item value is retrieved by using its lower-case variant.
        """

        return self._data[self._lower_key(item)][1]

    def __delitem__(self, item: Union[str, Any]):
        """
        Delete given item from the case-insensitive :class:`dict`-like object.

        Parameters
        ----------
        item
            Item to delete from the case-insensitive :class:`dict`-like object.

        Notes
        -----
        -   The item is deleted by using its lower-case variant.
        """

        del self._data[self._lower_key(item)]

    def __contains__(self, item: Union[str, Any]) -> bool:
        """
        Return whether the case-insensitive :class:`dict`-like object contains
        given item.

        Parameters
        ----------
        item
            Item to find whether it is in the case-insensitive
            :class:`dict`-like object.

        Returns
        -------
        :class:`bool`
            Whether given item is in the case-insensitive :class:`dict`-like
            object.
        """

        return self._lower_key(item) in self._data

    def __iter__(self) -> Generator:
        """
        Iterate over the items of the case-insensitive :class:`dict`-like
        object.

        Yields
        ------
        Generator
            Item generator.

        Notes
        -----
        -   The iterated items are the original items.
        """

        return (item for item, value in self._data.values())

    def __len__(self) -> Integer:
        """
        Return the items count.

        Returns
        -------
        :class:`numpy.integer`
            Items count.
        """

        return len(self._data)

    def __eq__(self, other: Any) -> bool:
        """
        Return whether the case-insensitive :class:`dict`-like object is equal
        to given other object.

        Parameters
        ----------
        other
            Object to test whether it is equal to the case-insensitive
            :class:`dict`-like object

        Returns
        -------
        :class:`bool`
            Whether given object is equal to the case-insensitive
            :class:`dict`-like object.
        """

        if isinstance(other, Mapping):
            other_mapping = CaseInsensitiveMapping(other)
        else:
            raise ValueError(
                f"Impossible to test equality with "
                f'"{other.__class__.__name__}" class type!'
            )

        return dict(self.lower_items()) == dict(other_mapping.lower_items())

    def __ne__(self, other: Any) -> bool:
        """
        Return whether the case-insensitive :class:`dict`-like object is not
        equal to given other object.

        Parameters
        ----------
        other
            Object to test whether it is not equal to the case-insensitive
            :class:`dict`-like object

        Returns
        -------
        :class:`bool`
            Whether given object is not equal to the case-insensitive
            :class:`dict`-like object.
        """

        return not (self == other)

    @staticmethod
    def _lower_key(key: Union[str, Any]) -> Union[str, Any]:
        """
        Return the lower-case variant of given key, if the key cannot be
        lower-cased, it is passed unmodified.

        Parameters
        ----------
        key
            Key to return the lower-case variant.

        Returns
        -------
        :class:`str` or :class:`object`
            Key lower-case variant.
        """

        try:
            return key.lower()
        except AttributeError:
            return key

    def copy(self) -> CaseInsensitiveMapping:
        """
        Return a copy of the case-insensitive :class:`dict`-like object.

        Returns
        -------
        :class:`CaseInsensitiveMapping`
            Case-insensitive :class:`dict`-like object copy.

        Warnings
        --------
        -   The :class:`CaseInsensitiveMapping` class copy returned is a
            *copy* of the object not a *deepcopy*!
        """

        return CaseInsensitiveMapping(dict(self._data.values()))

    def lower_items(self) -> Generator:
        """
        Iterate over the lower-case items of the case-insensitive
        :class:`dict`-like object.

        Yields
        ------
        Generator
            Item generator.

        Notes
        -----
        -   The iterated items are the lower-case items.
        """

        return ((item, value[1]) for (item, value) in self._data.items())


class LazyCaseInsensitiveMapping(CaseInsensitiveMapping):
    """
    Implement a lazy case-insensitive :class:`dict`-like object inheriting
    from :class:`CaseInsensitiveMapping` class.

    Allows lay values retrieving from keys while ignoring the key case.
    The keys are expected to be str or :class:`str`-like objects supporting the
    :meth:`str.lower` method.

    The lazy retrieval is performed as follows: If the value is a callable,
    then it is evaluated and its return value is stored in place of the current
    value.

    Parameters
    ----------
    data
        Data to store into the lazy case-insensitive :class:`dict`-like object
        at initialisation.

    Other Parameters
    ----------------
    kwargs
        Key / value pairs to store into the mapping at initialisation.

    Methods
    -------
    -   :meth:`~colour.utilities.LazyCaseInsensitiveMapping.__getitem__`

    Examples
    --------
    >>> def callable_a():
    ...     print(2)
    ...     return 2
    >>> methods = LazyCaseInsensitiveMapping(
    ...     {'McCamy': 1, 'Hernandez': callable_a})
    >>> methods['mccamy']
    1
    >>> methods['hernandez']
    2
    2
    """

    def __getitem__(self, item: Union[str, Any]) -> Any:
        """
        Return the value of given item from the case-insensitive
        :class:`dict`-like object.

        Parameters
        ----------
        item
            Item to retrieve the value of from the case-insensitive
            :class:`dict`-like object.

        Returns
        -------
        :class:`object`
            Item value.

        Notes
        -----
        -   The item value is retrieved by using its lower-case variant.
        """

        import colour

        value = super().__getitem__(item)

        if callable(value) and hasattr(colour, "__disable_lazy_load__"):
            value = value()
            super().__setitem__(item, value)

        return value


class Node:
    """
    Represent a basic node supporting the creation of basic node trees.

    Parameters
    ----------
    name
        Node name.
    parent
        Parent of the node.
    children
        Children of the node.
    data
        The data belonging to this node.

    Attributes
    ----------
    -   :attr:`~colour.utilities.Node.name`
    -   :attr:`~colour.utilities.Node.parent`
    -   :attr:`~colour.utilities.Node.children`
    -   :attr:`~colour.utilities.Node.id`
    -   :attr:`~colour.utilities.Node.root`
    -   :attr:`~colour.utilities.Node.leaves`
    -   :attr:`~colour.utilities.Node.siblings`
    -   :attr:`~colour.utilities.Node.data`

    Methods
    -------
    -   :meth:`~colour.utilities.Node.__new__`
    -   :meth:`~colour.utilities.Node.__init__`
    -   :meth:`~colour.utilities.Node.__str__`
    -   :meth:`~colour.utilities.Node.__len__`
    -   :meth:`~colour.utilities.Node.is_root`
    -   :meth:`~colour.utilities.Node.is_inner`
    -   :meth:`~colour.utilities.Node.is_leaf`
    -   :meth:`~colour.utilities.Node.walk`
    -   :meth:`~colour.utilities.Node.render`

    Examples
    --------
    >>> node_a = Node('Node A')
    >>> node_b = Node('Node B', node_a)
    >>> node_c = Node('Node C', node_a)
    >>> node_d = Node('Node D', node_b)
    >>> node_e = Node('Node E', node_b)
    >>> node_f = Node('Node F', node_d)
    >>> node_g = Node('Node G', node_f)
    >>> node_h = Node('Node H', node_g)
    >>> [node.name for node in node_a.leaves]
    ['Node H', 'Node E', 'Node C']
    >>> print(node_h.root.name)
    Node A
    >>> len(node_a)
    7
    """

    _INSTANCE_ID: Integer = 1
    """
    Node id counter.

    _INSTANCE_ID
    """

    def __new__(cls, *args: Any, **kwargs: Any) -> Node:
        """
        Return a new instance of the :class:`colour.utilities.Node` class.

        Other Parameters
        ----------------
        args
            Arguments.
        kwargs
            Keywords arguments.
        """

        instance = super().__new__(cls)

        instance._id = Node._INSTANCE_ID  # type: ignore[attr-defined]
        Node._INSTANCE_ID += 1

        return instance

    def __init__(
        self,
        name: Optional[str] = None,
        parent: Optional[Node] = None,
        children: Optional[List[Node]] = None,
        data: Optional[Any] = None,
    ):
        self._name: str = f"{self.__class__.__name__}#{self.id}"
        self.name = self._name if name is None else name
        self._parent: Optional[Node] = None
        self.parent = parent
        self._children: List[Node] = []
        self.children = self._children if children is None else children
        self._data: Optional[Any] = data

    @property
    def name(self) -> str:
        """
        Getter and setter property for the name.

        Parameters
        ----------
        value
            Value to set the name with.

        Returns
        -------
        :class:`str`
            Node name.
        """

        return self._name

    @name.setter
    def name(self, value: str):
        """Setter for the **self.name** property."""

        attest(
            isinstance(value, str),
            f'"name" property: "{value}" type is not "str"!',
        )

        self._name = value

    @property
    def parent(self) -> Optional[Node]:
        """
        Getter and setter property for the node parent.

        Parameters
        ----------
        value
            Parent to set the node with.

        Returns
        -------
        :class:`Node` or :py:data:`None`
            Node parent.
        """

        return self._parent

    @parent.setter
    def parent(self, value: Optional[Node]):
        """Setter for the **self.parent** property."""

        if value is not None:
            attest(
                issubclass(value.__class__, Node),
                f'"parent" property: "{value}" is not a '
                f'"{Node.__class__.__name__}" subclass!',
            )

            value.children.append(self)

        self._parent = value

    @property
    def children(self) -> List[Node]:
        """
        Getter and setter property for the node children.

        Parameters
        ----------
        value
            Children to set the node with.

        Returns
        -------
        :class:`list`
            Node children.
        """

        return self._children

    @children.setter
    def children(self, value: List[Node]):
        """Setter for the **self.children** property."""

        attest(
            isinstance(value, list),
            f'"children" property: "{value}" type is not a "list" instance!',
        )

        for element in value:
            attest(
                issubclass(element.__class__, Node),
                f'"children" property: A "{element}" element is not a '
                f'"{Node.__class__.__name__}" subclass!',
            )

        for node in value:
            node.parent = self

        self._children = value

    @property
    def id(self) -> Integer:
        """
        Getter property for the node id.

        Returns
        -------
        :class:`numpy.integer`
            Node id.
        """

        return self._id  # type: ignore[attr-defined]

    @property
    def root(self) -> Node:
        """
        Getter property for the node tree.

        Returns
        -------
        :class:`Node`
            Node root.
        """

        if self.is_root():
            return self
        else:
            return list(self.walk(ascendants=True))[-1]

    @property
    def leaves(self) -> Generator:
        """
        Getter property for the node leaves.

        Yields
        ------
        Generator
            Node leaves.
        """

        if self.is_leaf():
            return (node for node in (self,))
        else:
            return (node for node in self.walk() if node.is_leaf())

    @property
    def siblings(self) -> Generator:
        """
        Getter property for the node siblings.

        Returns
        -------
        Generator
            Node siblings.
        """

        if self.parent is None:
            return (sibling for sibling in ())  # type: ignore[var-annotated]
        else:
            return (
                sibling
                for sibling in self.parent.children
                if sibling is not self
            )

    @property
    def data(self) -> Any:
        """
        Getter property for the node data.

        Returns
        -------
        :class:`object`
            Node data.
        """

        return self._data

    @data.setter
    def data(self, value: Any):
        """Setter for the **self.data** property."""

        self._data = value

    def __str__(self) -> str:
        """
        Return a formatted string representation of the node.

        Returns
        -------
        :class`str`
            Formatted string representation.
        """

        return f"{self.__class__.__name__}#{self.id}({self._data})"

    def __len__(self) -> Integer:
        """
        Return the number of children of the node.

        Returns
        -------
        :class:`numpy.integer`
            Number of children of the node.
        """

        return len(list(self.walk()))

    def is_root(self) -> Boolean:
        """
        Return whether the node is a root node.

        Returns
        -------
        :class:`bool`
            Whether the node is a root node.

        Examples
        --------
        >>> node_a = Node('Node A')
        >>> node_b = Node('Node B', node_a)
        >>> node_c = Node('Node C', node_b)
        >>> node_a.is_root()
        True
        >>> node_b.is_root()
        False
        """

        return self.parent is None

    def is_inner(self) -> Boolean:
        """
        Return whether the node is an inner node.

        Returns
        -------
        :class:`bool`
            Whether the node is an inner node.

        Examples
        --------
        >>> node_a = Node('Node A')
        >>> node_b = Node('Node B', node_a)
        >>> node_c = Node('Node C', node_b)
        >>> node_a.is_inner()
        False
        >>> node_b.is_inner()
        True
        """

        return all([not self.is_root(), not self.is_leaf()])

    def is_leaf(self) -> Boolean:
        """
        Return whether the node is a leaf node.

        Returns
        -------
        :class:`bool`
            Whether the node is a leaf node.

        Examples
        --------
        >>> node_a = Node('Node A')
        >>> node_b = Node('Node B', node_a)
        >>> node_c = Node('Node C', node_b)
        >>> node_a.is_leaf()
        False
        >>> node_c.is_leaf()
        True
        """

        return len(self._children) == 0

    def walk(self, ascendants: Boolean = False) -> Generator:
        """
        Return a generator used to walk into :class:`colour.utilities.Node`
        trees.

        Parameters
        ----------
        ascendants
            Whether to walk up the node tree.

        Yields
        ------
        Generator
            Node tree walker.

        Examples
        --------
        >>> node_a = Node('Node A')
        >>> node_b = Node('Node B', node_a)
        >>> node_c = Node('Node C', node_a)
        >>> node_d = Node('Node D', node_b)
        >>> node_e = Node('Node E', node_b)
        >>> node_f = Node('Node F', node_d)
        >>> node_g = Node('Node G', node_f)
        >>> node_h = Node('Node H', node_g)
        >>> for node in node_a.walk():
        ...     print(node.name)
        Node B
        Node D
        Node F
        Node G
        Node H
        Node E
        Node C
        """

        attribute = "children" if not ascendants else "parent"

        nodes = getattr(self, attribute)
        nodes = nodes if isinstance(nodes, list) else [nodes]

        for node in nodes:
            yield node

            if not getattr(node, attribute):
                continue

            yield from node.walk(ascendants=ascendants)

    def render(self, tab_level: Integer = 0):
        """
        Render the current node and its children as a string.

        Parameters
        ----------
        tab_level
            Initial indentation level

        Returns
        -------
        :class:`str`
            Rendered node tree.

        Examples
        --------
        >>> node_a = Node('Node A')
        >>> node_b = Node('Node B', node_a)
        >>> node_c = Node('Node C', node_a)
        >>> print(node_a.render())
        |----"Node A"
            |----"Node B"
            |----"Node C"
        <BLANKLINE>
        """

        output = ""

        for _i in range(tab_level):
            output += "    "

        tab_level += 1

        output += f'|----"{self.name}"\n'

        for child in self._children:
            output += child.render(tab_level)

        tab_level -= 1

        return output
