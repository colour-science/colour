# -*- coding: utf-8 -*-
"""
Data Structures
===============

Defines various data structures classes:

-   :class:`colour.utilities.Structure`: An object similar to C/C++ structured
    type.
-   :class:`colour.utilities.Lookup`: A *dict* sub-class acting as a lookup to
    retrieve keys by values.
-   :class:`colour.utilities.CaseInsensitiveMapping`: A case insensitive
    mapping allowing values retrieving from keys while ignoring the key case.
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
-   :cite:`Mansencald` : Mansencal, T. (n.d.). Structure.
    https://github.com/KelSolaar/Foundations/blob/develop/foundations/\
data_structures.py
-   :cite:`Reitza` : Reitz, K. (n.d.). CaseInsensitiveDict.
    https://github.com/kennethreitz/requests/blob/v1.2.3/requests/\
structures.py#L37
"""

from collections.abc import Mapping, MutableMapping, Sequence

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'attest',
    'Structure',
    'Lookup',
    'CaseInsensitiveMapping',
    'LazyCaseInsensitiveMapping',
    'Node',
]


def attest(condition, message=str()):
    """
    A replacement for `assert` that is not removed by optimised Python
    execution.

    See :func:`colour.utilities.assert` for more information.

    Notes
    -----
    -   This definition name is duplicated to avoid import circular dependency.
    """

    # Avoiding circular dependency.
    import colour.utilities

    colour.utilities.attest(condition, message)


class Structure(dict):
    """
    Defines a dict-like object allowing to access key values using dot syntax.

    Other Parameters
    ----------------
    \\*args : list, optional
        Arguments.
    \\**kwargs : dict, optional
        Key / Value pairs.

    Methods
    -------
    -   :meth:`~colour.utilities.Structure.__init__`

    References
    ----------
    :cite:`Mansencald`

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

    def __init__(self, *args, **kwargs):
        super(Structure, self).__init__(*args, **kwargs)
        self.__dict__ = self


class Lookup(dict):
    """
    Extends *dict* type to provide a lookup by value(s).

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

    def keys_from_value(self, value):
        """
        Gets the keys with given value.

        Parameters
        ----------
        value : object
            Value.
        Returns
        -------
        object
            Keys.
        """

        keys = []
        for key, data in self.items():
            matching = data == value
            try:
                matching = all(matching)

            except TypeError:
                matching = all((matching, ))

            if matching:
                keys.append(key)

        return keys

    def first_key_from_value(self, value):
        """
        Gets the first key with given value.

        Parameters
        ----------
        value : object
            Value.
        Returns
        -------
        object
            Key.
        """

        return self.keys_from_value(value)[0]


class CaseInsensitiveMapping(MutableMapping):
    """
    Implements a case-insensitive mutable mapping / *dict* object.

    Allows values retrieving from keys while ignoring the key case.
    The keys are expected to be str or string-like objects supporting the
    :meth:`str.lower` method.

    Parameters
    ----------
    data : dict
        *dict* of data to store into the mapping at initialisation.

    Other Parameters
    ----------------
    \\**kwargs : dict, optional
        Key / Value pairs to store into the mapping at initialisation.

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

    Warnings
    --------
    The keys are expected to be str or string-like objects.

    References
    ----------
    :cite:`Reitza`

    Examples
    --------
    >>> methods = CaseInsensitiveMapping({'McCamy': 1, 'Hernandez': 2})
    >>> methods['mccamy']
    1
    """

    def __init__(self, data=None, **kwargs):
        self._data = dict()

        self.update({} if data is None else data, **kwargs)

    @property
    def data(self):
        """
        Getter property for the data.

        Returns
        -------
        dict
            Data.
        """

        return self._data

    def __repr__(self):
        """
        Returns the mapping representation with the original item names.

        Returns
        -------
        str
            Mapping representation.
        """

        return '{0}({1})'.format(self.__class__.__name__, dict(self.items()))

    def __setitem__(self, item, value):
        """
        Sets given item with given value.

        The item is stored as lower in the mapping while the original name and
        its value are stored together as the value in a *tuple*:

        {"item.lower()": ("item", value)}

        Parameters
        ----------
        item : object
            Attribute.
        value : object
            Value.
        """

        self._data[item.lower()] = (item, value)

    def __getitem__(self, item):
        """
        Returns the value of given item.

        The item value is retrieved using its lower name in the mapping.

        Parameters
        ----------
        item : str
            Item name.

        Returns
        -------
        object
            Item value.
        """

        return self._data[item.lower()][1]

    def __delitem__(self, item):
        """
        Deletes the item with given name.

        The item is deleted from the mapping using its lower name.

        Parameters
        ----------
        item : str
            Item name.
        """

        del self._data[item.lower()]

    def __contains__(self, item):
        """
        Returns if the mapping contains given item.

        Parameters
        ----------
        item : str
            Item name.

        Returns
        -------
        bool
            Is item in mapping.
        """

        return item.lower() in self._data

    def __iter__(self):
        """
        Iterates over the items names in the mapping.

        The item names returned are the original input ones.

        Returns
        -------
        generator
            Item names.
        """

        return (item for item, value in self._data.values())

    def __len__(self):
        """
        Returns the items count.

        Returns
        -------
        int
            Items count.
        """

        return len(self._data)

    def __eq__(self, item):
        """
        Returns the equality with given object.

        Parameters
        ----------
        item
            Object item.

        Returns
        -------
        bool
            Equality.
        """

        if isinstance(item, Mapping):
            item_mapping = CaseInsensitiveMapping(item)
        else:
            raise ValueError(
                'Impossible to test equality with "{0}" class type!'.format(
                    item.__class__.__name__))

        return dict(self.lower_items()) == dict(item_mapping.lower_items())

    def __ne__(self, item):
        """
        Returns the inequality with given object.

        Parameters
        ----------
        item
            Object item.

        Returns
        -------
        bool
            Inequality.
        """

        return not (self == item)

    def copy(self):
        """
        Returns a copy of the mapping.

        Returns
        -------
        CaseInsensitiveMapping
            Mapping copy.

        Notes
        -----
        -   The :class:`colour.utilities.CaseInsensitiveMapping` class copy
            returned is a simple *copy* not a *deepcopy*.
        """

        return CaseInsensitiveMapping(self._data.values())

    def lower_items(self):
        """
        Iterates over the lower items names.

        Returns
        -------
        generator
            Lower item names.
        """

        return ((item, value[1]) for (item, value) in self._data.items())


class LazyCaseInsensitiveMapping(CaseInsensitiveMapping):
    """
    Implements a lazy case-insensitive mutable mapping / *dict* object by
    inheriting from :class:`colour.utilities.CaseInsensitiveMapping` class.

    Allows lazy values retrieving from keys while ignoring the key case.
    The keys are expected to be str or string-like objects supporting the
    :meth:`str.lower` method. The lazy retrieval is performed as follows:
    If the value is a callable, then it is evaluated and its return value is
    stored in place of the current value.

    Parameters
    ----------
    data : dict
        *dict* of data to store into the mapping at initialisation.

    Other Parameters
    ----------------
    \\**kwargs : dict, optional
        Key / Value pairs to store into the mapping at initialisation.

    Methods
    -------
    -   :meth:`~colour.utilities.LazyCaseInsensitiveMapping.__getitem__`

    Warnings
    --------
    The keys are expected to be str or string-like objects.

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

    def __getitem__(self, item):
        """
        Returns the value of given item.

        The item value is retrieved using its lower name in the mapping. If
        the value is a callable, then it is evaluated and its return value is
        stored in place of the current value.

        Parameters
        ----------
        item : str
            Item name.

        Returns
        -------
        object
            Item value.
        """

        import colour

        value = super(LazyCaseInsensitiveMapping, self).__getitem__(item)

        if callable(value) and hasattr(colour, '__disable_lazy_load__'):
            value = value()
            super(LazyCaseInsensitiveMapping, self).__setitem__(item, value)

        return value


class Node:
    """
    Represents a basic node supporting the creation of basic node trees.

    Parameters
    ----------
    parent : Node, optional
        Parent of the node.
    children : Node, optional
        Children of the node.
    data : object
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

    _INSTANCE_ID = 1
    """
    Node id counter.

    _INSTANCE_ID : int
    """

    def __new__(cls, *args, **kwargs):
        """
        Constructor of the class.

        Other Parameters
        ----------
        \\*args : list, optional
            Arguments.
        \\**kwargs : dict, optional
            Keywords arguments.

        Returns
        -------
        Node
            Class instance.
        """

        instance = super(Node, cls).__new__(cls)

        instance._id = Node._INSTANCE_ID
        Node._INSTANCE_ID += 1

        return instance

    def __init__(self, name=None, parent=None, children=None, data=None):
        self._name = '{0}#{1}'.format(self.__class__.__name__, self._id)
        self.name = name

        self._parent = None
        self.parent = parent

        self._children = None
        self._children = [] if children is None else children

        self._data = data

    @property
    def name(self):
        """
        Getter and setter property for the name.

        Parameters
        ----------
        value : str
            Value to set the name with.

        Returns
        -------
        str
            Node name.
        """

        return self._name

    @name.setter
    def name(self, value):
        """
        Setter for the **self.name** property.
        """

        if value is not None:
            attest(
                isinstance(value, str),
                '"{0}" attribute: "{1}" is not a "string" like object!'.format(
                    'name', value))

            self._name = value

    @property
    def parent(self):
        """
        Getter and setter property for the node parent.

        Parameters
        ----------
        value : Node
            Parent to set the node with.

        Returns
        -------
        Node
            Node parent.
        """

        return self._parent

    @parent.setter
    def parent(self, value):
        """
        Setter for the **self.parent** property.
        """

        if value is not None:
            attest(
                issubclass(value.__class__, Node),
                '"{0}" attribute: "{1}" is not a "{2}" subclass!'.format(
                    'parent', value, Node.__class__.__name__))

            value.children.append(self)

            self._parent = value

    @property
    def children(self):
        """
        Getter and setter property for the node children.

        Parameters
        ----------
        value : list
            Children to set the node with.

        Returns
        -------
        list
            Node children.
        """

        return self._children

    @children.setter
    def children(self, value):
        """
        Setter for the **self.children** property.
        """

        if value is not None:
            attest(
                isinstance(value, Sequence) and not isinstance(value, str),
                '"{0}" attribute: "{1}" type is not a "Sequence" instance!'
                .format('children', value))

            for element in value:
                attest(
                    issubclass(element.__class__, Node),
                    '"{0}" attribute: A "{1}" element is not a "{2}" subclass!'
                    .format('children', element, Node.__class__.__name__))

            for node in value:
                node.parent = self

            self._children = list(value)

    @property
    def id(self):
        """
        Getter property for the node id.

        Returns
        -------
        int
            Node id.
        """

        return self._id

    @property
    def root(self):
        """
        Getter property for the node tree.

        Returns
        -------
        Node
            Node root.
        """

        if self.is_root():
            return self
        else:
            return list(self.walk(ascendants=True))[-1]

    @property
    def leaves(self):
        """
        Getter property for the node leaves.

        Returns
        -------
        generator
            Node leaves.
        """

        if self.is_leaf():
            return (node for node in (self, ))
        else:
            return (node for node in self.walk() if node.is_leaf())

    @property
    def siblings(self):
        """
        Getter property for the node siblings.

        Returns
        -------
        list
            Node siblings.
        """

        if self.parent is None:
            return (sibling for sibling in ())
        else:
            return (sibling for sibling in self.parent.children
                    if sibling is not self)

    @property
    def data(self):
        """
        Getter property for the node data.

        Returns
        -------
        Data
            Node data.
        """

        return self._data

    @data.setter
    def data(self, value):
        """
        Setter for the **self.data** property.
        """

        self._data = value

    def __str__(self):
        """
        Returns a formatted string representation of the node.

        Returns
        -------
        str
            Formatted string representation.
        """

        return '{0}#{1}({2})'.format(self.__class__.__name__, self._id,
                                     self._data)

    def __len__(self):
        """
        Returns the number of children of the node.

        Returns
        -------
        int
            Number of children of the node.
        """

        return len(list(self.walk()))

    def is_root(self):
        """
        Returns whether the node is a root node.

        Returns
        -------
        bool
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

    def is_inner(self):
        """
        Returns whether the node is an inner node.

        Returns
        -------
        bool
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

    def is_leaf(self):
        """
        Returns whether the node is a leaf node.

        Returns
        -------
        bool
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

    def walk(self, ascendants=False):
        """
        Returns a generator used to walk into :class:`colour.utilities.Node`
        trees.

        Parameters
        ----------
        ascendants : bool, optional
            Whether to walk up the node tree.

        Returns
        -------
        generator
            Node tree walker

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

        attribute = 'children' if not ascendants else 'parent'

        nodes = getattr(self, attribute)
        nodes = nodes if isinstance(nodes, list) else [nodes]

        for node in nodes:
            yield node

            if not getattr(node, attribute):
                continue

            for relative in node.walk(ascendants=ascendants):
                yield relative

    def render(self, tab_level=0):
        """
        Renders the current node and its children as a string.

        Parameters
        ----------
        tab_level : int, optional
            Initial indentation level

        Returns
        ------
        str
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

        output = ''

        for i in range(tab_level):
            output += '    '

        tab_level += 1

        output += '|----"{0}"\n'.format(self.name)

        for child in self._children:
            output += child.render(tab_level)

        tab_level -= 1

        return output
