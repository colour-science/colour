"""
Network
=======

Define various node-graph / network related classes:

-   :class:`colour.utilities.TreeNode`: A basic node object supporting creation
    of basic node trees.
-   :class:`colour.utilities.Port`: An object that can be added either as an
    input or output port.
-   :class:`colour.utilities.PortMode`: A node with support for input and
    output ports.
-   :class:`colour.utilities.PortGraph`: A graph for nodes with input and
    output ports.
-   :class:`colour.utilities.ExecutionPort`: An object for nodes
    supporting execution input and output ports.
-   :class:`colour.utilities.ExecutionNode`: A node with builtin input and
    output execution ports.
-   :class:`colour.utilities.ControlFlowNode`: A node inherited by control flow
    nodes.
-   :class:`colour.utilities.For`: A node performing for loops in the
    node-graph.
-   :class:`colour.utilities.ParallelForThread`: A node performing for loops in
    parallel in the node-graph using threads.
-   :class:`colour.utilities.ParallelForMultiprocess`: A node performing for
    loops in parallel in the node-graph using multiprocessing.
"""

from __future__ import annotations

import concurrent.futures
import multiprocessing
import os
import threading

from colour.hints import (
    Any,
    Dict,
    Generator,
    List,
    Self,
    Tuple,
    Type,
)
from colour.utilities import MixinLogging, attest, optional, required

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TreeNode",
    "Port",
    "PortNode",
    "ControlFlowNode",
    "PortGraph",
    "ExecutionPort",
    "ExecutionNode",
    "ControlFlowNode",
    "For",
    "ParallelForThread",
    "ParallelForMultiprocess",
]


class TreeNode:
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
    -   :attr:`~colour.utilities.Node.id`
    -   :attr:`~colour.utilities.Node.name`
    -   :attr:`~colour.utilities.Node.parent`
    -   :attr:`~colour.utilities.Node.children`
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
    >>> node_a = TreeNode("Node A")
    >>> node_b = TreeNode("Node B", node_a)
    >>> node_c = TreeNode("Node C", node_a)
    >>> node_d = TreeNode("Node D", node_b)
    >>> node_e = TreeNode("Node E", node_b)
    >>> node_f = TreeNode("Node F", node_d)
    >>> node_g = TreeNode("Node G", node_f)
    >>> node_h = TreeNode("Node H", node_g)
    >>> [node.name for node in node_a.leaves]
    ['Node H', 'Node E', 'Node C']
    >>> print(node_h.root.name)
    Node A
    >>> len(node_a)
    7
    """

    _INSTANCE_ID: int = 1
    """
    Node id counter.

    _INSTANCE_ID
    """

    def __new__(cls, *args: Any, **kwargs: Any) -> Self:  # noqa: ARG003
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

        instance._id = TreeNode._INSTANCE_ID  # pyright: ignore
        TreeNode._INSTANCE_ID += 1

        return instance

    def __init__(
        self,
        name: str | None = None,
        parent: Self | None = None,
        children: List[Self] | None = None,
        data: Any | None = None,
    ) -> None:
        self._name: str = f"{self.__class__.__name__}#{self.id}"
        self.name = optional(name, self._name)
        self._parent: Self | None = None
        self.parent = parent
        self._children: List[Self] = []
        self.children = optional(children, self._children)
        self._data: Any | None = data

    @property
    def id(self) -> int:
        """
        Getter property for the node id.

        Returns
        -------
        :class:`int`
            Node id.
        """

        return self._id  # pyright: ignore

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
    def parent(self) -> Self | None:
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
    def parent(self, value: Self | None):
        """Setter for the **self.parent** property."""

        from colour.utilities import attest

        if value is not None:
            attest(
                issubclass(value.__class__, TreeNode),
                f'"parent" property: "{value}" is not a '
                f'"{self.__class__.__name__}" subclass!',
            )

            value.children.append(self)

        self._parent = value

    @property
    def children(self) -> List[Self]:
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
    def children(self, value: List[Self]):
        """Setter for the **self.children** property."""

        from colour.utilities import attest

        attest(
            isinstance(value, list),
            f'"children" property: "{value}" type is not a "list" instance!',
        )

        for element in value:
            attest(
                issubclass(element.__class__, TreeNode),
                f'"children" property: A "{element}" element is not a '
                f'"{self.__class__.__name__}" subclass!',
            )

        for node in value:
            node.parent = self

        self._children = value

    @property
    def root(self) -> Self:
        """
        Getter property for the node tree.

        Returns
        -------
        :class:`TreeNode`
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
            return (sibling for sibling in ())
        else:
            return (sibling for sibling in self.parent.children if sibling is not self)

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

    def __len__(self) -> int:
        """
        Return the number of children of the node.

        Returns
        -------
        :class:`int`
            Number of children of the node.
        """

        return len(list(self.walk()))

    def is_root(self) -> bool:
        """
        Return whether the node is a root node.

        Returns
        -------
        :class:`bool`
            Whether the node is a root node.

        Examples
        --------
        >>> node_a = TreeNode("Node A")
        >>> node_b = TreeNode("Node B", node_a)
        >>> node_c = TreeNode("Node C", node_b)
        >>> node_a.is_root()
        True
        >>> node_b.is_root()
        False
        """

        return self.parent is None

    def is_inner(self) -> bool:
        """
        Return whether the node is an inner node.

        Returns
        -------
        :class:`bool`
            Whether the node is an inner node.

        Examples
        --------
        >>> node_a = TreeNode("Node A")
        >>> node_b = TreeNode("Node B", node_a)
        >>> node_c = TreeNode("Node C", node_b)
        >>> node_a.is_inner()
        False
        >>> node_b.is_inner()
        True
        """

        return all([not self.is_root(), not self.is_leaf()])

    def is_leaf(self) -> bool:
        """
        Return whether the node is a leaf node.

        Returns
        -------
        :class:`bool`
            Whether the node is a leaf node.

        Examples
        --------
        >>> node_a = TreeNode("Node A")
        >>> node_b = TreeNode("Node B", node_a)
        >>> node_c = TreeNode("Node C", node_b)
        >>> node_a.is_leaf()
        False
        >>> node_c.is_leaf()
        True
        """

        return len(self._children) == 0

    def walk(self, ascendants: bool = False) -> Generator:
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
        >>> node_a = TreeNode("Node A")
        >>> node_b = TreeNode("Node B", node_a)
        >>> node_c = TreeNode("Node C", node_a)
        >>> node_d = TreeNode("Node D", node_b)
        >>> node_e = TreeNode("Node E", node_b)
        >>> node_f = TreeNode("Node F", node_d)
        >>> node_g = TreeNode("Node G", node_f)
        >>> node_h = TreeNode("Node H", node_g)
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

    def render(self, tab_level: int = 0):
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
        >>> node_a = TreeNode("Node A")
        >>> node_b = TreeNode("Node B", node_a)
        >>> node_c = TreeNode("Node C", node_a)
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


class Port(MixinLogging):
    """
    Define an object that can be added either as an input or output port,
    i.e., a pin, to a :class:`colour.utilities.PortNode` class and connected to
    another input or output port.

    Parameters
    ----------
    name
        Port name.
    value
        Initial value to set the port with.
    description
        Port description
    node
        Node to add the port to.

    Attributes
    ----------
    -   :attr:`~colour.utilities.Port.name`
    -   :attr:`~colour.utilities.Port.value`
    -   :attr:`~colour.utilities.Port.description`
    -   :attr:`~colour.utilities.Port.node`
    -   :attr:`~colour.utilities.Port.connections`

    Methods
    -------
    -   :meth:`~colour.utilities.Port.__init__`
    -   :meth:`~colour.utilities.Port.__str__`
    -   :meth:`~colour.utilities.Port.is_input_port`
    -   :meth:`~colour.utilities.Port.is_output_port`
    -   :meth:`~colour.utilities.Port.connect`
    -   :meth:`~colour.utilities.Port.disconnect`
    -   :meth:`~colour.utilities.Port.to_graphviz`

    Examples
    --------
    >>> port = Port("a", 1, "Port A Description")
    >>> port.name
    'a'
    >>> port.value
    1
    >>> port.description
    'Port A Description'
    """

    def __init__(
        self,
        name: str | None = None,
        value: Any = None,
        description: str | None = None,
        node: PortNode | None = None,
    ) -> None:
        super().__init__()

        # TODO: Consider using an ordered set instead of a dict.
        self._connections: Dict[Port, None] = {}

        self._node: PortNode | None = None
        self.node = optional(node, self._node)
        self._name: str = self.__class__.__name__
        self.name = optional(name, self._name)
        self._value = None
        self.value = optional(value, self._value)
        self._description = description
        self.description = optional(description, self._description)

    @property
    def name(self) -> str:
        """
        Getter and setter property for the port name.

        Parameters
        ----------
        value
            Value to set the port name with.

        Returns
        -------
        :class:`str`
            Port name.
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
    def value(self) -> Any:
        """
        Getter and setter property for the port value.

        Parameters
        ----------
        value
            Value to set the port value with.

        Returns
        -------
        :class:`object`
            Port value.
        """

        # NOTE: Assumption is that if the public API is used to set values, the
        # actual port value is coming from the connected port. Any connected
        # port is valid as they should all carry the same value, thus the first
        # connected port is returned.
        for connection in self._connections:
            return connection._value

        return self._value

    @value.setter
    def value(self, value: Any):
        """Setter for the **self.value** property."""

        self._value = value

        if self._node is not None:
            self.log(f'Dirtying "{self._node}".', "debug")
            self._node.dirty = True

        # NOTE: Setting the port value implies that all the connected ports
        # should be also set to the same given value.
        for direct_connection in self._connections:
            self.log(f'Setting "{direct_connection.node}" value to {value}.', "debug")
            direct_connection._value = value

            if direct_connection.node is not None:
                self.log(f'Dirtying "{direct_connection.node}".', "debug")
                direct_connection.node.dirty = True

            for indirect_connection in direct_connection.connections:
                if indirect_connection == self:
                    continue

                self.log(
                    f'Setting "{indirect_connection.node}" value to {value}.', "debug"
                )
                indirect_connection._value = value

                if indirect_connection.node is not None:
                    self.log(f'Dirtying "{indirect_connection.node}".', "debug")
                    indirect_connection.node.dirty = True

        self._value = value

    @property
    def description(self) -> str | None:
        """
        Getter and setter property for the port description.

        Parameters
        ----------
        value
            Value to set the port description with.

        Returns
        -------
        :class:`str` or None
            Port description.
        """

        return self._description

    @description.setter
    def description(self, value: str | None):
        """Setter for the **self.description** property."""

        attest(
            value is None or isinstance(value, str),
            f'"description" property: "{value}" is not "None" or '
            f'its type is not "str"!',
        )

        self._description = value

    @property
    def node(self) -> PortNode | None:
        """
        Getter property for the port node.

        Returns
        -------
        :class:`PortNode` or None
            Port node.
        """

        return self._node

    @node.setter
    def node(self, value: PortNode | None):
        """Setter for the **self.node** property."""

        attest(
            value is None or isinstance(value, PortNode),
            f'"node" property: "{value}" is not "None" or '
            f'its type is not "PortNode"!',
        )

        self._node = value

    @property
    def connections(self) -> Dict[Port, None]:
        """
        Getter property for the port connections.

        Returns
        -------
        :class:`dict`
            Port connections.
        """

        return self._connections

    def __str__(self) -> str:
        """
        Return a formatted string representation of the port.

        Returns
        -------
        :class:`str`
            Formatted string representation of the port.

        Examples
        --------
        >>> print(Port("a"))
        None.a (-> [])
        >>> print(Port("a", node=PortNode("Port Node")))
        Port Node.a (-> [])
        """

        connections = [
            (
                f"{connection.node.name}.{connection.name}"
                if connection.node is not None
                else "None.{connection.name}"
            )
            for connection in self._connections
        ]

        direction = "<-" if self.is_input_port() else "->"

        node_name = self._node.name if self._node is not None else "None"

        return f"{node_name}.{self._name} ({direction} {connections})"

    def is_input_port(self) -> bool:
        """
        Return whether the port is an input port.

        Returns
        -------
        :class:`bool`
            Whether the port is an input port.

        Examples
        --------
        >>> Port().is_input_port()
        False
        >>> node = PortNode()
        >>> node.add_input_port("a").is_input_port()
        True
        """

        if self._node is not None:
            return self._name in self._node.input_ports

        return False

    def is_output_port(self) -> bool:
        """
        Return whether the port is an output port.

        Returns
        -------
        :class:`bool`
            Whether the port is an output port.

        Examples
        --------
        >>> Port().is_output_port()
        False
        >>> node = PortNode()
        >>> node.add_output_port("output").is_output_port()
        True
        """

        if self._node is not None:
            return self._name in self._node.output_ports

        return False

    def connect(self, port: Port) -> None:
        """
        Connect the port to the other given port.

        Parameters
        ----------
        port
            Port to connect to.

        Raises
        ------
        ValueError
            if an attempt is made to connect an input port to multiple output
            ports.

        Examples
        --------
        >>> port_a = Port()
        >>> port_b = Port()
        >>> port_a.connections
        {}
        >>> port_b.connections
        {}
        >>> port_a.connect(port_b)
        >>> port_a.connections  # doctest: +ELLIPSIS
        {<...Port object at 0x...>: None}
        >>> port_b.connections  # doctest: +ELLIPSIS
        {<...Port object at 0x...>: None}
        """

        attest(isinstance(port, Port), f'"{port}" is not a "Port" instance!')

        self.log(f'Connecting "{self.name}" to "{port.name}".', "debug")

        self.connections[port] = None
        port.connections[self] = None

    def disconnect(self, port: Port) -> None:
        """
        Disconnect the port from the other given port.

        Parameters
        ----------
        port
            Port to disconnect from.

        Examples
        --------
        >>> port_a = Port()
        >>> port_b = Port()
        >>> port_a.connect(port_b)
        >>> port_a.connections  # doctest: +ELLIPSIS
        {<...Port object at 0x...>: None}
        >>> port_b.connections  # doctest: +ELLIPSIS
        {<...Port object at 0x...>: None}
        >>> port_a.disconnect(port_b)
        >>> port_a.connections
        {}
        >>> port_b.connections
        {}
        """

        attest(isinstance(port, Port), f'"{port}" is not a "Port" instance!')

        self.log(f'Disconnecting "{self.name}" from "{port.name}".', "debug")

        self.connections.pop(port)
        port.connections.pop(self)

    def to_graphviz(self) -> str:
        """
        Return a string representation for visualisation of the port with
        *Graphviz*.

        Returns
        -------
        :class:`str`
            String representation for visualisation of the port with *Graphviz*.

        Examples
        --------
        >>> Port("a").to_graphviz()
        '<a> a'
        """

        return f"<{self._name}> {self.name}"


class PortNode(MixinLogging):
    """
    Define a node with support for input and output ports.

    Other Parameters
    ----------------
    name
        Node name.

    Attributes
    ----------
    -   :attr:`~colour.utilities.PortNode.id`
    -   :attr:`~colour.utilities.PortNode.name`
    -   :attr:`~colour.utilities.PortNode.input_ports`
    -   :attr:`~colour.utilities.PortNode.output_ports`
    -   :attr:`~colour.utilities.PortNode.dirty`
    -   :attr:`~colour.utilities.PortNode.edges`
    -   :attr:`~colour.utilities.PortNode.description`

    Methods
    -------
    -   :meth:`~colour.utilities.PortNode.__new__`
    -   :meth:`~colour.utilities.PortNode.__init__`
    -   :meth:`~colour.utilities.PortNode.__str__`
    -   :meth:`~colour.utilities.PortNode.add_input_port`
    -   :meth:`~colour.utilities.PortNode.remove_input_port`
    -   :meth:`~colour.utilities.PortNode.add_output_port`
    -   :meth:`~colour.utilities.PortNode.remove_output_port`
    -   :meth:`~colour.utilities.PortNode.get_input`
    -   :meth:`~colour.utilities.PortNode.set_input`
    -   :meth:`~colour.utilities.PortNode.get_output`
    -   :meth:`~colour.utilities.PortNode.set_output`
    -   :meth:`~colour.utilities.PortNode.connect`
    -   :meth:`~colour.utilities.PortNode.disconnect`
    -   :meth:`~colour.utilities.PortNode.process`
    -   :meth:`~colour.utilities.PortNode.to_graphviz`

    Examples
    --------
    >>> class NodeAdd(PortNode):
    ...     def __init__(self, *args, **kwargs):
    ...         super().__init__(*args, **kwargs)
    ...
    ...         self.description = "Perform the addition of the two input port values."
    ...
    ...         self.add_input_port("a")
    ...         self.add_input_port("b")
    ...         self.add_output_port("output")
    ...
    ...     def process(self):
    ...         a = self.get_input("a")
    ...         b = self.get_input("b")
    ...
    ...         if a is None or b is None:
    ...             return
    ...
    ...         self._output_ports["output"].value = a + b
    ...
    ...         self.dirty = False
    >>> node = NodeAdd()
    >>> node.set_input("a", 1)
    >>> node.set_input("b", 1)
    >>> node.process()
    >>> node.get_output("output")
    2
    """

    _INSTANCE_ID: int = 1
    """
    Node id counter.

    _INSTANCE_ID
    """

    def __new__(cls, *args, **kwargs) -> Self:  # noqa: ARG003
        """
        Return a new instance of the :class:`colour.utilities.PortNode` class.

        Other Parameters
        ----------------
        args
            Arguments.
        kwargs
            Keywords arguments.
        """

        instance = super().__new__(cls)

        instance._id = PortNode._INSTANCE_ID  # pyright: ignore
        PortNode._INSTANCE_ID += 1

        return instance

    def __init__(self, name: str | None = None, description: str | None = None):
        self._name: str = f"{self.__class__.__name__}#{self.id}"
        self.name = optional(name, self._name)
        self._description = description
        self.description = optional(description, self._description)

        self._input_ports = {}
        self._output_ports = {}
        self._dirty = True

    @property
    def id(self) -> int:
        """
        Getter property for the node id.

        Returns
        -------
        :class:`int`
            Node id.
        """

        return self._id  # pyright: ignore

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
    def input_ports(self) -> Dict[str, Port]:
        """
        Getter property for the input ports.

        Returns
        -------
        :class:`dict`
            Input ports.
        """

        return self._input_ports

    @property
    def output_ports(self) -> Dict[str, Port]:
        """
        Getter property for the output ports.

        Returns
        -------
        :class:`dict`
            Output ports.
        """

        return self._output_ports

    @property
    def dirty(self) -> bool:
        """
        Getter and setter property for the node dirty state.

        Parameters
        ----------
        value
            Value to set the node dirty state.

        Returns
        -------
        :class:`bool`
            Whether the node is dirty.
        """

        return self._dirty

    @dirty.setter
    def dirty(self, value: bool):
        """Setter for the **self.dirty** property."""

        attest(
            isinstance(value, bool),
            f'"dirty" property: "{value}" type is not "bool"!',
        )

        self._dirty = value

    @property
    def edges(
        self,
    ) -> Tuple[Dict[Tuple[Port, Port], None], Dict[Tuple[Port, Port], None]]:
        """
        Return the edges of the node.

        Each edge represent a port and one of its connections.

        Returns
        -------
        :class:`tuple`
            Edges of the node as a tuple of input and output edge dictionaries.
        """

        # TODO: Consider using ordered set.
        input_edges = {}
        for port in self.input_ports.values():
            for connection in port.connections:
                input_edges[(port, connection)] = None

        # TODO: Consider using ordered set.
        output_edges = {}
        for port in self.output_ports.values():
            for connection in port.connections:
                output_edges[(port, connection)] = None

        return input_edges, output_edges

    @property
    def description(self) -> str | None:
        """
        Getter and setter property for the node description.

        Parameters
        ----------
        value
            Value to set the node description with.

        Returns
        -------
        :class:`str` or None
            Node description.
        """

        return self._description

    @description.setter
    def description(self, value: str | None):
        """Setter for the **self.description** property."""

        attest(
            value is None or isinstance(value, str),
            f'"description" property: "{value}" is not "None" or '
            f'its type is not "str"!',
        )

        self._description = value

    def __str__(self) -> str:
        """
        Return a formatted string representation of the node.

        Returns
        -------
        :class`str`
            Formatted string representation.
        """

        return f"{self.__class__.__name__}#{self.id}"

    def add_input_port(
        self,
        name: str,
        value: Any = None,
        description: str | None = None,
        port_type: Type[Port] = Port,
    ) -> Port:
        """
        Add an input port with given name and value to the node.

        Parameters
        ----------
        name
            Name of the input port.
        value
            Value of the input port
        description
            Description of the input port.
        port_type
            Type of the input port.

        Returns
        -------
        :class:`colour.utilities.Port`
            Input port.

        Examples
        --------
        >>> node = PortNode()
        >>> node.add_input_port("a")  # doctest: +ELLIPSIS
        <...Port object at 0x...>
        """

        self._input_ports[name] = port_type(name, value, description, self)

        return self._input_ports[name]

    def remove_input_port(
        self,
        name: str,
    ) -> Port:
        """
        Remove the input port with given name from the node.

        Parameters
        ----------
        name
            Name of the input port.

        Returns
        -------
        :class:`colour.utilities.Port`
            Input port.

        Examples
        --------
        >>> node = PortNode()
        >>> port = node.add_input_port("a")
        >>> node.remove_input_port("a")  # doctest: +ELLIPSIS
        <...Port object at 0x...>
        """

        attest(
            name in self._input_ports,
            f'"{name}" port is not a member of {self} input ports!',
        )

        port = self._input_ports.pop(name)

        for connection in port.connections:
            port.disconnect(connection)

        return port

    def add_output_port(
        self,
        name: str,
        value: Any = None,
        description: str | None = None,
        port_type: Type[Port] = Port,
    ) -> None:
        """
        Add an output port with given name and value to the node.

        Parameters
        ----------
        name
            Name of the output port.
        value
            Value of the output port
        description
            Description of the output port.
        port_type
            Type of the output port.

        Returns
        -------
        :class:`colour.utilities.Port`
            Output port.

        Examples
        --------
        >>> node = PortNode()
        >>> node.add_output_port("output")  # doctest: +ELLIPSIS
        <...Port object at 0x...>
        """

        self._output_ports[name] = port_type(name, value, description, self)

        return self._output_ports[name]

    def remove_output_port(
        self,
        name: str,
    ) -> Port:
        """
        Remove the output port with given name from the node.

        Parameters
        ----------
        name
            Name of the output port.

        Returns
        -------
        :class:`colour.utilities.Port`
            Output port.

        Examples
        --------
        >>> node = PortNode()
        >>> port = node.add_output_port("a")
        >>> node.remove_output_port("a")  # doctest: +ELLIPSIS
        <...Port object at 0x...>
        """

        attest(
            name in self._output_ports,
            f'"{name}" port is not a member of {self} output ports!',
        )

        port = self._output_ports.pop(name)

        for connection in port.connections:
            port.disconnect(connection)

        return port

    def get_input(self, name: str) -> Any:
        """
        Return the value of the input port with given name.

        Parameters
        ----------
        name
            Name of the input port.

        Returns
        -------
        :class:`object`:
            Value of the input port.

        Raises
        ------
        AssertionError
            If the input port is not a member of the node input ports.

        Examples
        --------
        >>> node = PortNode()
        >>> port = node.add_input_port("a", 1)  # doctest: +ELLIPSIS
        >>> node.get_input("a")
        1
        """

        attest(
            name in self._input_ports,
            f'"{name}" is not a member of "{self._name}" input ports!',
        )

        return self._input_ports[name].value

    def set_input(self, name: str, value: Any) -> None:
        """
        Set the value of the input port with given name.

        Parameters
        ----------
        name
            Name of the input port.
        value
            Value of the input port

        Raises
        ------
        AssertionError
            If the input port is not a member of the node input ports.

        Examples
        --------
        >>> node = PortNode()
        >>> port = node.add_input_port("a")  # doctest: +ELLIPSIS
        >>> port.value
        >>> node.set_input("a", 1)
        >>> port.value
        1
        """

        attest(
            name in self._input_ports,
            f'"{name}" is not a member of "{self._name}" input ports!',
        )

        self._input_ports[name].value = value

    def get_output(self, name: str) -> None:
        """
        Return the value of the output port with given name.

        Parameters
        ----------
        name
            Name of the output port.

        Returns
        -------
        :class:`object`:
            Value of the output port.

        Raises
        ------
        AssertionError
            If the output port is not a member of the node output ports.

        Examples
        --------
        >>> node = PortNode()
        >>> port = node.add_output_port("output", 1)  # doctest: +ELLIPSIS
        >>> node.get_output("output")
        1
        """

        attest(
            name in self._output_ports,
            f'"{name}" is not a member of "{self._name}" output ports!',
        )

        return self._output_ports[name].value

    def set_output(self, name: str, value: Any) -> None:
        """
        Set the value of the output port with given name.

        Parameters
        ----------
        name
            Name of the output port.
        value
            Value of the output port

        Raises
        ------
        AssertionError
            If the output port is not a member of the node output ports.

        Examples
        --------
        >>> node = PortNode()
        >>> port = node.add_output_port("output")  # doctest: +ELLIPSIS
        >>> port.value
        >>> node.set_output("output", 1)
        >>> port.value
        1
        """

        attest(
            name in self._output_ports,
            f'"{name}" is not a member of "{self._name}" input ports!',
        )

        self._output_ports[name].value = value

    def connect(
        self,
        source_port: str,
        target_node: PortNode,
        target_port: str,
    ) -> None:
        """
        Connect the given source port to given node target port.

        The source port can be an input port but the target port must be
        an output port and conversely, if the source port is an output port, the
        target port must be an input port.

        Parameters
        ----------
        source_port
            Source port of the node to connect to the other node target port.
        target_node
            Target node that the target port is the member of.
        target_port
            Target port from the target node to connect the source port to.

        Examples
        --------
        >>> node_1 = PortNode()
        >>> port = node_1.add_output_port("output")
        >>> node_2 = PortNode()
        >>> port = node_2.add_input_port("a")
        >>> node_1.connect("output", node_2, "a")
        >>> node_1.edges  # doctest: +ELLIPSIS
        ({}, {(<...Port object at 0x...>, <...Port object at 0x...>): None})
        """

        port_source = self._output_ports.get(
            source_port, self.input_ports.get(source_port)
        )
        port_target = target_node.input_ports.get(
            target_port, target_node.output_ports.get(target_port)
        )

        port_source.connect(port_target)

    def disconnect(
        self,
        source_port: str,
        target_node: PortNode,
        target_port: str,
    ) -> None:
        """
        Disconnect the given source port from given node target port.

        The source port can be an input port but the target port must be
        an output port and conversely, if the source port is an output port, the
        target port must be an input port.

        Parameters
        ----------
        source_port
            Source port of the node to disconnect from the other node target port.
        target_node
            Target node that the target port is the member of.
        target_port
            Target port from the target node to disconnect the source port from.

        Examples
        --------
        >>> node_1 = PortNode()
        >>> port = node_1.add_output_port("output")
        >>> node_2 = PortNode()
        >>> port = node_2.add_input_port("a")
        >>> node_1.connect("output", node_2, "a")
        >>> node_1.edges  # doctest: +ELLIPSIS
        ({}, {(<...Port object at 0x...>, <...Port object at 0x...>): None})
        >>> node_1.disconnect("output", node_2, "a")
        >>> node_1.edges
        ({}, {})
        """

        port_source = self._output_ports.get(
            source_port, self.input_ports.get(source_port)
        )
        port_target = target_node.input_ports.get(
            target_port, target_node.output_ports.get(target_port)
        )

        port_source.disconnect(port_target)

    def process(self) -> None:
        """
        Process the node, must be reimplemented by sub-classes.

        This definition is responsible to set the dirty state of the node
        according to processing outcome.

        Examples
        --------
        >>> class NodeAdd(PortNode):
        ...     def __init__(self, *args, **kwargs):
        ...         super().__init__(*args, **kwargs)
        ...
        ...         self.description = (
        ...             "Perform the addition of the two input port values."
        ...         )
        ...
        ...         self.add_input_port("a")
        ...         self.add_input_port("b")
        ...         self.add_output_port("output")
        ...
        ...     def process(self):
        ...         a = self.get_input("a")
        ...         b = self.get_input("b")
        ...
        ...         if a is None or b is None:
        ...             return
        ...
        ...         self._output_ports["output"].value = a + b
        ...
        ...         self.dirty = False
        >>> node = NodeAdd()
        >>> node.set_input("a", 1)
        >>> node.set_input("b", 1)
        >>> node.process()
        >>> node.get_output("output")
        2
        """

        self._dirty = False

    def to_graphviz(self) -> str:
        """
        Return a string representation for visualisation of the node with
        *Graphviz*.

        Returns
        -------
        :class:`str`
            String representation for visualisation of the node with *Graphviz*.

        Examples
        --------
        >>> node_1 = PortNode("PortNode")
        >>> port = node_1.add_input_port("a")
        >>> port = node_1.add_input_port("b")
        >>> port = node_1.add_output_port("output")
        >>> node_1.to_graphviz()  # doctest: +ELLIPSIS
        'PortNode (#...) | {{<a> a|<b> b} | {<output> output}}'
        """

        input_ports = "|".join(
            [port.to_graphviz() for port in self._input_ports.values()]
        )
        output_ports = "|".join(
            [port.to_graphviz() for port in self._output_ports.values()]
        )

        return f"{self.name} (#{self.id}) | {{{{{input_ports}}} | {{{output_ports}}}}}"


class PortGraph(PortNode):
    """
    Define a node-graph for :class:`colour.utilities.PortNode` class instances.

    Parameters
    ----------
    name
        Name of the node-graph.
    description
        Port description

    Attributes
    ----------
    -   :attr:`~colour.utilities.PortGraph.nodes`
    -   :attr:`~colour.utilities.PortGraph.is_subgraph`

    Methods
    -------
    -   :meth:`~colour.utilities.PortGraph.__str__`
    -   :meth:`~colour.utilities.PortGraph.add_node`
    -   :meth:`~colour.utilities.PortGraph.remove_node`
    -   :meth:`~colour.utilities.PortGraph.walk`
    -   :meth:`~colour.utilities.PortGraph.process`
    -   :meth:`~colour.utilities.PortGraph.to_graphviz`

    Examples
    --------
    >>> class NodeAdd(PortNode):
    ...     def __init__(self, *args, **kwargs):
    ...         super().__init__(*args, **kwargs)
    ...
    ...         self.description = "Perform the addition of the two input port values."
    ...
    ...         self.add_input_port("a")
    ...         self.add_input_port("b")
    ...         self.add_output_port("output")
    ...
    ...     def process(self):
    ...         a = self.get_input("a")
    ...         b = self.get_input("b")
    ...
    ...         if a is None or b is None:
    ...             return
    ...
    ...         self._output_ports["output"].value = a + b
    ...
    ...         self.dirty = False
    >>> node_1 = NodeAdd()
    >>> node_1.set_input("a", 1)
    >>> node_1.set_input("b", 1)
    >>> node_2 = NodeAdd()
    >>> node_1.connect("output", node_2, "a")
    >>> node_2.set_input("b", 1)
    >>> graph = PortGraph()
    >>> graph.add_node(node_1)
    >>> graph.add_node(node_2)
    >>> graph.nodes  # doctest: +ELLIPSIS
    {'NodeAdd#...': <...NodeAdd object at 0x...>, \
'NodeAdd#...': <...NodeAdd object at 0x...>}
    >>> graph.process()
    >>> node_2.get_output("output")
    3
    """

    def __init__(self, name: str | None = None, description: str | None = None):
        super().__init__(name, description)

        self._name: str = self.__class__.__name__
        self.name = optional(name, self._name)
        self._description = description
        self.description = optional(description, self._description)

        self._nodes = {}
        self._is_subgraph = False

    @property
    def nodes(self) -> Dict[str, PortNode]:
        """
        Getter property for the node-graph nodes.

        Returns
        -------
        :class:`dict`
            Node-graph nodes.
        """

        return self._nodes

    @property
    def is_subgraph(self) -> bool:
        """
        Getter and setter property for the sub-graph state.

        Parameters
        ----------
        value
            Value to set the sub-graph state with.

        Returns
        -------
        :class:`bool`
            Node sub-graph state.
        """

        return self._is_subgraph

    @is_subgraph.setter
    def is_subgraph(self, value: bool):
        """Setter for the **self.is_subgraph** property."""

        attest(
            isinstance(value, bool),
            f'"is_subgraph" property: "{value}" type is not "bool"!',
        )

        self._is_subgraph = value

    def __str__(self) -> str:
        """
        Return a formatted string representation of the node-graph.

        Returns
        -------
        :class`str`
            Formatted string representation.
        """

        return f"{self.__class__.__name__}({len(self._nodes)})"

    def add_node(self, node: PortNode) -> None:
        """
        Add given node to the node-graph.

        Parameters
        ----------
        node
            Node to add to the node-graph.

        Raises
        ------
        AsssertionError
            If the node is not a :class:`colour.utilities.PortNode` class
            instance.

        Examples
        --------
        >>> node_1 = PortNode()
        >>> node_2 = PortNode()
        >>> graph = PortGraph()
        >>> graph.nodes
        {}
        >>> graph.add_node(node_1)
        >>> graph.nodes  # doctest: +ELLIPSIS
        {'PortNode#...': <...PortNode object at 0x...>}
        >>> graph.add_node(node_2)
        >>> graph.nodes  # doctest: +ELLIPSIS
        {'PortNode#...': <...PortNode object at 0x...>, 'PortNode#...': \
<...PortNode object at 0x...>}
        """

        attest(isinstance(node, PortNode), f'"{node}" is not a "Node" instance!')

        attest(
            node.name not in self._nodes, f'"{node}" is already a member of the graph!'
        )

        self._nodes[node.name] = node

    def remove_node(self, node: PortNode) -> None:
        """
        Remove given node from the node-graph.

        The node input and output ports will be disconnected from all their
        connections.

        Parameters
        ----------
        node
            Node to remove from the node-graph.

        Raises
        ------
        AsssertionError
            If the node is not a member of the node-graph.

        Examples
        --------
        >>> node_1 = PortNode()
        >>> node_2 = PortNode()
        >>> graph = PortGraph()
        >>> graph.add_node(node_1)
        >>> graph.add_node(node_2)
        >>> graph.nodes  # doctest: +ELLIPSIS
        {'PortNode#...': <...PortNode object at 0x...>, \
'PortNode#...': <...PortNode object at 0x...>}
        >>> graph.remove_node(node_2)
        >>> graph.nodes  # doctest: +ELLIPSIS
        {'PortNode#...': <...PortNode object at 0x...>}
        >>> graph.remove_node(node_1)
        >>> graph.nodes
        {}
        """

        attest(isinstance(node, PortNode), f'"{node}" is not a "Node" instance!')

        attest(
            node.name in self._nodes,
            f'"{node}" is not a member of "{self._name}" node-graph!',
        )

        for port in node.input_ports.values():
            for connection in port.connections.copy():
                port.disconnect(connection)

        for port in node.output_ports.values():
            for connection in port.connections.copy():
                port.disconnect(connection)

        self._nodes.pop(node.name)

    @required("NetworkX")
    def walk(self) -> Generator:
        """
        Return a generator used to walk into the node-graph.

        The node is walked according to a topological sorted order. A
        topological sort is a non-unique permutation of the nodes of a directed
        graph such that an edge from :math:`u` to :math:`v` implies that
        :math:`u` appears before :math:`v` in the topological sort order.
        This ordering is valid only if the graph has no directed cycles.

        To walk the node-graph, an *NetworkX* graph is constructed by
        connecting the ports together and in turn connecting them to the nodes.

        Yields
        ------
        Generator
            Node-graph walker.

        Examples
        --------
        >>> node_1 = PortNode()
        >>> port = node_1.add_output_port("output")
        >>> node_2 = PortNode()
        >>> port = node_2.add_input_port("a")
        >>> graph = PortGraph()
        >>> graph.add_node(node_1)
        >>> graph.add_node(node_2)
        >>> node_1.connect("output", node_2, "a")
        >>> list(graph.walk())  # doctest: +ELLIPSIS
        [<...PortNode object at 0x...>, <...PortNode object at 0x...>]
        """

        import networkx as nx

        graph = nx.DiGraph()

        for node in self._nodes.values():
            input_edges, output_edges = node.edges

            graph.add_node(node.name, node=node)

            for edge in input_edges:
                # PortGraph is used a container, it is common to connect its
                # input ports to other node input ports and other node output
                # ports to its output ports. The graph generated is thus not
                # acyclic.
                if self in (edge[0].node, edge[1].node):
                    continue

                # Node -> Port -> Port -> Node
                # Connected Node Output Port Node -> Connected Node Output Port
                graph.add_edge(edge[1].node.name, str(edge[1]), edge=edge)
                # Connected Node Output Port -> Node Input Port
                graph.add_edge(str(edge[1]), str(edge[0]), edge=edge)
                # Input Port - Input Port Node
                graph.add_edge(str(edge[0]), edge[0].node.name, edge=edge)

            for edge in output_edges:
                if self in (edge[0].node, edge[1].node):
                    continue

                # Node -> Port -> Port -> Node
                # Output Port Node -> Output Port
                graph.add_edge(edge[0].node.name, str(edge[0]), edge=edge)
                # Node Output Port -> Connected Node Input Port
                graph.add_edge(str(edge[0]), str(edge[1]), edge=edge)
                # Connected Node Input Port -> Connected Node Input Port Node
                graph.add_edge(str(edge[1]), edge[1].node.name, edge=edge)

        try:
            for name in nx.topological_sort(graph):
                node = graph.nodes[name].get("node")
                if node is not None:
                    yield node
        except nx.NetworkXUnfeasible as error:
            filename = "AGraph.png"
            self.log(
                f'A "NetworkX" error occurred, debug graph image has been '
                f'saved to "{os.path.join(os.getcwd(), filename)}"!'
            )
            agraph = nx.nx_agraph.to_agraph(graph)
            agraph.draw(filename, prog="dot")

            raise error  # noqa: TRY201

    def process(self, **kwargs: Dict) -> None:
        """
        Process the node-graph by walking it and calling the
        :func:`colour.utilities.PortNode.process` method.

        Other Parameters
        ----------------
        kwargs
            Keyword arguments.

        Examples
        --------
        >>> class NodeAdd(PortNode):
        ...     def __init__(self, *args, **kwargs):
        ...         super().__init__(*args, **kwargs)
        ...
        ...         self.description = (
        ...             "Perform the addition of the two input port values."
        ...         )
        ...
        ...         self.add_input_port("a")
        ...         self.add_input_port("b")
        ...         self.add_output_port("output")
        ...
        ...     def process(self):
        ...         a = self.get_input("a")
        ...         b = self.get_input("b")
        ...
        ...         if a is None or b is None:
        ...             return
        ...
        ...         self._output_ports["output"].value = a + b
        ...
        ...         self.dirty = False
        >>> node_1 = NodeAdd()
        >>> node_1.set_input("a", 1)
        >>> node_1.set_input("b", 1)
        >>> node_2 = NodeAdd()
        >>> node_1.connect("output", node_2, "a")
        >>> node_2.set_input("b", 1)
        >>> graph = PortGraph()
        >>> graph.add_node(node_1)
        >>> graph.add_node(node_2)
        >>> graph.nodes  # doctest: +ELLIPSIS
        {'NodeAdd#...': <...NodeAdd object at 0x...>, \
'NodeAdd#...': <...NodeAdd object at 0x...>}
        >>> graph.process()
        >>> node_2.get_output("output")
        3
        >>> node_2.dirty
        False
        """

        dry_run = kwargs.get("dry_run", False)

        for_node_reached = False
        for node in self.walk():
            if for_node_reached:
                break

            # Processing currently stops once a control flow node is reached.
            # TODO: Implement solid control flow based processing using a stack.
            if isinstance(node, ControlFlowNode):
                for_node_reached = True

            if not node.dirty:
                self.log(f'Skipping "{node}" computed node.')
                continue

            self.log(f'Processing "{node}" node...')

            if dry_run:
                continue

            node.process()

    @required("Graphviz")
    def to_graphviz(self) -> AGraph:  # noqa: F821  # pyright: ignore
        """
        Return a visualisation node-graph for *Graphviz*.

        Returns
        -------
        :class:`pygraphviz.AGraph`
            String representation for visualisation of the node-graph with
            *Graphviz*.

        Examples
        --------
        >>> node_1 = PortNode()
        >>> port = node_1.add_output_port("output")
        >>> node_2 = PortNode()
        >>> port = node_2.add_input_port("a")
        >>> graph = PortGraph()
        >>> graph.add_node(node_1)
        >>> graph.add_node(node_2)
        >>> node_1.connect("output", node_2, "a")
        >>> graph.to_graphviz()  # doctest: +SKIP
        <AGraph <Swig Object of type 'Agraph_t *' at 0x...>>
        """

        if self.is_subgraph:
            return super().to_graphviz()

        from pygraphviz import AGraph

        agraph = AGraph(strict=False)
        agraph.graph_attr["rankdir"] = "LR"
        agraph.graph_attr["splines"] = "polyline"

        for node in self.walk():
            agraph.add_node(
                f"{node.name} (#{node.id})", label=node.to_graphviz(), shape="record"
            )
            input_edges, output_edges = node.edges

            for edge in input_edges:
                is_ii_or_oo_connection = False

                if edge[0].is_input_port() and edge[1].is_input_port():
                    is_ii_or_oo_connection = True

                if edge[0].is_output_port() and edge[1].is_output_port():
                    is_ii_or_oo_connection = True

                if (
                    isinstance(edge[0].node, PortGraph)
                    and edge[0].node.is_subgraph
                    and is_ii_or_oo_connection
                ):
                    continue

                if (
                    isinstance(edge[1].node, PortGraph)
                    and edge[1].node.is_subgraph
                    and is_ii_or_oo_connection
                ):
                    continue

                agraph.add_edge(
                    f"{edge[1].node.name} (#{edge[1].node.id})",
                    f"{edge[0].node.name} (#{edge[0].node.id})",
                    tailport=edge[1].name,
                    headport=edge[0].name,
                    key=f"{edge[1]} => {edge[0]}",
                    dir="forward",
                )

        return agraph


class ExecutionPort(Port):
    """
    Define a special port for nodes supporting execution input and output
    ports.
    """

    @property
    def value(self) -> Any:
        """
        Getter and setter property for the port value.

        Parameters
        ----------
        value
            Value to set the port value with.

        Returns
        -------
        :class:`object`
            Port value.
        """

    @value.setter
    def value(self, value: Any):
        """Setter for the **self.value** property."""


class ExecutionNode(PortNode):
    """
    Define a special node with execution input and output ports.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.add_input_port(
            "execution_input", None, "Port for input execution", ExecutionPort
        )
        self.add_output_port(
            "execution_output", None, "Port for output execution", ExecutionPort
        )


class ControlFlowNode(ExecutionNode):
    """Define a class inherited by control flow nodes."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class For(ControlFlowNode):
    """
    Define a ``for`` loop node.

    The node loops over the input port ``array``, sets the ``index`` and
    ``element`` output ports at each iteration and call the
    :meth:`colour.utilities.ExecutionNode.process` method of the object
    connected to the ``loop_output`` output port.

    Upon completion, the :meth:`colour.utilities.ExecutionNode.process` method
    of the object connected to the ``execution_output`` output port is called.

    Notes
    -----
    -   The :class:`colour.utilities.For` loop node does not currently call
        more than the two aforementioned
        :meth:`colour.utilities.ExecutionNode.process` methods, if a series of
        nodes is attached to the `loop_output`` or ``execution_output`` output
        ports, only the left-most node will be processed. To circumvent this
        limitation, it is recommended to use a
        :class:`colour.utilities.PortGraph` class instance.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.add_input_port("array", [], "Array to loop onto")
        self.add_output_port("index", None, "Index of the current element of the array")
        self.add_output_port("element", None, "Current element of the array")
        self.add_output_port("loop_output", None, "Port for loop Output", ExecutionPort)

    def process(self) -> None:
        """
        Process the ``for`` loop node.
        """

        connection = next(iter(self.output_ports["loop_output"].connections), None)
        if connection is None:
            return

        node = connection.node

        if node is None:
            return

        self.log(f'Processing "{node}" node...')

        for i, element in enumerate(self.get_input("array")):
            self.log(f"Index {i}, Element {element}", "debug")
            self.set_output("index", i)
            self.set_output("element", element)

            node.process()

        execution_output_connection = next(
            iter(self.output_ports["execution_output"].connections), None
        )
        if execution_output_connection is None:
            return

        execution_output_node = execution_output_connection.node

        if execution_output_node is None:
            return

        execution_output_node.process()

        self.dirty = False


_THREADING_LOCK = threading.Lock()


def _task_thread(args):
    """
    Define the default task for the
    :class:`colour.utilities.ParallelForThread` loop node

    Parameters
    ----------
    args
        Processing arguments.
    """

    i, element, sub_graph, node = args

    node.log(f"Index {i}, Element {element}", "info")

    with _THREADING_LOCK:
        node.set_output("index", i)
        node.set_output("element", element)

        sub_graph.process()

    return i, sub_graph.get_output("output")


class ParallelForThread(ControlFlowNode):
    """
    Define an advanced ``for`` loop node distributing the work across multiple
    threads.

    Each generated task receives one ``index`` and ``element`` output ports
    value. The tasks are then executed by a
    :class:`concurrent.futures.ThreadPoolExecutor` class instance, the futures
    result are then collected, sorted and the ``results`` output port value is
    set with them.

    Upon completion, the :meth:`colour.utilities.ExecutionNode.process` method
    of the object connected to the ``execution_output`` output port is called.

    Notes
    -----
    -   The :class:`colour.utilities.ParallelForThread` loop node does not
        currently call more than the two aforementioned
        :meth:`colour.utilities.ExecutionNode.process` methods, if a series of
        nodes is attached to the `loop_output`` or ``execution_output`` output
        ports, only the left-most node will be processed. To circumvent this
        limitation, it is recommended to use a
        :class:`colour.utilities.PortGraph` class instance.
    -   As the graph being processed is shared across the threads, a lock must
        be taken in the task callable. This might nullify any speed gains for
        heavy processing tasks, in such eventuality, it is recommended to use
        the :class:`colour.utilities.ParallelForMultiprocess` loop node
        instead.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.add_input_port("array", [], "Array to loop onto")
        self.add_input_port("task", _task_thread, "Task to execute")
        self.add_input_port("workers", 16, "Maximum number of workers")
        self.add_output_port("index", None, "Index of the current element of the array")
        self.add_output_port("element", None, "Current element of the array")
        self.add_output_port("results", [], "Results from the parallel loop")
        self.add_output_port("loop_output", None, "Port for loop output", ExecutionPort)

    def process(self) -> None:
        """
        Process the ``for`` loop node.
        """

        connection = next(iter(self.output_ports["loop_output"].connections), None)
        if connection is None:
            return

        node = connection.node

        if node is None:
            return

        self.log(f'Processing "{node}" node...')

        results = {}
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.get_input("workers")
        ) as executor:
            futures = [
                executor.submit(self.get_input("task"), (i, element, node, self))
                for i, element in enumerate(self.get_input("array"))
            ]

            for future in concurrent.futures.as_completed(futures):
                index, element = future.result()
                self.log(f'Processed "{element}" element with index "{index}".')
                results[index] = element

        results = dict(sorted(results.items()))
        self.set_output("results", list(results.values()))

        execution_output_connection = next(
            iter(self.output_ports["execution_output"].connections), None
        )
        if execution_output_connection is None:
            return

        execution_output_node = execution_output_connection.node

        if execution_output_node is None:
            return

        execution_output_node.process()

        self.dirty = False


def _task_multiprocess(args):
    """
    Define the default task for the
    :class:`colour.utilities.ParallelForMultiprocess` loop node

    Parameters
    ----------
    args
        Processing arguments.
    """

    i, element, sub_graph, node = args

    node.log(f"Index {i}, Element {element}", "info")

    node.set_output("index", i)
    node.set_output("element", element)

    sub_graph.process()

    return i, sub_graph.get_output("output")


class ParallelForMultiprocess(ControlFlowNode):
    """
    Define an advanced ``for`` loop node distributing the work across multiple
    processes.

    Each generated task receives one ``index`` and ``element`` output ports
    value. The tasks are then executed by a
    :class:`multiprocessing.Pool` class instance, the results are then
    collected, sorted and the ``results`` output port value is set with them.

    Upon completion, the :meth:`colour.utilities.ExecutionNode.process` method
    of the object connected to the ``execution_output`` output port is called.

    Notes
    -----
    -   The :class:`colour.utilities.ParallelForMultiprocess` loop node does
        not currently call more than the two aforementioned
        :meth:`colour.utilities.ExecutionNode.process` methods, if a series of
        nodes is attached to the `loop_output`` or ``execution_output``
        output ports, only the left-most node will be processed. To circumvent
        this limitation, it is recommended to use a
        :class:`colour.utilities.PortGraph` class instance.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.add_input_port("array", [], "Array to loop onto")
        self.add_input_port("task", _task_multiprocess, "Task to execute")
        self.add_input_port("processes", 4, "Number of processes")
        self.add_output_port("index", None, "Index of the current element of the array")
        self.add_output_port("element", None, "Current element of the array")
        self.add_output_port("results", [], "Results from the parallel loop")
        self.add_output_port("loop_output", None, "Port for loop output", ExecutionPort)

    def process(self) -> None:
        """
        Process the ``for`` loop node.
        """

        connection = next(iter(self.output_ports["loop_output"].connections), None)
        if connection is None:
            return

        node = connection.node

        if node is None:
            return

        self.log(f'Processing "{node}" node...')

        with multiprocessing.Pool(processes=self.get_input("processes")) as pool:
            results = dict(
                pool.map(
                    self.get_input("task"),
                    [
                        (i, element, node, self)
                        for i, element in enumerate(self.get_input("array"))
                    ],
                )
            )

        results = dict(sorted(results.items()))
        self.set_output("results", list(results.values()))

        execution_output_connection = next(
            iter(self.output_ports["execution_output"].connections), None
        )
        if execution_output_connection is None:
            return

        execution_output_node = execution_output_connection.node

        if execution_output_node is None:
            return

        execution_output_node.process()

        self.dirty = False
