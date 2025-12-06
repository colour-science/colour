"""
Network
=======

Node-graph and network infrastructure for computational workflows.

-   :class:`colour.utilities.TreeNode`: Basic node object supporting
    creation of hierarchical node trees.
-   :class:`colour.utilities.Port`: Object that can be added as either an
    input or output port for data flow.
-   :class:`colour.utilities.PortMode`: Node with support for input and
    output ports.
-   :class:`colour.utilities.PortGraph`: Graph structure for nodes with
    input and output ports.
-   :class:`colour.utilities.ExecutionPort`: Object for nodes supporting
    execution input and output ports.
-   :class:`colour.utilities.ExecutionNode`: Node with built-in input and
    output execution ports.
-   :class:`colour.utilities.ControlFlowNode`: Base node inherited by
    control flow nodes.
-   :class:`colour.utilities.For`: Node performing for loops in the
    node-graph.
-   :class:`colour.utilities.ParallelForThread`: Node performing for loops
    in parallel in the node-graph using threads.
-   :class:`colour.utilities.ParallelForMultiprocess`: Node performing for
    loops in parallel in the node-graph using multiprocessing.
"""

from __future__ import annotations

import atexit
import concurrent.futures
import multiprocessing
import os
import threading
import typing

if typing.TYPE_CHECKING:
    from colour.hints import (
        Any,
        Dict,
        Generator,
        List,
        Self,
        Sequence,
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
    "ThreadPoolExecutorManager",
    "ParallelForThread",
    "ProcessPoolExecutorManager",
    "ParallelForMultiprocess",
]


class TreeNode:
    """
    Define a basic node supporting the creation of hierarchical node
    trees.

    Parameters
    ----------
    name
        Node name.
    parent
        Parent of the node.
    children
        Children of the node.
    data
        Data belonging to this node.

    Attributes
    ----------
    -   :attr:`~colour.utilities.TreeNode.id`
    -   :attr:`~colour.utilities.TreeNode.name`
    -   :attr:`~colour.utilities.TreeNode.parent`
    -   :attr:`~colour.utilities.TreeNode.children`
    -   :attr:`~colour.utilities.TreeNode.root`
    -   :attr:`~colour.utilities.TreeNode.leaves`
    -   :attr:`~colour.utilities.TreeNode.siblings`
    -   :attr:`~colour.utilities.TreeNode.data`

    Methods
    -------
    -   :meth:`~colour.utilities.TreeNode.__new__`
    -   :meth:`~colour.utilities.TreeNode.__init__`
    -   :meth:`~colour.utilities.TreeNode.__str__`
    -   :meth:`~colour.utilities.TreeNode.__len__`
    -   :meth:`~colour.utilities.TreeNode.is_root`
    -   :meth:`~colour.utilities.TreeNode.is_inner`
    -   :meth:`~colour.utilities.TreeNode.is_leaf`
    -   :meth:`~colour.utilities.TreeNode.walk_hierarchy`
    -   :meth:`~colour.utilities.TreeNode.render`

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

    def __new__(cls, *args: Any, **kwargs: Any) -> Self:  # noqa: ARG004
        """
        Return a new instance of the :class:`colour.utilities.TreeNode` class.

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
        Getter for the node identifier.

        Returns
        -------
        :class:`int`
            Node identifier.
        """

        return self._id  # pyright: ignore

    @property
    def name(self) -> str:
        """
        Getter and setter for the node name.

        Parameters
        ----------
        value
            Value to set the node name with.

        Returns
        -------
        :class:`str`
            Node name.
        """

        return self._name

    @name.setter
    def name(self, value: str) -> None:
        """Setter for the **self.name** property."""

        attest(
            isinstance(value, str),
            f'"name" property: "{value}" type is not "str"!',
        )

        self._name = value

    @property
    def parent(self) -> Self | None:
        """
        Getter and setter for the node parent.

        Parameters
        ----------
        value
            Parent to set the node with.

        Returns
        -------
        :class:`TreeNode` or :py:data:`None`
            Node parent.
        """

        return self._parent

    @parent.setter
    def parent(self, value: Self | None) -> None:
        """Setter for the **self.parent** property."""

        from colour.utilities import attest  # noqa: PLC0415

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
        Getter and setter for the node children.

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
    def children(self, value: List[Self]) -> None:
        """Setter for the **self.children** property."""

        from colour.utilities import attest  # noqa: PLC0415

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
        Getter for the root node of the tree hierarchy.

        Returns
        -------
        :class:`TreeNode`
            Root node of the tree.
        """

        if self.is_root():
            return self

        return list(self.walk_hierarchy(ascendants=True))[-1]

    @property
    def leaves(self) -> Generator:
        """
        Getter for all leaf nodes in the hierarchy.

        Returns
        -------
        Generator
            Generator yielding all leaf nodes (nodes without children) in
            the hierarchy.
        """

        if self.is_leaf():
            return (node for node in (self,))

        return (node for node in self.walk_hierarchy() if node.is_leaf())

    @property
    def siblings(self) -> Generator:
        """
        Getter for the sibling nodes at the same hierarchical level.

        Returns
        -------
        Generator
            Generator yielding sibling nodes that share the same parent
            node in the hierarchy.
        """

        if self.parent is None:
            return (sibling for sibling in ())

        return (sibling for sibling in self.parent.children if sibling is not self)

    @property
    def data(self) -> Any:
        """
        Getter and setter for the node data.

        Parameters
        ----------
        value
            Data to assign to the node.

        Returns
        -------
        :class:`object`
            Data stored in the node.
        """

        return self._data

    @data.setter
    def data(self, value: Any) -> None:
        """Setter for the **self.data** property."""

        self._data = value

    def __str__(self) -> str:
        """
        Return a formatted string representation of the node.

        Returns
        -------
        :class:`str`
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

        return len(list(self.walk_hierarchy()))

    def is_root(self) -> bool:
        """
        Determine whether the node is a root node.

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
        Determine whether the node is an inner node.

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
        Determine whether the node is a leaf node.

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

    def walk_hierarchy(self, ascendants: bool = False) -> Generator:
        """
        Generate a generator to walk the :class:`colour.utilities.TreeNode`
        tree hierarchy.

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
        >>> for node in node_a.walk_hierarchy():
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

            yield from node.walk_hierarchy(ascendants=ascendants)

    def render(self, tab_level: int = 0) -> str:
        """
        Render the node and its children as a formatted tree string.

        Parameters
        ----------
        tab_level
            Initial indentation level for the tree structure.

        Returns
        -------
        :class:`str`
            Formatted tree representation of the node hierarchy.

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
    Define a port object that serves as an input or output port (i.e., a
    pin) for a :class:`colour.utilities.PortNode` class and connects to
    other input or output ports.

    Parameters
    ----------
    name
        Port name.
    value
        Initial value to set the port with.
    description
        Port description.
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
        description: str = "",
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
        self.description = description

    @property
    def name(self) -> str:
        """
        Getter and setter for the port name.

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
    def name(self, value: str) -> None:
        """Setter for the **self.name** property."""

        attest(
            isinstance(value, str),
            f'"name" property: "{value}" type is not "str"!',
        )

        self._name = value

    @property
    def value(self) -> Any:
        """
        Getter and setter for the port value.

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
            return connection._value  # noqa: SLF001

        return self._value

    @value.setter
    def value(self, value: Any) -> None:
        """Setter for the **self.value** property."""

        self._value = value

        if self._node is not None:
            self.log(f'Dirtying "{self._node}".', "debug")
            self._node.dirty = True

        # NOTE: Setting the port value implies that all the connected ports
        # should be also set to the same specified value.
        for direct_connection in self._connections:
            self.log(f'Setting "{direct_connection.node}" value to {value}.', "debug")
            direct_connection._value = value  # noqa: SLF001

            if direct_connection.node is not None:
                self.log(f'Dirtying "{direct_connection.node}".', "debug")
                direct_connection.node.dirty = True

            for indirect_connection in direct_connection.connections:
                if indirect_connection == self:
                    continue

                self.log(
                    f'Setting "{indirect_connection.node}" value to {value}.', "debug"
                )
                indirect_connection._value = value  # noqa: SLF001

                if indirect_connection.node is not None:
                    self.log(f'Dirtying "{indirect_connection.node}".', "debug")
                    indirect_connection.node.dirty = True

        self._value = value

    @property
    def description(self) -> str:
        """
        Getter and setter for the port description.

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
    def description(self, value: str) -> None:
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
        Getter and setter for the port node.

        Parameters
        ----------
        value : PortNode or None
            Port node to set.

        Returns
        -------
        :class:`PortNode` or None
            Port node.
        """

        return self._node

    @node.setter
    def node(self, value: PortNode | None) -> None:
        """Setter for the **self.node** property."""

        attest(
            value is None or isinstance(value, PortNode),
            f'"node" property: "{value}" is not "None" or its type is not "PortNode"!',
        )

        self._node = value

    @property
    def connections(self) -> Dict[Port, None]:
        """
        Getter for the port connections.

        Returns
        -------
        :class:`dict`
            Port connections mapping each :class:`Port` instance to
            ``None``.
        """

        return self._connections

    def __str__(self) -> str:
        """
        Return a formatted string representation of the port.

        Returns
        -------
        :class:`str`
            Formatted string representation.

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
        Determine whether the port is an input port.

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
        Determine whether the port is an output port.

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
        Connect this port to the specified port.

        Parameters
        ----------
        port
            Port to connect to.

        Raises
        ------
        ValueError
            If an attempt is made to connect an input port to multiple
            output ports.

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
        Disconnect from the specified port.

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
        Generate a string representation for port visualisation with
        *Graphviz*.

        Returns
        -------
        :class:`str`
            String representation for visualisation of the port with
            *Graphviz*.

        Examples
        --------
        >>> Port("a").to_graphviz()
        '<a> a'
        """

        return f"<{self._name}> {self.name}"


class PortNode(TreeNode, MixinLogging):
    """
    Define a node with support for input and output ports.

    Other Parameters
    ----------------
    name
        Node name.

    Attributes
    ----------
    -   :attr:`~colour.utilities.PortNode.input_ports`
    -   :attr:`~colour.utilities.PortNode.output_ports`
    -   :attr:`~colour.utilities.PortNode.dirty`
    -   :attr:`~colour.utilities.PortNode.edges`
    -   :attr:`~colour.utilities.PortNode.description`

    Methods
    -------
    -   :meth:`~colour.utilities.PortNode.__init__`
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
    ...     def __init__(self, *args: Any, **kwargs: Any):
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

    def __init__(self, name: str | None = None, description: str = "") -> None:
        super().__init__(name)
        self.description = description

        self._input_ports = {}
        self._output_ports = {}
        self._dirty = True

    @property
    def input_ports(self) -> Dict[str, Port]:
        """
        Getter for the input ports of the node.

        Returns
        -------
        :class:`dict`
            Dictionary mapping port names to their corresponding input port
            instances.
        """

        return self._input_ports

    @property
    def output_ports(self) -> Dict[str, Port]:
        """
        Getter for the output ports of the node.

        Returns
        -------
        :class:`dict`
            Mapping of output port names to their corresponding :class:`Port`
            instances.
        """

        return self._output_ports

    @property
    def dirty(self) -> bool:
        """
        Getter and setter for the node's dirty state.

        Parameters
        ----------
        value
            Value to set the node dirty state with.

        Returns
        -------
        :class:`bool`
            Whether the node is in a dirty state.
        """

        return self._dirty

    @dirty.setter
    def dirty(self, value: bool) -> None:
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
        Getter for the edges of the node.

        Retrieve the edges representing ports and their connections. Each
        edge corresponds to a port and one of its connections within the
        node structure.

        Returns
        -------
        :class:`tuple`
            Edges of the node as a tuple of input and output edge
            dictionaries.
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
    def description(self) -> str:
        """
        Getter and setter for the node description.

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
    def description(self, value: str) -> None:
        """Setter for the **self.description** property."""

        attest(
            value is None or isinstance(value, str),
            f'"description" property: "{value}" is not "None" or '
            f'its type is not "str"!',
        )

        self._description = value

    def add_input_port(
        self,
        name: str,
        value: Any = None,
        description: str = "",
        port_type: Type[Port] = Port,
    ) -> Port:
        """
        Add an input port with specified name and value to the node.

        Parameters
        ----------
        name
            Name of the input port.
        value
            Value of the input port.
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
        Remove the input port with the specified name from the node.

        Parameters
        ----------
        name
            Name of the input port to remove.

        Returns
        -------
        :class:`colour.utilities.Port`
            Removed input port.

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

        for connection in port.connections.copy():
            port.disconnect(connection)

        return port

    def add_output_port(
        self,
        name: str,
        value: Any = None,
        description: str = "",
        port_type: Type[Port] = Port,
    ) -> Port:
        """
        Add an output port with the specified name and value to the node.

        Parameters
        ----------
        name
            Name of the output port.
        value
            Value of the output port.
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
        Remove the output port with the specified name from the node.

        Parameters
        ----------
        name
            Name of the output port to remove.

        Returns
        -------
        :class:`colour.utilities.Port`
            Removed output port.

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

        for connection in port.connections.copy():
            port.disconnect(connection)

        return port

    def get_input(self, name: str) -> Any:
        """
        Return the value of the input port with the specified name.

        Parameters
        ----------
        name
            Name of the input port.

        Returns
        -------
        :class:`object`
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
        Set the value of an input port with the specified name.

        Parameters
        ----------
        name
            Name of the input port to set.
        value
            Value to assign to the input port.

        Raises
        ------
        AssertionError
            If the specified input port is not a member of the node's
            input ports.

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

    def get_output(self, name: str) -> Any:
        """
        Return the value of the output port with the specified name.

        Parameters
        ----------
        name
            Name of the output port.

        Returns
        -------
        :class:`object`
            Value of the output port.

        Raises
        ------
        AssertionError
            If the output port is not a member of the node output
            ports.

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
        Set the value of the output port with the specified name.

        Parameters
        ----------
        name
            Name of the output port.
        value
            Value to assign to the output port.

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
        Connect the specified source port to the specified target port of
        another node.

        The source port can be an input port but the target port must be
        an output port and conversely, if the source port is an output
        port, the target port must be an input port.

        Parameters
        ----------
        source_port
            Source port of the node to connect to the other node target
            port.
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
        Disconnect the specified source port from the specified target node
        port.

        The source port can be an input port but the target port must be an
        output port and conversely, if the source port is an output port,
        the target port must be an input port.

        Parameters
        ----------
        source_port
            Source port of the node to disconnect from the other node target
            port.
        target_node
            Target node that the target port is the member of.
        target_port
            Target port from the target node to disconnect the source port
            from.

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

        This definition is responsible for setting the dirty state of the
        node according to the processing outcome.

        Examples
        --------
        >>> class NodeAdd(PortNode):
        ...     def __init__(self, *args: Any, **kwargs: Any):
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
        Generate a string representation for node visualisation with
        *Graphviz*.

        Returns
        -------
        :class:`str`
            String representation for visualisation of the node with
            *Graphviz*.

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
    Define a node-graph for :class:`colour.utilities.PortNode` class
    instances.

    Parameters
    ----------
    name
        Name of the node-graph.
    description
        Description of the node-graph's purpose or functionality.

    Attributes
    ----------
    -   :attr:`~colour.utilities.PortGraph.nodes`

    Methods
    -------
    -   :meth:`~colour.utilities.PortGraph.__str__`
    -   :meth:`~colour.utilities.PortGraph.add_node`
    -   :meth:`~colour.utilities.PortGraph.remove_node`
    -   :meth:`~colour.utilities.PortGraph.walk_ports`
    -   :meth:`~colour.utilities.PortGraph.process`
    -   :meth:`~colour.utilities.PortGraph.to_graphviz`

    Examples
    --------
    >>> class NodeAdd(PortNode):
    ...     def __init__(self, *args: Any, **kwargs: Any):
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

    def __init__(self, name: str | None = None, description: str = "") -> None:
        super().__init__(name, description)

        self._name: str = self.__class__.__name__
        self.name = optional(name, self._name)
        self.description = description

        self._nodes = {}

    @property
    def nodes(self) -> Dict[str, PortNode]:
        """
        Getter for the node-graph nodes.

        Returns
        -------
        :class:`dict`
            Node-graph nodes as a mapping from node identifiers to their
            corresponding :class:`PortNode` instances.
        """

        return self._nodes

    def __str__(self) -> str:
        """
        Return a formatted string representation of the node-graph.

        Returns
        -------
        :class:`str`
            Formatted string representation.
        """

        return f"{self.__class__.__name__}({len(self._nodes)})"

    def add_node(self, node: PortNode) -> None:
        """
        Add specified node to the node-graph.

        Parameters
        ----------
        node
            Node to add to the node-graph.

        Raises
        ------
        AssertionError
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

        attest(isinstance(node, PortNode), f'"{node}" is not a "PortNode" instance!')

        attest(
            node.name not in self._nodes, f'"{node}" is already a member of the graph!'
        )

        self._nodes[node.name] = node
        self._children.append(node)  # pyright: ignore
        node._parent = self  # noqa: SLF001

    def remove_node(self, node: PortNode) -> None:
        """
        Remove the specified node from the node-graph.

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

        attest(isinstance(node, PortNode), f'"{node}" is not a "PortNode" instance!')

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
        self._children.remove(node)  # pyright: ignore
        node._parent = None  # noqa: SLF001

    @required("NetworkX")
    def walk_ports(self) -> Generator:
        """
        Return a generator to walk the node-graph in topological order.

        Walk the node according to topologically sorted order. A topological
        sort is a non-unique permutation of the nodes of a directed graph
        such that an edge from :math:`u` to :math:`v` implies that :math:`u`
        appears before :math:`v` in the topological sort order. This ordering
        is valid only if the graph has no directed cycles.

        To walk the node-graph, a *NetworkX* graph is constructed by
        connecting the ports together and in turn connecting them to the
        nodes.

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
        >>> list(graph.walk_ports())  # doctest: +ELLIPSIS
        [<...PortNode object at 0x...>, <...PortNode object at 0x...>]
        """

        import networkx as nx  # noqa: PLC0415

        graph = nx.DiGraph()

        for node in self._children:
            input_edges, output_edges = node.edges

            graph.add_node(node.name, node=node)

            if len(node.children) != 0:
                continue

            for edge in input_edges:
                # PortGraph is used a container, it is common to connect its
                # input ports to other node input ports and other node output
                # ports to its output ports. The graph generated is thus not
                # acyclic.
                if self in (edge[0].node, edge[1].node):
                    continue

                # Node -> Port -> Port -> Node
                # Connected Node Output Port Node -> Connected Node Output Port
                graph.add_edge(
                    edge[1].node.name,  # pyright: ignore
                    str(edge[1]),
                    edge=edge,
                )
                # Connected Node Output Port -> Node Input Port
                graph.add_edge(str(edge[1]), str(edge[0]), edge=edge)
                # Input Port - Input Port Node
                graph.add_edge(
                    str(edge[0]),
                    edge[0].node.name,  # pyright: ignore
                    edge=edge,
                )

            for edge in output_edges:
                if self in (edge[0].node, edge[1].node):
                    continue

                # Node -> Port -> Port -> Node
                # Output Port Node -> Output Port
                graph.add_edge(
                    edge[0].node.name,  # pyright: ignore
                    str(edge[0]),
                    edge=edge,
                )
                # Node Output Port -> Connected Node Input Port
                graph.add_edge(str(edge[0]), str(edge[1]), edge=edge)
                # Connected Node Input Port -> Connected Node Input Port Node
                graph.add_edge(
                    str(edge[1]),
                    edge[1].node.name,  # pyright: ignore
                    edge=edge,
                )

        try:
            for name in nx.topological_sort(graph):
                node = graph.nodes[name].get("node")
                if node is not None:
                    yield node
        except nx.NetworkXUnfeasible as error:
            filename = "AGraph.png"
            self.log(  # pyright: ignore
                f'A "NetworkX" error occurred, debug graph image has been '
                f'saved to "{os.path.join(os.getcwd(), filename)}"!'
            )

            def rename_reserved(data: dict) -> dict:
                """Rename DOT reserved keywords by prefixing with underscore."""

                reserved = {"node", "edge", "graph"}
                return {
                    f"_{key}" if key in reserved else key: value
                    for key, value in data.items()
                }

            unfeasible_graph = nx.DiGraph()
            for node, data in graph.nodes(data=True):
                unfeasible_graph.add_node(node, **rename_reserved(data))
            for source, target, data in graph.edges(data=True):
                unfeasible_graph.add_edge(source, target, **rename_reserved(data))

            dot = nx.drawing.nx_pydot.to_pydot(unfeasible_graph)
            dot.write_png(filename)  # type: ignore[attr-defined]

            raise error  # noqa: TRY201

    def process(self, **kwargs: Any) -> None:
        """
        Process the node-graph by traversing it and executing the
        :func:`colour.utilities.PortNode.process` method for each node.

        Other Parameters
        ----------------
        kwargs
            Keyword arguments.

        Examples
        --------
        >>> class NodeAdd(PortNode):
        ...     def __init__(self, *args: Any, **kwargs: Any):
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
        for node in self.walk_ports():
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

    @required("Pydot")
    def to_graphviz(self) -> Dot:  # noqa: F821  # pyright: ignore
        """
        Generate a node-graph visualisation for *Graphviz*.

        Returns
        -------
        :class:`pydot.Dot`
            *Pydot* graph.

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
        <pydot.core.Dot object at 0x...>
        """

        if self._parent is not None:
            return PortNode.to_graphviz(self)

        import pydot  # noqa: PLC0415

        dot = pydot.Dot(
            "digraph", graph_type="digraph", rankdir="LR", splines="polyline"
        )

        graphs = [node for node in self.walk_ports() if isinstance(node, PortGraph)]

        def is_graph_member(node: PortNode) -> bool:
            """Determine whether the specified node is member of a graph."""

            return any(node in graph.nodes.values() for graph in graphs)

        for node in self.walk_ports():
            dot.add_node(
                pydot.Node(
                    f"{node.name} (#{node.id})",
                    label=node.to_graphviz(),
                    shape="record",
                )
            )
            input_edges, output_edges = node.edges

            for edge in input_edges:
                # Not drawing node edges that involve a node member of graph.
                if is_graph_member(edge[0].node) or is_graph_member(edge[1].node):
                    continue

                dot.add_edge(
                    pydot.Edge(
                        f"{edge[1].node.name} (#{edge[1].node.id})",
                        f"{edge[0].node.name} (#{edge[0].node.id})",
                        tailport=edge[1].name,
                        headport=edge[0].name,
                        key=f"{edge[1]} => {edge[0]}",
                        dir="forward",
                    )
                )

        return dot


class ExecutionPort(Port):
    """
    Define a specialised port for execution flow control in node graphs.

    Attributes
    ----------
    value
        Port value accessor for execution state transmission.
    """

    @property
    def value(self) -> Any:
        """
        Getter and setter for the port value.

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
    def value(self, value: Any) -> None:
        """Setter for the **self.value** property."""


class ExecutionNode(PortNode):
    """
    Define a specialised node that manages execution flow through
    dedicated input and output ports.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.add_input_port(
            "execution_input", None, "Port for input execution", ExecutionPort
        )
        self.add_output_port(
            "execution_output", None, "Port for output execution", ExecutionPort
        )


class ControlFlowNode(ExecutionNode):
    """
    Define a base class for control flow nodes in computational graphs.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)


class For(ControlFlowNode):
    """
    Define a ``for`` loop node for iterating over arrays.

    The node iterates over the input port ``array``, setting the
    ``index`` and ``element`` output ports at each iteration and calling
    the :meth:`colour.utilities.ExecutionNode.process` method of the
    object connected to the ``loop_output`` output port.

    Upon completion, the :meth:`colour.utilities.ExecutionNode.process`
    method of the object connected to the ``execution_output`` output
    port is called.

    Notes
    -----
    -   The :class:`colour.utilities.For` loop node does not currently
        call more than the two aforementioned
        :meth:`colour.utilities.ExecutionNode.process` methods, if a
        series of nodes is attached to the ``loop_output`` or
        ``execution_output`` output ports, only the left-most node will
        be processed. To circumvent this limitation, it is recommended
        to use a :class:`colour.utilities.PortGraph` class instance.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.add_input_port("array", [], "Array to loop onto")
        self.add_output_port("index", None, "Index of the current element of the array")
        self.add_output_port("element", None, "Current element of the array")
        self.add_output_port("loop_output", None, "Port for loop Output", ExecutionPort)

    def process(self) -> None:
        """Process the *for* loop node execution."""

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


def _task_thread(args: Sequence) -> tuple[int, Any]:  # pragma: no cover
    """
    Execute the default task for the
    :class:`colour.utilities.ParallelForThread` loop node.

    Parameters
    ----------
    args
        Processing arguments for the parallel thread task.

    Returns
    -------
    :class:`tuple`
        Index and result pair from the executed task.
    """

    i, element, sub_graph, node = args

    node.log(f"Index {i}, Element {element}", "info")

    with _THREADING_LOCK:
        node.set_output("index", i)
        node.set_output("element", element)

        sub_graph.process()

    return i, sub_graph.get_output("output")


class ThreadPoolExecutorManager:
    """
    Define a singleton :class:`concurrent.futures.ThreadPoolExecutor`
    manager.

    Attributes
    ----------
    -   :attr:`~colour.utilities.ThreadPoolExecutorManager.ThreadPoolExecutor`

    Methods
    -------
    -   :meth:`~colour.utilities.ThreadPoolExecutorManager.get_executor`
    -   :meth:`~colour.utilities.ThreadPoolExecutorManager.shutdown_executor`
    """

    ThreadPoolExecutor: concurrent.futures.ThreadPoolExecutor | None = None

    @staticmethod
    def get_executor(
        max_workers: int | None = None,
    ) -> concurrent.futures.ThreadPoolExecutor:
        """
        Return the :class:`concurrent.futures.ThreadPoolExecutor` class
        instance or create it if not existing.

        Parameters
        ----------
        max_workers
            Maximum worker count.

        Returns
        -------
        :class:`concurrent.futures.ThreadPoolExecutor`
            Thread pool executor instance.

        Notes
        -----
        The :class:`concurrent.futures.ThreadPoolExecutor` class instance is
        automatically shutdown on process exit.
        """

        if ThreadPoolExecutorManager.ThreadPoolExecutor is None:
            ThreadPoolExecutorManager.ThreadPoolExecutor = (
                concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
            )

        return ThreadPoolExecutorManager.ThreadPoolExecutor

    @atexit.register
    @staticmethod
    def shutdown_executor() -> None:
        """
        Shut down the :class:`concurrent.futures.ThreadPoolExecutor` class
        instance.
        """

        if ThreadPoolExecutorManager.ThreadPoolExecutor is not None:
            ThreadPoolExecutorManager.ThreadPoolExecutor.shutdown(wait=True)
            ThreadPoolExecutorManager.ThreadPoolExecutor = None


class ParallelForThread(ControlFlowNode):
    """
    Define an advanced ``for`` loop node that distributes work across
    multiple threads for parallel execution.

    Each generated task receives one ``index`` and ``element`` output port
    value. The tasks are executed by a
    :class:`concurrent.futures.ThreadPoolExecutor` class instance. The
    futures results are collected, sorted, and assigned to the ``results``
    output port.

    Upon completion, the :meth:`colour.utilities.ExecutionNode.process`
    method of the object connected to the ``execution_output`` output port
    is called.

    Notes
    -----
    -   The :class:`colour.utilities.ParallelForThread` loop node does not
        currently call more than the two aforementioned
        :meth:`colour.utilities.ExecutionNode.process` methods. If a series
        of nodes is attached to the ``loop_output`` or ``execution_output``
        output ports, only the left-most node will be processed. To
        circumvent this limitation, it is recommended to use a
        :class:`colour.utilities.PortGraph` class instance.
    -   As the graph being processed is shared across the threads, a lock
        must be taken in the task callable. This might nullify any speed
        gains for heavy processing tasks. In such eventuality, it is
        recommended to use the
        :class:`colour.utilities.ParallelForMultiprocess` loop node
        instead.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
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
        Process the parallel loop node execution.
        """

        connection = next(iter(self.output_ports["loop_output"].connections), None)
        if connection is None:
            return

        node = connection.node

        if node is None:
            return

        self.log(f'Processing "{node}" node...')

        results = {}
        thread_pool_executor = ThreadPoolExecutorManager.get_executor(
            max_workers=self.get_input("workers")
        )
        futures = [
            thread_pool_executor.submit(
                self.get_input("task"), (i, element, node, self)
            )
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


def _task_multiprocess(args: Sequence) -> tuple[int, Any]:  # pragma: no cover
    """
    Execute the default processing task for
    :class:`colour.utilities.ParallelForMultiprocess` loop node instances.

    Parameters
    ----------
    args
        Processing arguments for the parallel execution task.

    Returns
    -------
    :class:`tuple`
        Tuple containing the task index and computed result.
    """

    i, element, sub_graph, node = args

    node.log(f"Index {i}, Element {element}", "info")

    node.set_output("index", i)
    node.set_output("element", element)

    sub_graph.process()

    return i, sub_graph.get_output("output")


class ProcessPoolExecutorManager:
    """
    Define a singleton :class:`concurrent.futures.ProcessPoolExecutor`
    manager for parallel processing.

    Attributes
    ----------
    -   :attr:`~colour.utilities.ProcessPoolExecutorManager.ProcessPoolExecutor`

    Methods
    -------
    -   :meth:`~colour.utilities.ProcessPoolExecutorManager.get_executor`
    -   :meth:`~colour.utilities.ProcessPoolExecutorManager.shutdown_executor`
    """

    ProcessPoolExecutor: concurrent.futures.ProcessPoolExecutor | None = None

    @staticmethod
    def get_executor(
        max_workers: int | None = None,
    ) -> concurrent.futures.ProcessPoolExecutor:
        """
        Return the :class:`concurrent.futures.ProcessPoolExecutor` class
        instance or create it if not existing.

        Parameters
        ----------
        max_workers
            Maximum number of worker processes. If ``None``, it will
            default to the number of processors on the machine.

        Returns
        -------
        :class:`concurrent.futures.ProcessPoolExecutor`
            Process pool executor instance for parallel execution.

        Notes
        -----
        The :class:`concurrent.futures.ProcessPoolExecutor` class instance is
        automatically shut down on process exit.
        """

        if ProcessPoolExecutorManager.ProcessPoolExecutor is None:
            context = multiprocessing.get_context("spawn")
            ProcessPoolExecutorManager.ProcessPoolExecutor = (
                concurrent.futures.ProcessPoolExecutor(
                    mp_context=context, max_workers=max_workers
                )
            )

        return ProcessPoolExecutorManager.ProcessPoolExecutor

    @atexit.register
    @staticmethod
    def shutdown_executor() -> None:
        """
        Shut down the :class:`concurrent.futures.ProcessPoolExecutor` class
        instance.
        """

        if ProcessPoolExecutorManager.ProcessPoolExecutor is not None:
            ProcessPoolExecutorManager.ProcessPoolExecutor.shutdown(wait=True)
            ProcessPoolExecutorManager.ProcessPoolExecutor = None


class ParallelForMultiprocess(ControlFlowNode):
    """
    Define a parallel ``for`` loop node that distributes operations across
    multiple processes.

    Distribute iteration work by assigning each task one ``index`` and
    ``element`` output port value. Execute tasks using a
    :class:`multiprocessing.Pool` instance, then collect, sort, and assign
    results to the ``results`` output port.

    Upon completion, invoke the :meth:`colour.utilities.ExecutionNode.process`
    method of the object connected to the ``execution_output`` output port.

    Notes
    -----
    -   The :class:`colour.utilities.ParallelForMultiprocess` loop node
        currently invokes only the two aforementioned
        :meth:`colour.utilities.ExecutionNode.process` methods. When a series
        of nodes connects to the ``loop_output`` or ``execution_output``
        output ports, only the left-most node processes. To circumvent this
        limitation, use a :class:`colour.utilities.PortGraph` class instance.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
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
        Process the ``for`` loop node execution.
        """

        connection = next(iter(self.output_ports["loop_output"].connections), None)
        if connection is None:
            return

        node = connection.node

        if node is None:
            return

        self.log(f'Processing "{node}" node...')

        results = {}
        process_pool_executor = ProcessPoolExecutorManager.get_executor(
            max_workers=self.get_input("processes")
        )
        futures = [
            process_pool_executor.submit(
                self.get_input("task"), (i, element, node, self)
            )
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
