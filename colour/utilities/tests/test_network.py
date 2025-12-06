"""Define the unit tests for the :mod:`colour.utilities.network` module."""

from __future__ import annotations

import sys

import pytest

if sys.version_info >= (3, 14, 1):
    pytest.skip(
        "networkx 3.6 is incompatible with Python 3.14.1+",
        allow_module_level=True,
    )

import re
import typing

import networkx as nx
import numpy as np

if typing.TYPE_CHECKING:
    from colour.hints import Any

from colour.utilities import (
    ExecutionNode,
    For,
    ParallelForMultiprocess,
    ParallelForThread,
    Port,
    PortGraph,
    PortNode,
    TreeNode,
    is_pydot_installed,
)
from colour.utilities.network import (
    ExecutionPort,
    ProcessPoolExecutorManager,
    ThreadPoolExecutorManager,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestTreeNode",
    "TestPort",
    "TestPortNode",
    "TestPortGraph",
    "TestFor",
    "TestThreadPoolExecutorManager",
    "TestParallelForThread",
    "TestProcessPoolExecutorManager",
    "TestParallelForMultiProcess",
]


class TestTreeNode:
    """
    Define :class:`colour.utilities.network.TreeNode` class unit tests
    methods.
    """

    def setup_method(self) -> None:
        """Initialise the common tests attributes."""

        self._data = {"John": "Doe"}

        self._node_a = TreeNode("Node A", data=self._data)
        self._node_b = TreeNode("Node B", self._node_a)
        self._node_c = TreeNode("Node C", self._node_a)
        self._node_d = TreeNode("Node D", self._node_b)
        self._node_e = TreeNode("Node E", self._node_b)
        self._node_f = TreeNode("Node F", self._node_d)
        self._node_g = TreeNode("Node G", self._node_f)
        self._node_h = TreeNode("Node H", self._node_g)

        self._tree = self._node_a

    def test_required_attributes(self) -> None:
        """Test the presence of required attributes."""

        required_attributes = (
            "id",
            "name",
            "parent",
            "children",
            "root",
            "leaves",
            "siblings",
            "data",
        )

        for attribute in required_attributes:
            assert attribute in dir(TreeNode)

    def test_required_methods(self) -> None:
        """Test the presence of required methods."""

        required_methods = (
            "__new__",
            "__init__",
            "__str__",
            "__len__",
            "is_root",
            "is_inner",
            "is_leaf",
            "walk_hierarchy",
            "render",
        )

        for method in required_methods:
            assert method in dir(TreeNode)

    def test_name(self) -> None:
        """Test :attr:`colour.utilities.network.TreeNode.name` property."""

        assert self._tree.name == "Node A"
        assert "Node#" in TreeNode().name

    def test_parent(self) -> None:
        """Test :attr:`colour.utilities.network.TreeNode.parent` property."""

        assert self._node_b.parent is self._node_a
        assert self._node_h.parent is self._node_g

    def test_children(self) -> None:
        """Test :attr:`colour.utilities.network.TreeNode.children` property."""

        assert self._node_a.children == [self._node_b, self._node_c]

    def test_id(self) -> None:
        """Test :attr:`colour.utilities.network.TreeNode.id` property."""

        assert isinstance(self._node_a.id, int)

    def test_root(self) -> None:
        """Test :attr:`colour.utilities.network.TreeNode.root` property."""

        assert self._node_a.root is self._node_a
        assert self._node_f.root is self._node_a
        assert self._node_g.root is self._node_a
        assert self._node_h.root is self._node_a

    def test_leaves(self) -> None:
        """Test :attr:`colour.utilities.network.TreeNode.leaves` property."""

        assert list(self._node_h.leaves) == [self._node_h]

        assert list(self._node_a.leaves) == [self._node_h, self._node_e, self._node_c]

    def test_siblings(self) -> None:
        """Test :attr:`colour.utilities.network.TreeNode.siblings` property."""

        assert list(self._node_a.siblings) == []

        assert list(self._node_b.siblings) == [self._node_c]

    def test_data(self) -> None:
        """Test :attr:`colour.utilities.network.TreeNode.data` property."""

        assert self._node_a.data is self._data

    def test__str__(self) -> None:
        """Test :attr:`colour.utilities.network.TreeNode.__str__` method."""

        assert "TreeNode#" in str(self._node_a)
        assert "{'John': 'Doe'})" in str(self._node_a)

    def test__len__(self) -> None:
        """Test :attr:`colour.utilities.network.TreeNode.__len__` method."""

        assert len(self._node_a) == 7

    def test_is_root(self) -> None:
        """Test :attr:`colour.utilities.network.TreeNode.is_root` method."""

        assert self._node_a.is_root()
        assert not self._node_b.is_root()
        assert not self._node_c.is_root()
        assert not self._node_h.is_root()

    def test_is_inner(self) -> None:
        """Test :attr:`colour.utilities.network.TreeNode.is_inner` method."""

        assert not self._node_a.is_inner()
        assert self._node_b.is_inner()
        assert not self._node_c.is_inner()
        assert not self._node_h.is_inner()

    def test_is_leaf(self) -> None:
        """Test :attr:`colour.utilities.network.TreeNode.is_leaf` method."""

        assert not self._node_a.is_leaf()
        assert not self._node_b.is_leaf()
        assert self._node_c.is_leaf()
        assert self._node_h.is_leaf()

    def test_walk_hierarchy(self) -> None:
        """Test :attr:`colour.utilities.network.TreeNode.walk_hierarchy` method."""

        assert list(self._node_a.walk_hierarchy()) == [
            self._node_b,
            self._node_d,
            self._node_f,
            self._node_g,
            self._node_h,
            self._node_e,
            self._node_c,
        ]

        assert list(self._node_h.walk_hierarchy(ascendants=True)) == [
            self._node_g,
            self._node_f,
            self._node_d,
            self._node_b,
            self._node_a,
        ]

    def test_render(self) -> None:
        """Test :attr:`colour.utilities.network.TreeNode.render` method."""

        assert isinstance(self._node_a.render(), str)


class TestPort:
    """
    Define :class:`colour.utilities.network.Port` class unit tests methods.
    """

    def setup_method(self) -> None:
        """Initialise the common tests attributes."""

        class Node(PortNode): ...

        self._node_a = Node("Node A")
        self._port_a_node_a = self._node_a.add_input_port("a", 1, "Port A")
        self._port_b_node_a = self._node_a.add_input_port("b")
        self._port_output_node_a = self._node_a.add_output_port(
            "output", description="Output"
        )

        self._node_b = Node("Node B")
        self._port_a_node_b = self._node_b.add_input_port("a", 1, "Port A")
        self._port_b_node_b = self._node_b.add_input_port("b")
        self._port_output_node_b = self._node_b.add_output_port(
            "output", description="Output"
        )

        self._ports = [
            self._port_a_node_a,
            self._port_b_node_a,
            self._port_output_node_a,
            self._port_a_node_b,
            self._port_b_node_b,
            self._port_output_node_b,
        ]

    def test_required_attributes(self) -> None:
        """Test the presence of required attributes."""

        required_attributes = (
            "name",
            "value",
            "description",
            "node",
            "connections",
        )

        for attribute in required_attributes:
            assert attribute in dir(Port)

    def test_required_methods(self) -> None:
        """Test the presence of required methods."""

        required_methods = (
            "__init__",
            "__str__",
            "is_input_port",
            "is_output_port",
            "connect",
            "disconnect",
            "to_graphviz",
        )

        for method in required_methods:
            assert method in dir(Port)

    def test_name(self) -> None:
        """Test :attr:`colour.utilities.network.Port.name` property."""

        assert self._port_a_node_a.name == "a"
        assert self._port_b_node_a.name == "b"
        assert self._port_output_node_a.name == "output"
        assert self._port_a_node_b.name == "a"
        assert self._port_b_node_b.name == "b"
        assert self._port_output_node_b.name == "output"

    def test_value(self) -> None:
        """Test :attr:`colour.utilities.network.Port.value` property."""

        assert self._port_a_node_a.value == 1
        assert self._port_b_node_a.value is None
        assert self._port_output_node_a.value is None
        assert self._port_a_node_b.value == 1
        assert self._port_b_node_b.value is None
        assert self._port_output_node_b.value is None

        self._port_output_node_a.connect(self._port_a_node_b)
        self._port_output_node_a.connect(self._port_b_node_b)

        self._port_output_node_a.value = 2
        assert self._port_output_node_a.node is not None
        assert self._port_output_node_a.node.dirty is True
        assert self._port_a_node_b.node is not None
        assert self._port_a_node_b.node.dirty is True
        assert self._port_a_node_b.value == 2
        assert self._port_b_node_b.value == 2

        self._port_a_node_b.value = 3
        assert self._port_a_node_b.node is not None
        assert self._port_a_node_b.node.dirty is True
        assert self._port_output_node_a.node is not None
        assert self._port_output_node_a.node.dirty is True
        assert self._port_output_node_a.value == 3
        assert self._port_b_node_b.value == 3

        self._port_output_node_a.disconnect(self._port_a_node_b)

        self._port_output_node_a.value = 2
        self._port_output_node_a.node.process()
        assert self._port_output_node_a.node is not None
        assert self._port_output_node_a.node.dirty is False
        assert self._port_a_node_b.node.dirty is True
        assert self._port_a_node_b.value == 3
        assert self._port_b_node_b.value == 2

        self._port_output_node_a.disconnect(self._port_b_node_b)

    def test_description(self) -> None:
        """Test :attr:`colour.utilities.network.Port.description` property."""

        assert self._port_a_node_a.description == "Port A"
        assert self._port_b_node_a.description == ""
        assert self._port_output_node_a.description == "Output"
        assert self._port_a_node_b.description == "Port A"
        assert self._port_b_node_b.description == ""
        assert self._port_output_node_b.description == "Output"

    def test_node(self) -> None:
        """Test :attr:`colour.utilities.network.Port.node` property."""

        assert self._port_a_node_a.node is self._node_a
        assert self._port_a_node_b.node is self._node_b

        port = Port("a", 1, "Port A Description")

        assert port.node is None

    def test_connections(self) -> None:
        """Test :attr:`colour.utilities.network.Port.connections` property."""

        for port in self._ports:
            assert len(port.connections) == 0

        self._port_output_node_a.connect(self._port_a_node_b)
        self._port_output_node_a.connect(self._port_b_node_b)

        assert len(self._port_output_node_a.connections) == 2
        assert len(self._port_a_node_b.connections) == 1
        assert len(self._port_b_node_b.connections) == 1

        self._port_output_node_a.disconnect(self._port_a_node_b)
        self._port_output_node_a.disconnect(self._port_b_node_b)

        for port in self._ports:
            assert len(port.connections) == 0

    def test___str__(self) -> None:
        """Test :meth:`colour.utilities.network.Port.__str__` method."""

        assert str(self._port_a_node_a) == "Node A.a (<- [])"
        assert str(self._port_b_node_a) == "Node A.b (<- [])"
        assert str(self._port_output_node_a) == "Node A.output (-> [])"

        assert str(self._port_a_node_b) == "Node B.a (<- [])"
        assert str(self._port_b_node_b) == "Node B.b (<- [])"
        assert str(self._port_output_node_b) == "Node B.output (-> [])"

        self._port_output_node_a.connect(self._port_a_node_b)
        self._port_output_node_a.connect(self._port_b_node_b)

        assert str(self._port_a_node_a) == "Node A.a (<- [])"
        assert str(self._port_b_node_a) == "Node A.b (<- [])"
        assert (
            str(self._port_output_node_a)
            == "Node A.output (-> ['Node B.a', 'Node B.b'])"
        )

        assert str(self._port_a_node_b) == "Node B.a (<- ['Node A.output'])"
        assert str(self._port_b_node_b) == "Node B.b (<- ['Node A.output'])"
        assert str(self._port_output_node_b) == "Node B.output (-> [])"

        self._port_output_node_a.disconnect(self._port_a_node_b)
        self._port_output_node_a.disconnect(self._port_b_node_b)

    def test_is_input_port(self) -> None:
        """Test :meth:`colour.utilities.network.Port.is_input_port` method."""

        assert self._port_a_node_a.is_input_port() is True
        assert self._port_b_node_a.is_input_port() is True
        assert self._port_output_node_a.is_input_port() is False

        assert self._port_a_node_b.is_input_port() is True
        assert self._port_b_node_b.is_input_port() is True
        assert self._port_output_node_b.is_input_port() is False

        # Test port without node
        port_without_node = Port("test_port")
        assert port_without_node.is_input_port() is False

    def test_is_output_port(self) -> None:
        """Test :meth:`colour.utilities.network.Port.is_output_port` method."""

        assert self._port_a_node_a.is_output_port() is False
        assert self._port_b_node_a.is_output_port() is False
        assert self._port_output_node_a.is_output_port() is True

        assert self._port_a_node_b.is_output_port() is False
        assert self._port_b_node_b.is_output_port() is False
        assert self._port_output_node_b.is_output_port() is True

        # Test port without node
        port_without_node = Port("test_port")
        assert port_without_node.is_output_port() is False

    def test_connect(self) -> None:
        """Test :meth:`colour.utilities.network.Port.connect` method."""

        self.test_connections()

    def test_disconnect(self) -> None:
        """Test :meth:`colour.utilities.network.Port.disconnect` method."""

        self.test_connections()

    def test_to_graphviz(self) -> None:
        """Test :meth:`colour.utilities.network.Port.test_to_graphviz` method."""

        assert self._port_a_node_a.to_graphviz() == "<a> a"
        assert self._port_b_node_a.to_graphviz() == "<b> b"
        assert self._port_output_node_a.to_graphviz() == "<output> output"

        assert self._port_a_node_b.to_graphviz() == "<a> a"
        assert self._port_b_node_b.to_graphviz() == "<b> b"
        assert self._port_output_node_b.to_graphviz() == "<output> output"


class _NodeAdd(ExecutionNode):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.description = "Perform the addition of the two input port values."

        self.add_input_port("a")
        self.add_input_port("b")
        self.add_output_port("output")

    def process(self) -> None:
        a = self.get_input("a")
        b = self.get_input("b")

        if a is None or b is None:
            return

        self.set_output("output", a + b)

        self.dirty = False


class _NodeMultiply(ExecutionNode):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.description = "Perform the multiplication of the two input port values."

        self.add_input_port("a")
        self.add_input_port("b")
        self.add_output_port("output")

    def process(self) -> None:
        a = self.get_input("a")
        b = self.get_input("b")

        if a is None or b is None:
            return

        self.set_output("output", a * b)

        self.dirty = False


class TestPortNode:
    """
    Define :class:`colour.utilities.network.PortNode` class unit tests methods.
    """

    def setup_method(self) -> None:
        """Initialise the common tests attributes."""

        self._add_node_1 = _NodeAdd("Node Add 1")
        self._multiply_node_1 = _NodeMultiply("Node Multiply 1")
        self._add_node_2 = _NodeAdd("Node Add 2")

        self._nodes = [self._add_node_1, self._multiply_node_1, self._add_node_2]

    def test_required_attributes(self) -> None:
        """Test the presence of required attributes."""

        required_attributes = (
            "input_ports",
            "output_ports",
            "dirty",
            "edges",
            "description",
        )

        for attribute in required_attributes:
            assert attribute in dir(PortNode)

    def test_required_methods(self) -> None:
        """Test the presence of required methods."""

        required_methods = (
            "__init__",
            "add_input_port",
            "remove_input_port",
            "add_output_port",
            "remove_output_port",
            "get_input",
            "set_input",
            "get_output",
            "set_output",
            "connect",
            "disconnect",
            "process",
            "to_graphviz",
        )

        for method in required_methods:
            assert method in dir(PortNode)

    def test_input_ports(self) -> None:
        """Test :attr:`colour.utilities.network.PortNode.input_ports` property."""

        for name in ("a", "b"):
            assert name in self._add_node_1.input_ports
            assert name in self._multiply_node_1.input_ports
            assert name in self._add_node_2.input_ports

    def test_output_ports(self) -> None:
        """Test :attr:`colour.utilities.network.PortNode.output_ports` property."""

        for name in ("output",):
            assert name in self._add_node_1.output_ports
            assert name in self._multiply_node_1.output_ports
            assert name in self._add_node_2.output_ports

    def test_dirty(self) -> None:
        """Test :attr:`colour.utilities.network.PortNode.dirty` property."""

        assert self._add_node_1.dirty is True
        assert self._multiply_node_1.dirty is True
        assert self._add_node_2.dirty is True

        self._add_node_1.process()
        self._multiply_node_1.process()
        self._add_node_2.process()

        assert self._add_node_1.dirty is True
        assert self._multiply_node_1.dirty is True
        assert self._add_node_2.dirty is True

        self._add_node_1.set_input("a", 1)
        self._add_node_1.set_input("b", 1)
        self._multiply_node_1.set_input("a", 1)
        self._multiply_node_1.set_input("b", 1)
        self._add_node_2.set_input("a", 1)
        self._add_node_2.set_input("b", 1)

        self._add_node_1.process()
        self._multiply_node_1.process()
        self._add_node_2.process()

        assert self._add_node_1.dirty is False
        assert self._multiply_node_1.dirty is False
        assert self._add_node_2.dirty is False

        self._add_node_1.set_input("a", None)
        self._add_node_1.set_input("b", None)
        self._multiply_node_1.set_input("a", None)
        self._multiply_node_1.set_input("b", None)
        self._add_node_2.set_input("a", None)
        self._add_node_2.set_input("b", None)

        assert self._add_node_1.dirty is True
        assert self._multiply_node_1.dirty is True
        assert self._add_node_2.dirty is True

    def test_edges(self) -> None:
        """Test :attr:`colour.utilities.network.PortNode.edges` property."""

        assert self._add_node_1.edges == ({}, {})
        assert self._multiply_node_1.edges == ({}, {})
        assert self._add_node_2.edges == ({}, {})

        self._add_node_1.connect("output", self._multiply_node_1, "a")
        self._multiply_node_1.connect("output", self._add_node_2, "a")

        assert self._add_node_1.edges == (
            {},
            {
                (
                    self._add_node_1.output_ports["output"],
                    self._multiply_node_1.input_ports["a"],
                ): None,
            },
        )
        assert self._multiply_node_1.edges == (
            {
                (
                    self._multiply_node_1.input_ports["a"],
                    self._add_node_1.output_ports["output"],
                ): None,
            },
            {
                (
                    self._multiply_node_1.output_ports["output"],
                    self._add_node_2.input_ports["a"],
                ): None,
            },
        )
        assert self._add_node_2.edges == (
            {
                (
                    self._add_node_2.input_ports["a"],
                    self._multiply_node_1.output_ports["output"],
                ): None,
            },
            {},
        )

        self._add_node_1.disconnect("output", self._multiply_node_1, "a")
        self._multiply_node_1.disconnect("output", self._add_node_2, "a")

        assert self._add_node_1.edges == ({}, {})
        assert self._multiply_node_1.edges == ({}, {})
        assert self._add_node_2.edges == ({}, {})

    def test_description(self) -> None:
        """Test :attr:`colour.utilities.network.PortNode.description` property."""

        assert (
            self._add_node_1.description
            == "Perform the addition of the two input port values."
        )
        assert (
            self._multiply_node_1.description
            == "Perform the multiplication of the two input port values."
        )
        assert (
            self._add_node_2.description
            == "Perform the addition of the two input port values."
        )

    def test_add_input_port(self) -> None:
        """Test :meth:`colour.utilities.network.PortNode.add_input_port` method."""

        node = PortNode()
        node.add_input_port("a", 1, 'Input Port "a"')

        assert node.input_ports["a"].value == 1
        assert node.input_ports["a"].description == 'Input Port "a"'

    def test_remove_input_port(self) -> None:
        """Test :meth:`colour.utilities.network.PortNode.remove_input_port` method."""

        node = PortNode()
        node.add_input_port("a", 1, 'Input Port "a"')
        node.remove_input_port("a")

        assert len(node.input_ports) == 0

        # Test removing input port with connections
        node_a = PortNode()
        node_b = PortNode()
        port_a = node_a.add_input_port("a", 1)
        port_b = node_b.add_output_port("output", 1)
        port_a.connect(port_b)
        assert len(port_a.connections) == 1
        assert len(port_b.connections) == 1

        removed_port = node_a.remove_input_port("a")
        assert len(node_a.input_ports) == 0
        assert len(removed_port.connections) == 0
        assert len(port_b.connections) == 0

    def test_add_output_port(self) -> None:
        """Test :meth:`colour.utilities.network.PortNode.add_output_port` method."""

        node = PortNode()
        node.add_output_port("output", 1, 'Output Port "output"')

        assert node.output_ports["output"].value == 1
        assert node.output_ports["output"].description == 'Output Port "output"'

    def test_remove_output_port(self) -> None:
        """Test :meth:`colour.utilities.network.PortNode.remove_output_port` method."""

        node = PortNode()
        node.add_output_port("output", 1, 'Output Port "output"')
        node.remove_output_port("output")

        assert len(node.output_ports) == 0

        # Test removing output port with connections
        node_a = PortNode()
        node_b = PortNode()
        port_a = node_a.add_output_port("output", 1)
        port_b = node_b.add_input_port("a", 1)
        port_a.connect(port_b)
        assert len(port_a.connections) == 1
        assert len(port_b.connections) == 1

        removed_port = node_a.remove_output_port("output")
        assert len(node_a.output_ports) == 0
        assert len(removed_port.connections) == 0
        assert len(port_b.connections) == 0

    def test_get_input(self) -> None:
        """Test :meth:`colour.utilities.network.PortNode.get_input` method."""

        node = PortNode()
        node.add_input_port("a", 1, 'Input Port "a"')

        assert node.get_input("a") == 1

    def test_set_input(self) -> None:
        """Test :meth:`colour.utilities.network.PortNode.set_input` method."""

        node = PortNode()
        node.add_input_port("a", 1, 'Input Port "a"')

        assert node.input_ports["a"].value == 1

        node.set_input("a", 2)

        assert node.input_ports["a"].value == 2

    def test_get_output(self) -> None:
        """Test :meth:`colour.utilities.network.PortNode.get_output` method."""

        node = PortNode()
        node.add_output_port("output", 1, 'Output Port "output"')

        assert node.get_output("output") == 1

    def test_set_output(self) -> None:
        """Test :meth:`colour.utilities.network.PortNode.set_output` method."""

        node = PortNode()
        node.add_output_port("output", 1, 'Output Port "output"')

        assert node.output_ports["output"].value == 1

        node.set_output("output", 2)

        assert node.output_ports["output"].value == 2

    def test_connect(self) -> None:
        """Test :meth:`colour.utilities.network.PortNode.connect` method."""

        self.test_edges()

    def test_disconnect(self) -> None:
        """Test :meth:`colour.utilities.network.PortNode.disconnect` method."""

        self.test_edges()

    def test_process(self) -> None:
        """Test :meth:`colour.utilities.network.PortNode.process` method."""

        self._add_node_1.connect("output", self._multiply_node_1, "a")
        self._multiply_node_1.connect("output", self._add_node_2, "a")

        self._add_node_1.set_input("a", 1)
        self._add_node_1.set_input("b", 1)
        self._multiply_node_1.set_input("b", 2)
        self._add_node_2.set_input("b", 1)

        assert self._add_node_2.get_output("output") is None

        self._add_node_1.process()
        self._multiply_node_1.process()
        self._add_node_2.process()

        assert self._add_node_2.get_output("output") == 5

        self._add_node_1.disconnect("output", self._multiply_node_1, "a")
        self._multiply_node_1.disconnect("output", self._add_node_2, "a")

    def test_to_graphviz(self) -> None:
        """Test :meth:`colour.utilities.network.PortNode.to_graphviz` method."""

        assert (
            re.sub(r"\(#\d+\)", "(#)", self._add_node_1.to_graphviz())
            == "Node Add 1 (#) | {{<execution_input> execution_input|<a> "
            "a|<b> b} | {<execution_output> execution_output|<output> output}}"
        )
        assert (
            re.sub(r"\(#\d+\)", "(#)", self._multiply_node_1.to_graphviz())
            == "Node Multiply 1 (#) | {{<execution_input> execution_input|<a> "
            "a|<b> b} | {<execution_output> execution_output|<output> output}}"
        )
        assert (
            re.sub(r"\(#\d+\)", "(#)", self._add_node_2.to_graphviz())
            == "Node Add 2 (#) | {{<execution_input> execution_input|<a> "
            "a|<b> b} | {<execution_output> execution_output|<output> output}}"
        )


class TestPortGraph:
    """
    Define :class:`colour.utilities.network.PortGraph` class unit tests methods.
    """

    def setup_method(self) -> None:
        """Initialise the common tests attributes."""

        self._add_node_1 = _NodeAdd("Node Add 1")
        self._multiply_node_1 = _NodeMultiply("Node Multiply 1")
        self._add_node_2 = _NodeAdd("Node Add 2")

        self._add_node_1.connect("output", self._multiply_node_1, "a")
        self._multiply_node_1.connect("output", self._add_node_2, "a")

        self._add_node_1.set_input("a", 1)
        self._add_node_1.set_input("b", 1)
        self._multiply_node_1.set_input("b", 2)
        self._add_node_2.set_input("b", 1)

        self._nodes = {
            self._add_node_1.name: self._add_node_1,
            self._multiply_node_1.name: self._multiply_node_1,
            self._add_node_2.name: self._add_node_2,
        }

        self._graph = PortGraph("Port Graph")

        for node in self._nodes.values():
            self._graph.add_node(node)

    def test_required_attributes(self) -> None:
        """Test the presence of required attributes."""

        required_attributes = ("nodes",)

        for attribute in required_attributes:
            assert attribute in dir(PortGraph)

    def test_required_methods(self) -> None:
        """Test the presence of required methods."""

        required_methods = (
            "__init__",
            "__str__",
            "add_node",
            "remove_node",
            "walk_ports",
            "process",
            "to_graphviz",
        )

        for method in required_methods:
            assert method in dir(PortGraph)

    def test_nodes(self) -> None:
        """Test :attr:`colour.utilities.network.PortGraph.nodes` property."""

        assert self._graph.nodes == self._nodes

    def test___str__(self) -> None:
        """Test :meth:`colour.utilities.network.PortGraph.__str__` method."""

        assert str(self._graph) == "PortGraph(3)"

    def test_add_node(self) -> None:
        """Test :meth:`colour.utilities.network.PortGraph.add_node` method."""

        for node in self._nodes.values():
            self._graph.remove_node(node)

        assert len(self._graph.nodes) == 0

        for node in self._nodes.values():
            self._graph.add_node(node)

        assert len(self._graph.nodes) == 3

    def test_remove_node(self) -> None:
        """Test :meth:`colour.utilities.network.PortGraph.remove_node` method."""

        self.test_add_node()

        # Test removing a node with connections
        graph = PortGraph("Test Graph")
        node_a = PortNode("Node A")
        node_b = PortNode("Node B")

        port_out = node_a.add_output_port("output", 1)
        port_in = node_b.add_input_port("input", 1)
        port_out.connect(port_in)

        graph.add_node(node_a)
        graph.add_node(node_b)

        assert len(port_out.connections) == 1
        assert len(port_in.connections) == 1

        # Remove node_b which has an input port with a connection
        graph.remove_node(node_b)

        assert len(graph.nodes) == 1
        assert len(port_out.connections) == 0
        assert len(port_in.connections) == 0

        # Test removing a node with output port connections
        graph2 = PortGraph("Test Graph 2")
        node_c = PortNode("Node C")
        node_d = PortNode("Node D")

        port_out2 = node_c.add_output_port("output", 1)
        port_in2 = node_d.add_input_port("input", 1)
        port_out2.connect(port_in2)

        graph2.add_node(node_c)
        graph2.add_node(node_d)

        # Remove node_c which has an output port with a connection
        graph2.remove_node(node_c)

        assert len(graph2.nodes) == 1
        assert len(port_out2.connections) == 0
        assert len(port_in2.connections) == 0

    def test_walk_ports(self) -> None:
        """Test :meth:`colour.utilities.network.PortGraph.walk_ports` method."""

        assert list(self._graph.walk_ports()) == list(self._nodes.values())

        # Test with a node that has children (nested graph)
        nested_graph = PortGraph()
        child_node = PortNode()
        nested_graph.add_node(child_node)

        outer_graph = PortGraph()
        outer_graph.add_node(nested_graph)

        # Call walk_ports to trigger the check for nodes with children
        list(outer_graph.walk_ports())

        # Test with a PortGraph that has ports connected to other nodes
        # This triggers the check where self is in the edge nodes
        graph_with_ports = PortGraph()
        node_a = PortNode()
        node_b = PortNode()

        graph_with_ports.add_node(node_a)
        graph_with_ports.add_node(node_b)

        # Add ports to the graph itself
        graph_input = graph_with_ports.add_input_port("graph_input", 0)
        graph_output = graph_with_ports.add_output_port("graph_output", 0)

        # Connect graph ports to nodes
        node_a_output = node_a.add_output_port("output", 0)
        node_b_input = node_b.add_input_port("input", 0)

        graph_input.connect(node_b_input)
        node_a_output.connect(graph_output)

        # Call walk_ports to trigger the edge checks
        list(graph_with_ports.walk_ports())

        # Test with a cyclic graph to trigger NetworkXUnfeasible exception
        # Skip if pydot is not installed (needed for debug visualization)
        if not is_pydot_installed():  # pragma: no cover
            return

        cyclic_graph = PortGraph()
        node_1 = PortNode()
        node_2 = PortNode()
        node_3 = PortNode()

        cyclic_graph.add_node(node_1)
        cyclic_graph.add_node(node_2)
        cyclic_graph.add_node(node_3)

        # Create a cycle: node_1 -> node_2 -> node_3 -> node_1
        output_port_1 = node_1.add_output_port("out", 0)
        input_port_2 = node_2.add_input_port("in", 0)
        output_port_2 = node_2.add_output_port("out", 0)
        input_port_3 = node_3.add_input_port("in", 0)
        output_port_3 = node_3.add_output_port("out", 0)
        input_port_1 = node_1.add_input_port("in", 0)

        output_port_1.connect(input_port_2)
        output_port_2.connect(input_port_3)
        output_port_3.connect(input_port_1)  # This creates the cycle

        # Try to walk the cyclic graph - should raise NetworkXUnfeasible
        with pytest.raises(nx.NetworkXUnfeasible):
            list(cyclic_graph.walk_ports())

    def test_process(self) -> None:
        """Test :meth:`colour.utilities.network.PortGraph.process` method."""

        self._graph.process()

        assert self._add_node_2.get_output("output") == 5

        # Test with dry_run=True
        graph2 = PortGraph()
        node_a = PortNode()
        node_a.add_input_port("in", 1)
        node_a.add_output_port("out", 1)
        graph2.add_node(node_a)
        graph2.process(dry_run=True)

        # Test with ControlFlowNode
        graph3 = PortGraph()
        node_before_for = PortNode()
        node_before_for.add_output_port("out", 1)
        for_node = For()
        node_after_for = PortNode()
        node_after_for.add_input_port("in", 0)

        graph3.add_node(node_before_for)
        graph3.add_node(for_node)
        graph3.add_node(node_after_for)

        # Connect: node_before_for -> for_node -> node_after_for
        graph3.process()

        # Test with non-dirty node
        graph4 = PortGraph()
        node_b = PortNode()
        node_b.add_output_port("out", 1)
        node_b._dirty = False  # noqa: SLF001
        graph4.add_node(node_b)
        graph4.process()

    def test_to_graphviz(self) -> None:
        """Test :meth:`colour.utilities.network.PortGraph.to_graphviz` method."""

        if not is_pydot_installed():  # pragma: no cover
            return

        import pydot  # noqa: PLC0415

        assert isinstance(self._graph.to_graphviz(), pydot.Dot)

        # Test nested graph with parent
        parent_graph = PortGraph("Parent")
        child_graph = PortGraph("Child")
        parent_graph.add_node(child_graph)
        child_graph._parent = parent_graph  # noqa: SLF001

        # This should call PortNode.to_graphviz when graph has parent
        result = child_graph.to_graphviz()
        assert result is not None

        # Test with graph members to trigger edge skipping
        graph_with_members = PortGraph("Graph With Members")
        inner_graph = PortGraph("Inner Graph")
        node_a = PortNode("Node A")
        node_b = PortNode("Node B")

        inner_graph.add_node(node_a)
        graph_with_members.add_node(inner_graph)
        graph_with_members.add_node(node_b)

        # Connect nodes to create edges that involve graph members
        port_a = node_a.add_output_port("out", 1)
        port_b = node_b.add_input_port("in", 0)
        port_a.connect(port_b)

        # This should skip edges involving graph members
        result = graph_with_members.to_graphviz()
        assert isinstance(result, pydot.Dot)


class _AddItem(ExecutionNode):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.description = "Add the item with input key and value to the input mapping."

        self.add_input_port("key")
        self.add_input_port("value")
        self.add_input_port("mapping", {})

    def process(self) -> None:
        """
        Process the node.
        """

        key = self.get_input("key")
        value = self.get_input("value")

        if key is None or value is None:  # pragma: no cover
            return

        self.get_input("mapping")[key] = value

        self.dirty = False


class _NodeSumMappingValues(ExecutionNode):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.description = "Sum the input mapping values."

        self.add_input_port("mapping", {})
        self.add_output_port("summation")

    def process(self) -> None:
        mapping = self.get_input("mapping")
        if len(mapping) == 0:  # pragma: no cover
            return

        self.set_output("summation", np.sum(list(mapping.values())))

        self.dirty = False


class _SubGraph1(ExecutionNode, PortGraph):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.add_input_port("input")
        self.add_output_port("output", {})

        for node in [
            _NodeAdd("Add 1"),
            _NodeMultiply("Multiply 1"),
            _NodeAdd("Add 2"),
            _AddItem("Add Item"),
        ]:
            self.add_node(node)

        for connection in [
            (
                ("Add 1", "output"),
                ("Multiply 1", "a"),
            ),
            (
                ("Add 1", "execution_output"),
                ("Multiply 1", "execution_input"),
            ),
            (
                ("Multiply 1", "output"),
                ("Add 2", "a"),
            ),
            (
                ("Multiply 1", "execution_output"),
                ("Add 2", "execution_input"),
            ),
            (
                ("Add 2", "execution_output"),
                ("Add Item", "execution_input"),
            ),
            (
                ("Add 2", "output"),
                ("Add Item", "value"),
            ),
        ]:
            (input_node, input_port), (output_node, output_port) = connection
            self.nodes[input_node].connect(
                input_port,
                self.nodes[output_node],
                output_port,
            )

        self.connect("input", self.nodes["Add 1"], "b")
        self.connect("input", self.nodes["Add Item"], "key")
        self.nodes["Add Item"].connect("mapping", self, "output")

    def process(self, **kwargs: Any) -> None:
        # Coverage can't track execution in subprocesses  # pragma: no cover
        self.nodes["Add 1"].set_input("a", 1)  # pragma: no cover
        self.nodes["Multiply 1"].set_input("b", 2)  # pragma: no cover
        self.nodes["Add 2"].set_input("b", 3)  # pragma: no cover
        super().process(**kwargs)  # pragma: no cover


class TestFor:
    """
    Define :class:`colour.utilities.network.For` class unit tests methods.
    """

    def test_For(self) -> None:
        """Test :class:`colour.utilities.network.For` class."""

        sum_mapping_values = _NodeSumMappingValues()

        sub_graph = _SubGraph1()
        sub_graph.connect("output", sum_mapping_values, "mapping")

        loop = For()
        loop.connect("loop_output", sub_graph, "execution_input")
        loop.connect("index", sub_graph, "input")
        loop.connect("execution_output", sum_mapping_values, "execution_input")
        loop.set_input("array", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        loop.process()

        assert sum_mapping_values.get_output("summation") == 140

        # Test For node without loop_output connection - covers early return
        loop_no_connection = For()
        loop_no_connection.set_input("array", [1, 2, 3])
        loop_no_connection.process()  # Should return early without error

        # Test For node without execution_output connection
        loop_no_exec_output = For()
        sub_graph_2 = _SubGraph1()
        loop_no_exec_output.connect("loop_output", sub_graph_2, "execution_input")
        loop_no_exec_output.set_input("array", [1, 2, 3])
        loop_no_exec_output.process()  # Should complete loop but not have exec output

        # Test For node with connection.node = None (edge case)
        loop_none_node = For()
        orphan_port = ExecutionPort(node=None)
        loop_none_node.output_ports["loop_output"].connect(orphan_port)
        loop_none_node.set_input("array", [1, 2, 3])
        loop_none_node.process()  # Should return early when node is None

        # Test For node with execution_output.node = None (edge case)
        loop_exec_none = For()
        sub_graph_3 = _SubGraph1()
        loop_exec_none.connect("loop_output", sub_graph_3, "execution_input")
        orphan_exec_port = ExecutionPort(node=None)
        loop_exec_none.output_ports["execution_output"].connect(orphan_exec_port)
        loop_exec_none.set_input("array", [1, 2, 3])
        loop_exec_none.process()


class _NodeSumArray(ExecutionNode):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.description = "Sum the input array."

        self.add_input_port("array", [])
        self.add_output_port("summation")

    def process(self) -> None:
        array = self.get_input("array")
        if len(array) == 0:  # pragma: no cover
            return

        self.set_output("summation", np.sum(array))

        self.dirty = False


class _SubGraph2(ExecutionNode, PortGraph):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.add_input_port("input")
        self.add_output_port("output")

        for node in [
            _NodeAdd("Add 1"),
            _NodeMultiply("Multiply 1"),
            _NodeAdd("Add 2"),
        ]:
            self.add_node(node)

        for connection in [
            (
                ("Add 1", "output"),
                ("Multiply 1", "a"),
            ),
            (
                ("Add 1", "execution_output"),
                ("Multiply 1", "execution_input"),
            ),
            (
                ("Multiply 1", "output"),
                ("Add 2", "a"),
            ),
            (
                ("Multiply 1", "execution_output"),
                ("Add 2", "execution_input"),
            ),
        ]:
            (input_node, input_port), (output_node, output_port) = connection
            self.nodes[input_node].connect(
                input_port,
                self.nodes[output_node],
                output_port,
            )

        self.connect("input", self.nodes["Add 1"], "b")
        self.nodes["Add 2"].connect("output", self, "output")

    def process(self, **kwargs: Any) -> None:  # pragma: no cover
        # Coverage can't track execution in subprocesses

        self.nodes["Add 1"].set_input("a", 1)
        self.nodes["Multiply 1"].set_input("b", 2)
        self.nodes["Add 2"].set_input("b", 3)

        super().process(**kwargs)


class TestThreadPoolExecutorManager:
    """
    Define :class:`colour.utilities.network.ThreadPoolExecutorManager` class unit tests
    methods.
    """

    def test_ThreadPoolExecutorManager(self) -> None:
        """Test :class:`colour.utilities.network.ThreadPoolExecutorManager` class."""

        executor = ThreadPoolExecutorManager.get_executor()

        assert executor is not None

        # Test shutdown_executor method
        ThreadPoolExecutorManager.shutdown_executor()
        assert ThreadPoolExecutorManager.ThreadPoolExecutor is None

        # Recreate for other tests
        executor = ThreadPoolExecutorManager.get_executor()
        assert executor is not None


class TestParallelForThread:
    """
    Define :class:`colour.utilities.network.ParallelForThread` class unit tests
    methods.
    """

    def test_ParallelForThread(self) -> None:
        """Test :class:`colour.utilities.network.ParallelForThread` class."""

        sum_array = _NodeSumArray()

        sub_graph = _SubGraph2()

        loop = ParallelForThread()
        loop.connect("loop_output", sub_graph, "execution_input")
        loop.connect("index", sub_graph, "input")
        loop.connect("execution_output", sum_array, "execution_input")
        loop.set_input("array", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        loop.connect("results", sum_array, "array")
        loop.process()

        assert sum_array.get_output("summation") == 140

        # Test ParallelForThread without loop_output connection
        loop_no_connection = ParallelForThread()
        loop_no_connection.set_input("array", [1, 2, 3])
        loop_no_connection.process()

        # Test ParallelForThread with connection.node = None
        loop_none_node = ParallelForThread()
        orphan_port = ExecutionPort(node=None)
        loop_none_node.output_ports["loop_output"].connect(orphan_port)
        loop_none_node.set_input("array", [1, 2, 3])
        loop_none_node.process()

        # Test ParallelForThread without execution_output connection
        loop_no_exec = ParallelForThread()
        sub_graph_2 = _SubGraph2()
        loop_no_exec.connect("loop_output", sub_graph_2, "execution_input")
        loop_no_exec.set_input("array", [1, 2, 3])
        loop_no_exec.process()

        # Test ParallelForThread with execution_output.node = None
        loop_exec_none = ParallelForThread()
        sub_graph_3 = _SubGraph2()
        loop_exec_none.connect("loop_output", sub_graph_3, "execution_input")
        orphan_exec_port = ExecutionPort(node=None)
        loop_exec_none.output_ports["execution_output"].connect(orphan_exec_port)
        loop_exec_none.set_input("array", [1, 2, 3])
        loop_exec_none.process()


class TestProcessPoolExecutorManager:
    """
    Define :class:`colour.utilities.network.ProcessPoolExecutorManager` class unit tests
    methods.
    """

    def test_ProcessPoolExecutorManager(self) -> None:
        """Test :class:`colour.utilities.network.ProcessPoolExecutorManager` class."""

        executor = ProcessPoolExecutorManager.get_executor()

        assert executor is not None

        # Test shutdown_executor method
        ProcessPoolExecutorManager.shutdown_executor()
        assert ProcessPoolExecutorManager.ProcessPoolExecutor is None

        # Recreate for other tests
        executor = ProcessPoolExecutorManager.get_executor()
        assert executor is not None


class TestParallelForMultiProcess:
    """
    Define :class:`colour.utilities.network.ParallelForMultiProcess` class unit
    tests methods.
    """

    def test_ParallelForMultiProcess(self) -> None:
        """Test :class:`colour.utilities.network.ParallelForMultiProcess` class."""

        sum_array = _NodeSumArray()

        sub_graph = _SubGraph2()

        loop = ParallelForMultiprocess()
        loop.connect("loop_output", sub_graph, "execution_input")
        loop.connect("index", sub_graph, "input")
        loop.connect("execution_output", sum_array, "execution_input")
        loop.set_input("array", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        loop.connect("results", sum_array, "array")
        loop.process()

        assert sum_array.get_output("summation") == 140

        # Test ParallelForMultiprocess without loop_output connection
        loop_no_connection = ParallelForMultiprocess()
        loop_no_connection.set_input("array", [1, 2, 3])
        loop_no_connection.process()

        # Test ParallelForMultiprocess with connection.node = None
        loop_none_node = ParallelForMultiprocess()
        orphan_port = ExecutionPort(node=None)
        loop_none_node.output_ports["loop_output"].connect(orphan_port)
        loop_none_node.set_input("array", [1, 2, 3])
        loop_none_node.process()

        # Test ParallelForMultiprocess without execution_output connection
        loop_no_exec = ParallelForMultiprocess()
        sub_graph_2 = _SubGraph2()
        loop_no_exec.connect("loop_output", sub_graph_2, "execution_input")
        loop_no_exec.set_input("array", [1, 2, 3])
        loop_no_exec.process()

        # Test ParallelForMultiprocess with execution_output.node = None
        loop_exec_none = ParallelForMultiprocess()
        sub_graph_3 = _SubGraph2()
        loop_exec_none.connect("loop_output", sub_graph_3, "execution_input")
        orphan_exec_port = ExecutionPort(node=None)
        loop_exec_none.output_ports["execution_output"].connect(orphan_exec_port)
        loop_exec_none.set_input("array", [1, 2, 3])
        loop_exec_none.process()
