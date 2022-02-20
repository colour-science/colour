"""Showcases geometry primitives generation examples."""

import numpy as np

import colour
from colour.utilities import message_box

message_box("Geometry Primitives Generation")

message_box('Generating a "grid":')
print(
    colour.primitive(
        "Grid",
        width=0.2,
        height=0.4,
        width_segments=1,
        height_segments=2,
        axis="+z",
    )
)
print(
    colour.geometry.primitive_grid(
        width=0.2, height=0.4, width_segments=1, height_segments=2, axis="+z"
    )
)

print("\n")

message_box('Generating a "cube":')
print(
    colour.primitive(
        "Cube",
        width=0.2,
        height=0.4,
        depth=0.6,
        width_segments=1,
        height_segments=2,
        depth_segments=3,
    )
)
print(
    colour.geometry.primitive_cube(
        width=0.2,
        height=0.4,
        depth=0.6,
        width_segments=1,
        height_segments=2,
        depth_segments=3,
    )
)

message_box('Generating the vertices of a "quad" for "Matplotlib":')
print(
    colour.primitive_vertices(
        "Quad MPL",
        width=0.2,
        height=0.4,
        depth=0.6,
        origin=np.array([0.2, 0.4]),
        axis="+z",
    )
)
print(
    colour.geometry.primitive_vertices_quad_mpl(
        width=0.2,
        height=0.4,
        depth=0.6,
        origin=np.array([0.2, 0.4]),
        axis="+z",
    )
)

print("\n")

message_box('Generating the vertices of a "grid" for "Matplotlib":')
print(
    colour.primitive_vertices(
        "Grid MPL",
        width=0.2,
        height=0.4,
        depth=0.6,
        width_segments=1,
        height_segments=2,
        origin=np.array([0.2, 0.4]),
        axis="+z",
    )
)
print(
    colour.geometry.primitive_vertices_grid_mpl(
        width=0.2,
        height=0.4,
        depth=0.6,
        width_segments=1,
        height_segments=2,
        origin=np.array([0.2, 0.4]),
        axis="+z",
    )
)

print("\n")

message_box('Generating the vertices of a "cube" for "Matplotlib":')
print(
    colour.primitive_vertices(
        "Cube MPL",
        width=0.2,
        height=0.4,
        depth=0.6,
        width_segments=1,
        height_segments=2,
        depth_segments=3,
        origin=np.array([0.2, 0.4, 0.6]),
    )
)
print(
    colour.geometry.primitive_vertices_cube_mpl(
        width=0.2,
        height=0.4,
        depth=0.6,
        width_segments=1,
        height_segments=2,
        depth_segments=3,
        origin=np.array([0.2, 0.4, 0.6]),
    )
)

print("\n")

message_box('Generating the vertices of a "sphere":')
print(
    colour.primitive_vertices(
        "Sphere",
        radius=100,
        segments=6,
        origin=np.array([-0.2, -0.4, -0.6]),
        axis="+x",
    )
)
print(
    colour.geometry.primitive_vertices_sphere(
        radius=100, segments=6, origin=np.array([-0.2, -0.4, -0.6]), axis="+x"
    )
)
