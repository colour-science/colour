# *************************************************************************
# Gamut Mapping Playground
#
# Plot.ly and Trimesh are required dependencies to run this code.
#
# *************************************************************************
print('*' * 79)
print('*' * 79)
# *************************************************************************
# Plotting the two spheres with Plot.ly.
# *************************************************************************
# source /Users/kelsolaar/Library/Caches/pypoetry/virtualenvs/colour-AKjB3F3b-py3.7/bin/activate
# python /Users/kelsolaar/Documents/Development/colour-science/colour/colour/gamut/playground.py

import numpy as np
import sys
import trimesh.smoothing
import plotly.graph_objects as go

import colour.plotting
from colour.gamut import (gamut_boundary_descriptor_Morovic2000,
                          tessellate_volume_boundary_descriptor)
from colour.geometry import (primitive_cube, primitive_vertices_cube_mpl,
                             primitive_vertices_sphere)
from colour.algebra import spherical_to_cartesian
from colour.utilities import linear_conversion, tsplit, tstack

np.set_printoptions(
    formatter={'float': '{:0.3f}'.format},
    linewidth=2048,
    suppress=True,
    threshold=sys.maxsize)

colour.plotting.colour_style()

# *************************************************************************
# Samples for a 8x8 Latitude-Longitude segments sphere.
# *************************************************************************
Jab_88s = primitive_vertices_sphere(
    origin=np.array([50, 0, 0]),
    segments=8,
    radius=50,
    axis='+x',
)

# *************************************************************************
# Samples for a 8x7 Latitude-Longitude segments sphere contained within
# the sectors of a 8x8 Latitude-Longitude segments sphere.
# *************************************************************************
Jab_87s = primitive_vertices_sphere(
    origin=np.array([50, 0, 0]),
    segments=8,
    radius=50,
    axis='+x',
    intermediate=True,
)

# *************************************************************************
# Plotting the two spheres with Plot.ly.
# *************************************************************************
if True:
    X_i_88s, Y_i_88s, Z_i_88s = tsplit(Jab_88s)
    X_i_87s, Y_i_87s, Z_i_87s = tsplit(Jab_87s)

    fig = go.Figure(data=[
        go.Scatter3d(
            x=np.ravel(Y_i_88s),
            y=np.ravel(Z_i_88s),
            z=np.ravel(X_i_88s),
            mode='markers',
            marker=dict(size=3),
        ),
        go.Scatter3d(
            x=np.ravel(Y_i_87s),
            y=np.ravel(Z_i_87s),
            z=np.ravel(X_i_87s),
            mode='markers',
            marker=dict(size=6),
        ),
    ])

    fig.show()

# *************************************************************************
# Ideal sRGB cube.
# *************************************************************************
vertices, faces, outline = primitive_cube(1, 1, 1, 32, 32, 32)
with colour.utilities.domain_range_scale(1):
    RGB_r = colour.colorimetry.luminance_CIE1976(vertices['position'] + 0.5)
Jab_32c = colour.convert(
    RGB_r, 'RGB', 'CIE Lab', verbose={'describe': 'short'}) * 100
mesh_r = trimesh.Trimesh(
    vertices=Jab_32c.reshape([-1, 3]), faces=faces, validate=True)
mesh_r.fix_normals

if False:
    mesh_r.show()

# *************************************************************************
# Alternate sRGB cube.
# *************************************************************************
segments = 32
RGB_t = primitive_vertices_cube_mpl(
    width_segments=segments, height_segments=segments, depth_segments=segments)
print(RGB_t)

Jab_32c = colour.convert(
    RGB_t, 'RGB', 'CIE Lab', verbose={'describe': 'short'}) * 100

# *************************************************************************
# Alternate sRGB cube.
# *************************************************************************
if False:
    XYZ = np.random.rand(1000, 3)
    XYZ = XYZ[colour.is_within_visible_spectrum(XYZ)]

    Jab_1000r = colour.convert(
        XYZ, 'CIE XYZ', 'CIE Lab', verbose={'describe': 'short'}) * 100

# *************************************************************************
# Reference data from Jan: https://github.com/colour-science/PGMA_v2.1/blob/master/test_data/test_input/ogamut_m.abl
# *************************************************************************
ogamut_m = '/Users/kelsolaar/Documents/Development/colour-science/PGMA_v2.1/test_data/test_input/ogamut_m.abl'
Jab_oabl = np.roll(np.loadtxt(ogamut_m, delimiter='\t'), 1, -1)

# *************************************************************************
# Plotting the reference data from Jan.
# *************************************************************************
if False:
    X_i_oabl, Y_i_oabl, Z_i_oabl = tsplit(Jab_oabl)

    fig = go.Figure(data=[
        go.Scatter3d(
            x=np.ravel(Y_i_oabl),
            y=np.ravel(Z_i_oabl),
            z=np.ravel(X_i_oabl),
            mode='markers',
            marker=dict(size=3),
        )
    ])
    fig.show()

# *************************************************************************
# Principal "Jab" array to plot.
# *************************************************************************
Jab = Jab_87s
# Jab = Jab_32c
# Jab = Jab_1000r
# Jab = Jab_oabl

# *************************************************************************
# "GBD" generation settings.
# *************************************************************************
v_r = 8
h_r = 8
# v_r = 16
# h_r = 16
# v_r = 32
# h_r = 32
v_h = [
    (v_r, h_r),
    # (v_r // 2, h_r // 2),
]
slc_v = v_r
slc_h = h_r

GBD_n = []
for v, h in v_h:
    GBD = gamut_boundary_descriptor_Morovic2000(Jab, [50, 0, 0], v, h)
    GBD_n.append(GBD)

GBD_oabl = gamut_boundary_descriptor_Morovic2000(Jab_oabl, [50, 0, 0], v, h)
GBD_n.append(GBD_oabl)
# *************************************************************************
# "plot_multi_segment_maxima_gamut_boundaries_in_hue_segments"
# *************************************************************************
if True:
    colour.plotting.plot_multi_segment_maxima_gamut_boundaries_in_hue_segments(
        GBD_n,
        # angles=[5, 10, 45, 180, 270],
        # columns=8,
        transparent_background=False)

# *************************************************************************
# Debugging image.
# *************************************************************************
if False:
    GBD_d = np.copy(GBD_n[0])
    ranges = [[-50, 50], [0, np.pi], [-np.pi, np.pi]]
    for i in range(3):
        GBD_d[..., i] = linear_conversion(GBD_d[..., i],
                                          (ranges[i][0], ranges[i][1]), (0, 1))
    colour.plotting.plot_image(GBD_d)

# *************************************************************************
# "plot_Jab_samples_in_segment_maxima_gamut_boundary"
# *************************************************************************
if True:
    colour.plotting.plot_Jab_samples_in_segment_maxima_gamut_boundary(
        [Jab, Jab_oabl],
        'ab',
        gamut_boundary_descriptor_kwargs={
            'm': v_r,
            'n': h_r
        })

if True:
    colour.plotting.plot_Jab_samples_in_segment_maxima_gamut_boundary(
        [Jab, Jab_oabl],
        'Ja',
        gamut_boundary_descriptor_kwargs={
            'm': v_r,
            'n': h_r
        },
        show_debug_circles=True)

if True:
    colour.plotting.plot_Jab_samples_in_segment_maxima_gamut_boundary(
        [Jab, Jab_oabl],
        'Jb',
        gamut_boundary_descriptor_kwargs={
            'm': v_r,
            'n': h_r
        },
        show_debug_markers=True)

GBD_n_t = [tessellate_volume_boundary_descriptor(GBD) for GBD in GBD_n]
# GBD_n_t = [tessellate_volume_boundary_descriptor(GBD_oabl)]

GBD_t_r = GBD_n_t[0]
for i, GBD_t in enumerate(GBD_n_t[1:]):
    # GBD_t.vertices += [(i + 1) * 100, 0, 0]
    GBD_t_r = GBD_t_r + GBD_t

# mesh_r.vertices -= [50, 150, 0]
# GBD_t_r = GBD_t_r + mesh_r

GBD_t_r.export('/Users/kelsolaar/Downloads/mesh.obj', 'obj')

if True:
    GBD_t_r.show()
