# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**plots.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Colour** package **Autodesk Maya** plotting objects.

**Others:**

"""

import maya.cmds as cmds
import maya.OpenMaya as OpenMaya
import numpy

import colour.computation.colourspaces.cie_xyy
import colour.computation.colourspaces.cie_lab
import colour.dataset.illuminants.chromaticity_coordinates
import colour.utilities.data_structures

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["get_dag_path",
           "get_mpoint",
           "get_shapes",
           "set_attributes",
           "RGB_to_Lab",
           "RGB_identity_cube",
           "Lab_colourspace_cube",
           "Lab_coordinates_system_representation"]


def get_dag_path(node):
    """
    Returns a dag path from given node.

    :param node: Node name.
    :type node: str or unicode
    :return: MDagPath.
    :rtype: MDagPath
    """

    selection_list = OpenMaya.MSelectionList()
    selection_list.add(node)
    dag_path = OpenMaya.MDagPath()
    selection_list.getDagPath(0, dag_path)
    return dag_path


def get_mpoint(point):
    """
    Converts a tuple to MPoint.

    :param point: Point.
    :type point: tuple
    :return: MPoint.
    :rtype: MPoint
    """

    return OpenMaya.MPoint(point[0], point[1], point[2])


def get_shapes(object, full_path=False, no_intermediate=True):
    """
    Returns shapes of given object.

    :param object: Current object.
    :type object: str or unicode
    :param full_path: Current full path state.
    :type full_path: bool
    :param no_intermediate: Current no intermediate state.
    :type no_intermediate: bool
    :return: Objects shapes.
    :rtype: list
    """

    object_shapes = []
    shapes = cmds.listRelatives(object, fullPath=full_path, shapes=True, noIntermediate=no_intermediate)
    if shapes is not None:
        object_shapes = shapes

    return object_shapes


def set_attributes(attributes):
    """
    Sets given attributes.

    :param attributes: Attributes to set.
    :type attributes: dict
    :return: Definition success.
    :rtype: bool
    """

    for attribute, value in attributes.iteritems():
        cmds.setAttr(attribute, value)
    return True


def RGB_to_Lab(RGB, colourspace):
    """
    Converts given *RGB* value from given colourspace to *CIE Lab* colourspace.

    :param RGB: *RGB* value.
    :type RGB: array_like
    :param colourspace: *RGB* colourspace.
    :type colourspace: RGB_Colourspace
    :return: Definition success.
    :rtype: bool
    """

    return colour.computation.colourspaces.cie_lab.XYZ_to_Lab(
        colour.computation.colourspaces.cie_xyy.RGB_to_XYZ(numpy.array(RGB).reshape((3, 1)),
                                                           colourspace.whitepoint,
                                                           colour.dataset.illuminants.chromaticity_coordinates.ILLUMINANTS.get(
                                                               "CIE 1931 2 Degree Standard Observer").get(
                                                               "E"),
                                                           "Bradford",
                                                           colourspace.to_XYZ,
                                                           colourspace.inverse_transfer_function),
        colourspace.whitepoint)


def RGB_identity_cube(name, density=20):
    """
    Creates an RGB identity cube with given name and geometric density.

    :param name: Cube name.
    :type name: str or unicode
    :param density: Cube divisions count.
    :type density: int
    :return: Cube.
    :rtype: unicode
    """

    cube = cmds.polyCube(w=1, h=1, d=1, sx=density, sy=density, sz=density, ch=False)[0]
    set_attributes({"{0}.translateX".format(cube): .5,
                    "{0}.translateY".format(cube): .5,
                    "{0}.translateZ".format(cube): .5})
    cmds.setAttr("{0}.displayColors".format(cube), True)

    vertex_colour_array = OpenMaya.MColorArray()
    vertex_index_array = OpenMaya.MIntArray()
    point_array = OpenMaya.MPointArray()
    fn_mesh = OpenMaya.MFnMesh(get_dag_path(get_shapes(cube)[0]))
    fn_mesh.getPoints(point_array, OpenMaya.MSpace.kWorld)
    for i in range(point_array.length()):
        vertex_colour_array.append(point_array[i][0], point_array[i][1], point_array[i][2])
        vertex_index_array.append(i)
    fn_mesh.setVertexColors(vertex_colour_array, vertex_index_array, None)

    cmds.makeIdentity(cube, apply=True, t=True, r=True, s=True)
    cmds.xform(cube, a=True, rotatePivot=(0., 0., 0.), scalePivot=(0., 0., 0.))
    return cmds.rename(cube, name)


def Lab_colourspace_cube(colourspace, density=20):
    """
    Creates a *CIE Lab* colourspace cube with geometric density.

    :param colourspace: *RGB* Colourspace description.
    :type colourspace: RGB_Colourspace
    :param density: Cube divisions count.
    :type density: int
    :return: *CIE Lab* Colourspace cube.
    :rtype: unicode
    """

    cube = RGB_identity_cube(colourspace.name, density)
    it_mesh_vertex = OpenMaya.MItMeshVertex(get_dag_path(cube))
    while not it_mesh_vertex.isDone():
        position = it_mesh_vertex.position(OpenMaya.MSpace.kObject)
        it_mesh_vertex.setPosition(get_mpoint(list(numpy.ravel(RGB_to_Lab((position[0], position[1], position[2],),
                                                                          colourspace)))))
        it_mesh_vertex.next()
    set_attributes({"{0}.rotateX".format(cube): 180,
                    "{0}.rotateZ".format(cube): 90})
    cmds.makeIdentity(cube, apply=True, t=True, r=True, s=True)
    return cube


def Lab_coordinates_system_representation():
    """
    Creates a *CIE Lab* coordinates system representation.

    :return: Definition success.
    :rtype: bool
    """

    group = cmds.createNode("transform")

    cube = cmds.polyCube(w=600, h=100, d=600, sx=12, sy=2, sz=12, ch=False)[0]
    set_attributes({"{0}.translateY".format(cube): 50,
                    "{0}.overrideEnabled".format(cube): True,
                    "{0}.overrideDisplayType".format(cube): 2,
                    "{0}.overrideShading".format(cube): False})
    cmds.makeIdentity(cube, apply=True, t=True, r=True, s=True)
    cmds.select(["{0}.f[0:167]".format(cube), "{0}.f[336:359]".format(cube)])
    cmds.delete()

    cmds.nurbsToPolygonsPref(polyType=1, chordHeightRatio=0.975)

    for label, position, name in (("-a*", (-350, 0), "minus_a"),
                                  ("+a*", (350, 0), "plus_a"),
                                  ("-b*", (0, 350), "minus_b"),
                                  ("+b*", (0, -350), "plus_b")):
        curves = cmds.listRelatives(cmds.textCurves(f="Arial Black Bold", t=label)[0])
        mesh = cmds.polyUnite(*map(lambda x: cmds.planarSrf(x, ch=False, o=True, po=1), curves), ch=False)[0]
        cmds.xform(mesh, cp=True)
        cmds.xform(mesh, translation=(0., 0., 0.), absolute=True)
        cmds.makeIdentity(cube, apply=True, t=True, r=True, s=True)
        cmds.select(mesh)
        cmds.polyColorPerVertex(rgb=(0, 0, 0), cdo=True)
        set_attributes({"{0}.translateX".format(mesh): position[0],
                        "{0}.translateZ".format(mesh): position[1],
                        "{0}.rotateX".format(mesh): -90,
                        "{0}.scaleX".format(mesh): 50,
                        "{0}.scaleY".format(mesh): 50,
                        "{0}.scaleY".format(mesh): 50,
                        "{0}.overrideEnabled".format(mesh): True,
                        "{0}.overrideDisplayType".format(mesh): 2})
        cmds.delete(cmds.listRelatives(curves, parent=True))
        cmds.makeIdentity(mesh, apply=True, t=True, r=True, s=True)
        mesh = cmds.rename(mesh, name)
        cmds.parent(mesh, group)

    cube = cmds.rename(cube, "grid")
    cmds.parent(cube, group)
    cmds.rename(group, "Lab_coordinates_system_representation")

    return True
