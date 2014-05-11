#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**plots.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Color** package **Autodesk Maya** plotting objects.

**Others:**

"""

import maya.cmds as cmds
import maya.mel as mel
import maya.OpenMaya as OpenMaya
import numpy

import color.colorspaces
import color.illuminants
import color.transformations
import color.data_structures
import color.verbose
import foundations.common

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["LOGGER",
           "getDagPath",
           "getMPoint",
           "getShapes",
           "setAttributes",
           "RGB_to_Lab",
           "RGB_identityCube",
           "Lab_colorspaceCube",
           "Lab_coordinatesSystemRepresentation"]

LOGGER = color.verbose.install_logger()


def getDagPath(node):
    """
    Returns a dag path from given node.

    :param node: Node name.
    :type node: str or unicode
    :return: MDagPath.
    :rtype: MDagPath
    """

    selectionList = OpenMaya.MSelectionList()
    selectionList.add(node)
    dagPath = OpenMaya.MDagPath()
    selectionList.getDagPath(0, dagPath)
    return dagPath


def getMPoint(point):
    """
    Converts a tuple to MPoint.

    :param point: Point.
    :type point: tuple
    :return: MPoint.
    :rtype: MPoint
    """

    return OpenMaya.MPoint(point[0], point[1], point[2])


def getShapes(object, fullPath=False, noIntermediate=True):
    """
    Returns shapes of given object.

    :param object: Current object.
    :type object: str or unicode
    :param fullPath: Current full path state.
    :type fullPath: bool
    :param noIntermediate: Current no intermediate state.
    :type noIntermediate: bool
    :return: Objects shapes.
    :rtype: list
    """

    objectShapes = []
    shapes = cmds.listRelatives(object, fullPath=fullPath, shapes=True, noIntermediate=noIntermediate)
    if shapes != None:
        objectShapes = shapes

    return objectShapes


def setAttributes(attributes):
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


def RGB_to_Lab(RGB, colorspace):
    return color.transformations.XYZ_to_Lab(color.transformations.RGB_to_XYZ(numpy.matrix(RGB).reshape((3, 1)),
                                                                             colorspace.whitepoint,
                                                                             color.illuminants.ILLUMINANTS.get(
                                                                                 "Standard CIE 1931 2 Degree Observer").get(
                                                                                 "E"),
                                                                             "Bradford",
                                                                             colorspace.to_XYZ,
                                                                             colorspace.inverse_transfer_function),
                                            colorspace.whitepoint)


def RGB_identityCube(name, density=20):
    """
    Creates an RGB identity cube with given name and geometric density.

    :param name: Cube name.
    :type name: str or unicode
    :param density: Cube divisions count.
    :type density: int
    :return: Cube.
    :rtype: unicode
    """

    cube = foundations.common.get_first_item(
        cmds.polyCube(w=1, h=1, d=1, sx=density, sy=density, sz=density, ch=False))
    setAttributes({"{0}.translateX".format(cube): .5,
                   "{0}.translateY".format(cube): .5,
                   "{0}.translateZ".format(cube): .5})
    cmds.setAttr("{0}.displayColors".format(cube), True)

    vertexColorArray = OpenMaya.MColorArray()
    vertexIndexArray = OpenMaya.MIntArray()
    pointArray = OpenMaya.MPointArray()
    fnMesh = OpenMaya.MFnMesh(getDagPath(foundations.common.get_first_item(getShapes(cube))))
    fnMesh.getPoints(pointArray, OpenMaya.MSpace.kWorld)
    for i in range(pointArray.length()):
        vertexColorArray.append(pointArray[i][0], pointArray[i][1], pointArray[i][2])
        vertexIndexArray.append(i)
    fnMesh.setVertexColors(vertexColorArray, vertexIndexArray, None)

    cmds.makeIdentity(cube, apply=True, t=True, r=True, s=True)
    cmds.xform(cube, a=True, rotatePivot=(0., 0., 0.), scalePivot=(0., 0., 0.))
    return cmds.rename(cube, name)


def Lab_colorspaceCube(colorspace, density=20):
    """
    Creates a **CIE Lab** colorspace cube with geometric density.

    :param colorspace: Colorspace description.
    :type colorspace: Colorspace
    :param density: Cube divisions count.
    :type density: int
    :return: Colorspace cube.
    :rtype: unicode
    """

    cube = RGB_identityCube(colorspace.name, density)
    itMeshVertex = OpenMaya.MItMeshVertex(getDagPath(cube))
    while not itMeshVertex.isDone():
        position = itMeshVertex.position(OpenMaya.MSpace.kObject)
        itMeshVertex.setPosition(getMPoint(list(numpy.ravel(RGB_to_Lab((position[0], position[1], position[2],),
                                                                       colorspace)))))
        itMeshVertex.next()
    setAttributes({"{0}.rotateX".format(cube): 180,
                   "{0}.rotateZ".format(cube): 90})
    cmds.makeIdentity(cube, apply=True, t=True, r=True, s=True)
    return cube


def Lab_coordinatesSystemRepresentation():
    """
    Creates a **CIE Lab** coordinates system representation.

    :return: Definition success.
    :rtype: bool
    """

    group = cmds.createNode("transform")

    cube = foundations.common.get_first_item(cmds.polyCube(w=600, h=100, d=600, sx=12, sy=2, sz=12, ch=False))
    setAttributes({"{0}.translateY".format(cube): 50,
                   "{0}.overrideEnabled".format(cube): True,
                   "{0}.overrideDisplayType".format(cube): 2,
                   "{0}.overrideShading".format(cube): False})
    cmds.makeIdentity(cube, apply=True, t=True, r=True, s=True)
    cmds.select(["{0}.f[0:167]".format(cube), "{0}.f[336:359]".format(cube)])
    mel.eval("doDelete;")

    cmds.nurbsToPolygonsPref(polyType=1, chordHeightRatio=0.975)

    for label, position, name in (("-a*", (-350, 0), "minus_a"),
                                  ("+a*", (350, 0), "plus_a"),
                                  ("-b*", (0, 350), "minus_b"),
                                  ("+b*", (0, -350), "plus_b")):
        curves = cmds.listRelatives(foundations.common.get_first_item(cmds.textCurves(f="Arial Black Bold", t=label)))
        mesh = foundations.common.get_first_item(
            cmds.polyUnite(*map(lambda x: cmds.planarSrf(x, ch=False, o=True, po=1), curves), ch=False))
        cmds.xform(mesh, cp=True)
        cmds.xform(mesh, translation=(0., 0., 0.), absolute=True)
        cmds.makeIdentity(cube, apply=True, t=True, r=True, s=True)
        cmds.select(mesh)
        cmds.polyColorPerVertex(rgb=(0, 0, 0), cdo=True)
        setAttributes({"{0}.translateX".format(mesh): position[0],
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
    cmds.rename(group, "Lab_coordinatesSystemRepresentation")

    return True
