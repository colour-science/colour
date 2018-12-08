from __future__ import division, unicode_literals

import numpy as np
import xml.etree.ElementTree as ET
import re

from colour.utilities import as_float_array, as_numeric
from colour.constants import DEFAULT_FLOAT_DTYPE
from colour.io.luts import (AbstractLUTSequenceOperator,
                            ASC_CDL,
                            LUT1D,
                            LUT2D,
                            LUT3D,
                            LUTSequence,
                            Matrix,
                            Range)

__all__ = ['half_to_uint16', 'uint16_to_half', 'halfDomain', 'read_IndexMap',
           'string_to_array', 'simple_clf_parse']

def half_to_uint16(x):
    x_half = np.copy(np.asarray(x)).astype(np.float16)
    return np.frombuffer(x_half.tobytes(),
                         dtype=np.uint16).reshape(x_half.shape)

def uint16_to_half(y):
    y_int = np.copy(np.asarray(y)).astype(np.uint16)
    return as_numeric(np.frombuffer(y_int.tobytes(),
        dtype=np.float16).reshape(y_int.shape).astype(DEFAULT_FLOAT_DTYPE))

class halfDomain(AbstractLUTSequenceOperator):
    def __init__(self,
                 table=None,
                 name=None,
                 comments=None,
                 rawHalfs=False):
        self.table = self.linear_table(rawHalfs=rawHalfs)
        self.name = name
        self.coments = comments
        self.rawHalfs = rawHalfs

    def _validate_table(self, table):
        table = as_float_array(table)

        assert table.shape == (65536,), 'The table must be a 1 by 65536 array!'

        return table

    @staticmethod
    def linear_table(rawHalfs=False):
        if rawHalfs:
            return np.arange(65536, dtype='uint16')
        else:
            return uint16_to_half(np.arange(65536, dtype='uint16'))

    def apply(self, RGB):
        if self.rawHalfs:
            return uint16_to_half(self.table[half_to_uint16(RGB)])
        else:
            return self.table[half_to_uint16(RGB)]

def read_IndexMap(IndexMap):
    IndexMap = IndexMap.strip().split()
    IndexMap = np.core.defchararray.partition(IndexMap, '@')
    domain, at, table = np.hsplit(IndexMap, 3)
    domain = domain.reshape(-1).astype(np.float)
    table = table.reshape(-1).astype(np.float)
    return LUT1D(table=table, domain=domain)

def string_to_array(string, dim=None):
    assert not dim is None, 'dim must be set to dimensions of array!'
    return np.fromstring(string, count=dim[0]*dim[1], sep=' ').reshape(dim)

def ns_strip(s):
    return re.sub('{.+}', '', s)

def add_LUT1D(LUT, node):
    shaper = None
    for child in node:
        if child.tag.endswith('IndexMap'):
            shaper = read_IndexMap(child.text)
        if child.tag.endswith('Array'):
            dim = child.attrib['dim']
            r = int(dim.split()[0])
            c = int(dim.split()[1])
            array = string_to_array(child.text, dim=(r, c))
            if c == 1:
                LUT_1 = LUT1D(table=array.reshape(-1))
            else:
                LUT_1 = LUT2D(table=array)
    if shaper:
        if np.all(shaper.table == np.arange(shaper.size)): #fully enumerated
            LUT_1.domain = shaper.domain
        else:
            LUT.append(shaper)
    LUT.append(LUT_1)
    return LUT

def add_LUT3D(LUT, node):
    shaper = None
    for child in node:
        if child.tag.endswith('IndexMap'):
            shaper = read_IndexMap(child.text)
            shaper.table /= shaper.domain[-1]
        if child.tag.endswith('Array'):
            dim = child.attrib['dim']
            size_r = int(dim.split()[0])
            size_g = int(dim.split()[1])
            size_b = int(dim.split()[2])
            length = size_r * size_g * size_b
            chans = int(dim.split()[3])
            array = string_to_array(child.text, dim=(length, chans))
            array = array.reshape([size_r, size_g, size_b, 3], order='C')
            LUT_1 = LUT3D(table=array)
    if shaper:
        LUT.append(shaper)
    LUT.append(LUT_1)
    return LUT

def add_Range(LUT, node):
    range = Range()
    for child in node:
        if child.tag.endswith('minInValue'):
            range.minInValue = float(child.text)
        if child.tag.endswith('maxInValue'):
            range.maxInValue = float(child.text)
        if child.tag.endswith('minOutValue'):
            range.minOutValue = float(child.text)
        if child.tag.endswith('maxOutValue'):
            range.maxOutValue = float(child.text)
    LUT.append(range)
    return LUT

def add_Matrix(LUT, node):
    mat = Matrix()
    for child in node:
        if child.tag.endswith('Array'):
            dim = child.attrib['dim']
            r = int(dim.split()[0])
            c = int(dim.split()[1])
            m = string_to_array(child.text, dim=(r, c))
            mat.array = m
    LUT.append(mat)
    return LUT

def add_ASC_CDL(LUT, node):
    cdl = ASC_CDL()
    for child in node:
        if child.tag.endswith('SOPNode'):
            for grandchild in child:
                if grandchild.tag.endswith('Slope'):
                    cdl.slope = string_to_array(grandchild.text, (3,))
                if grandchild.tag.endswith('Offset'):
                    cdl.offset = string_to_array(grandchild.text, (3,))
                if grandchild.tag.endswith('Power'):
                    cdl.power = string_to_array(grandchild.text, (3,))
        if child.tag.endswith('SATNode'):
            for grandchild in child:
                if grandchild.tag.endswith('Saturation'):
                    cdl.sat = float(grandchild.text)
    LUT.append(cdl)
    return LUT

def simple_clf_parse(path):
    LUT = LUTSequence()
    tree = ET.parse(path)
    pl = tree.getroot()
    #if 'name' in pl.keys():
        #LUT.name = pl.attrib['name']
    for node in pl:
        #if node.tag.endswith('Description'):
            #LUT.comments.append(node.text)
        #if node.tag.endswith('InputDescriptor'):
            #LUT.InputDescriptor = node.text
        #if node.tag.endswith('OutputDescriptor'):
            #LUT.OutputDescriptor = node.text
        if node.tag.endswith('LUT1D'):
            LUT = add_LUT1D(LUT, node)
        if node.tag.endswith('LUT3D'):
            LUT = add_LUT3D(LUT, node)
        if node.tag.endswith('Range'):
            LUT = add_Range(LUT, node)
        if node.tag.endswith('Matrix'):
            LUT = add_Matrix(LUT, node)
        if node.tag.endswith('ASC_CDL'):
            LUT = add_ASC_CDL(LUT, node)
    return LUT
