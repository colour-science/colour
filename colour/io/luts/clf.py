from __future__ import division, unicode_literals

import numpy as np
import xml.etree.ElementTree as ET
import re

from colour.utilities import as_float_array, as_numeric, tsplit, tstack
from colour.constants import DEFAULT_FLOAT_DTYPE
from colour.io.luts import (AbstractLUTSequenceOperator,
                            ASC_CDL,
                            LUT1D,
                            LUT2D,
                            LUT3D,
                            LUTSequence,
                            Matrix,
                            Range)

__all__ = ['half_to_uint16', 'uint16_to_half', 'halfDomain_lookup',
           'halfDomain1D', 'halfDomain2D', 'read_IndexMap', 'string_to_array',
           'simple_clf_parse']

def half_to_uint16(x):
    x_half = np.copy(np.asarray(x)).astype(np.float16)
    return np.frombuffer(x_half.tobytes(),
                         dtype=np.uint16).reshape(x_half.shape)

def uint16_to_half(y):
    y_int = np.copy(np.asarray(y)).astype(np.uint16)
    return as_numeric(np.frombuffer(y_int.tobytes(),
        dtype=np.float16).reshape(y_int.shape).astype(DEFAULT_FLOAT_DTYPE))

def halfDomain_lookup(x, LUT, rawHalfs=True):
    h0 = half_to_uint16(x)
    f0 = uint16_to_half(h0)
    f1 = uint16_to_half(h0 - 1)
    f2 = uint16_to_half(h0 + 1)
    h1 = np.where((x - f0) * (x - f1) > 0.0, h0 - 1, h0 + 1)
    f = (x - uint16_to_half(h1)) / (f0 - uint16_to_half(h1))
    out0 = LUT[h0]
    out1 = LUT[h1]
    if rawHalfs:
        out0 = uint16_to_half(out0)
        out1 = uint16_to_half(out1)
    return np.where(np.isinf(f0), out0, f * out0 + (1 - f) * out1)

class halfDomain1D(AbstractLUTSequenceOperator):
    def __init__(self,
                 table=None,
                 name=None,
                 comments=None,
                 rawHalfs=False):
        if table is not None:
            self.table = table
        else:
            self.table = self.linear_table(rawHalfs=rawHalfs)
        self.name = name
        self.coments = comments
        self.rawHalfs = rawHalfs

    def _validate_table(self, table):
        table = np.asarray(table)

        assert table.shape == (65536,), 'The table must be 65536 lines!'

        return table

    @staticmethod
    def linear_table(rawHalfs=False):
        if rawHalfs:
            return np.arange(65536, dtype='uint16')
        else:
            return uint16_to_half(np.arange(65536, dtype='uint16'))

    def apply(self, RGB):
        return halfDomain_lookup(RGB, self.table, rawHalfs=self.rawHalfs)

class halfDomain2D(AbstractLUTSequenceOperator):
    def __init__(self,
                 table=None,
                 name=None,
                 comments=None,
                 rawHalfs=False):
        if table is not None:
            self.table = table
        else:
            self.table = self.linear_table(rawHalfs=rawHalfs)
        self.name = name
        self.coments = comments
        self.rawHalfs = rawHalfs

    def _validate_table(self, table):
        table = np.asarray(table)

        assert table.shape == (65536, 3), 'The table must be 65536 x 3!'

        return table

    @staticmethod
    def linear_table(rawHalfs=False):
        if rawHalfs:
            c = np.arange(65536, dtype='uint16')
        else:
            c = uint16_to_half(np.arange(65536, dtype='uint16'))
        return tstack((c, c, c)).astype(np.uint16)

    def apply(self, RGB):
        r, g, b = tsplit(as_float_array(RGB))
        table_r, table_g, table_b = tsplit(self.table)
        r = halfDomain_lookup(r, table_r, rawHalfs=self.rawHalfs)
        g = halfDomain_lookup(g, table_g, rawHalfs=self.rawHalfs)
        b = halfDomain_lookup(b, table_b, rawHalfs=self.rawHalfs)
        return tstack((r, g, b))

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

def parse_array(array):
    return np.array(array.strip().split()).astype(DEFAULT_FLOAT_DTYPE)

def ns_strip(s):
    return re.sub('{.+}', '', s)

def add_LUT1D(LUT, node):
    shaper, is_halfDomain, is_rawHalfs = None, None, None
    if 'halfDomain' in node.keys():
        is_halfDomain = bool(node.attrib['halfDomain'])
        if 'rawHalfs' in node.keys():
            is_rawHalfs = bool(node.attrib['rawHalfs'])
    for child in node:
        if child.tag.lower().endswith('indexmap'):
            shaper = read_IndexMap(child.text)
        if child.tag.lower().endswith('array'):
            dim = child.attrib['dim']
            r = int(dim.split()[0])
            c = int(dim.split()[1])
            array = string_to_array(child.text, dim=(r, c))
            if c == 1:
                if is_halfDomain:
                    LUT_1 = halfDomain1D(table=array.reshape(-1),
                                         rawHalfs=is_rawHalfs)
                else:
                    LUT_1 = LUT1D(table=array.reshape(-1))
            else:
                if is_halfDomain:
                    LUT_1 = halfDomain2D(table=array, rawHalfs=is_rawHalfs)
                else:
                    LUT_1 = LUT2D(table=array)
    if shaper:
        if np.all(shaper.table == np.arange(shaper.size)): #fully enumerated
            LUT_1.domain = shaper.domain
        else:
            shaper.table /= np.max(shaper.table)
            LUT.append(shaper)
    LUT.append(LUT_1)
    return LUT

def add_LUT3D(LUT, node):
    shaper = None
    for child in node:
        if child.tag.lower().endswith('indexmap'):
            shaper = read_IndexMap(child.text)
            shaper.table /= np.max(shaper.table)
        if child.tag.lower().endswith('array'):
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
    if 'style' in node.keys():
        if node.attrib['style'].lower() == 'noclamp':
            range.noClamp = True
        if node.attrib['style'].lower() == 'clamp':
            range.noClamp = False
    for child in node:
        if child.tag.lower().endswith('mininvalue'):
            range.minInValue = float(child.text)
        if child.tag.lower().endswith('maxinvalue'):
            range.maxInValue = float(child.text)
        if child.tag.lower().endswith('minoutvalue'):
            range.minOutValue = float(child.text)
        if child.tag.lower().endswith('maxoutvalue'):
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
    if 'style' in node.keys():
        if node.attrib['style'].lower() == 'fwdnoclamp':
            cdl.clamp, cdl.rev = False, False
        if node.attrib['style'].lower() == 'revnoclamp':
            cdl.clamp, cdl.rev = False, True
        if node.attrib['style'].lower() == 'fwd':
            cdl.clamp, cdl.rev = True, False
        if node.attrib['style'].lower() == 'rev':
            cdl.clamp, cdl.rev = True, True
    for child in node:
        if child.tag.lower().endswith('sopnode'):
            for grandchild in child:
                if grandchild.tag.lower().endswith('slope'):
                    cdl.slope = parse_array(grandchild.text)
                if grandchild.tag.lower().endswith('offset'):
                    cdl.offset = parse_array(grandchild.text)
                if grandchild.tag.lower().endswith('power'):
                    cdl.power = parse_array(grandchild.text)
        if child.tag.lower().endswith('satnode'):
            for grandchild in child:
                if grandchild.tag.lower().endswith('saturation'):
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
        if node.tag.lower().endswith('lut1d'):
            LUT = add_LUT1D(LUT, node)
        if node.tag.lower().endswith('lut3d'):
            LUT = add_LUT3D(LUT, node)
        if node.tag.lower().endswith('range'):
            LUT = add_Range(LUT, node)
        if node.tag.lower().endswith('matrix'):
            LUT = add_Matrix(LUT, node)
        if node.tag.lower().endswith('asc_cdl'):
            LUT = add_ASC_CDL(LUT, node)
    return LUT
