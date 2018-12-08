# -*- coding: utf-8 -*-
"""
ASC CDL Input Utilities
=======================

Defines *ASC CDL* *AbstractLUTSequenceOperator* related input utilities
objects.

-   :func:`colour.io.read_LUT_cdl_xml`
-   :func:`colour.io.read_LUT_cdl_edl`
-   :func:`colour.io.read_LUT_cdl_ale`
"""

from __future__ import division, unicode_literals

import numpy as np
import os
import re

from colour.constants import DEFAULT_FLOAT_DTYPE, DEFAULT_INT_DTYPE
from colour.io.luts import LUTSequence, ASC_CDL
from colour.utilities import as_float_array, warning
from xml.dom import minidom

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2018 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['read_LUT_cdl_xml', 'read_LUT_cdl_edl', 'read_LUT_cdl_ale']


def read_LUT_cdl_xml(path):
    def _parse_array(array):
        return np.array(list(map(DEFAULT_FLOAT_DTYPE, array.split())))
    title = re.sub('_|-|\\.', ' ', os.path.splitext(os.path.basename(path))[0])
    data = minidom.parse(path)
    LUT = LUTSequence()
    corrections = data.getElementsByTagName('ColorCorrection')
    for idx, correction in enumerate(corrections):
        event = ASC_CDL()
        slope = correction.getElementsByTagName('Slope')
        slope = '1 1 1' if not slope else slope[0].firstChild.data
        offset = correction.getElementsByTagName('Offset')
        offset = '0 0 0' if not offset else offset[0].firstChild.data
        power = correction.getElementsByTagName('Power')
        power = '1 1 1' if not power else power[0].firstChild.data
        sat = correction.getElementsByTagName('Saturation')
        sat = '1' if not sat else sat[0].firstChild.data
        if 'id' in correction.attributes.keys():
            event.id = correction.attributes['id'].value
        event.slope = _parse_array(slope)
        event.offset = _parse_array(offset)
        event.power = _parse_array(power)
        event.sat = _parse_array(sat)
        event.name = '{0} ({1})'.format(title, idx + 1)
        LUT.append(event)
    if len(LUT) == 1:
        LUT[0].name = title
        return LUT[0]
    else:
        return LUT


def read_LUT_cdl_edl(path):
    with open(path) as edl_file:
        edl_lines = edl_file.readlines()
    if 'TITLE' in edl_lines[0]:
        title = edl_lines[0].split()[1]
    else:
        title = re.sub('_|-|\\.', ' ',
                       os.path.splitext(os.path.basename(path))[0])
    event_cdl = None
    has_cdl = False
    LUT = LUTSequence()
    for line in edl_lines:
        if len(line.split()) == 0:
            continue
        if line.split()[0].isdigit():
            if has_cdl:
                LUT.append(event_cdl)
                event_cdl = None
                has_cdl = False
            event_number = int(line.split()[0])
            event_cdl = ASC_CDL(name='{0} EV{1:04d}'.format(title,
                                                            event_number))
            event_cdl.comments = []
            continue
        if event_cdl:
            if line[0] == '*':
                trimmed = line[1:].lstrip()
                if trimmed.startswith('ASC_SOP'):
                    sop = re.sub('\)\s*\(|\s*\(|\s*\)', ' ', trimmed).split()
                    event_cdl.slope = np.array(sop[1:4]).astype(np.float)
                    event_cdl.offset = np.array(sop[4:7]).astype(np.float)
                    event_cdl.power = np.array(sop[7:]).astype(np.float)
                    has_cdl = True
                elif trimmed.startswith('ASC_SAT'):
                    event_cdl.sat = float(trimmed.split()[1])
                    has_cdl = True
                else:
                    event_cdl.comments.append(trimmed)
    if event_cdl:
        LUT.append(event_cdl)
    return LUT

def read_LUT_cdl_ale(path):
    with open(path, 'rU') as ale_file:
        ale_lines = ale_file.readlines()
    title = re.sub('_|-|\\.', ' ',
                       os.path.splitext(os.path.basename(path))[0])
    event_cdl = None
    LUT = LUTSequence()
    try:
        header_line = ale_lines.index('Column\n') + 1
    except:
        raise ValueError('ALE format error')
    headers = ale_lines[header_line].split('\t')
    try:
        sop_index = headers.index('ASC_SOP')
        sat_index = headers.index('ASC_SAT')
        name_index = headers.index('Name')
    except:
        raise ValueError('No ASC CDL data')
    try:
        first_data = ale_lines.index('Data\n') + 1
    except:
        raise ValueError('ALE format error')
    for line in ale_lines[first_data:]:
        line_data = line.split('\t')
        sop = re.sub('\)\s*\(|\s*\(|\s*\)', ' ', line_data[sop_index]).split()
        sat = line_data[sat_index]
        name = line_data[name_index]
        event_cdl = ASC_CDL(name=name)
        event_cdl.slope = np.array(sop[0:3]).astype(np.float)
        event_cdl.offset = np.array(sop[3:6]).astype(np.float)
        event_cdl.power = np.array(sop[6:]).astype(np.float)
        event_cdl.sat = float(sat)
        LUT.append(event_cdl)
    return LUT
