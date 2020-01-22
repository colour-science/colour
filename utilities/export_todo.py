#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Export TODOs
============
"""

from __future__ import division, unicode_literals

import codecs
import os
from collections import OrderedDict

__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['TODO_FILE_TEMPLATE', 'extract_todo_items', 'export_todo_items']

TODO_FILE_TEMPLATE = """
Colour - TODO
=============

TODO
----

{0}

About
-----

| **Colour** by Colour Developers
| Copyright © 2013-2020 – Colour Developers – \
`colour-developers@colour-science.org <colour-developers@colour-science.org>`__
| This software is released under terms of New BSD License: \
https://opensource.org/licenses/BSD-3-Clause
| `https://github.com/colour-science/colour \
<https://github.com/colour-science/colour>`__
""" [1:]


def extract_todo_items(root_directory):
    """
    Extracts the TODO items from given directory.

    Parameters
    ----------
    root_directory : unicode
        Directory to extract the TODO items from.

    Returns
    -------
    OrderedDict
        TODO items.
    """

    todo_items = OrderedDict()
    for root, dirnames, filenames in os.walk(root_directory):
        for filename in filenames:
            if not filename.endswith('.py'):
                continue

            filename = os.path.join(root, filename)
            with codecs.open(filename, encoding='utf8') as file_handle:
                content = file_handle.readlines()

            in_todo = False
            line_number = -1
            todo_item = []
            for i, line in enumerate(content):
                line = line.strip()
                if line.startswith('# TODO:'):
                    in_todo = True
                    line_number = i
                    todo_item.append(line)
                    continue

                if in_todo and line.startswith('#'):
                    todo_item.append(line.replace('#', '').strip())
                elif len(todo_item):
                    key = filename.replace('../', '')
                    if not todo_items.get(key):
                        todo_items[key] = []

                    todo_items[key].append((line_number, ' '.join(todo_item)))
                    in_todo = False
                    line_number
                    todo_item = []

    return todo_items


def export_todo_items(todo_items, file_path):
    """
    Exports TODO items to given file.

    Parameters
    ----------
    todo_items : OrderedDict
        TODO items.
    file_path : unicode
        File to write the TODO items to.
    """

    todo_rst = []
    for module, todo_items in todo_items.items():
        todo_rst.append('-   {0}\n'.format(module))
        for line_numer, todo_item in todo_items:
            todo_rst.append('    -   Line {0} : {1}'.format(
                line_numer, todo_item))

        todo_rst.append('\n')

    with codecs.open(file_path, 'w', encoding='utf8') as todo_file:
        todo_file.write(TODO_FILE_TEMPLATE.format('\n'.join(todo_rst[:-1])))


if __name__ == '__main__':
    export_todo_items(
        extract_todo_items(os.path.join('..', 'colour')),
        os.path.join('..', 'TODO.rst'))
