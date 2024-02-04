#!/usr/bin/env python
"""
Export TODOs
============
"""

from __future__ import annotations

import codecs
import os

__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TODO_FILE_TEMPLATE",
    "extract_todo_items",
    "export_todo_items",
]

TODO_FILE_TEMPLATE = """
Colour - TODO
=============

TODO
----

{0}

About
-----

| **Colour** by Colour Developers
| Copyright 2013 Colour Developers - \
`colour-developers@colour-science.org <colour-developers@colour-science.org>`__
| This software is released under terms of BSD-3-Clause: \
https://opensource.org/licenses/BSD-3-Clause
| `https://github.com/colour-science/colour \
<https://github.com/colour-science/colour>`__
"""[1:]


def extract_todo_items(root_directory: str) -> dict:
    """
    Extract the TODO items from given directory.

    Parameters
    ----------
    root_directory
        Directory to extract the TODO items from.

    Returns
    -------
    :class:`dict`
        TODO items.
    """

    todo_items = {}
    for root, _dirnames, filenames in os.walk(root_directory):
        for filename in filenames:
            if not filename.endswith(".py"):
                continue

            filename = os.path.join(root, filename)  # noqa: PLW2901
            with codecs.open(filename, encoding="utf8") as file_handle:
                content = file_handle.readlines()

            in_todo = False
            line_number = -1
            todo_item = []
            for i, line in enumerate(content):
                line = line.strip()  # noqa: PLW2901
                if line.startswith("# TODO:"):
                    in_todo = True
                    line_number = i + 1
                    todo_item.append(line)
                    continue

                if in_todo and line.startswith("#"):
                    todo_item.append(line.replace("#", "").strip())
                elif len(todo_item):
                    key = filename.replace("../", "")
                    if not todo_items.get(key):
                        todo_items[key] = []

                    todo_items[key].append((line_number, " ".join(todo_item)))
                    in_todo = False
                    todo_item = []

    return todo_items


def export_todo_items(todo_items: dict, file_path: str):
    """
    Export TODO items to given file.

    Parameters
    ----------
    todo_items
        TODO items.
    file_path
        File to write the TODO items to.
    """

    todo_rst = []
    for module, module_todo_items in todo_items.items():
        todo_rst.append(f"-   {module}\n")
        for line_numer, todo_item in module_todo_items:
            todo_rst.append(f"    -   Line {line_numer} : {todo_item}")

        todo_rst.append("\n")

    with codecs.open(file_path, "w", encoding="utf8") as todo_file:
        todo_file.write(TODO_FILE_TEMPLATE.format("\n".join(todo_rst[:-1])))


if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))

    export_todo_items(
        extract_todo_items(os.path.join("..", "colour")),
        os.path.join("..", "TODO.rst"),
    )
