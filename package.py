# -*- coding: utf-8 -*-

name = u'colour'

version = '0.3.14.z1'

description = u'Colour Science for Python'

requires = [
    'imageio-2.0.0+',
    'six-1.10.0+',
    'scipy-0.16.0+',
    'networkx',
    'pygraphviz',
]

variants = [
    ['platform-osx', 'arch-x86_64', 'os-osx-10.14.5', 'python-2.7'],
]

def commands():
    env.PYTHONPATH.append('{root}')

timestamp = 1562044253

format_version = 2

hashed_variants = True

