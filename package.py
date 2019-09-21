# -*- coding: utf-8 -*-

name = u'colour'

version = '0.3.14.z0'

description = u'Colour Science for Python'

requires = [
    'imageio-2.0.0+',
    'six-1.10.0+',
    'scipy-0.16.0+',
    'networkx',
]

variants = [
    ['platform-osx', 'arch-x86_64', 'os-osx-10.14.5', 'python-2.7'],
]

def commands():
    env.PYTHONPATH.append('{root}/python')

timestamp = 1562044253

format_version = 2

hashed_variants = True

is_pure_python = True

#from_pip = True

pip_name = u'colour-science (0.3.14)'
