# -*- coding: utf-8 -*-

name = 'colour'

version = '0.3.16.0.dev082501'

description = 'Colour Science for Python'

authors = ["Colour Developers"]

help = 'http://colour-science.org'

license = 'BSD-3-Clause'

requires = [
    '~python-2.7|3.5+',
    'imageio-2.0.0+',
    'six-1.10.0+',
    'scipy-0.16.0+',
    'networkx-2.2+<3',
    'pygraphviz-1.5+',
    'matplotlib-2+<4',
    'openimageio-1+<3',
    'numpy',
]

private_build_requires = [
    'setuptools',
    'pip',
]

variants = [
    ['python-2.7'],
    ['python-3.7'],
]


def pre_build_commands():
    env.PYTHONUSERBASE = '{build.install_path}'


build_command = "pip install {root} --no-deps --user --no-compile --global-option='build_ext'"


def commands():
    pyver = '.'.join(
        map(str, [resolve.python.version.major, resolve.python.version.minor]))
    env.PYTHONPATH.prepend('{root}/lib/python{pyver}/site-packages')


format_version = 2