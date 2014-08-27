#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pypi Setup
==========
"""

from __future__ import unicode_literals

import sys

from setuptools import setup
from setuptools import find_packages

import colour

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['SHORT_DESCRIPTION',
           'LONG_DESCRIPTION',
           'INSTALLATION_REQUIREMENTS',
           'TESTS_REQUIREMENTS',
           'DOCS_REQUIREMENTS']

SHORT_DESCRIPTION = 'Colour Science for Python'
LONG_DESCRIPTION = ('Colour is a Python colour science package implementing a '
                    'comprehensive number of colour theory transformations '
                    'and algorithms.')

INSTALLATION_REQUIREMENTS = [
    'matplotlib>=1.3.1',
    'numpy>=1.8.2',
    'scipy>=0.14.0']

if sys.version_info[:2] <= (2, 7):
    INSTALLATION_REQUIREMENTS += [
        'backports.functools_lru_cache>=1.0.1']

if sys.version_info[:2] <= (2, 6):
    INSTALLATION_REQUIREMENTS += [
        'ordereddict>=1.1',
        'unittest2>=0.5.1']

TESTS_REQUIREMENTS = (
    'nose>=1.3.4',)

DOCS_REQUIREMENTS = (
    'sphinx>=1.2.2',)

setup(name='colour-science',
      version=colour.__version__,
      author=colour.__author__,
      author_email=colour.__email__,
      include_package_data=True,
      packages=find_packages(),
      scripts=[],
      url='http://github.com/colour-science/colour',
      license='',
      description=SHORT_DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      install_requires=INSTALLATION_REQUIREMENTS,
      extras_require={
          'tests': TESTS_REQUIREMENTS,
          'docs': DOCS_REQUIREMENTS},
      classifiers=['Development Status :: 4 - Beta',
                   'Environment :: Console',
                   'Intended Audience :: Developers',
                   'Natural Language :: English',
                   'Operating System :: OS Independent',
                   'Programming Language :: Python :: 2.6',
                   'Programming Language :: Python :: 2.7',
                   'Programming Language :: Python :: 3.4',
                   'Topic :: Utilities'])
