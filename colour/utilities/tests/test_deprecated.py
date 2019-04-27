# -*- coding: utf-8 -*-

from __future__ import absolute_import

import sys
from colour.utilities.deprecation import ModuleAPI, Renamed, Removed


class deprecated(ModuleAPI):
    def __getattr__(self, attribute):
        return super(deprecated, self).__getattr__(attribute)


NAME = None
"""
An non-deprecated module attribute.

NAME : object
"""

NEW_NAME = None
"""
A module attribute with a new name.

NAME : object
"""

sys.modules['colour.utilities.tests.test_deprecated'] = (deprecated(
    sys.modules['colour.utilities.tests.test_deprecated'], {
        'OLD_NAME':
            Renamed(
                name='colour.utilities.tests.test_deprecated.OLD_NAME',
                new_name='colour.utilities.tests.test_deprecated.NEW_NAME'),
        'REMOVED':
            Removed(name='colour.utilities.tests.test_deprecated.REMOVED')
    }))

del ModuleAPI
del Renamed
del Removed
del sys
