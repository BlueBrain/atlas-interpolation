# SPDX-License-Identifier: MIT
# Source: https://github.com/microsoft/MaskFlownet/tree/5cba12772e2201f0d1c1e27161d224e585334571
class Reader:
    def __init__(self, obj, full_attr=""):
        self._object = obj
        self._full_attr = full_attr
    
    def __getattr__(self, name):
        if self._object is None:
            ret = None
        else:
            ret = self._object.get(name, None)
        return Reader(ret, self._full_attr + '.' + name)

    def get(self, default=None):
        if self._object is None:
            print('Default FLAGS.{} to {}'.format(self._full_attr, default))
            return default
        else:
            return self._object

    @property
    def value(self):
        return self._object