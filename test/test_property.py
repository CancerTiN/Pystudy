# -*- coding: utf-8 -*-
# __author__ = 'qinjincheng'

class TestClass(object):
    def __init__(self):
        self._a = 'a'
        self._b = 'b'

    @property
    def a(self):
        return self._a

    def b(self):
        return self._b

def main():
    inst = TestClass()
    print inst.a
    print inst.b
    print inst.b()
    try:
        inst.a = 'c'
    except Exception, e:
        print '{} when set value to property-decorated attribute of {}'.format(e, inst)
    try:
        inst.b = 'd'
    except Exception, e:
        print e

if __name__ == '__main__':
    main()
