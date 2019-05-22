# -*- coding: utf-8 -*-
# __author__ = 'qinjincheng'

import gevent
from gevent.event import AsyncResult
import sys

a = AsyncResult()

def setter():
    print 'Here is {}'.format(sys._getframe().f_code.co_name)
    gevent.sleep(1)
    a.set('Hello, waiter!')
    gevent.sleep(1)
    a.set('Hello, again!')

def waiter1():
    print 'Here is {}'.format(sys._getframe().f_code.co_name)
    print '{} in {}'.format(a.get(), sys._getframe().f_code.co_name)
    gevent.sleep(3)
    print '{} in {}'.format(a.get(), sys._getframe().f_code.co_name)

def waiter2():
    print 'Here is {}'.format(sys._getframe().f_code.co_name)
    print '{} in {}'.format(a.get(), sys._getframe().f_code.co_name)

def main():
    gevent.joinall([
        gevent.spawn(setter),
        gevent.spawn(waiter1),
        gevent.spawn(waiter2),
    ])

if __name__ == '__main__':
    main()
