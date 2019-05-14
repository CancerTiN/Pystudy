# -*- coding: utf-8 -*-
# __author__ = 'qinjincheng'

import gevent
from gevent.event import Event
import sys
import time

evt = Event()

def setter():
    print 'Here is {}'.format(sys._getframe().f_code.co_name)
    print 'Hey, wait for me, I have to do something, {}'.format(time.time())
    gevent.sleep(3)
    print 'Ok, I am done, {}'.format(time.time())
    evt.set()

def waiter():
    print 'Here is {}'.format(sys._getframe().f_code.co_name)
    print 'I will wait for you, {}'.format(time.time())
    evt.wait()
    print 'It is about time, {}'.format(time.time())

def main():
    gevent.joinall([
        gevent.spawn(setter),
        gevent.spawn(waiter),
        gevent.spawn(waiter),
    ])

if __name__ == '__main__':
    main()
