# -*- coding: utf-8 -*-
# __author__ = 'qinjincheng'

import gevent
import urllib2
import sys
from gevent.lock import Semaphore
import time

situation = 3

def func0(url):
    print 'GET: {}'.format(url)
    resp = urllib2.urlopen(url)
    data = resp.read()
    print '{} bytes received from {}'.format(len(data), url)

def func1():
    for i in range(5):
        print 'this is {} in {}'.format(i, sys._getframe().f_code.co_name)
        gevent.sleep(4)

def func2():
    for i in range(5):
        print 'this is {} in {}'.format(i, sys._getframe().f_code.co_name)
        gevent.sleep(1)

def func3():
    for i in range(5):
        sem.acquire()
        print 'this is {} in {}'.format(i, sys._getframe().f_code.co_name)
        time.sleep(0.3)
        sem.release()

def func4():
    for i in range(10):
        sem.acquire()
        print 'this is {} in {}'.format(i, sys._getframe().f_code.co_name)
        time.sleep(0.1)
        sem.release()

def func5():
    for i in range(5):
        sem.acquire()
        print 'this is {} in {}'.format(i, sys._getframe().f_code.co_name)
        sem.release()
        gevent.sleep(2)

def func6():
    for i in range(10):
        sem.acquire()
        print 'this is {} in {}'.format(i, sys._getframe().f_code.co_name)
        time.sleep(0.5)
        sem.release()

if __name__ == '__main__':
    if situation == 0:
        gevent.monkey.patch_all()
        gevent.joinall([
            gevent.spawn(func0, 'https://www.baidu.com/'),
            gevent.spawn(func0, 'https://www.alibaba.com/'),
            gevent.spawn(func0, 'https://www.tencent.com/')
        ])
    elif situation == 1:
        gevent.joinall([
            gevent.spawn(func1),
            gevent.spawn(func2)
        ])
    elif situation == 2:
        sem = Semaphore(1)
        gevent.joinall([
            gevent.spawn(func3),
            gevent.spawn(func4)
        ])
    elif situation == 3:
        sem = Semaphore(1)
        gevent.joinall([
            gevent.spawn(func5),
            gevent.spawn(func6)
        ])
