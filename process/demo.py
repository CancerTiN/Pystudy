# -*- coding: utf-8 -*-
# __author__ = 'qinjincheng'

import datetime
import os
import time
from multiprocessing import Process

def decorator(func):
    def wrapper(*args, **kwargs):
        begin_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print('{} INFO: begin of the function ({})'.format(begin_time, func.__qualname__))
        result = func(*args, **kwargs)
        final_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print('{} INFO: final of the function ({})'.format(final_time, func.__qualname__))
        return result
    return wrapper

def test():
    print('I am subprocess')

def child_1(interval):
    print('child process ({}) is start from parent process ({})'.format(os.getpid(), os.getppid()))
    ta = time.time()
    time.sleep(interval)
    tz = time.time()
    print('child process ({}) is stop, the elapsed time is ({})'.format(os.getpid(), tz - ta))

def child_2(interval):
    print('child process ({}) is start from parent process ({})'.format(os.getpid(), os.getppid()))
    ta = time.time()
    time.sleep(interval)
    tz = time.time()
    print('child process ({}) is stop, the elapsed time is ({})'.format(os.getpid(), tz - ta))

@decorator
def main():
    print('parent process ({}) is start'.format(os.getpid()))
    p1 = Process(target=child_1, args=(1,))
    p2 = Process(target=child_2, name='mrsoft', args=(2,))
    p1.start()
    p2.start()
    print('p1.is_alive -> {}'.format(p1.is_alive()))
    print('p2.is_alive -> {}'.format(p2.is_alive()))
    print('p1.name -> {}'.format(p1.name))
    print('p1.pid -> {}'.format(p1.pid))
    print('p2.name -> {}'.format(p2.name))
    print('p2.pid -> {}'.format(p2.pid))

    p3 = Process(target=test, args=())
    p3.start()

    print('wait for child processes')
    # use method join() to wait for subprocess
    p1.join()
    p2.join()

if __name__ == '__main__':
    main()
