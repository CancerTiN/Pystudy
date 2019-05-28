# -*- coding: utf-8 -*-
# __author__ = 'qinjincheng'

import datetime
from multiprocessing import Process
import time
import os

def decorator(func):
    def wrapper(*args, **kwargs):
        begin_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print('{} INFO: begin of the function ({})'.format(begin_time, func.__qualname__))
        result = func(*args, **kwargs)
        final_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print('{} INFO: final of the function ({})'.format(final_time, func.__qualname__))
        return result
    return wrapper

class Subprocess(Process):
    def __init__(self, interval: int, name=str()):
        Process.__init__(self)
        self.interval = interval
        if name:
            self.name = name

    @decorator
    def run(self):
        print('child process ({}) is start from parent process ({})'.format(os.getpid(), os.getppid()))
        ta = time.time()
        time.sleep(self.interval)
        tz = time.time()
        print('child process ({}) is stop, the elapsed time is ({})'.format(os.getpid(), tz - ta))

@decorator
def main():
    print('parent process ({}) is start'.format(os.getpid()))
    p1 = Subprocess(interval=1, name='mrsoft')
    p2 = Subprocess(interval=2)
    # call method start() of the Process class which do not contain property target will call method run()
    # the method run() of Subprocess class will be call as follows
    p1.start()
    p2.start()
    print('p1.is_alive -> {}'.format(p1.is_alive()))
    print('p2.is_alive -> {}'.format(p2.is_alive()))
    print('p1.name -> {}'.format(p1.name))
    print('p1.pid -> {}'.format(p1.pid))
    print('p2.name -> {}'.format(p2.name))
    print('p2.pid -> {}'.format(p2.pid))
    print('wait for child processes')
    # use method join() to wait for subprocess
    p1.join()
    p2.join()

if __name__ == '__main__':
    main()
