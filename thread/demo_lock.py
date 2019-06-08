# -*- coding: utf-8 -*-
# __author__ = 'qinjincheng'

from thread.demo_class import decorator, SubThread
from threading import Lock
import time

n = 100
mutex = Lock()

def task():
    global n
    mutex.acquire()
    time.sleep(0.5)
    n -= 1
    print('succeed in buying ticket, there are {} tickets left'.format(n))
    mutex.release()

def main():
    lst = list()
    for i in range(10):
        t = SubThread(task)
        lst.append(t)
        t.start()
    else:
        for t in lst:
            t.join()

if __name__ == '__main__':
    main()
