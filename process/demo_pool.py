# -*- coding: utf-8 -*-
# __author__ = 'qinjincheng'

import datetime
import os
import time
from multiprocessing import Pool
import random

def decorator(func):
    def wrapper(*args, **kwargs):
        begin_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print('{} INFO: begin of the function ({}) at pid ({})'.format(begin_time, func.__qualname__, os.getpid()))
        result = func(*args, **kwargs)
        final_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print('{} INFO: final of the function ({}) at pid ({})'.format(final_time, func.__qualname__, os.getpid()))
        return result
    return wrapper

def runtask(name, interval: int):
    print('task {} is start at pid ({})'.format(name, os.getpid()))
    time.sleep(interval)
    print('task {} is stop at pid ({})'.format(name, os.getpid()))

@decorator
def main():
    p = Pool(3)
    for i in range(10):
        p.apply_async(runtask, args=(i, random.randint(1, 9)))
    print('wait for all child processes')
    p.close()
    p.join()

if __name__ == '__main__':
    main()
