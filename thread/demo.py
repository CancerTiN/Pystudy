# -*- coding: utf-8 -*-
# __author__ = 'qinjincheng'

import datetime
import threading
import time

def decorator(func):
    def wrapper(*args, **kwargs):
        begin_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print('{} INFO: begin of the function ({})'.format(begin_time, func.__qualname__))
        result = func(*args, **kwargs)
        final_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print('{} INFO: final of the function ({})'.format(final_time, func.__qualname__))
        return result
    return wrapper

def process():
    for i in range(3):
        time.sleep(1)
        print('thread name is {}'.format(threading.current_thread().name))

@decorator
def main():
    threads = [threading.Thread(target=process) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

if __name__ == '__main__':
    main()
