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

class SubThread(threading.Thread):
    def __init__(self, func=None):
        threading.Thread.__init__(self)
        self._func = func

    @decorator
    def run(self):
        if self._func:
            print('start calling function ({})'.format(self._func.__name__))
            self._func()
        else:
            for i in range(3):
                time.sleep(1)
                print('{} is running at {}'.format(self.name, i))

@decorator
def main():
    t1 = SubThread()
    t2 = SubThread()
    t1.start()
    t2.start()
    t1.join()
    t2.join()

if __name__ == '__main__':
    main()
