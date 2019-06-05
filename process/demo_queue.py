# -*- coding: utf-8 -*-
# __author__ = 'qinjincheng'

import datetime
import os
from multiprocessing import Queue, Process
import time

def decorator(func):
    def wrapper(*args, **kwargs):
        begin_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print('{} INFO: begin of the function ({}) at pid ({})'.format(begin_time, func.__qualname__, os.getpid()))
        result = func(*args, **kwargs)
        final_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print('{} INFO: final of the function ({}) at pid ({})'.format(final_time, func.__qualname__, os.getpid()))
        return result
    return wrapper

class Subprocess(Process):
    def __init__(self, quene, action, number=None):
        Process.__init__(self)
        self.quene = quene
        self.action = action
        self.number = number

    @decorator
    def run(self):
        {'w': self.put_message, 'r': self.get_message}[self.action]()

    @decorator
    def put_message(self):
        number = self.number
        while number:
            message = 'message {}'.format(number)
            self.quene.put(message)
            print('put {} into {}'.format(message, self.quene))
            number -= 1

    @decorator
    def get_message(self):
        while not self.quene.empty():
            time.sleep(0.5)
            print('get {} from {}'.format(self.quene.get(), self.quene))

@decorator
def main():
    q = Queue(3)
    pw = Subprocess(q, 'w', 9)
    pr = Subprocess(q, 'r')
    pw.start()
    pr.start()
    pw.join()
    pr.join()

if __name__ == '__main__':
    main()
