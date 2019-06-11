# -*- coding: utf-8 -*-
# __author__ = 'qinjincheng'

import threading
import time
import random
from thread.demo_class import decorator
from queue import Queue

class Producer(threading.Thread):
    def __init__(self, name, queue):
        threading.Thread.__init__(self, name=name)
        self.data = queue
    def run(self):
        for i in range(5):
            self.data.put(i)
            print('{} put {} into {}'.format(self.getName(), i, self.data))
            time.sleep(random.random())
        else:
            print('{} has completed job'.format(self.getName()))

class Consumer(threading.Thread):
    def __init__(self, name, queue):
        threading.Thread.__init__(self, name=name)
        self.data = queue
    def run(self):
        for i in range(5):
            print('{} get {} from {}'.format(self.getName(), i, self.data))
            time.sleep(random.random())
        else:
            print('{} has completed job'.format(self.getName()))

@decorator
def main():
    queue = Queue()
    producer = Producer('Producer', queue)
    consumer = Consumer('Consumer', queue)
    producer.start()
    consumer.start()
    producer.join()
    consumer.join()

if __name__ == '__main__':
    main()
