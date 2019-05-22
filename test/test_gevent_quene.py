# -*- coding: utf-8 -*-
# __author__ = 'qinjincheng'

import gevent
from gevent.queue import Queue

tasks = Queue()

def worker(name):
    while not tasks.empty():
        task = tasks.get()
        print '{} got task {}'.format(name, task)
        gevent.sleep(0.5)
    else:
        print '{} quitting now'.format(name)

def boss():
    for i in range(25):
        tasks.put_nowait(i)
    else:
        print 'succeed in putting task'

def main():
    gevent.spawn(boss).join()
    gevent.joinall([
        gevent.spawn(worker, 'steve'),
        gevent.spawn(worker, 'john'),
        gevent.spawn(worker, 'nancy'),
    ])

if __name__ == '__main__':
    main()
