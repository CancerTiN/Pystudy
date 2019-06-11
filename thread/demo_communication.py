# -*- coding: utf-8 -*-
# __author__ = 'qinjincheng'

from thread.demo_class import decorator, SubThread

@decorator
def plus():
    global g_num
    g_num += 50
    print('g_num is {}'.format(g_num))

@decorator
def minus():
    global g_num
    g_num -= 50
    print('g_num is {}'.format(g_num))

@decorator
def main():
    global g_num
    g_num = 100
    print('g_num is {}'.format(g_num))
    t1 = SubThread(plus)
    t2 = SubThread(minus)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

if __name__ == '__main__':
    main()
