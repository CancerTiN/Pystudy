# -*- coding: utf-8 -*-
# __author__ = 'qinjincheng'

import urllib3
http = urllib3.PoolManager()

def get():
    respone = http.request('GET', 'https://www.baidu.com/')
    print(respone.data.decode())

def post():
    respone = http.request('POST', 'http://httpbin.org/post', fields={'hello': 'world'})
    print(respone.data.decode())

def main(k):
    {'get': get, 'post': post}[k]()

if __name__ == '__main__':
    main('post')
