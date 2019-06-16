# -*- coding: utf-8 -*-
# __author__ = 'qinjincheng'

import requests

def get():
    headers = {
        'firebox': {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64; rv:67.0) Gecko/20100101 Firefox/67.0'},
        'chrome': {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.90 Safari/537.36'},
        'sogou': {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3314.0 Safari/537.36 SE 2.X MetaSr 1.0'},
    }
    response = requests.get('http://www.baidu.com', headers=headers['firebox'])
    print(response.status_code)
    print(response.url)
    print(response.headers)
    print(response.cookies)
    print(response.text)
    print(response.content.decode())

def post():
    data = {'word': 'hello'}
    response = requests.post('http://httpbin.org/post', data=data)
    print(response.content.decode())

def put():
    data = {'key': 'value'}
    response = requests.put('http://httpbin.org/put', data=data)
    print(response.content.decode())

def delete():
    response = requests.delete('http://httpbin.org/delete')
    print(response.content.decode())

def head():
    response = requests.head('http://httpbin.org/get')
    print(response.content.decode())

def options():
    response = requests.options('http://httpbin.org/get')
    print(response.content.decode())

def main(k):
    {'get': get, 'post': post, 'put': put, 'delete': delete, 'head': head, 'options': options}[k]()

if __name__ == '__main__':
    main('options')
