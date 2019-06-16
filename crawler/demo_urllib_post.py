# -*- coding: utf-8 -*-
# __author__ = 'qinjincheng'

import urllib.parse
import urllib.request

def main():
    data = bytes(urllib.parse.urlencode({'word': 'hello'}), encoding='utf8')
    print(data)
    response = urllib.request.urlopen('http://httpbin.org/post', data=data)
    print(response)
    html = response.read()
    print(html.decode())

if __name__ == '__main__':
    main()
