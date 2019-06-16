# -*- coding: utf-8 -*-
# __author__ = 'qinjincheng'

import urllib.request

def main():
    response = urllib.request.urlopen('http://www.baidu.com')
    html = response.read()
    print(html.decode())

if __name__ == '__main__':
    main()
