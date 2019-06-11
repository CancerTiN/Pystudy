# -*- coding: utf-8 -*-
# __author__ = 'qinjincheng'

import socket
import random
import time

def main():
    host = '127.0.0.1'
    port = 8080
    for i in range(random.randrange(10, 20)):
        time.sleep(1)
        s = socket.socket()
        s.connect((host, port))
        print('client send ({}) to server'.format(i))
        s.send(str(i).encode())
        data = s.recv(1024)
        print('client recv raw data ({}) from server'.format(data))
        print('client recv dec data ({}) from server'.format(data.decode()))
        s.close()

if __name__ == '__main__':
    main()
