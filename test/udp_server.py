# -*- coding: utf-8 -*-
# __author__ = 'qinjincheng'

import socket

HOST = '0.0.0.0'
PROT = 3443

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind((HOST, PROT))

while True:
    data, addr = s.recvfrom(1024)
    print 'Received following data from {}:\n{}'.format(addr, data)

s.close()
