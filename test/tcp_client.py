# -*- coding: utf-8 -*-
# __author__ = 'qinjincheng'

import socket
import datetime

HOST = '127.0.0.1'
PROT = 3434

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

s.connect((HOST, PROT))
print('Connect {}:{} OK'.format(HOST, PROT))
data = s.recv(1024)
print('Received:\n{}'.format(data))
s.close()
