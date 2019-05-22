# -*- coding: utf-8 -*-
# __author__ = 'qinjincheng'

import socket
import datetime

HOST = '127.0.0.1'
PROT = 3443

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

data = 'Hello, UDP!'
s.sendto(data, (HOST, PROT))
print 'Sent following data to {}:{}:\n{}'.format(HOST, PROT, data)

s.close()
