# -*- coding: utf-8 -*-
# __author__ = 'qinjincheng'

import socket
import datetime

HOST = '0.0.0.0'
PROT = 3434

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PROT))
s.listen(1)

while True:
    conn, addr = s.accept()
    print 'Client {} connected!'.format(addr)
    dt = datetime.datetime.now()
    message = 'Current time is {}'.format(dt)
    conn.send(message)
    print 'Sent following data to {}:\n{}'.format(conn, message)
    conn.close()
