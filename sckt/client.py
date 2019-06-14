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

def main(sk):
    {'tcp': tcp_client, 'udp': udp_client}[sk]()

def tcp_client():
    host = socket.gethostname()
    port = 12345
    print('Connect -> {}:{}'.format(host, port))
    s = socket.socket()
    s.connect((host, port))
    print('Connection already established')
    info = str()
    while info != 'byebye':
        data = input('Content send: ')
        s.send(data.encode())
        if data == 'byebye':
            break
        info = s.recv(1024).decode()
        print('Content received:\n{}'.format(info))
    s.close()

def udp_client():
    host = socket.gethostname()
    port = 54321
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    data = input('Please input a Centigrade degree: ')
    s.sendto(data.encode(), (host, port))
    print(s.recv(1024).decode())
    s.close()

if __name__ == '__main__':
    main(sk='udp')
