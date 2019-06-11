# -*- coding: utf-8 -*-
# __author__ = 'qinjincheng'

import socket

def main():
    host = '127.0.0.1'
    port = 8080
    s = socket.socket()
    s.bind((host, port))
    s.listen(5)
    print('server is waiting for client now')
    while True:
        conn, addr = s.accept()
        data = conn.recv(1024)
        print(data)
        conn.sendall(b'HTTP/1.1 200 OK\r\n\r\nHello World')
        conn.close()

if __name__ == '__main__':
    main()
