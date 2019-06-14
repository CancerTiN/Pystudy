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

def main(sk):
    {'tcp': tcp_server, 'udp': udp_server}[sk]()

def tcp_server():
    host = socket.gethostname()
    port = 12345
    print('Bind -> {}:{}'.format(host, port))
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((host, port))
    s.listen(1)
    sock, addr = s.accept()
    print('Connection already established -> {}'.format(s))
    info = sock.recv(1024).decode()
    while info != 'byebye':
        if info:
            print('Content received:\n{}'.format(info))
        data = input('Content send: ')
        sock.send(data.encode())
        if data == 'byebye':
            break
        info = sock.recv(1024).decode()
    sock.close()
    s.close()

def udp_server():
    host = socket.gethostname()
    port = 54321
    print('Bind -> {}:{}'.format(host, port))
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind((host, port))
    print('Connection already established -> {}'.format(s))
    info, addr = s.recvfrom(1024)
    print('Received from {}'.format(addr))
    data = 'Fahrenheit degree {}'.format(float(info) * 1.8 + 32)
    s.sendto(data.encode(), addr)
    s.close()

if __name__ == '__main__':
    main(sk='udp')
