# coding=utf-8

import socket

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
a=[b'Michael', b'Tracy', b'Sarah']
b=[str(12.666).encode('utf-8'),str(13.444).encode('utf-8')]
for data in b:
    # 发送数据:
    s.sendto(data, ('127.0.0.1', 9999))
    # 接收数据:
    #print(s.recv(1024).decode('utf-8'))

#s.close()