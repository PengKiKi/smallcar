# coding=utf-8

import pygame
from sys import exit
import socket

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind(('192.168.0.122', 9999))
bg = (255, 255, 255)

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

pygame.init()
screen = pygame.display.set_mode((800, 450))
pygame.display.set_caption("Hello, World!")
background = pygame.image.load('bg.jpg').convert_alpha()
background2 = pygame.image.load('bg.jpg').convert_alpha()
plane = pygame.image.load('sundy2.png').convert_alpha()

while True:
    screen.blit(background2, (0, 0))
    screen.blit(background, (0, 0))
    x, y = pygame.mouse.get_pos()

    horiz_axis_pos = 400
    vert_axis_pos = 0

    # UDP reveive
    data, addr = s.recvfrom(1024)
    horiz_axis_pos = float(data.decode('utf-8'))
    # temp=str(addr)
    s.sendto(addr[0].encode('utf-8'), addr)

    print("x: ", horiz_axis_pos, "  Y:", vert_axis_pos)
    a = plane.get_width() + 800 * horiz_axis_pos / 2 - 50
    b = plane.get_height() / 2 + 400 * vert_axis_pos / 2

    screen.fill(bg)
    screen.blit(plane, (a, b))
    pygame.display.flip()

    pygame.display.update()
    pygame.time.delay(1)
