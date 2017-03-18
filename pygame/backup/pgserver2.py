# coding=utf-8

# -*- coding: utf-8 -*-
import pygame
from sys import exit
import socket


s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind(('127.0.0.1', 9999))


BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


class TextPrint(object):
    """
    This is a simple class that will help us print to the screen
    It has nothing to do with the joysticks, just outputting the
    information.
    """

    def __init__(self):
        """ Constructor """
        self.reset()
        self.x_pos = 10
        self.y_pos = 10
        self.font = pygame.font.Font(None, 20)

    def print(self, my_screen, text_string):
        """ Draw text onto the screen. """
        text_bitmap = self.font.render(text_string, True, BLACK)
        my_screen.blit(text_bitmap, [self.x_pos, self.y_pos])
        self.y_pos += self.line_height

    def reset(self):
        """ Reset text to the top of the screen. """
        self.x_pos = 10
        self.y_pos = 10
        self.line_height = 15

    def indent(self):
        """ Indent the next line of text """
        self.x_pos += 10

    def unindent(self):
        """ Unindent the next line of text """
        self.x_pos -= 10


pygame.init()
screen = pygame.display.set_mode((800, 450))
pygame.display.set_caption("Hello, World!")
background = pygame.image.load('bg.jpg').convert_alpha()
background2 = pygame.image.load('bg.jpg').convert_alpha()
plane = pygame.image.load('sundy2.png').convert_alpha()
#加载飞机图像


joystick_count = pygame.joystick.get_count()
if joystick_count == 0:
    # 没检查到游戏手柄！
    print("Error, I didn't find any joysticks.")
else:
    # 使用手柄0并初始化
    my_joystick = pygame.joystick.Joystick(0)
    my_joystick.init()

textPrint = TextPrint()






while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

    screen.blit(background2, (0, 0))
    screen.blit(background, (0, 0))
    x, y = pygame.mouse.get_pos()

    horiz_axis_pos = my_joystick.get_axis(0)
    vert_axis_pos = my_joystick.get_axis(2)-my_joystick.get_axis(3)
    if my_joystick.get_button(8):
        plane = pygame.image.load('sundy2.png').convert_alpha()
    else:
        plane = pygame.image.load('sundy3.png').convert_alpha()


    #UDP reveive
    data, addr = s.recvfrom(1024)
    horiz_axis_pos=float(data.decode('utf-8'))


    #获取鼠标位置
    print("x: ",horiz_axis_pos , "  Y:",vert_axis_pos)
    a= plane.get_width()  + 800*horiz_axis_pos/2 -50
    b= plane.get_height() / 2 + 400*vert_axis_pos/2
    #计算飞机的左上角位置
    screen.blit(plane, (a,b))
    #把飞机画到屏幕上
    pygame.display.update()
