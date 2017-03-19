# coding=utf-8

import pygame
from sys import exit
import socket
import threading, time, random
import struct
import spidev as SPI


s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind(('192.168.0.122', 9999))

bus=0
device=0
spi=SPI.SpiDev(bus,device)
spi.open (0,0 )
spi.max_speed_hz = 500000
spi.mode = 0b00

# Define some colors
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

    def printg(self, my_screen, text_string):
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

axisa=[]
buttona=[]
hata=[]


pygame.init()

# Set the width and height of the screen [width,height]
size = [500, 700]
screen = pygame.display.set_mode(size)

pygame.display.set_caption("My Game")

# Loop until the user clicks the close button.
done = False

# Used to manage how fast the screen updates
clock = pygame.time.Clock()

# Initialize the joysticks
pygame.joystick.init()

# Get ready to print
textPrint = TextPrint()

# -------- Main Program Loop -----------
while not done:

    data, addr = s.recvfrom(1024)
    a = struct.unpack('5f2i23?', data)
    a=list(a)
    axisa = a[0:5]
    hata = a[5:7]
    buttona = a[7:30]
    # EVENT PROCESSING STEP
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

        # Possible joystick actions: JOYAXISMOTION JOYBALLMOTION JOYBUTTONDOWN
        # JOYBUTTONUP JOYHATMOTION
        if event.type == pygame.JOYBUTTONDOWN:
            print("Joystick button pressed.")
        if event.type == pygame.JOYBUTTONUP:
            print("Joystick button released.")

    # DRAWING STEP
    # First, clear the screen to white. Don't put other drawing commands
    # above this, or they will be erased with this command.
    screen.fill(WHITE)
    textPrint.reset()

    # Get count of joysticks
    joystick_count = 1

    textPrint.printg(screen, "Number of joysticks: {}".format(joystick_count))
    textPrint.indent()

    # For each joystick:
    for i in range(joystick_count):


        textPrint.printg(screen, "Joystick {}".format(i))
        textPrint.indent()

        # Get the name from the OS for the controller/joystick
        name = 'Pengkiki Pi'
        textPrint.printg(screen, "Joystick name: {}".format(name))

        # Usually axis run in pairs, up/down for one, and left/right for
        # the other.
        axes = 5
        textPrint.printg(screen, "Number of axes: {}".format(axes))
        textPrint.indent()

        for i in range(axes):
            axis = axisa[i]
            textPrint.printg(screen, "Axis {} value: {:>6.3f}".format(i, axis))

        textPrint.unindent()

        # Hat switch. All or nothing for direction, not like joysticks.
        # Value comes back in an array.
        hats = 1
        textPrint.printg(screen, "Number of hats: {}".format(hats))
        textPrint.indent()

        for i in range(hats):
            hat = [hata[0],hata[1]]

            textPrint.printg(screen, "Hat {} value: {}".format(i, str(hat)))
        textPrint.unindent()

        textPrint.unindent()




        buttons = 23
        textPrint.printg(screen, "Number of buttons: {}".format(buttons))
        textPrint.indent()

        for i in range(buttons):
            button = buttona[i]
            textPrint.printg(screen, "Button {:>2} value: {}".format(i, button))
        textPrint.unindent()



    # ALL CODE TO DRAW SHOULD GO ABOVE THIS COMMENT

    # Go ahead and update the screen with what we've drawn.
    pygame.display.flip()

#    updatedata=struct.pack('5f23?ii',for i in packdata)


    steer = (axisa[0]+1)*65535/2
    steer = int(steer)
    acc = (axisa[2]+ 1)*256/2
    acc = int(acc)
    dcc = (axisa[3]+ 1)*256/2
    dcc = int(dcc)
    r = spi.xfer2([0, 10, 80, 10, steer/256, steer%256, acc, dcc])

    # Limit to 60 frames per second
    clock.tick(30)

# Close the window and quit.
# If you forget this line, the program will 'hang'
# on exit if running from IDLE.
pygame.quit()