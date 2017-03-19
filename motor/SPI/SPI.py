# coding=utf-8

import serial
import struct
import time
import spidev as SPI

bus=0
device=0
spi=SPI.SpiDev(bus,device)
spi.open (0,0 )
spi.max_speed_hz = 500000
spi.mode = 0b00

button=[ 1.2, 2.2, 3.2]
r=spi.xfer2([0,10,80,10,1,2,3,54])

t = struct.pack('fff', *button)
a=b'\xff\xfe'
t = b'\xff\xfe' + t

ser.write(t)


while 1:

    time.sleep(1)
    ser.write(t)
