#潇神 2023.7.4 颜色识别
#红队的代码
#红队的代码
#红队的代码

import sensor, image, time,math,pyb
import json
from pyb import Servo
from pyb import UART
import ustruct


bule_threshold  = (25, 47, -49, 36, -78, -24)
red_threshold  = (28, 54, 15, 53, 1, 47)
yellow_threshold  = (62, 85, -33, -6, 37, 78)
green_threshold  = (28, 56, -61, -9, -17, 24)
uart=UART(1,115200,timeout_char=1000)
uart.init(115200,bits=8,parity=None,stop=1,timeout_char=1000)
u_start=bytearray([0xb3,0xb3])
u_over=bytearray([0x0d,0x0a])

sensor.reset() # Initialize the camera sensor.
sensor.set_vflip(True)
sensor.set_pixformat(sensor.RGB565) # use RGB565.
sensor.set_framesize(sensor.QQVGA) # use QQVGA for speed.
sensor.set_windowing((0,20,320,200))
sensor.skip_frames(10) # Let new settings take affect.
sensor.set_auto_whitebal(False) # turn this off.
#sensor.set_vflip(True)
clock = time.clock() # Tracks FPS.
temp1=5
temp2=5
uart=UART(3,115200)

def find_max(blobs):
    max_pixels=0
    for blob in blobs:
        if blob[4] > max_pixels:
            max_blob=blob
            max_pixels = blob[4]
    return max_blob

# while(True):
#     times=0
#     clock.tick()
#     img=sensor.snapshot()

#     row_data=[0,0,0,0]
#     uart_buf=bytearray(row_data)
#     uart.write(u_start)
#     uart.write(uart_buf)
#     uart.write(u_over)

#while(True):
    #times=0
    #clock.tick() # Track elapsed milliseconds between snapshots().
    #img = sensor.snapshot() # Take a picture and return the image.
    #row_data=[0,0,0,0]
    #uart_buf=bytearray(row_data)
    #uart.write(u_start)
    #uart.write(uart_buf)
    #uart.write(u_over)

















####################################################################################################################
while(True):
    times=0
    clock.tick() # Track elapsed milliseconds between snapshots().
    img = sensor.snapshot() # Take a picture and return the image.
    temp1=5
    temp2=5
    blobs_blue = img.find_blobs([bule_threshold],pixels_threshold=200, area_threshold=200, merge=True)
    blobs_red = img.find_blobs([red_threshold],pixels_threshold=200, area_threshold=200, merge=True)
    blobs_yellow = img.find_blobs([yellow_threshold],pixels_threshold=200, area_threshold=200, merge=True)
    blobs_green = img.find_blobs([green_threshold],pixels_threshold=200, area_threshold=200, merge=True)

    if blobs_blue:
        # print("blue")
        temp1=0
    if blobs_red:
        # print("red")
        temp1=1
    if blobs_yellow:
        # print("yellow")
        temp2=0
    if blobs_green:
        # print("green")
        temp2=1
#green 0 blue and yellow 1 red and yellow 2
    #else:
        #print("not found!")

#红队的代码

#情况2 有绿色和红色 则为红色方宝藏
    if temp1==1 and temp2==1:
      row_data=[0,0,0,0]
      uart_buf=bytearray(row_data)
      uart.write(u_start)
      uart.write(uart_buf)
      uart.write(u_over)
    #   print("红色")
    # else:
    #   print("不是红色")
















############################################################################
#情况2 有黄色和红色 则为红色宝藏
   #  elif temp1==1 and temp2==0:
      #   output_str_zhen=ustruct.pack("bbbb",
      #              0x2C,                      #帧头1
      #              0x12,                      #帧头2
      #              0x01,                      #数据
      #              0x5B)
      #   uart.write(output_str_zhen+'\n')
       # print("红色")
#情况3 有绿色则直接为假
   #  elif temp2==1:
   #    circumstance=0
      #print("绿色")







# while(True):
#     clock.tick() # Track elapsed milliseconds between snapshots().
#     img = sensor.snapshot() # Take a picture and return the image.
#     temp1=5
#     temp2=5
#     blobs_blue = img.find_blobs([bule_threshold],pixels_threshold=200, area_threshold=200, merge=True)
#     blobs_red = img.find_blobs([red_threshold],pixels_threshold=200, area_threshold=200, merge=True)
#     blobs_yellow = img.find_blobs([yellow_threshold],pixels_threshold=200, area_threshold=200, merge=True)
#     blobs_green = img.find_blobs([green_threshold],pixels_threshold=200, area_threshold=200, merge=True)

#     if blobs_blue:
#         # print("blue")
#         temp1=0
#     if blobs_red:
#         # print("red")
#         temp1=1
#     if blobs_yellow:
#         # print("yellow")
#         temp2=0
#     if blobs_green:
#         # print("green")
#         temp2=1
# #green 0 blue and yellow 1 red and yellow 2
#     #else:
#         #print("not found!")

# #情况1 有绿色则直接为假
#     if temp2==1:
#         circumstance=0
#         #print("绿色")
# #情况2 有黄色和蓝色 则为蓝色方宝藏
#     elif temp1==0 and temp2==0:
#         circumstance=1
#       #   output_str_jia=ustruct.pack("bbbb",
#       #              0x2C,                      #帧头1
#       #              0x12,                      #帧头2
#       #              0x00,                      #数据
#       #              0x5B)

#       #   uart.write(output_str_jia+'\n')
#         #print("蓝色")
# #情况3 有黄色和红色 则为红色宝藏
#     elif temp1==1 and temp2==0:
#         circumstance=2
#         output_str_zhen=ustruct.pack("bbbb",
#                    0x2C,                      #帧头1
#                    0x12,                      #帧头2
#                    0x01,                      #数据
#                    0x5B)
#         uart.write(output_str_zhen+'\n')
#        # print("红色")
# #if blobs:s
#         ##print (blobs)
#         #max_blob = find_max(blobs)
#         #img.draw_rectangle(max_blob.rect()) # rect
#         #img.draw_cross(max_blob.cx(), max_blob.cy()) # cx, cy
#         #pcx=max_blob.cx()
#         #pcy=max_blob.cy()
#         #data={
#         #"cx":pcx,
#         #"cy":pcy}
#         #print('you send:',data_out)
#     #else:
#         #print("not found!")
