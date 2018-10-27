# -*- coding: utf-8 -*-
import cv2
import numpy as np

drawing = False # true if mouse is pressed
ix,iy = -1,-1

class mouseParam:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.ix = 0
        self.iy = 0
        self.eventType = 0
        self.flags = 0

def draw_point(event,x,y,flags,param):
    global ix,iy,drawing
    
    if event == cv2.EVENT_LBUTTONDOWN:
        print(2)
        drawing = True
        param.ix = x
        param.iy = y

    elif event == cv2.EVENT_MOUSEMOVE:
        print(3)
        if drawing == True:
            # cv2.rectangle(frame,(ix,iy),(x,y),(0,255,0),-1)
            param.x = x
            param.y = y

    elif event == cv2.EVENT_LBUTTONUP:
        print(4)
        drawing = False
   
    elif event == cv2.EVENT_RBUTTONUP:
        param.ix = -1
        param.iy = -1
        param.x = -1
        param.y = -1

 
cap = cv2.VideoCapture(0)

mouseData = mouseParam()
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_point, mouseData)

while(1):
    # VideoCaptureから1フレーム読み込む
    ret, frame = cap.read()

    cv2.line(frame, (mouseData.ix, mouseData.iy), (mouseData.x, mouseData.y), (0, 0, 255), 3)
    cv2.imshow('image',frame)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

    print("ix={},iy={},x={},y={}".format(mouseData.ix,mouseData.iy, mouseData.x, mouseData.y))
    # 描画する

cap.release()
cv2.destroyAllWindows()
