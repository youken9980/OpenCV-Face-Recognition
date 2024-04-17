#!/usr/bin/env python3
# -*- coding:utf-8 -*-

''''
Real Time Face Recogition
    ==> Each face stored on dataset/ dir, should have a unique numeric integer ID as 1, 2, 3, etc
    ==> LBPH computed model (trained faces) should be on trainer/ dir

Based on original code by Anirban Kar: https://github.com/thecodacus/Face-Recognition

Developed by Marcelo Rovai - MJRoBot.org @ 21Feb18
'''

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
faceCascade = cv2.CascadeClassifier("cascade.xml")
fontSize = 20
font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", fontSize, encoding="utf-8")
# names related to ids: example ==> Marcelo: id=1, etc
names = ['None', 'xxx']


# 文字转换为图片并添加到图片上
def cv2ImgAddText(img, x,y,w,h, text, textColor=(255,255,255)):
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    draw.line((x, y, x+w, y), textColor)
    draw.line((x, y+h, x+w, y+h), textColor)
    draw.line((x, y, x, y+h), textColor)
    draw.line((x+w, y, x+w, y+h), textColor)
    # 绘制文本，参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体
    draw.text((x+5, y-fontSize-5), text, textColor, font=font)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def detect_face(img, minSize):
    # img = cv2.flip(img, 1) # Flip vertically
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # cv2和PIL中颜色的hex码的储存顺序不同
    pilimg = Image.fromarray(cv2img)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = minSize,
    )

    for(x,y,w,h) in faces:
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        # Check if confidence is less them 100 ==> "0" is perfect match
        if (confidence < 100):
            id = names[id]
        else:
            id = "unknown"
        confidence = "  {0}%".format(round(100 - confidence))
        img = cv2ImgAddText(pilimg, x, y, w, h, str(id) + str(confidence), (255,255,255))

    return img


if __name__ == '__main__':
    # 读取视频文件
    # cam = cv2.VideoCapture("*.mp4")
    # 读取视频流
    # cam = cv2.VideoCapture('rtmp://')

    # # Initialize and start realtime video capture
    cam = cv2.VideoCapture(0)
    cam.set(3, 960) # set video width
    cam.set(4, 720) # set video height
    # # Define min window size to be recognized as a face
    minW = 0.1*cam.get(3)
    minH = 0.1*cam.get(4)
    minSize = (int(minW), int(minH))

    try:
        while True:
            ret, frame = cam.read()
            if ret:
                cv2charimg = detect_face(frame, minSize)
                cv2.namedWindow('face_recognition', cv2.WINDOW_KEEPRATIO)
                cv2.imshow('face_recognition', cv2charimg)
                cv2.resizeWindow('face_recognition', 960, 720)

                k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
                if k == 27:
                    break
            else:
                break
    except Exception as e:
        print(e)
    finally:
        # Do a bit of cleanup
        print("\n [INFO] Exiting Program and cleanup stuff")
        cam.release()
        cv2.destroyAllWindows()
