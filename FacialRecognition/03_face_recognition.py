#!/usr/bin/env python3
#coding=utf-8
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
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

#iniciate id counter
id = 0

# names related to ids: example ==> Marcelo: id=1, etc
names = ['None', 'Marcelo', 'Paula', 'Ilza', 'Z', 'W']

# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video widht
cam.set(4, 480) # set video height

# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

while True:

    ret, img =cam.read()
    img = cv2.flip(img, 1) # Flip vertically

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # cv2和PIL中颜色的hex码的储存顺序不同
    pilimg = Image.fromarray(cv2img)

    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
    )

    for(x,y,w,h) in faces:
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        # Check if confidence is less them 100 ==> "0" is perfect match
        if (confidence < 100):
            id = names[id]
        else:
            id = "unknown"
        confidence = "  {0}%".format(round(100 - confidence))

        # PIL图片上打印汉字
        draw = ImageDraw.Draw(pilimg)
        draw.line((x, y, x+w, y), (0,255,0))
        draw.line((x, y+h, x+w, y+h), (0,255,0))
        draw.line((x, y, x, y+h), (0,255,0))
        draw.line((x+w, y, x+w, y+h), (0,255,0))
        # 参数1：字体文件路径，参数2：字体大小
        font = ImageFont.truetype("msyh.ttf", 20, encoding="utf-8")
        # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体
        draw.text((x+5,y-25), str(id) + str(confidence), (255,255,255), font=font)

    # PIL图片转cv2 图片
    cv2charimg = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
    cv2.imshow('camera',cv2charimg)

    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
