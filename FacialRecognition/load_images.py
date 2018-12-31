#!/usr/bin/env python3
#coding=utf-8
''''
Capture multiple Faces from multiple users to be stored on a DataBase (dataset directory)
    ==> Faces will be stored on a directory: dataset/ (if does not exist, pls create one)
    ==> Each face will have a unique numeric integer ID as 1, 2, 3, etc

Based on original code by Anirban Kar: https://github.com/thecodacus/Face-Recognition

Developed by Marcelo Rovai - MJRoBot.org @ 21Feb18

'''

import cv2
import os

img_src = "/Volumes/Destiny/tmp/dataset"
img_dest = "dataset"
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_id = 0
names = ['None']

def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            os.remove(c_path)

del_file(img_dest)
os.system("find " + img_src + " -name '.DS_Store' | xargs rm")

for folder in [os.path.join(img_src,f) for f in os.listdir(img_src)]:
    if os.path.isfile(folder):
        continue
    face_id += 1
    count = 0

    # 扫描图片源目录下所有子目录
    for file in [os.path.join(folder,f) for f in os.listdir(folder)]:
        print(file)
        img = cv2.imread(file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            count += 1
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
            file_path = img_dest + "/User." + str(face_id) + '.' + str(count) + ".jpg"
            print(file_path)
            cv2.imwrite(file_path, gray[y:y+h,x:x+w])

    if count > 0:
        names.append(os.path.split(folder)[1])

cv2.destroyAllWindows()
print("names = %s" % names)
