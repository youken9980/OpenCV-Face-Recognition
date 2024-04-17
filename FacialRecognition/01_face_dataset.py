#!/usr/bin/env python3
# -*- coding:utf-8 -*-

''''
Capture multiple Faces from multiple users to be stored on a DataBase (dataset directory)
    ==> Faces will be stored on a directory: dataset/ (if does not exist, pls create one)
    ==> Each face will have a unique numeric integer ID as 1, 2, 3, etc

Based on original code by Anirban Kar: https://github.com/thecodacus/Face-Recognition

Developed by Marcelo Rovai - MJRoBot.org @ 21Feb18
'''

import cv2
import os


face_detector = cv2.CascadeClassifier('cascade.xml')


if __name__ == '__main__':
    # cam = cv2.VideoCapture("*.mp4")
    cam = cv2.VideoCapture(0)
    cam.set(3, 960) # set video width
    cam.set(4, 720) # set video height

    # For each person, enter one numeric face id
    face_id = input('\n enter user id and press <return> ==>  ')
    # Initialize individual sampling face count
    seq = input('\n enter seq and press <return> ==>  ')
    if not seq.strip():
        count = 0
    else:
        count = int(seq)

    print("\n [INFO] Initializing face capture. Look the camera and wait ...")

    try:
        while(True):
            ret, frame = cam.read()
            if ret:
                # frame = cv2.flip(frame, 1) # flip video image vertically
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_detector.detectMultiScale(gray, 1.3, 5)

                for (x,y,w,h) in faces:
                    cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
                    count += 1
                    # Save the captured image into the datasets folder
                    cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
                    # 显示图像
                    cv2.namedWindow('face_capture', cv2.WINDOW_KEEPRATIO)
                    cv2.imshow('face_capture', frame)
                    cv2.resizeWindow('face_capture', 960, 720)

                k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
                if k == 27:
                    break
                elif count >= 100: # Take 30 face sample and stop video
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
