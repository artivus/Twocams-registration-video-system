#import PySimpleGUI as sg
import cv2, sys, numpy, os
import os
import time, datetime

def read_config_cams():
    # читаем конфиг
    f = open('config.ini', 'r')
    for line in f:
        l = [line.strip() for line in f]
    f.close()
    input1 = f'{l[0]}'
    input2 = f'{l[1]}'
    return input1, input2

file1 = 'proj1'
haar_file = 'haarcascade_frontalface_default.xml'
datasets = 'datasets'
try:
    os.mkdir(datasets)
except:
    pass
sub_data = file1
path = os.path.join(datasets, sub_data)
#print(path)
if not os.path.isdir(path):
    os.mkdir(path)
(width, height) = (130, 100)
face_cascade = cv2.CascadeClassifier(haar_file)
webcam = cv2.VideoCapture(read_config_cams()[0])
#webcam = cv2.VideoCapture(0)
count = 1
time_start = time.time()
# print(time_start)
while True:
    #if (not int(time_start) % 3):
        #print(count)
        (ret, im) = webcam.read()
        if (ret):
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.2, 3)
            #print(faces)
            for (x, y, w, h) in faces:
                cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
                face = gray[y:y + h, x:x + w]
                face_resize = cv2.resize(face, (width, height))
                cv2.imwrite('% s/% s.png' % (path, count), face_resize)
                count += 1
            cv2.imshow(file1, im)
            #info = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            #cv2.putText(im, info, (8, 8), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 2)
        key = cv2.waitKey(1)
        if key == 27:
            break
