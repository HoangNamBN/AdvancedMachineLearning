import os
import cv2
import numpy as np

def collect_faces(path):
    print('Press ENTER to collect faces...\nPress ESC to exit...\n')
    video_capture = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    while(video_capture.isOpened()):
        _, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = face_cascade.detectMultiScale(gray, 1.1, 2, 0|2, (100,100))
        for (x, y ,w, h) in faces:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        k = cv2.waitKey(100)
        if k==27:
            break
        elif k==13:
            face_id = input('\n enter face id and press <return> ===>  ')
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (100,100))
            cv2.imwrite(os.path.join(path, face_id+'.jpg'), face)
        cv2.imshow('collect_faces', frame)
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    path = 'dataset'
    if not os.path.exists(path):
        os.mkdir(path)
    collect_faces(path)
