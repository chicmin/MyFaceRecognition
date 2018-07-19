#CropImage

import cv2 as cv
import glob

cascade = cv.CascadeClassifier('/home/pi/opencv-2.4.13.4/data/haarcascades/haarcascade_frontalface_default.xml')

images = glob.glob('/home/pi/MyFaceRecognition/myFace/*.JPG')
number = 0

print(images)

for fname in images :
    print(fname)
    image = cv.imread(fname)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    #reimage = cv.resize(gray, (640,480)) 
    face= cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in face :
        print('enter')
        cropped = gray[y:y+h, x:x+w]
        cv.imwrite('/home/pi/MyFaceRecognition/images/17/picture' + str(number) + '.jpg', cropped)
        number += 1
        print('completed')

