"""
ECE196 Face Recognition Project
Author: Will Chen

Prerequisite: You need to install OpenCV before running this code
The code here is an example of what you can write to print out 'Hello World!'
Now modify this code to process a local image and do the following:
1. Read geisel.jpg
2. Convert color to gray scale
3. Resize to half of its original dimensions
4. Draw a box at the center the image with size 100x100
5. Save image with the name, "geisel-bw-rectangle.jpg" to the local directory
All the above steps should be in one function called process_image()
"""

# TODO: Import OpenCV
import cv2
# TODO: Edit this function
def process_image():
    img=cv2.imread('geisel.jpg',0)
    size = img.shape
    small = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    height, width  = small.shape
    cv2.rectangle(small, (width/2-50,height/2-50) , (width/2+50, height/2+50), (255,255,255), 4)
    cv2.imshow('image', small)
    cv2.waitKey(0)
    cv2.imwrite('new geisel.jpg',small)
    cv2.destroyAllWindow()
    return

# Just prints 'Hello World! to screen.
def hello_world():
    print('Hello World!')
    return

# TODO: Call process_image function.
def main():
    hello_world()
    process_image()
    return


if(__name__ == '__main__'):
    main()
