#!/usr/env/bin python3

import sys
import string
import numpy as np
import cv2
import random
from functools import reduce

# font                   = cv2.FONT_HERSHEY_SIMPLEX
# bottomLeftCornerOfText = (10,25)
# bottomLeftCornerOfText = (0,17)
blcot = []
for i in range(30):
    for j in range(17, 26):
        blcot.append((i, j))
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2

char_list = string.ascii_letters+string.digits

random.seed()

def pick_char():
    return char_list[random.randint(0, len(char_list) - 1)]

def pick_text():
    l = random.randint(1, 20)
    return reduce(lambda s, c: s + c, map(lambda _: pick_char(), list(range(l))))

fonts = ['Arial.ttf', 'Roboto-Regular.ttf', 'Microsoft Sans Serif.ttf']
# fonts = ['Arial.ttf', 'Microsoft Sans Serif.ttf']
fts = []
for font in fonts:
    ft = cv2.freetype.createFreeType2()
    ft.loadFontData(fontFileName=font,
                    id=0)
    fts.append(ft)

font_sizes = list(range(13, 21))
# font_sizes = [20]

def generate_dataset(count, path):
    data = set()
    i = -1

    while len(data) < count:
        i += 1

        ft = fts[i%len(fts)]
        img = np.zeros((32, 400, 3), np.uint8)
        s = pick_text()
        data.add(s.lower())
        ft.putText(
            img=img, text=s,
            # org=bottomLeftCornerOfText,
            org=blcot[i%len(blcot)],
            fontHeight=font_sizes[i%len(font_sizes)],
            color=fontColor,
            thickness=-1,
            line_type=cv2.LINE_AA,
            bottomLeftOrigin=True
        )
        # cv2.putText(img, s,
        #     bottomLeftCornerOfText, 
        #     font, 
        #     fontScale,
        #     fontColor,
        #     lineType)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # # extract interests
        invert_img = 255 - img
        ret,thresh1 = cv2.threshold(invert_img,180,255,cv2.THRESH_BINARY_INV)
        kernel = np.ones((5,5),np.uint8)
        dilated = cv2.dilate(thresh1,kernel,iterations = 2)
        contours, hierarchy = cv2.findContours(dilated,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for cnt in contours:
            (x, y, w, h) = cv2.boundingRect(cnt)
            boxes.append([x,y, x+w,y+h])

        boxes = np.asarray(boxes)
        left = np.min(boxes[:,0])
        top = np.min(boxes[:,1])
        right = np.max(boxes[:,2])
        bottom = np.max(boxes[:,3])
        area = img[top:bottom, left:right]
        h, w = area.shape
        if h > 32 or w > 400:
            print(area.shape, font_sizes[i%len(font_sizes)])
            exit(-1)
        area = 255 - area
        cv2.imwrite(path+s+".jpg", area)

        if count == 1:
            return s + ".jpg"
        # img = 255 - img
        # cv2.imwrite("ocr/"+s+".jpg", img)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        s = generate_dataset(int(sys.argv[1]), '')
        print(s)
    else:
        generate_dataset(50000, 'ocr/')


# font                   = cv2.FONT_HERSHEY_SIMPLEX
# ft = cv2.freetype.createFreeType2()
# ft.loadFontData(fontFileName='Arial.ttf',
#                 id=0)
# bottomLeftCornerOfText = (12,20)
# fontScale              = 1
# fontColor              = (255,255,255)
# lineType               = 2

# img = np.zeros((32, 256, 3), np.uint8)
# s = "Python"
# ft.putText(
#     img=img, text=s,
#     org=bottomLeftCornerOfText,
#     fontHeight=15,
#     color=fontColor,
#     thickness=-1,
#     line_type=cv2.LINE_AA,
#     bottomLeftOrigin=True
# )
# # cv2.putText(img, s,
# #     bottomLeftCornerOfText, 
# #     font, 
# #     fontScale,
# #     fontColor,
# #     lineType)
# cv2.imwrite("Python_arial15.jpg", img)


# # extract words from image
# im = 255 - img # invert image
# im1 = im.copy()
# ret,thresh1 = cv2.threshold(im1,180,255,cv2.THRESH_BINARY_INV)
# kernel = np.ones((5,5),np.uint8)
# dilated = cv2.dilate(thresh1,kernel,iterations = 2)
# _,contours, hierarchy = cv2.findContours(dilated,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# cordinates = []
# for cnt in contours:
#     x,y,w,h = cv2.boundingRect(cnt)
#     cordinates.append((x,y,w,h))
#     #bound the images
#     cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),1)
#     area = img[y:y+h, x:x+w]

# cv2.namedWindow('BindingBox', cv2.WINDOW_NORMAL)
# cv2.imwrite('data/BindingBox4.jpg',im)