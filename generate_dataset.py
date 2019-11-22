#!/usr/env/bin python3

import string
import numpy as np
import cv2
import random
from functools import reduce

img = np.zeros((50, 500, 1), np.uint8)

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,30)
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

def generate_dataset(count):
    data = set()
    while len(data) < count:
        s = pick_text()
        data.add(s.lower())
        cv2.putText(img, s,
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)
        cv2.imwrite("ocr/"+s+".jpg", img)

generate_dataset(20000)
