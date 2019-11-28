from numpy import mean
from numpy import std
from numpy import dstack
from numpy import expand_dims
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import BatchNormalization
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import to_categorical
import string
import fnmatch
import sys
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, LSTM, Reshape, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional
from keras.models import Model
from keras.activations import relu, sigmoid, softmax
from keras.backend import squeeze
import keras.backend as K
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn import metrics
from matplotlib import pyplot

char_list = string.ascii_letters+string.digits

width, height = 400, 32

# model = Sequential()
# model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(50, 500, 1)))
# model.add(MaxPool2D())
# model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
# model.add(MaxPool2D())
# model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
# model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
# model.add(MaxPool2D(pool_size=(2, 1)))
# model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
# model.add(BatchNormalization())
# model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
# model.add(MaxPool2D(pool_size=(2, 1)))
# model.add(Conv2D(512, (3, 2), activation='relu'))
# model.add(Lambda(lambda x: squeeze(x, 1)))
# model.add(Bidirectional(LSTM(500, return_sequences=True, dropout = 0.2)))
# model.add(Bidirectional(LSTM(500, return_sequences=True, dropout = 0.2)))
# model.add(Dense(len(char_list)+1, activation = 'softmax'))

# model.add(LSTM(100))
# model.add(Dropout(0.5))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(n_outputs, activation='softmax'))

# input with shape of height=32 and width=200
inputs = Input(shape=(height,width,1))

# convolution layer with kernel size (3,3)
conv_1 = Conv2D(64, (3,3), activation = 'relu', padding='same')(inputs)
# poolig layer with kernel size (2,2)
pool_1 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_1)
 
conv_2 = Conv2D(128, (3,3), activation = 'relu', padding='same')(pool_1)
pool_2 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_2)
 
conv_3 = Conv2D(256, (3,3), activation = 'relu', padding='same')(pool_2)
 
conv_4 = Conv2D(256, (3,3), activation = 'relu', padding='same')(conv_3)
# poolig layer with kernel size (2,1)
pool_4 = MaxPool2D(pool_size=(2, 1))(conv_4)
 
conv_5 = Conv2D(512, (3,3), activation = 'relu', padding='same')(pool_4)
# Batch normalization layer
batch_norm_5 = BatchNormalization()(conv_5)
 
conv_6 = Conv2D(512, (3,3), activation = 'relu', padding='same')(batch_norm_5)
batch_norm_6 = BatchNormalization()(conv_6)
pool_6 = MaxPool2D(pool_size=(2, 1))(batch_norm_6)
 
conv_7 = Conv2D(512, (2,2), activation = 'relu')(pool_6)
 
squeezed = Lambda(lambda x: K.squeeze(x, 1))(conv_7)

# bidirectional LSTM layers with units=128
blstm_1 = Bidirectional(LSTM(500, return_sequences=True, dropout = 0.2))(squeezed)
blstm_2 = Bidirectional(LSTM(500, return_sequences=True, dropout = 0.2))(blstm_1)
 
outputs = Dense(len(char_list)+1, activation = 'softmax')(blstm_2)

# model to be used at test time
act_model = Model(inputs, outputs)
act_model.load_weights('best_model.hdf5')


import cv2

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
#     fontHeight=25,
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
# cv2.imwrite("Python_arial25.jpg", img)

black_bg = False
# img = cv2.imread('cwhp.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
h, w = img.shape
if h > height or w > width:
    print(img.shape)
    # print("max shape is ({}, {})".format(width, height))
    exit(-1)

invert_img = img
if black_bg:
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
# cv2.imwrite("area.jpg", area)
img = area

if black_bg:
    img = 255 - img

# resize
# h, w = img.shape
# print(h,w)
# percent = float(height) / h
# _width = int(w * percent)
# resized = cv2.resize(img, (_width, height))
# cv2.imwrite("resized.jpg", resized)
# print(resized.shape)

# img = resized

# padding
h, w = img.shape
if h > height or w > width:
    raise Exception("width: {}, height: {}".format(w, h))
if h < height:
    add_zeros = np.ones((height-h, w))*255
    img = np.concatenate((img, add_zeros))
if w < width:
    add_zeros = np.ones((height, width-w))*255
    img = np.concatenate((img, add_zeros), axis=1)

# bg = np.zeros((50, 500), np.uint8)
# bg[:img.shape[0], :img.shape[1]] = img
# img = bg
# img = 255 - img
# cv2.imwrite("bg.jpg", img)
img = img/255.
img = img.reshape(height, width, 1)
# valid_img = np.array([img])
valid_img = np.expand_dims(img, axis=0)
prediction = act_model.predict(valid_img)
out = K.get_value(K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0])*prediction.shape[1],
                         greedy=True)[0][0])

for x in out:
    # print(valid_orig_txt[i])
    for p in x:  
        if int(p) != -1:
            print(char_list[int(p)], end = '')
    print('\n')
