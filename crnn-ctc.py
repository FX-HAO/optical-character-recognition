#!/usr/env/bin python3

import string
import fnmatch
import os
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

char_list = string.ascii_letters+string.digits

def encode_to_labels(txt):
    # encoding each output word into digits
    dig_lst = []
    for index, char in enumerate(txt):
        try:
            dig_lst.append(char_list.index(char))
        except:
            print(char)
        
    return dig_lst

# lists for training dataset
training_img = []
training_txt = []
train_input_length = []
train_label_length = []
orig_txt = []

valid_img = []
valid_txt = []
valid_input_length = []
valid_label_length = []
valid_orig_txt = []

max_label_len = 0

i = 1
flag = 0
path = 'ocr'

width, height = 400, 32

for root, dirnames, filenames in os.walk(path):
    for f_name in fnmatch.filter(filenames, '*.jpg'):
        img = np.array(load_img(os.path.join(root, f_name), color_mode='grayscale'))
        # img = 255 - img # invert image

        # convert each image of shape (50, 500, 1)
        # w, h = img.shape
        h, w = img.shape
        if h > height or w > width:
            print(f_name, img.shape)
            continue
        if h < height:
            add_zeros = np.ones((height-h, w))*255
            img = np.concatenate((img, add_zeros))
        if w < width:
            add_zeros = np.ones((height, width-w))*255
            img = np.concatenate((img, add_zeros), axis=1)
        img = np.expand_dims(img, axis = 2)

        # Normalize each image
        img = img/255.

        # get the text from the image
        txt = f_name.split('.')[0]

        # compute maximum length of the text
        if len(txt) > max_label_len:
            max_label_len = len(txt)
        
        if i%10 == 0:     
            valid_orig_txt.append(txt)   
            valid_label_length.append(len(txt))
            valid_input_length.append(49)
            valid_img.append(img)
            valid_txt.append(encode_to_labels(txt))
        else:
            orig_txt.append(txt)   
            train_label_length.append(len(txt))
            train_input_length.append(49)
            training_img.append(img)
            training_txt.append(encode_to_labels(txt)) 

        if i == 20000:
            flag = 1
            break
        i+=1
    if flag == 1:
        break

train_padded_txt = pad_sequences(training_txt, maxlen=max_label_len, padding='post', value = len(char_list))
valid_padded_txt = pad_sequences(valid_txt, maxlen=max_label_len, padding='post', value = len(char_list))


from numpy import mean
from numpy import std
from numpy import dstack
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
from matplotlib import pyplot

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

labels = Input(name='the_labels', shape=[max_label_len], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    # y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

 
loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([outputs, labels, input_length, label_length])
model = Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out)

model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer = 'adam')
 
filepath="best_model.hdf5"
checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
early = EarlyStopping(monitor='loss', min_delta=0.001, patience=3, mode='auto')
callbacks_list = [early, checkpoint]

training_img = np.array(training_img)
train_input_length = np.array(train_input_length)
train_label_length = np.array(train_label_length)

valid_img = np.array(valid_img)
valid_input_length = np.array(valid_input_length)
valid_label_length = np.array(valid_label_length)
 
model.summary()

model.fit(
    x=[training_img, train_padded_txt, train_input_length, train_label_length], 
    y=np.zeros(45000), 
    batch_size=256, 
    epochs = 100, 
    validation_data = ([valid_img, valid_padded_txt, valid_input_length, valid_label_length], [np.zeros(2000)]), 
    verbose = 1, 
    callbacks = callbacks_list
)

# load the saved best model weights
act_model.load_weights('best_model.hdf5')

# predict outputs on validation images
prediction = act_model.predict(valid_img)

# use CTC decoder
out = K.get_value(K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0])*prediction.shape[1],
                         greedy=True)[0][0])

# see the results
i = 0
for x in out:
    print(valid_orig_txt[i])
    for p in x:  
        if int(p) != -1:
            print(char_list[int(p)], end = '')       
    print('\n')
    i+=1

text_len = 10

valid_padded_txt[valid_padded_txt == 62] = -1
eval_valid_padded_txt = np.sum(valid_padded_txt[:, :text_len], axis=1)
eval_out = np.sum(out[:, :text_len], axis=1)
print("Accuracy:", metrics.accuracy_score(eval_valid_padded_txt, eval_out))
