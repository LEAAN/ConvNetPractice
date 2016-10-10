from __future__ import print_function
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import RMSprop
from keras.utils import np_utils

from keras.layers import Convolution2D, MaxPooling2D, Flatten
from keras.layers import Input, LSTM, Embedding, Dense, merge, Dropout, Activation, Flatten
from keras.models import Model, Sequential
from keras.layers import TimeDistributed
import numpy as np
import set_para
import sys

sys.setrecursionlimit(10000)
modelname = raw_input("Ask me == ")
modelpath = "/home/workstation/Documents/PIROPO/"
# path = '/home/workstation/Documents/humandataset/'
batch_size = 16
nb_classes = 7
nb_epoch = 100
nb_frames = 50
img_channels = 3
img_rows, img_cols = 64, 64
# img_rows, img_cols = 32,32
ds1 = 256
ds2 = nb_classes
# classes now are number of available description words

# X_train = X_train.reshape(500, 100, 3, 32 , 32)
# Y_train = np.random.randint(100, size = (500, 100))


# Going to read 500 videos from 5 folders, read 100 videos from each folder. 
# Each video contains 100 frams.
# vocab = [Apply, Eye, Makeup, Lipstick, Basketball, Dunk, Archery]
lacab = [[0,1,2], [0,3], [4], [4,5], [6]]
num_vid_perlabel = 100

rootfolder = '/home/workstation/Documents/UCF101/UCF-101/'
x0 = set_para.readvideo(video_folder = rootfolder + 'ApplyEyeMakeup/', num_video = num_vid_perlabel)
x1 = set_para.readvideo(video_folder = rootfolder + 'ApplyLipstick/', num_video = num_vid_perlabel)
x2 = set_para.readvideo(video_folder = rootfolder + 'Basketball/', num_video = num_vid_perlabel)
x3 = set_para.readvideo(video_folder = rootfolder + 'BasketballDunk/', num_video = num_vid_perlabel)
x4 = set_para.readvideo(video_folder = rootfolder + 'Archery/', num_video = num_vid_perlabel +1)
X_test = x4[100][:nb_frames].reshape(1,nb_frames, img_channels, img_rows, img_cols)
Y_test = [0,0,0,0,0,0,1]
Y_test = np.asarray(Y_test).T.reshape((1,7))

X_train = x0 + x1 + x2+ x3+ x4[0:num_vid_perlabel]
X_train = [x[:nb_frames, :,:, :] for x in X_train]
X_train = np.rollaxis(np.asarray(X_train), 4,2)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

y_train = np.zeros((len(X_train), nb_classes))
for i in range(5):
    y_train[10*i: 10*i +10, lacab[i]] = 1

# dup = np.zeros((50, 45, 7))
# for i in range(45):
#     dup[:,i,:] = y_train
# Y_train = dup

Y_train = y_train
# this model will encode an image into a vector.
vision_model = Sequential()
vision_model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same', input_shape=(3, img_rows, img_cols)))
vision_model.add(Convolution2D(64, 3, 3, activation='relu'))
vision_model.add(MaxPooling2D((2, 2)))
vision_model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
vision_model.add(Convolution2D(128, 3, 3, activation='relu'))
vision_model.add(MaxPooling2D((2, 2)))
vision_model.add(Convolution2D(256, 3, 3, activation='relu', border_mode='same'))
vision_model.add(Convolution2D(256, 3, 3, activation='relu'))
vision_model.add(Convolution2D(256, 3, 3, activation='relu'))
vision_model.add(MaxPooling2D((2, 2)))
vision_model.add(Flatten())

# Video part
video_input = Input(shape = (nb_frames, img_channels, img_rows, img_cols))
encoded_frame_sequence = TimeDistributed(vision_model)(video_input)  # the output will be a sequence of vectors
encoded_video = LSTM(256)(encoded_frame_sequence)  # the output will be a vector
# out1 = Dense(256, activation = 'relu')(encoded_video)
# out2 = Dense(nb_classes, activation = 'softmax')(out1)
# out2 = Dense(ds2, activation = 'relu')(out1)
out2 = Dense(nb_classes, activation = 'softmax')(encoded_video)
final_model = Model(input = video_input, output = out2)

# Loss shall be changed?
rms = RMSprop(lr=0.0001)
final_model.compile(loss='categorical_crossentropy',
              optimizer=rms)

model_hist = final_model.fit(X_train, Y_train,
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          validation_data=(X_test, Y_test),
          shuffle=True)


def fake_cifar10():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    X_train = X_train.reshape(500, 100, 3, 32 , 32)
    Y_train = np.random.randint(100, size = (500, 100))
    X_test = X_test.reshape(100, 100, 3, 32, 32)
    Y_test = Y_train[:100]