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
import csv
import sys

from keras import backend as K
from keras.engine.topology import Layer

class MyLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    # def build(self, input_shape):
    #     input_dim = input_shape[1]
        # initial_weight_value = np.random.random((input_dim, output_dim))
        # self.W = K.variable(initial_weight_value)
        # self.trainable_weights = [self.W]

    def call(self, x, mask=None):
        return K.mean(x, axis = 1)
        # print x.shape
        # return x.mean(axis = 0)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim)

sys.setrecursionlimit(10000)
modelname = raw_input("Ask me == ")
modelpath = "/home/workstation/Documents/PIROPO/"
gru_on = False
batch_size = 16
nb_people = 6
nb_epoch = 2
nb_frames = 10
img_channels = 3
img_rows, img_cols = 64, 64

ds1 = 256

# Read sequences from piropo
filepath = "/home/workstation/Documents/PIROPO/omni_1A/"
filename = 'groundTruth_omni1A_training.cvs'
x, fakey = set_para.readimage(path = filepath + "omni1A_training/",
 cutborder = True, rew = 64, reh = 64, resize = False, cutsize = False)
info = []
with open('/home/workstation/Documents/PIROPO/omni_1A/Ground_Truth_Annotations/groundTruth_omni1A_training.csv', 'rb') as csvfile:
  reader = csv.reader(csvfile)
  for row in reader:
    # row = row[1:] + (11-len(row))*['0']
    info.append(row[1:])
labels = np.asarray(info, dtype = int)
y = labels

x = x[:10960]
y = y[:len(x)]
x = x.reshape((1096, 10, 3, 64, 64))
y = y.reshape((1096, 10, 6))
y = y.mean(axis = 1)
X_train, Y_train, X_test, Y_test = set_para.preprocessing(x = x, y = y, split = True, to_categorical = False)

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


if not gru_on:
  # Video without GRU
  sequence_input = Input(shape = (nb_frames, img_channels, img_rows, img_cols))
  encoded_video = TimeDistributed(vision_model)(sequence_input)
  out1 = TimeDistributed(Dense(256, activation = 'relu'))(encoded_video)
  out2 = TimeDistributed(Dense(nb_people, activation = 'relu'))(out1)
  out3 = MyLayer(output_dim = (nb_people))(out2)
  final_model = Model(input = sequence_input, output = out3)

else: 
  # Video part
  video_input = Input(shape = (nb_frames, img_channels, img_rows, img_cols))
  encoded_frame_sequence = TimeDistributed(vision_model)(video_input)  # the output will be a sequence of vectors
  encoded_video = LSTM(256)(encoded_frame_sequence)  # the output will be a vector

  out2 = Dense(nb_people, activation = 'softmax')(encoded_video)
  final_model = Model(input = video_input, output = out2)

# Loss shall be changed?
rms = RMSprop(lr=0.001)
final_model.compile(
  loss = 'poisson',
  # loss='categorical_crossentropy',
              optimizer=rms)

model_hist = final_model.fit(X_train, Y_train,
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          validation_data=(X_test, Y_test)
          # ,shuffle=True
          )

