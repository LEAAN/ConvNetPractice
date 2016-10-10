from __future__ import print_function
from keras.datasets import cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
import numpy as np
import cPickle
import pickle
import sys
import set_para
import plot

sys.setrecursionlimit(10000)
modelpath = "/home/workstation/Documents/Cifar100/"
modelname = raw_input("Ask me == ")
batch_size = 32
nb_classes = 2
nb_epoch = 10
data_augmentation = True

img_rows, img_cols = 32, 32
img_channels = 3
(X_train, y_train), (X_test, y_test) = cifar100.load_data()
y_train = np.where ((y_train < 74), y_train, 0)
y_train = np.where ((y_train > 69), 1, 0)
y_test = np.where ((y_test < 74), y_test, 0)
y_test = np.where ((y_test > 69), 1, 0)

# X_train, y_train, X_test, y_test = set_para.slicing(X_train, y_train, X_test, y_test)
X_train, Y_train, X_test, Y_test = set_para.preprocessing(X_train, y_train, X_test, y_test, nb_classes = nb_classes)


acc = []
avg_acc = []
m_hist = []
model = Sequential()
model.add(Convolution2D(32, 5, 5, border_mode='same',
                        input_shape=(img_channels, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(Convolution2D(32, 5, 5))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# THE BEST IS: sgd = SGD(lr=0.01, decay=1e-5, momentum=0.9, nesterov=True)
lr = 0.1
decay = 1e-2
sgd = SGD(lr=lr, decay=decay, momentum=0.9, nesterov=True)
print (lr, decay)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_data=(X_test, Y_test),
              shuffle=True)
else:
    print('Using real-time data augmentation.')

    # this will do preprocessing and realtime data augmentation
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0 
        featurewise_std_normalization=False,  # divide inputs by std of the dataset 
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(X_train)

    # fit the model on the batches generated by datagen.flow()
    model_hist = model.fit_generator(datagen.flow(X_train, Y_train,
                        batch_size=batch_size),
                        samples_per_epoch=X_train.shape[0],
                        nb_epoch=nb_epoch,
                        validation_data=(X_test, Y_test))
    acc.append(model_hist.history['val_acc'])
    # avg_acc.append(np.mean(model_hist.history['val_acc'][90:]))
    m_hist.append(model_hist)

    #Final evaluation of the model
pickle.dump(m_hist, open(modelpath + modelname + "_hist.p", "wb" ))
pickle.dump(model.summary(), open(modelpath + modelname + "_summa.p", "wb" ))
pickle.dump(model, open(modelpath + modelname + "_model.p", "wb" ))
# print (avg_acc)

# scores=model.evaluate(X_test,Y_test,batch_size= 32,verbose =1,sample_weight=None)
# print("Accuracy: %.2f%%" % (scores[1]*100))
# print(model.summary())
