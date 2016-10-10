from __future__ import print_function
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Convolution2D, MaxPooling2D, Input 
from keras.optimizers import SGD
import numpy as np
import pickle
import sys
import set_para
import evalu
import plot

sys.setrecursionlimit(10000)
# modelpath = "/home/workstation/Documents/humandataset/animals/"
# path = '/home/workstation/Documents/humandataset/'
modelname = raw_input("Ask me == ")

batch_size = 32
nb_classes = 2
nb_epoch = 100
data_augmentation = True

img_rows, img_cols = 36,18
img_channels = 3


x, y = set_para.readimage(label = 1, cutsize = False, resize = False, 
    path = '/home/workstation/Documents/humandataset/pedestrain_benchmarks/Stereo/TrainingData/Pedestrians/18x36/')
x, y = set_para.readimage(label = 0, cutsize = True, resize = False,
    x = x, labelist = y, reh = 36, rew = 18,
    path = '/home/workstation/Documents/humandataset/pedestrain_benchmarks/Stereo/TrainingData/NonPedestrians/')
X_train, Y_train, X_test, Y_test = set_para.preprocessing(x = x, y = y, split = True, nb_classes = nb_classes,
    to_categorical = False)


acc = []
avg_acc = []
m_hist = []
inputs = Input(shape = (img_channels, img_rows, img_cols))

cn1     = Convolution2D(32,5,5, border_mode = 'same')(inputs)
relu1   = Activation('relu')(cn1)
pool1   = MaxPooling2D(pool_size = (2,2))(relu1)
do1     = Dropout(0.2)(pool1)

cn2     = Convolution2D(64,3,3, border_mode = 'same')(do1)
relu2   = Activation('relu')(cn2)
pool2   = MaxPooling2D(pool_size = (2,2))(relu2)
do2     = Dropout(0.3)(pool2)

# flat = Flatten(())
re      = Reshape((2304,))(do2)
ds1     = Dense(512)(re)
relu3   = Activation('relu')(ds1)
do3     = Dropout(0.15)(relu3)
ds2     = Dense(nb_classes)(do3)
soft    = Activation('softmax')(ds2)

model = Model(input=inputs, output=soft)

# larger decay? 
sgd = SGD(lr=0.01, decay=1e-4, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

if not data_augmentation:
    print('Not using data augmentation.')
    model_hist = model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_data=(X_test, Y_test),
              shuffle=True)
    acc.append(model_hist.history['val_acc'])
    avg_acc.append(np.mean(model_hist.history['val_acc'][90:]))
    m_hist.append(model_hist)
    
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
    # if run this part again, the result should be epoch += epoch right?
    model_hist = model.fit_generator(datagen.flow(X_train, Y_train,
                        batch_size=batch_size),
                        samples_per_epoch=X_train.shape[0],
                        nb_epoch=nb_epoch,
                        validation_data=(X_test, Y_test))
    acc.append(model_hist.history['val_acc'])
    avg_acc.append(np.mean(model_hist.history['val_acc'][len(acc)-10:]))
    m_hist.append(model_hist)

    #Final evaluation of the model
# pickle.dump(m_hist, open(modelpath + modelname + "_hist.p", "wb" ))
# pickle.dump(model.summary(), open(modelpath + modelname + "_summa.p", "wb" ))
print (avg_acc)

# Y_predict = model.predict_classes(X_test, batch_size=batch_size, verbose=0)
# plot.confusion(Y_predict, Y_test[:,1], savename = modelname, normalize = True, title = modelname)


# scores=model.evaluate(X_test,Y_test,batch_size= 32,verbose =1,sample_weight=None)
# print("Accuracy: %.2f%%" % (scores[1]*100))

# plot.weight(model=model, modelname = modelname)
''' 
See what featrues/ outputs we have
'''
do2_output = Model(input = inputs, output = do2)
X_output = do2_output.predict(X_test)

cn1_output = Model(input = inputs, output = cn1)
X_output_cn1 = cn1_output.predict(X_test)

input1 = Model(input = inputs, output = inputs)
X_input1 = input1.predict(X_test)
layeroutput(X_output_cn1[0], 'Stereo_cn1_output_1', nrow=8, ncol = 4)

do1_output = Model(input = inputs, output = do1)
X_output_do2 = do2_output.predict(X_test)
layeroutput(X_output_do2[0], 'Stereo_do1_output_1', nrow=8, ncol = 8)


w0 = model.layers[1].get_weights()[0]

plot.layeroutput(X_output[0], 'Stereo_cn2_out_1')
plot.layeroutput(X_input1[0], 'Stereo_input_1', nrow=1, ncol = 3)