from __future__ import print_function
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import Callback
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import hyperopt.pyll.stochastic
from time import gmtime, strftime
import numpy as np
import cPickle
import pickle
import sys
import set_para
import plot


class EarlyStoppingByLossVal(Callback):
    def __init__(self, monitor='val_acc', value=0.5, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("It raise warning here...Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current < self.value:
            if self.verbose > 0:
                print("What's the stuff...Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True


callbacks = [
    EarlyStoppingByLossVal(monitor='val_acc', value=0.5, verbose=1)
    # ModelCheckpoint(kfold_weights_path, monitor='val_loss', save_best_only=True, verbose=0),
]

sys.setrecursionlimit(10000)
modelpath = "/home/workstation/Documents/Cifar10/best/hyper_best/"
print (modelpath)
modelname = raw_input("Ask me == ")
global info_s
global info_h
info_s = []
info_h = []

nb_classes = 10
data_augmentation = True
img_rows, img_cols = 32, 32
img_channels = 3
nb_epoch = 100
batch_size = 32

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
# X_train, y_train, X_test, y_test = set_para.slicing(X_train, y_train, X_test, y_test)
X_train, Y_train, X_test, Y_test = set_para.preprocessing(X_train, y_train, X_test, y_test)


# Hyperopt on cnn
space4cnn = {
    # "drop_out_1": hp.normal("drop_out_1", 0.176, 0.10),  # 0.25
    # "drop_out_2": hp.normal("drop_out_2", 0.30, 0.10),   # 0.5
    # "drop_out_3": hp.normal("drop_out_3", 0.143, 0.10)   # 0.5

    # "drop_out_1": hp.normal("drop_out_1", 0.25, 0.10),  # 0.25
    # "drop_out_2": hp.normal("drop_out_2", 0.25, 0.10),   # 0.5
    # "drop_out_3": hp.normal("drop_out_3", 0.5, 0.20)   # 0.5
    # "momentum"  : hp.normal("momentum", 0.9, 0.1),
    # "decay" : hp.normal("decay", 1e-6),
    # "n_filter_1": hp.choice("n_filter_1", [32, 64]),
    # "perspect_size_1" : hp.choice("perspect_size", [3,5,8,16,32])
    "dense" : hp.normal("dense", 512, 150),
}

print (str(space4cnn.keys()) + ' changed' + strftime("%Y-%m-%d %H:%M:%S", gmtime()))
def f(params):
    # do1 = params["drop_out_1"]
    # do2 = params["drop_out_2"]
    # do3 = params["drop_out_3"]
    # if (do1<0):
    #     do1 = 0.12
    # if (do2<0):
    #     do2 = 0.30
    # if (do3<0):
    #     do3 = 0.20

    # print ("1st: ",do1,"2nd: ",do2, "3rd: ", do3)    
    do1 = 0.12
    do2 = 0.3
    do3 = 0.143
    ds = int(params['dense'])
    print ('ds: ', ds)

    model = Sequential()
    model.add(Convolution2D(32, 5, 5, border_mode='same',
                            input_shape=(img_channels, img_rows, img_cols)))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 5, 5))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(do1))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(do2))

    model.add(Flatten())
    model.add(Dense(ds))
    model.add(Activation('relu'))
    model.add(Dropout(do3))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.01, decay=1e-5, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    print('Using real-time data augmentation.')
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

    datagen.fit(X_train)
    model_hist = model.fit_generator(datagen.flow(X_train, Y_train,
              batch_size=batch_size), samples_per_epoch=X_train.shape[0],
              nb_epoch=nb_epoch, validation_data=(X_test, Y_test),
             verbose = 1, callbacks = callbacks)

    scores= model.evaluate(X_test,Y_test,batch_size= 32,verbose =1,sample_weight=None)
    history = (model_hist.history)
    avg_acc = np.mean(model_hist.history['val_acc'][nb_epoch-20:])

    info_s.append(scores)
    info_h.append(history)
    return {'loss': -avg_acc, 'status': STATUS_OK}


trials = Trials()
best = fmin(fn=f, space=space4cnn, algo=tpe.suggest, max_evals=25, trials=trials)

print ('best:')
print (best)



# cPickle.dump(model,open(modelpath + modelname + ".pkl","wb")) 
pickle.dump(info_h, open(modelpath + modelname + "_h.p", "wb" )) 
pickle.dump(info_s, open(modelpath + modelname + "_s.p", "wb" ))   
pickle.dump(trials, open(modelpath + modelname + "_trails.p", "wb" ))


# print ("%.10f" % 1e-6)

