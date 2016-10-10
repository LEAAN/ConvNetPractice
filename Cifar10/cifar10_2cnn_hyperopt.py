from __future__ import print_function
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.utils import np_utils
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import hyperopt.pyll.stochastic
import numpy as np
import cPickle
import pickle
import sys
import set_para
import plot

sys.setrecursionlimit(10000)
modelpath = "/home/workstation/Documents/Cifar10/hyperopt/"
modelname = "1st"
global info_s
global info_h
info_s = []
info_h = []

nb_classes = 10
data_augmentation = False
img_rows, img_cols = 32, 32
img_channels = 3
nb_epoch = 100
batch_size = 32

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
# X_train, y_train, X_test, y_test = set_para.slicing(X_train, y_train, X_test, y_test)
X_train, Y_train, X_test, Y_test = set_para.preprocessing(X_train, y_train, X_test, y_test)

# Hyperopt on cnn
space4cnn = {
    "perspect_size_1" : hp.choice("perspect_size", [3,5,8,16,32]),
    "n_filter_1": hp.choice("n_filter_1", [32, 64]),
    "drop_out_1": hp.uniform("drop_out_1", 0.1, 0.7),   # 0.25
    "drop_out_2": hp.uniform("drop_out_2", 0.1, 0.7) ,  # 0.5
    "dense_1"   : hp.normal("dense_1", 512, 10)
    # "drop_out_1": hp.normal("drop_out_1", 0.25, 0.5),   # 0.25
    # "drop_out_2": hp.uniform("drop_out_2", 0.5, 0.5)   # 0.5
    # "momentum"  : hp.normal("momentum", 0.9, 0.1),
    # "decay" : hp.normal("decay", 1e-6),
}

def f(params):
    # int?
    perspect_size_2 = {
    "3": 3,
    "5": 3,
    "8": 4,
    "16": 8,
    "32": 16}
    ps1 = params["perspect_size_1"]
    ps2 = perspect_size_2[str(ps1)]
    do1 = params["drop_out_1"]
    do2 = params["drop_out_2"]
    nf1 = params["n_filter_1"]
    nf2 = 2* nf1
    # mom = params["momentum"]
    # ds1 = params["dense_1"]

    lr = 0.01
    dec = 1e-6
    mom = 0.9
    # do1 = 0.5
    # do2 = 0.25

    # do1 = 6
    # do1 = np.around(do1, decimals=2)
    # do1 = np.float32(np.clip(do1, 0.1, 0.9))
    # print (do1)
    # do2 = np.float32(np.clip(do1, 0.01, 0.99))


    # mom = max(0.9, max(0, mom))
    # ds1 = int(max(512, max(0, ds1)))
    # ds1 = 512

    # dec = max(1e-6, max(0, dec))
    
    model = Sequential()
    model.add(Convolution2D(nf1, ps1, ps1, border_mode='same',subsample = (1,1),
                            input_shape=(img_channels, img_rows, img_cols)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(do1))

    model.add(Convolution2D(nf2, ps2, ps2, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(do1))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(do2))
    model.add(Dense(nb_classes))    
    model.add(Activation('softmax'))

    sgd = SGD(lr=lr, decay=dec, momentum=mom, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    model_hist = model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_data=(X_test, Y_test),
              shuffle=True)

    scores= model.evaluate(X_test,Y_test,batch_size= 32,verbose =1,sample_weight=None)
    history = (model_hist.history)

    info_s.append(scores)
    info_h.append(history)
    return {'loss': -scores[1], 'status': STATUS_OK}

trials = Trials()
best = fmin(fn=f, space=space4cnn, algo=tpe.suggest, max_evals=100, trials=trials)

print ('best:')
print (best)
print ('trials:')
for t in trials.trials:
    # t['misc']['idxs']
    print (t['misc']['vals'])
    print ('==========================')


# cPickle.dump(model,open(modelpath + modelname + ".pkl","wb")) 
pickle.dump(info_h, open(modelpath + modelname + "_h.p", "wb" )) 
pickle.dump(info_s, open(modelpath + modelname + "_s.p", "wb" ))   
pickle.dump(trials, open(modelpath + modelname + "_trails.p", "wb" ))


# print ("%.10f" % 1e-6)

# if __name__ == "__main__":
#     run()