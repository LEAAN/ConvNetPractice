from __future__ import print_function
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from sklearn.cross_validation import train_test_split
import pandas
from scipy import misc
import numpy as np
import pickle
import sys
import set_para
import evalu
import plot
import shutil

sys.setrecursionlimit(10000)
modelname = raw_input("Ask me == ")
modelpath = "/home/workstation/Documents/PIROPO/"
# path = '/home/workstation/Documents/humandataset/'
batch_size = 32
nb_classes = 2
nb_epoch = 100
data_augmentation = True
img_channels = 3
img_rows, img_cols = 64, 64
# img_rows, img_cols = 32, 64



def load_google():
    img_rows, img_cols = 32, 64
    x, y = set_para.readimage(label = 1, cutsize = False, resize = True, reh = img_rows, rew = img_cols,
        path = '/home/workstation/Documents/humandataset/beforethermal/1/pedestrain/')
    x, y = set_para.readimage(label = 1, cutsize = False, resize = True,
        x = x, labelist = y, reh = img_rows, rew = img_cols,
        path = '/home/workstation/Documents/humandataset/beforethermal/1/people_on_the_street/')
    x, y = set_para.readimage(label = 1, cutsize = False, resize = True,
        x = x, labelist = y, reh = img_rows, rew = img_cols,
        path = '/home/workstation/Documents/humandataset/beforethermal/1/traffic_warden/')
    x, y = set_para.readimage(label = 0, cutsize = False, resize = True,
        x = x, labelist = y, reh = img_rows, rew = img_cols,
        path = '/home/workstation/Documents/humandataset/beforethermal/0/empty_city_road/')
    x, y = set_para.readimage(label = 0, cutsize = False, resize = True,
        x = x, labelist = y, reh = img_rows, rew = img_cols,
        path = '/home/workstation/Documents/humandataset/beforethermal/0/empty_street/')
    x, y = set_para.readimage(label = 0, cutsize = False, resize = True,
        x = x, labelist = y, reh = img_rows, rew = img_cols,
        path = '/home/workstation/Documents/humandataset/beforethermal/0/trees_street_empty/')
    X_train, Y_train, X_test, Y_test = set_para.preprocessing(x = x, y = y, split = True, nb_classes = nb_classes,
        to_categorical = False)
    return X_train, Y_train, X_test, Y_test

def load_beforethermal():
    # Seems these are random google pics resized into (2446, 3, 64, 32) ones
    m = pickle.load(open( '/home/workstation/Documents/humandataset/beforethermal/' +  'beforethermal_x' + '.p', 'rb'))
    y = pickle.load(open( '/home/workstation/Documents/humandataset/beforethermal/' +  'beforethermal_y' + '.p', 'rb'))
    X_train, Y_train, X_test, Y_test = set_para.preprocessing(x = x, y = y, split = True, nb_classes = nb_classes)
    return X_train, Y_train, X_test, Y_test

def load_stereo():
    img_rows, img_cols = 36,18
    x, y = set_para.readimage(label = 1, cutsize = False, resize = False, 
        path = '/home/workstation/Documents/humandataset/pedestrain_benchmarks/Stereo/TrainingData/Pedestrians/18x36/')
    x, y = set_para.readimage(label = 0, cutsize = True, resize = False,
        x = x, labelist = y, reh = 36, rew = 18,
        path = '/home/workstation/Documents/humandataset/pedestrain_benchmarks/Stereo/TrainingData/NonPedestrians/')
    X_train, Y_train, X_test, Y_test = set_para.preprocessing(x = x, y = y, split = True, nb_classes = nb_classes,
        to_categorical = False)
    return X_train, Y_train, X_test, Y_test

def load_piropo():
    # img_rows, img_cols = 128, 128
    ilist = set_para.lookfor(keyword = 'omni_1A', imagelist = None)
    ilist = set_para.lookfor(keyword = 'omni_1B', imagelist = ilist)
    ilist = set_para.lookfor(keyword = 'omni_2A', imagelist = ilist)
    ilist = set_para.lookfor(keyword = 'omni_3A', imagelist = ilist)
    frame = pandas.DataFrame.from_dict(ilist)

    train, test = train_test_split(frame, random_state = 1)
    # train = pandas.DataFrame.as_matrix(train)
    # test = pandas.DataFrame.as_matrix(test)
    Y_train = np.asarray(train.label.tolist())
    Y_test = np.asarray(test.label.tolist())
    X_train = np.asarray(train.picarray.tolist())
    X_test = np.asarray(test.picarray.tolist())

    X_train, Y_train, X_test, Y_test = set_para.preprocessing(X_train = X_train, y_train = Y_train, X_test = X_test, y_test = Y_test
        , nb_classes = nb_classes, to_categorical = True)
    # x = []
    # y = []
    # for item in ilist:
    #     x.append(item['picarray'])
    #     y.append(item['label'])
    # x = np.asarray(x)
    # y = np.asarray(y)
    # y = np.reshape(y, (len(y),1))
    # evalu.savexy(x,y, modelname, modelpath)
    # x, y = evalu.loadxy(modelname, modelpath)

    # X_train, Y_train, X_test, Y_test = set_para.preprocessing(x = x, y = y, split = True, nb_classes = nb_classes,
    #     to_categorical = True)
    evalu.savexy(X_train, Y_train, 'piropo_train', modelpath)
    evalu.savexy(X_test, Y_test, 'piropo_test', modelpath)
    evalu.savexy(train, test, 'dataframe', modelpath)
    return X_train, Y_train, X_test, Y_test, train, test


def load_traintest():
    X_train, Y_train = evalu.loadxy('piropo_train', modelpath)
    X_test, Y_test = evalu.loadxy('piropo_test', modelpath)
    train, test = evalu.loadxy('dataframe', modelpath)
    # X_train, Y_train, X_test, Y_test = set_para.preprocessing(X_train = X_train, y_train = Y_train, X_test = X_test, y_test = Y_test
    #     , nb_classes = nb_classes, to_categorical = True)
    return X_train, Y_train, X_test, Y_test, train, test


X_train, Y_train, X_test, Y_test, train, test = load_piropo()
# X_train, Y_train, X_test, Y_test, train, test = load_traintest()

avg_acc = []
m_hist = []
model = Sequential()
cnn1 = model.add(Convolution2D(32, 5, 5, border_mode='same',
                        input_shape=(img_channels, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(Convolution2D(32, 5, 5))
model.add(Activation('relu'))
model.add(Convolution2D(32, 5, 5))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.176))

model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.303))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.143))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
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
    acc =model_hist.history['val_acc']
    avg_acc.append(np.mean(model_hist.history['val_acc'][len(acc)-10:]))
    m_hist.append(model_hist)


print (avg_acc)

Y_predict = model.predict_classes(X_test, batch_size=batch_size, verbose=0)
# Y_predict = 1-Y_predict
plot.confusion(y_true = Y_test[:,1],y_pred = Y_predict , savename = modelname, normalize = False, title = modelname)




# Final evaluation of the model
# pickle.dump(m_hist, open(modelpath + modelname + "_hist.p", "wb" ))
# pickle.dump(model.summary(), open(modelpath + modelname + "_summa.p", "wb" ))
pickle.dump(model, open(modelpath + modelname + "_model.p", "wb" ))


# for i in range(len(X_test)):
#     if (Y_predict[i]==1 & int(Y_test[i][0]) ==0 ):
#         misc.imsave('/home/workstation/Documents/PIROPO/p1t0/' 
#             + str(i) + '.png',  np.rollaxis(X_test[i],0,3))



# for i in range(len(X_test)):
#     if (Y_predict[i]==0 & int(Y_test[i][0]) == 1 ):
#         print (i)
#         scipy.misc.imsave('/home/workstation/Documents/PIROPO/p0t1/'+
#          str(i) + 'png',  np.rollaxis(X_test[i],0,3))

# plot.weight(model=model, modelname = modelname)

def findimage(findpath, Y_test, Y_predict):
    namelist = []
    # Add names of wrongly predicted pics
    for i in range(len(Y_test)):
        if (Y_predict[i]==1 & int(Y_test[i][0]) ==0 ):
            namelist.extend(test.filename[i:i+1].tolist())
    # Copy wrongly predicte pics to folder
    for name in namelist:
        for root, dirs, files in os.walk(modelpath):
            if name in files:
                # shutil.copy(os.path.join(root, name), '/home/workstation/Documents/p1t0/' )
                try:
                    shutil.copy(os.path.join(root, name), '/home/workstation/Documents/p1t0/' )
                except: print (name)


#p0t1

    zero = test[test.label ==0]
    for index in zero.index.values:
        namelist.append(test.filename[index])

    namelist = []
    # Add names of no-people images
    for i in range(len(Y_test)):
        if (Y_test[i][0] ==1 ):
            # namelist.extend(test.filename.loc[i])
            namelist.extend(test.filename[i:i+1].tolist())
    # Copy wrongly predicte pics to folder
    for name in namelist:
        for root, dirs, files in os.walk(modelpath):
            if name in files:
                try:
                    shutil.copy(os.path.join(root, name), '/home/workstation/Documents/Y_test0/' )
                except: print (name)


    Y_predict_proba = model.predict(X_test, batch_size=batch_size, verbose=0)

    namelist = []
    proba_p0t1 = []
    # Add names of wrongly predicted pics
    for i in range(len(Y_test)):
        if ((Y_predict[i]==0) and (int(Y_test[i][1]) ==1) ):
        # if (Y_predict[i]==1):
            proba_p0t1.append(Y_predict_proba[i])
            namelist.extend(test.filename[i:i+1].tolist())
    # Copy wrongly predicte pics to folder
    for name in namelist:
        # shutil.copy(os.path.join(root, name), '/home/workstation/Documents/p1t0/' )
        try:
            shutil.copy(name, '/home/workstation/Documents/p0t1/' )
        except: print (name)
    proba_p0t1 = np.asarray(proba_p0t1)
    # Sort by p_without 
    proba_p0t1 = proba_p0t1[np.argsort(proba_p0t1[:,0])]
    plt.clf()
    plt.plot(np.arange(len(proba_p0t1)), proba_p0t1[:,0], 'g-', label = 'p0')
    plt.plot(np.arange(len(proba_p0t1)), proba_p0t1[:,1], '+',  label = 'p1')
    plt.savefig('try')

