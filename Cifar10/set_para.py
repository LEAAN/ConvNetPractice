import os
import re
import csv
import numpy as np
import cv2
import scipy.misc
import functools
import locale
from keras.utils import np_utils
from sklearn.cross_validation import train_test_split



def preprocessing(X_train = None, y_train = None, X_test = None, y_test = None,
  nb_classes = 10, x = None, y = None, split = False, split_percent = 0.75, 
  to_categorical = True):
  if y_train is not None:
    y_train = np.reshape(y_train,(len(y_train),1))
    y_test = np.reshape(y_test, (len(y_test),1))
  # # convert class vectors to binary class matrices
  if split:
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)
  if to_categorical:
    y_train = np_utils.to_categorical(np.asarray(y_train), nb_classes)
    y_test = np_utils.to_categorical(np.asarray(y_test), nb_classes) 
  X_train = X_train.astype('float32')
  X_test = X_test.astype('float32')
  # Normalizing
  X_train /= 255
  X_test /= 255
  # Submean
  X_train -= np.mean(X_train, axis= 0)
  X_test -= np.mean(X_test, axis= 0)
  return X_train, y_train, X_test, y_test
  
def slicing(X_train, y_train, X_test, y_test, x_cut= 100, y_cut= 100):
  # Return only first 100 samples of X and y
  X_train = X_train[0:x_cut,:]
  X_test = X_test[0:x_cut,:]
  y_train = y_train[0:y_cut,:]
  y_test = y_test[0:y_cut,:]
  print('X_train shape:', X_train.shape)
  print(X_train.shape[0], 'train samples')
  print(X_test.shape[0], 'test samples')
  return X_train, y_train, X_test, y_test

def readimage(labelist = None, label = None, x = None, color = True, resize = True, cutsize = True, 
    rew = 18, reh = 36, cuw = 18, cuh = 36, printname = False, f = None,
    path = None, cutborder = False):
    if label is None:
      label = 0
        # label = int(raw_input('Where is the label ==? 1 or 0 '))
    print ('read and label all images from folder: \n'), str(path), '\n', 'label of this folder is ' + str(label)
    pics = []
   # Either list of filenames or folder name is given.
   # If folder name is provided, list of filenames in this folder will be created
    if f is None:
      f = []
      for (dirpath, dirnames, filenames) in os.walk(path):
        f.extend(filenames)
      ff = []
      for name in f:
        if (re.search(r'jpg', name)): ff.append(path + name)
        if (re.search(r'pgm', name)): ff.append(path + name)
      f = ff
      f = sorted(f, key=functools.cmp_to_key(locale.strcoll))
   # If filenames are provided, directly read in images for all names in filenames
   # How can I change this ugly part of code ==
    if resize:
      print ('resize, no cut')
      for name in f:
        try:
            image = cv2.imread(name, cv2.CV_LOAD_IMAGE_COLOR)
            image = cv2.resize(image, (rew, reh)) 
            pics.append(image)
        except:
            print(name)
    elif cutsize: 
      print ('cut, no resize')
      for name in f:
        try:
            image = cv2.imread(name, cv2.CV_LOAD_IMAGE_COLOR)
            # image = cv2.imread(path + name)
            w = np.random.randint(image.shape[1]-cuw)
            h = np.random.randint(image.shape[0]-cuh) 
            pics.append(image[h:h+cuh, w:w+cuw, :])
        except:  pass
    elif cutborder:
      print ('cut border, and resize')
      for name in f:
        try:
            # Keep the middle part of an image, and then resize
            image = cv2.imread(name, cv2.CV_LOAD_IMAGE_COLOR)
            image = image[:, (image.shape[1]-image.shape[0])/2:(image.shape[1]+image.shape[0])/2,:]
            image = cv2.resize(image, (rew, reh)) 
            pics.append(image)
        except:
            print(name)
    else:
      print ('not cut, no resize')
      try:
        for name in f:
          # image = cv2.imread(name, cv2.CV_LOAD_IMAGE_COLOR)
          # image = cv2.imread(path+ name)
          image = cv2.imread(name)
          pics.append(image)
      except: pass
    print ((np.asarray(pics).shape))
    print ('%s %d %s %d' % ('len of filenames is: ', len(f), 'len of pics is: ', len(pics)))
    pics = np.rollaxis(np.asarray(pics, dtype = float),3,1)
  # Cut size for large test pics
    # if cutsize:
    #   w = np.random.randint(pics.shape[2]-cuw)
    #   h = np.random.randint(pics.shape[3]-cuh)
    #   pics = pics[:,:, h:h+cuh, w:w+cuw]
    # print (pics.shape)
    # Make labels
    y = np.repeat(label, len(pics))
    y = np.reshape(y, (len(y),1))
    y = np.concatenate((1-y, y), axis = 1)

    # append labellist
    if labelist is not None:
        print ('label in this foler is appended to label list')
        labelist = np.append(labelist, y, axis = 0)
        x = np.append(x, pics, axis = 0)
    else:
        print('A label list will be created')
        labelist = y
        x = pics
    return (x, labelist)


def lookfor(keyword = 'omni_1A', rootpath = '/home/workstation/Documents/PIROPO/', imagelist = None):
  # imagelist extend new pics under keyword folder to existing pics dict
  if imagelist is None:
    imagelist = []
  labelpath = rootpath + keyword + '/Ground_Truth_Annotations/'
  all_gt = [name for name in os.listdir(labelpath) if (os.path.isfile(labelpath + name) and re.search(r'groundTruth.', name, re.M))]
  available_gt = [name[len('groundTruth' + keyword)+1: len(name)-len('.csv')] for name in all_gt]
  for name in available_gt:
    print (name)
    labelname = 'groundTruth_' + keyword.replace('_', '') + '_' + name + '.csv'
    picpath = rootpath + keyword + '/' + keyword.replace('_', '') + '_' + name + '/'
    # Include all file names in the picpath folder
    f = []
    for (dirpath, dirnames, filenames) in os.walk(picpath):
      f.extend(filenames)
    ff = []
    # Remove filenames that do no end with 'jpg'
    # ff includes picpath
    for name in f:
      if (re.search(r'jpg', name)): ff.append(picpath + name)
    f = ff
    f = sorted(f, key=functools.cmp_to_key(locale.strcoll))

    picdict = imagetodict(labelpath = labelpath, labelname = labelname, picpath = picpath, filenames = f)
    imagelist.extend(picdict)
  return imagelist 


def imagetodict(labelpath = '/home/workstation/Documents/PIROPO/omni_1A/Ground_Truth_Annotations/',
  labelname = 'groundTruth_omni1A_training_seat1.csv', 
  picpath = '/home/workstation/Documents/PIROPO/omni_1A/omni1A_training_seat1/', label_of_the_folder = None,
  imagelist = None, filenames = None):
  if imagelist is None:
    imagelist = []

  if label_of_the_folder is not None:
    f = []
    for (dirpath, dirnames, filenames) in os.walk(picpath):
      f.extend(filenames)

    ff = []
    for name in f:
      if (re.search(r'jpg', name)): ff.append(picpath + name)
    f = ff
    f = sorted(f, key=functools.cmp_to_key(locale.strcoll))
    filenames = f

    pics, labels = readimage(path = picpath,label = label_of_the_folder,
     cutborder = True, resize = True, cutsize = False, rew = 64, reh = 64, color = True, f = filenames)
    labels = labels[:,1]

    info = np.zeros(len(filenames))

  else:  
    # labelname is a csv file of pics info
    info = []
    with open(labelpath+ labelname, 'rb') as csvfile:
      reader = csv.reader(csvfile)
      for row in reader:
        info.append(row)

    labels = np.asarray(info, dtype = int)[:,1]
    labels = np.asarray(labels>0, dtype = int)
    # print (labels[:100])
    pics, randomy = readimage(path = picpath, cutborder = True, resize = True, cutsize = False, rew = 64, reh = 64, color = True, f = filenames)
  
  print ('%s %s' % ('shape of labels is...', str(labels.shape)))
  print ('Number of readed in pics is' + str(len(imagelist)))
  keys = ['filename', 'info',  'picarray', 'label']
  picdict = [dict(zip(keys,row)) for row in zip(filenames, info, pics, labels)]
  # allinarray = zip(filenames, info, pics, labels)
  # return picdict
  imagelist.extend(picdict)
  return imagelist

# image = cv2.imread(path + name, cv2.CV_LOAD_IMAGE_COLOR)

# # read 50 files from a folder
# def readvideo (video_folder = None, num_video = 50):
#   # Get 50 names of videos in video_folder
#   video_names = []
#   for (dirpath, dirnames, filenames) in os.walk(video_folder):
#     video_names.extend(filenames)
#   # video_names = video_names[:num_video]

#   # Get array for each name
#   video_array = []
#   for name in video_names:  
#     print (name)
#     video = []
#     cap = cv2.VideoCapture(video_folder + name)

#     while(cap.isOpened()):
#         ret, frame = cap.read()
#         if frame is None: break
#         if len(frame)> 50:
#           frame = cv2.resize(frame,(64,64), interpolation = cv2.INTER_CUBIC)
#           video.append(frame)
#     cap.release()
#     # cv2.destroyAllWindows()
#     video_array.append(np.asarray(video))
#   return video_array


# read 50 files from a folder
def readvideo (video_folder = None, num_video = 50):
  # Get names of videos in video_folder
  video_names = []
  for (dirpath, dirnames, filenames) in os.walk(video_folder):
    video_names.extend(filenames)

  # Get array for each name
  pointer = 0
  video_array = []
  while (num_video):
    name = video_names[pointer]
    pointer += 1
    video = []
    cap = cv2.VideoCapture(video_folder + name)
    while(cap.isOpened()):
      ret, frame = cap.read()
      if frame is None: break
      frame = cv2.resize(frame,(64,64), interpolation = cv2.INTER_CUBIC)
      video.append(frame)
    cap.release()
    if (len(video) > 49):
      video_array.append(np.asarray(video))
      num_video -= 1
      print (name)
  return video_array

