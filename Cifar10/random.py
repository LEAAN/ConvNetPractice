import evalu
from keras.datasets import cifar10
import numpy as np
import pickle
import set_para
import evalu


modelpath = "/home/workstation/Documents/Cifar10/"
modelname = "keras_example_aug_submean"
modelname = "model"
model = evalu.open_file(modelpath, modelname,"pkl")

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train, Y_train, X_test, Y_test = set_para.preprocessing(X_train, y_train, X_test, y_test)

scores=model.evaluate(X_test,Y_test,batch_size= 32,verbose =1,sample_weight=None)
print("Accuracy: %.2f%%" % (scores[1]*100))
print(model.summary())


# plot.firstweight(model, modelname, modelpath = modelpath)
# plot.epoch(modelname, modelpath, loss, y_name = "loss", y_para = "loss")
# plot.epoch(modelname, modelpath, val_acc, y_para = "val_acc")
# acc_epoch(modelname, modelpath, y,y1= None,y_para = None, y1_para = None, y_name = 'Accuracy', x_name = 'epoch'):
# cPickle.dump(model,open(modelpath + modelname + "_model.pkl","wb")) 
# pickle.dump(info, open(modelpath + modelname + "_info.p", "wb" ))    



# Return in tuple
def f(x):
  y0 = x + 1
  y1 = x * 3
  y2 = y0 ** 6
  return (y0,y1,y2)


f = open(filename)
try:
    for line in f:
        content.append(line)
finally:
    f.close()

# Open stereo dataset
semicolon = [i for i,x in enumerate(content) if x == ';\n']
name = np.asarray(content)[np.asarray(semicolon) + 1]
label = np.asarray(content)[np.asarray(semicolon) + 3]
label = [val[2] for val in label]
label = [int(val) for val in label]

pickle.dump(label, open("Stereo_label.p", "wb" ))
label_c1 = 16401

label_c1 = slice_label('20m_00s_042803u','26m_54s_977025u',label, title)

def slice_label(start, end, obj, namelist):
    if (type(namelist) is list):
        namelist = np.asarray(namelist)
    s = namelist.tolist().index(start + '.pgm\n')
    e = namelist.tolist().index(end + '.pgm\n') +1
    slice_label = np.asarray(obj[s:e])
    return slice_label


# cv2 show image
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Open file in ways
with open(filename) as myfile:
    head = [next(myfile) for x in xrange(10)]
print head


with open(filename) as myfile:
    content = [next(myfile)]
print head



f = open(filename)
try:
    for line in f:
        content.append(line)
finally:
    f.close()



infoname = 'sequence1_training_seat1.info'

info = []
with open(labelpath+ labelname, 'rb') as csvfile:
  reader = csv.reader(csvfile)
  for row in reader:
    info.append(row)

infoname = infoname
names = []
f = open(picpath + infoname)
try:
    for line in f:
        names.append(line)
finally:
    f.close()
names = names[2:]






# for i in range(len(X_test)):
#     if (Y_predict[i]==1 & int(Y_test[i][0]) ==0 ):
#         misc.imsave('/home/workstation/Documents/PIROPO/p1t0/' 
#             + str(i) + '.png',  np.rollaxis(X_test[i],0,3))



# for i in range(len(X_test)):
#     if (Y_predict[i]==0 & int(Y_test[i][0]) == 1 ):
#         print (i)
#         scipy.misc.imsave('/home/workstation/Documents/PIROPO/p0t1/'+
#          str(i) + 'png',  np.rollaxis(X_test[i],0,3))

vie = []
cap = cv2.VideoCapture(name)

while(cap.isOpened()):
    ret, frame = cap.read()
    if frame is None:
      break
    frame = cv2.resize(frame,(64,64), interpolation = cv2.INTER_CUBIC)
    vie.append(frame)
cap.release()
cv2.destroyAllWindows()

video = np.asarray(video)
print (video.shape)



# Read info from csv file
def getInfo(filepath = None, filename = None):
  info = []
  with open(filepath + filename, 'rb') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
      info.append(row)
  return info

filename = 
filepath = '/home/workstation/Documents/PIROPO/omni_1A/Ground_Truth_Annotations/'
filenames = []
for (dirpath, dirnames, names) in os.walk(filepath):
  f.extend(names)