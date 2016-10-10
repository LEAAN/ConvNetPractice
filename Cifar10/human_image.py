import cv2
import numpy as np
from os import walk
import matplotlib.pyplot as plt



print [name for name in os.listdir(path) if os.path.isdir(name)]

path = '/home/workstation/Documents/humandataset/'
folder_list = os.listdir(path)

# def add_pic(folder_list, humanor_nonhuman):
f = []
for (dirpath, dirnames, filenames) in walk(path):
    f.extend(filenames)

max_h = 0
max_w = 0
# h0 w1
pics = []
w = []
h = []    
for name in f:
    image = cv2.imread(path + name, cv2.CV_LOAD_IMAGE_COLOR)
    print (name)
    h.append(image.shape[0])
    w.append(image.shape[1])
    pics.append(image)

hmin = min(np.asarray(h))
wmin = min(np.asarray(w))

# cut + make 
pics_cut = []
for p in pics:
    pics_cut.append(p[:hmin, :wmin])

pics_cut = np.asarray(pics_cut)    
X_test = np.rollaxis(pics_cut,3,1)

X_test = np.concatenate((people,animals))

Y_test = np.concatenate((np.ones(421),np.zeros(506)))
Y_test = np.concatenate((Y_test, 1-Y_test))
# 421 1 506 0

# hist of h & w
plt.clf()
plt.hist(w, bins= 20)
plt.title('distribution of width')
plt.show()
plt.clf()
plt.hist(h, bins= 20)
plt.title('distribution of height')


pickle.dump(X_test, open(path +  "Xtest.p", "wb" ))
pickle.dump(il, open(path +  "omni_1A.p", "wb" ))

# Make new Y_test?
np.reshape(Y_test,(len(Y_test),1)).shape
(Y_test).shape
Y_test=np.reshape(Y_test,(len(Y_test),1))
Y_test1=np.reshape(1-Y_test,(len(Y_test),1))
np.concatenate((Y_test, (Y_test1).T),axis=1).shape
np.concatenate((Y_test, (Y_test1).T),axis=1).shape
Y_test1=shape
Y_test1.shape
np.concatenate((Y_test, (Y_test1)),axis=1).shape
YY_test=np.concatenate((Y_test, (Y_test1)),axis=1)
YY_test[:10]
YY_test[420:431]



# 32x32 Make new X_test?
XX_test = np.zeros(len(X_test),3,32,32)
rbound = np.random.randint(204, size = len(X_test))
lbound = np.random.randint(214, size = len(X_test))
for i in range(len(X_test)):
    XX_test[i] = X_test[i,:,rbound[i]-32:rbound[i],lbound[i]-32:lbound[i]]
    break

XX_test = X_test[:,:,rbound-32:rbound,lbound-32:lbound]

YY_test = np.concatenate((Y_test, np.zeros((len(Y_test),8))),axis=1)



# Make X_test = 
pickle.load(open(name + ".p", "rb" ))