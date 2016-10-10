import pickle
import numpy as np
import shutil
import scipy.misc


def show_para(itte):
	for item in itte:
		print item['lr_values']
		print item['scores'][1]

# Update it like 'walk(dir)'
def open_file(modelpath, modelname):
    l = ['_h', '_s','_trails']
    stuff = []
    for item in l:
        stuff.append(pickle.load(open(modelpath + modelname + item + ".p", "rb" )))
    # return tuple(stuff)

def trials(trials, score, print_trials = True, print_num = 5):
    if print_trials:
        print ('sample trials:')
        print (trials.trials)[0]

        print ('trials:')
        for t in trials.trials:
            print (t['misc']['vals'])

    # Print 5 best acc and params used    
    n_trails = len(trials.trials)    
    s = np.asarray(score)
    max_val = s[np.argsort(s[:,1])][n_trails-print_num:]
    max_ind = np.argsort(s[:,1])[n_trails-print_num:]
    print ('params of ' + str(print_num) + ' best results')
    i = 0
    for ind in max_ind:
        print i
        print trials.trials[ind]['misc']['vals']
        i += 1
    print ('Best accs')    
    i = 0 
    for i in range(len(max_val)):
        print (i, np.float32(max_val[i][1]))

def predict(model, imagepath, savepath0, savepath1):
    X_predict, l, f = set_para.readimage(label = 0, 
        path = '/home/workstation/Documents/humandataset/thermal_image/')
    Y_predict = model.predict_classes(X_predict, batch_size=32, verbose=0)
    Y_predict = model.predict(X_predict, batch_size=32, verbose=0)
    # f = np.asarray(f)
    imagepath = '/home/workstation/Documents/humandataset/thermal_image/'
    savepath1 = '/home/workstation/Documents/humandataset/thermal1/'
    for i in range(len(Y_predict)):
        if (Y_predict[i][1] > 0.998): shutil.copy(imagepath + f[i], savepath1)



def savexy(x,y,filename, filepath):
  # model = pickle.load(open(modelpath + modelname + '.p', 'rb'))
  pickle.dump(x, open(filepath + filename + "_x.p", "wb" ))
  pickle.dump(y, open(filepath + filename + "_y.p", "wb" ))

def loadxy(filename, filepath):
    x = pickle.load(open(filepath + filename + "_x.p", "rb" ))
    y = pickle.load(open(filepath + filename + "_y.p", "rb" ))
    return x,y



def findimage(findpath, Y_test, Y_predict):
    # To be developed
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





# a = np.float32(h[43]['val_acc']).tolist()
# for index, elem in enumerate(a):
#         print(index, elem)
























# with open('4layers1e-5decay200epoch_summa.p', 'rb') as f:
#     data = pickle.load(f)

