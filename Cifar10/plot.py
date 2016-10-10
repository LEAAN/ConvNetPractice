# plot Weight of first layer
import matplotlib.pyplot as plt
import pickle
import numpy as np
import re
from matplotlib import cm
import matplotlib.gridspec as gridspec
from sklearn.metrics import confusion_matrix

def firstweight(model= None, modelname = None, n_perrow= 8, n_percolume= 4, modelpath = "/home/workstation/Documents/Cifar100/"):
	if modelname is None:
		name = raw_input('Assign model name plz:) ')
	if model is None:
		print ('open ' + modelpath + modelname)
		model = pickle.load(open(modelpath + modelname + '.p', 'rb'))
	# model = cPickle.load(open(modelpath+modelname+ ".pkl", "rb"))
	W0 = model.layers[0].get_weights()[0]
	bias0 = model.layers[0].get_weights()[1]
	print "First layer weight W0.shape is: "
	print W0.shape
	plt.clf()
	W0 = np.rollaxis(W0,1,4)
	for i in range(W0.shape[0]):
		plt.subplot(n_perrow,n_percolume,i+1)
		plt.imshow(W0[i],interpolation = 'none', cmap ='Greys')
		# plt.imshow(W0[i][:,:,0],interpolation = 'none', cmap = 'Greys')
		plt.axis('off')
	# plt.colorbar()
	plt.savefig(modelpath+modelname, bbox_inches='tight')

def epoch(modelname, modelpath, y,y1= None,y_para = None, y1_para = None, title = 'Accuracy', x_name = 'epoch'):
	x = np.arange(len(y))
	plt.clf()
	line = plt.plot(x,y, label = y_para)
	if y1:
		line1 = plt.plot(x,y1, label = y1_para)
	plt.legend()
	plt.title(y_name + '_against epoch')
	plt.xlabel(x_name)
	plt.ylabel(y_name)
	plt.savefig(modelpath + modelname + y_name + '_epoch.png')
	plt.show()
	# plt.axis([0,len(y),0,1])
	# plt.axis(np.arange(0,len(y), 6))

def weight(model= None, modelname = None, openpath = "/home/workstation/Documents/Cifar100/"):
	if modelname is None:
		modelname = raw_input('Model is loaded. Assign a model name plz:) ')
	if model is None:
		print ('open cnn model in ' + openpath + modelname)
		model = pickle.load(open(openpath + modelname + '.p', 'rb'))
	# model = cPickle.load(open(modelpath+modelname+ ".pkl", "rb"))
	cnn_layer_counter = 0
	w_max = []
	w_min = []
	weights = []
	for layer in model.layers:
		if (re.match(r'convolution.', str(layer.name), re.M)):
			w = layer.get_weights()[0]
			w_max.append(np.amax(w))
			w_min.append(np.amin(w))
			weights.append(w)
	print (min(w_min), max(w_max))
	for w in weights:
		cnn_layer_counter +=1
		drawsquares(w, modelname = modelname, title = 'cnn layer '+ str(cnn_layer_counter),
		vmin = min(w_min), vmax = max(w_max))


def drawsquares(w, modelname = None, title = None,
	savepath = "/home/workstation/Documents/Cifar100/", vmax = None, vmin = None):
	if modelname == None:
		modelname = raw_input('Assign name to weights figure: ')
	plt.clf()
	fig = plt.figure(figsize=(10,10))
	nrow = w.shape[0]
	ncol = w.shape[1]
	size = max(nrow, ncol)
	gs1 = gridspec.GridSpec(size, size)
	gs1.update(wspace=0.1, hspace=0.1) 
	for i in range(nrow*ncol):
		# plt.subplot(gs1[i])
		subfig = plt.subplot(gs1[i%nrow, i/nrow])
		print (i%nrow, i/nrow)
		plt.axis('off')
		im = plt.imshow(w[i%nrow, i/nrow] ,interpolation = 'none', 
			vmin = vmin, vmax = vmax, cmap = plt.get_cmap('Greys'))
	cbar_ax = fig.add_axes([0.1, 0, 0.2, 0.02])
	ticks = [vmin, 0, vmax]
	fig.colorbar(im, cax = cbar_ax,orientation='horizontal',ticks = ticks, ticklocation = 'top')
	plt.clim(vmin,vmax)
	if title:
		fig.suptitle('%s %s %s %s %d %d' % ( 'Model:', modelname, 'layer:', title, nrow, ncol), 
			 x = 0.2)
	else: print ('Sure of adding no titles?')
	plt.savefig(savepath + modelname+ title + '.png')
	# plt.savefig(savepath + modelname+ title + '.png', bbox_inches = 'tight')

def layeroutput(output, title, nrow = 8, ncol = 8,
	savepath = "/home/workstation/Documents/Cifar100/"):
	plt.clf()
	fig = plt.figure()
	gs1 = gridspec.GridSpec(nrow, ncol)
	gs1.update(wspace = 0.1, hspace = 0.1)
	print(nrow, ncol)
	for i in range(nrow* ncol):
		subfig = plt.subplot(gs1[i%nrow, i/nrow])
		print (i%nrow, i/nrow)
		plt.axis('off')
		# im = plt.imshow(output[i], interpolation = 'none')
		im = plt.imshow(output[i], interpolation = 'none', cmap = plt.get_cmap('magma'))
	# plt.colorbar()
	fig.subplots_adjust(right=0.8)
	cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
	fig.colorbar(im, cax=cbar_ax)
	fig.suptitle(title)
	plt.savefig(savepath + title + '.png')





def confusion(y_pred = None, y_true = None, cmap = plt.cm.Blues, title = 'Confusion mat',
	plot = True, savepath = "/home/workstation/Documents/Cifar100/", savename = None, normalize = True,
	text = True):
    cm = confusion_matrix(y_true = y_true, y_pred = y_pred)
    if normalize:
    	cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    	print('Confusion matrix, with normalization')
    np.set_printoptions(precision=2)
    print(cm)
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title)
    # plt.colorbar()
    ticks = ['without', 'with human']
    ax.set_xticks(np.arange(len(ticks)))
    ax.set_xticklabels(ticks, rotation = 45)
    ax.set_yticks(np.arange(len(ticks)))
    ax.set_yticklabels(ticks)

    plt.tight_layout()
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')

    cbar = fig.colorbar(cax)

    if text:
    	# ax = fig.add_subplot(111)
    	ax.text(-0.45,-0.4, str(cm[0]), color = 'white')
    	ax.text(-0.45,-0.3, str(cm[1]), color = 'white')
    if savename:
		plt.savefig(savepath + savename+ '.png')


# print("Accuracy: %.2f%%" % (scores[1]*100))



# def whatr():
# 	for i in range(len(W0)):
# 		plt.figure()
# 		plt.title("conv1 w_"+str(i+1))
# 		plt.imshow(W0[i])
# 		plt.axis('off')
# 		plt.savefig("W0"+str(i+1)+".png", bbox_inches='tight')

# 	plt.clf()
# 	for i in range(8):
# 		for j in range(4):
# 			plt.subplot(8,4,4*i+j+1)
# 			plt.imshow(W0[4*i+j])
# 			plt.axis('off')

# 	plt.savefig("whole.png", bbox_inches='tight')


# 	plt.clf()
# 	for i in range(8):
# 		for j in range(4):
# 			plt.subplot(8,4,4*i+j+1)
# 			plt.imshow(W0[4*i+j][:,0,:])
# 			plt.axis('off')
# 	plt.savefig("whole_G.png", bbox_inches='tight')

# 	plt.clf()
# 	plt.imshow(W0[0][0,:,:])
# 	plt.savefig("test_RGB.png")