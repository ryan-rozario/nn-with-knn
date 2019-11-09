import numpy as np
from sklearn.model_selection import KFold
import random

def loadfile(filename):
	dataset=np.genfromtxt(filename,delimiter=',')
	dataset=dataset[1:]
	np.random.shuffle(dataset)
	return dataset

def normalize(dataset):
	h=1
	l=0
	mins=np.min(dataset,axis=0)
	maxz=np.max(dataset,axis=0)
	rng=maxz-mins
	return h-(((h-l)*(maxz-dataset))/rng)

def kfold(dataset):
	kf=KFold(n_splits=10)
	kf.get_n_splits(dataset)
	for train_ind, test_ind in kf.split(dataset):
		train,test=dataset[train_ind],dataset[test_ind]
		
dataset=loadfile("IRIS.csv")
dataset=normalize(dataset)
kfold(dataset)




