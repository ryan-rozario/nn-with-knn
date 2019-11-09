import numpy as np
import random
def loadfile(filename):
	dataset=np.genfromtxt('filename.csv',delimiter=',')
	dataset=dataset[1:]
	np.random.shuffle(dataset)
	print(dataset.min())



def normalize(dataset):
	print("inside")
	h=1
	l=0
	mins=np.min(dataset,axis=0)
	maxz=np.max(dataset,axis=0)
	rng=maxz-mins
	return h-(((h-l)*(maxz-dataset))/rng)

x=np.array(
	[2500,0.15,12],
	[1200,0.65,20],
	[6200,0.35,19]
	)
x=normalize(x)
print(x)




