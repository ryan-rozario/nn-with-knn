import numpy as np
from sklearn.model_selection import KFold
import random
from pso import Swarm
from knn_code import accuracy



FILENAME = input("Enter filename")


def run(test,train):
	'''
	Run the algorithm for each test and training set pair
	'''
	s=Swarm()
	s.initialize_swarm(train)
	ans=s.omptimize()
	return accuracy(ans,test,train)



def normalize(dataset):
	'''
	Minmax Normalization of the features before running the algorithm
	'''

	#normalize the features between 0 and 1
	#print(dataset)
	h=1
	l=0
	mins=np.min(dataset,axis=0)
	maxz=np.max(dataset,axis=0)
	
	rng=maxz-mins
	#res containins he normalized features
	res=h-(((h-l)*(maxz-dataset))/rng)
	#print(res)

	#set the class_flag to -1 if the class label is the last column
	#set class_flag to 0 if class label is first column
	class_flag=int(input("Enter class flag"))

	#remove the class column and add back the unormalized class label
	#class labels should not be normalized
	if class_flag==0:
		res=res[:,1:]
		dataset=dataset[:,0]
		dataset=dataset.reshape(-1, 1)
	elif class_flag==-1:
		#print(res)
		res=res[:,:-1]
		dataset=dataset[:,-1]
		dataset=dataset.reshape(-1, 1)

	#concatanate the class labels with the normalized features along the column axis
	out=np.concatenate((dataset,res),axis=1)



	return out

def loadfile(filename):
	'''Load the data from the file and  normalize it'''
	dataset=np.genfromtxt(filename,delimiter=',')
	dataset=dataset[1:]
	np.random.shuffle(dataset)
	dataset=normalize(dataset)
	return dataset

def kfold(dataset):
	'''
	kfold validation to test for accuracy
	'''
	kf=KFold(n_splits=10)
	kf.get_n_splits(dataset)
	avg_acc=[]
	avg_fscr=[]
	for train_ind, test_ind in kf.split(dataset):
		train,test=dataset[train_ind],dataset[test_ind]
		acc,fscr=run(test,train)
		print("Accuracy ",acc," F-score ",fscr)
		avg_acc.append(acc)
		avg_fscr.append(fscr)
	avg_acc_ans=0
	avg_fscore_ans=0
	for i in range(len(avg_acc)):
		avg_acc_ans+= avg_acc[i]
		avg_fscore_ans += avg_fscr[i]
	avg_acc_ans /= len(avg_acc)
	avg_fscore_ans /= len(avg_fscr)
	print("Average accuracy ",avg_acc_ans)
	print("Average fscore ",avg_fscore_ans)


if __name__ == "__main__":
	'''Start the algorithm. Use k fold validation to test accuracy'''
	dataset=loadfile(FILENAME)
	print(FILENAME)
	kfold(dataset)

		





