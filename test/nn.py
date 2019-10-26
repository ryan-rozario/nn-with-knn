#import torch
import csv
import numpy as np
from scipy import stats
from sklearn.neighbors import NearestNeighbors


NEAREST_NEIGHBOURS=10

NUMBER_OF_INPUT_NODES = 22
NUMBER_OF_HIDDEN_NODES =10
NUMBER_OF_OUTPUT_NODES = 2


def loadTHEfile(fil):
	lines=csv.reader(open(fil, "r"))
	dataset=list(lines)
	dataset.pop(0)
	for i in range(len(dataset)):
		if dataset[i][0]=="Yes":
			dataset[i][0]=1
		else:
			dataset[i][0]=0
	for i in range(len(dataset)):   
		dataset[i]=[int(x) for x in dataset[i]]
	return dataset


class Particle:
    def  __init__(self):
        self.w1 =np.random.randn(NUMBER_OF_INPUT_NODES, NUMBER_OF_HIDDEN_NODES) # weight for hidden layer
        self.w2 =np.random.randn(NUMBER_OF_HIDDEN_NODES, NUMBER_OF_OUTPUT_NODES) # weight for output layer

        # initialize tensor variables for bias terms 
        self.b1 =np.random.randn(1, NUMBER_OF_HIDDEN_NODES) # bias for hidden layer
        self.b2 =np.random.randn(1, NUMBER_OF_OUTPUT_NODES)

        self.fitness=None  #this has to be done
        self.output=None   #this has to be done

        #this has to be set
        self.alpha=2
        self.weight_class=[2,3]

    '''
    def run_forward(self,inp_x):
        output = model.Model(self.w1,self.w2,self.b1,self.b2).forward_propogation(inp_x)

        return output
    '''
    '''
    def z_score(self,x):
        mean=np.mean(x)
        std=np.std(x)
        z=(x-mean)/std
        return z 
    '''

    def forward(self,inp_x):
        ## activation of hidden layer 
        z1 = np.dot(x, self.w1) + self.b1

        ## activation (output) of final layer 
        z2 = np.dot(z1, self.w2) + self.b2

        self.output=z2

    def calc_fitness(self,inp_x,out_y):

        n=len(inp_x)

        #run thorugh the neural network and give output in reduced dimensionality space

        #for i in range(n):
        #    self.output.append(model.Model(self.w1,self.w2,self.b1,self.b2).forward_propogation(inp_x[i]))

        self.forward(inp_x)

        

        
        self.output = stats.zscore(self.output)   #z-score function
        

        
        h=np.zeros((n,2))
        #normalized points constrained in hyperspace
        for i in range(n):
            x_dist = np.linalg.norm(self.output[i])
            numerator=1-np.exp(-(x_dist/2))
            denominator= x_dist*(1+np.exp(-(x_dist/2)))
            h[i]=self.output[i]*(numerator/denominator)

        
        #print(h)

        

        similarity_matrix = np.zeros((n,n))

        #gives similarity between every two points
        for i in range(n):
            for j in range(i,n):
                similarity = 2-(np.linalg.norm(h[i]-h[j]))
                similarity_matrix[i][j]=similarity
                similarity_matrix[j][i]=similarity



        #nearest neightbours
        nbrs = NearestNeighbors(n_neighbors=NEAREST_NEIGHBOURS).fit(self.output)
        distances, indices  = nbrs.kneighbors(self.output)

        print(indices)

        #calcualte fitness as per equation 6
        f=0

        for i in range(n):
            f_temp=0
            for j in indices[i]:
                if out_y[i]==out_y[j]:
                    f_temp+=similarity_matrix[i][j]
                else:
                    f_temp+=self.alpha*similarity_matrix[i][j]

            index = int(out_y[i])

            f+=self.weight_class[index]*f_temp

        return f



data = loadTHEfile("SPECT.csv")

data = np.array(data, dtype=np.float)

x=data[:,:-1]
y=data[:,-1]

p=Particle()
p.calc_fitness(x,y)



