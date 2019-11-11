'''
This file contains the main code for the neural network trained using pso
based on the paper L. Wang, B. Yang, Y. Chen, X. Zhang and J. Orchard, "Improving Neural-Network Classifiers Using Nearest Neighbor Partitioning," in IEEE Transactions on Neural Networks and Learning Systems, vol. 28, no. 10, pp. 2255-2267, Oct. 2017.
This basically reduces the dimensionality of data to use knn
'''
import copy 
import random
import numpy as np
from scipy import stats
from sklearn.neighbors import NearestNeighbors
import csv


#define the number of classes in the data
CLASS_NUM=int(input("Enter number of classes"))

#define the discrimination weight
ALPHA=float(input("Enter alpha"))
#print("alpha ",ALPHA)

#define the 
NEAREST_NEIGHBOURS=10

#number of input nodes should be equal to the number of features
NUMBER_OF_INPUT_NODES = 9
NUMBER_OF_HIDDEN_NODES =5
#number of output nodes should be equal to  dimensions of partition space
NUMBER_OF_OUTPUT_NODES = int(input("Enter number of output nodes"))

#maximum number of iterations
MAX_GENERATION = 100 
#number of indivisuals
POPULATION_SIZE =10
#maximum velocity
VMAX = 0.4

#constants related to pso
C1 = 1.8
C2 = 1.8

def tanh(X):
    return np.tanh(X)

def relu(X):
    '''relu activation function'''
    return np.maximum(0,X)


def sigmoid(Z):
    '''Applies sigmoid function for a particular input'''
    return 1/(1+np.exp(-Z))

'''
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

'''



class Swarm:
    '''
    Swarm class contains all the indivisuals
    '''
    def __init__(self,size=POPULATION_SIZE,phi_1=C1,phi_2=C2,iter=MAX_GENERATION):

        self.size=size
        self.phi_p=phi_1
        self.phi_g=phi_2
        #self.learning_parameter=lp   #not needed
        self.max_iter = iter

        self.velocity=[]
        self.group = []
        self.global_best=None
        self.local_best=[]
        self.x=None
        self.y=None

    def initialize_swarm(self,data):
        '''
        Initialize all the indivisuals with the given data
        '''
        #give file to each particle
        #data = loadTHEfile(filename)

        #data = np.array(data, dtype=np.float)

        x=data[:,1:]
        y=data[:,0]

        global NUMBER_OF_INPUT_NODES
        NUMBER_OF_INPUT_NODES = len(x[0])
        #print(NUMBER_OF_INPUT_NODES)
        

        self.x=x
        self.y=y
        #for i in self.y:
            #print(i)

        self.group = [Particle(self.x,self.y) for i in range(self.size)]    
        self.velocity = [Vel() for i in range(self.size)] 
        self.local_best=copy.deepcopy(self.group)


        self.global_best=self.local_best[0]
        for i in self.local_best:
            if self.global_best.fitness<i.fitness:
                self.global_best=copy.deepcopy(i)


    def omptimize(self):
        '''Run the algorithm will maximum iterations are done '''
        iteration=0
        #untill termination condition
        while iteration<self.max_iter:
            self.update()
            iteration+=1
        '''
        print("Global Best")
        print(self.global_best.fitness)
        print("\n\n")
        '''
        return self.global_best

    def update(self):
        '''
        Update the velocities and weights of the swarm
        Calculate the fitness
        Update the local best and global best fitness
        '''
        #random numbers
        r_p=random.random()
        r_g=random.random()
        for i in range(self.size):

            #update velocities
            '''
            self.velocity[i].w1 = self.learning_parameter*self.velocity.w1  + self.phi_p*r_p*(self.local_best[i].w1-self.group[i].w1)+ self.phi_g*r_g*(self.global_best.w1-self.group[i].w1)
            self.velocity[i].w2 = self.learning_parameter*self.velocity.w2  + self.phi_p*r_p*(self.local_best[i].w2-self.group[i].w2)+ self.phi_g*r_g*(self.global_best.w2-self.group[i].w2)
            self.velocity[i].b1 = self.learning_parameter*self.velocity.b1  + self.phi_p*r_p*(self.local_best[i].b1-self.group[i].b1)+ self.phi_g*r_g*(self.global_best.b1-self.group[i].b1)
            self.velocity[i].b2 = self.learning_parameter*self.velocity.b2  + self.phi_p*r_p*(self.local_best[i].b2-self.group[i].b2)+ self.phi_g*r_g*(self.global_best.b2-self.group[i].b2)
            '''
            self.velocity[i].w1 = self.velocity[i].w1  + self.phi_p*r_p*(self.local_best[i].w1-self.group[i].w1)+ self.phi_g*r_g*(self.global_best.w1-self.group[i].w1)
            self.velocity[i].w2 = self.velocity[i].w2  + self.phi_p*r_p*(self.local_best[i].w2-self.group[i].w2)+ self.phi_g*r_g*(self.global_best.w2-self.group[i].w2)
            self.velocity[i].b1 = self.velocity[i].b1  + self.phi_p*r_p*(self.local_best[i].b1-self.group[i].b1)+ self.phi_g*r_g*(self.global_best.b1-self.group[i].b1)
            self.velocity[i].b2 = self.velocity[i].b2  + self.phi_p*r_p*(self.local_best[i].b2-self.group[i].b2)+ self.phi_g*r_g*(self.global_best.b2-self.group[i].b2)
            

            #cap the velocity at VMAX
            #print("1")
            # self.velocity[i].w1=np.maximum(self.velocity[i].w1,VMAX)
            # self.velocity[i].w2=np.maximum(self.velocity[i].w2,VMAX)
            # self.velocity[i].b1=np.maximum(self.velocity[i].b1,VMAX)
            # self.velocity[i].b2=np.maximum(self.velocity[i].b2,VMAX)

            

            for j in range(len(self.velocity[i].w1)):
                for k in range(len(self.velocity[i].w1[j])):
                    if self.velocity[i].w1[j][k]>VMAX:
                        self.velocity[i].w1[j][k]=VMAX
            for j in range(len(self.velocity[i].w2)):
                for k in range(len(self.velocity[i].w2[j])):
                    if self.velocity[i].w2[j][k]>VMAX:
                        self.velocity[i].w2[j][k]=VMAX
            for j in range(len(self.velocity[i].b1)):
                for k in range(len(self.velocity[i].b1[j])):
                    if self.velocity[i].b1[j][k]>VMAX:
                        self.velocity[i].b1[j][k]=VMAX
            for j in range(len(self.velocity[i].b2)):
                for k in range(len(self.velocity[i].b2[j])):
                    if self.velocity[i].b2[j][k]>VMAX:
                        self.velocity[i].b2[j][k]=VMAX

            for j in range(len(self.velocity[i].w1)):
                for k in range(len(self.velocity[i].w1[j])):
                    if self.velocity[i].w1[j][k]<-VMAX:
                        self.velocity[i].w1[j][k]=-VMAX
            for j in range(len(self.velocity[i].w2)):
                for k in range(len(self.velocity[i].w2[j])):
                    if self.velocity[i].w2[j][k]<-VMAX:
                        self.velocity[i].w2[j][k]=-VMAX
            for j in range(len(self.velocity[i].b1)):
                for k in range(len(self.velocity[i].b1[j])):
                    if self.velocity[i].b1[j][k]<VMAX:
                        self.velocity[i].b1[j][k]=-VMAX
            for j in range(len(self.velocity[i].b2)):
                for k in range(len(self.velocity[i].b2[j])):
                    if self.velocity[i].b2[j][k]<VMAX:
                        self.velocity[i].b2[j][k]=-VMAX
 

            


            #update weights
            self.group[i].w1 = self.group[i].w1 + self.velocity[i].w1
            self.group[i].w2 = self.group[i].w2 + self.velocity[i].w2
            self.group[i].b1 = self.group[i].b1 + self.velocity[i].b1
            self.group[i].b2 = self.group[i].b2 + self.velocity[i].b2 
            
            #calculate the fitness
            self.group[i].calc_fitness(self.x,self.y)

        for i in range(self.size):
            #update local best
            if self.group[i].fitness > self.local_best[i].fitness:
                self.local_best[i] = copy.deepcopy(self.group[i])

                #calculate global best
                if self.group[i].fitness > self.global_best.fitness:
                    self.global_best = copy.deepcopy(self.group[i])
            '''
            print("Local Bests")
            for i in self.local_best:
                print(i.fitness)
            '''
            

class Particle:
    '''
    Each indivisual represents a neural network. We PSO to train the weights for the neural network.
    '''
    def  __init__(self,x=[],y=[]):
        #initial weights are set between -4 and +4#
        #refer R. Eberhart and J. Kennedy,A new optimizer using particle swarm theory
        weight_initial_min=-4
        weight_initial_max=4
        self.w1 =(weight_initial_max-weight_initial_min)*np.random.random_sample(size=(NUMBER_OF_INPUT_NODES, NUMBER_OF_HIDDEN_NODES))+weight_initial_min   #np.random.randn(NUMBER_OF_INPUT_NODES, NUMBER_OF_HIDDEN_NODES) # weight for hidden layer
        self.w2 =(weight_initial_max-weight_initial_min)*np.random.random_sample(size=(NUMBER_OF_HIDDEN_NODES, NUMBER_OF_OUTPUT_NODES))+weight_initial_min   #np.random.randn(NUMBER_OF_HIDDEN_NODES, NUMBER_OF_OUTPUT_NODES) # weight for output layer

        # initialize tensor variables for bias terms 
        self.b1 =(weight_initial_max-weight_initial_min)*np.random.random_sample(size=(1, NUMBER_OF_HIDDEN_NODES))+weight_initial_min#np.random.randn(1, NUMBER_OF_HIDDEN_NODES) # bias for hidden layer
        self.b2 =(weight_initial_max-weight_initial_min)*np.random.random_sample(size=(1, NUMBER_OF_OUTPUT_NODES))+weight_initial_min#np.random.randn(1, NUMBER_OF_OUTPUT_NODES)

        self.fitness=None  #stores fitness value for each neural network
        self.output=None   #stores output of the neural network in the reduced dimensionality partition space

        #discrimination weight
        self.alpha=ALPHA

        if x!=[] and y!=[]:
            #weight class contains the fraction of the data elements belonging to each class of the dataset
            self.weight_class=self.frac_class_wt(y)
            #initialize fitness
            self.calc_fitness(x,y)

        
    def frac_class_wt(self,arr):
        '''
        Gives the fraction of each class in the dataset
        '''        
        

        frac_arr=[0 for i in range(CLASS_NUM)]

        for j in arr:
            class_num=int(j)-1
            frac_arr[class_num]+=1
        #print(frac_arr)
        for i in range(len(frac_arr)):
            frac_arr[i]=frac_arr[i]/float(arr.size)
        
        return frac_arr

    def forward(self,inp_x,activation="sigmoid"):
        '''
        Gives the ouput of the neural network
        '''
        
        #specifies activation function not working right now
        
        if activation is "sigmoid":
            activation = sigmoid
        elif activation is "tanh":
            activation = tanh
        elif activation is "relu":
            activation = relu
        else:
            raise Exception('Non-supported activation function')
        
        ## activation of hidden layer 
        z1 = np.dot(inp_x, self.w1) + self.b1
        a1 = activation(z1)
        #print(self.w1)
        ## activation (output) of final layer 
        z2 = np.dot(a1, self.w2) + self.b2
        a2 = activation(z2)

        #print(z2)
        #self.output=z2
        self.output=a2

    def calc_fitness(self,inp_x,out_y):
        '''
        Calculate the fitness of each neural network using similarity measure given in the paper
        '''

        n=len(inp_x)

        #run thorugh the neural network and give output in reduced dimensionality space
        self.forward(inp_x)

        

        
        self.output = stats.zscore(self.output)   #z-score function
        

        
        h=np.zeros((n,NUMBER_OF_OUTPUT_NODES))
        #normalized points constrained in hyperspace
        #we constrain the normalized points into a hypersphere of radius 1
        for i in range(n):
            x_dist = np.linalg.norm(self.output[i])
            numerator=1-np.exp(-(x_dist/2))
            denominator= x_dist*(1+np.exp(-(x_dist/2)))
            h[i]=self.output[i]*(numerator/denominator)

        self.output=h
        #print(h)


        
        #similarity matrix gives the similarity between every two records in the dataset
        similarity_matrix = np.zeros((n,n))

        #gives similarity between every two points
        for i in range(n):
            for j in range(i,n):
                similarity = 2-(np.linalg.norm(h[i]-h[j]))
                similarity_matrix[i][j]=similarity
                similarity_matrix[j][i]=similarity



        #nearest neightbours
        #for i in self.output:
            #print(i)

        #get the nearest neighbours
        nbrs = NearestNeighbors(n_neighbors=NEAREST_NEIGHBOURS).fit(self.output)
        _distances, indices  = nbrs.kneighbors(self.output)


        #calcualte fitness as per equation 6 in the paper
        f=0

        for i in range(n):
            f_temp=0
            for j in indices[i]:
                if out_y[i]==out_y[j]:
                    #similarity for elements of same class
                    f_temp+=similarity_matrix[i][j]
                else:
                    #similarity for elements of different class
                    f_temp+=self.alpha*similarity_matrix[i][j]

            #index for the weight_class
            index = int(out_y[i])-1   

            f+=self.weight_class[index]*f_temp
        self.fitness=f
        return f

    def kmeans_eval(self,inp_x):
        '''
        Calculate the fitness of each neural network using similarity measure given in the paper
        '''

        n=len(inp_x)

        #run thorugh the neural network and give output in reduced dimensionality space
        self.forward(inp_x)

        

        
        self.output = stats.zscore(self.output)   #z-score function
        

        
        h=np.zeros((n,NUMBER_OF_OUTPUT_NODES))
        #normalized points constrained in hyperspace
        #we constrain the normalized points into a hypersphere of radius 1
        for i in range(n):
            x_dist = np.linalg.norm(self.output[i])
            numerator=1-np.exp(-(x_dist/2))
            denominator= x_dist*(1+np.exp(-(x_dist/2)))
            h[i]=self.output[i]*(numerator/denominator)

        self.output=h
        #print(h)




'''
    #gives similarity on inputing the h value of the respective points
    def similarity(self,h1,h2):
        return 2-np.linalg.norm(h1-h2)

'''




class Vel:
    def  __init__(self,x=[],y=[]):
        #initial weights are set between -4 and +4#
        #refer R. Eberhart and J. Kennedy,A new optimizer using particle swarm theory
        self.w1 =np.zeros((NUMBER_OF_INPUT_NODES, NUMBER_OF_HIDDEN_NODES))   #np.random.randn(NUMBER_OF_INPUT_NODES, NUMBER_OF_HIDDEN_NODES) # weight for hidden layer
        self.w2 =np.zeros((NUMBER_OF_HIDDEN_NODES, NUMBER_OF_OUTPUT_NODES))   #np.random.randn(NUMBER_OF_HIDDEN_NODES, NUMBER_OF_OUTPUT_NODES) # weight for output layer

        # initialize tensor variables for bias terms 
        self.b1 =np.zeros((1, NUMBER_OF_HIDDEN_NODES))#np.random.randn(1, NUMBER_OF_HIDDEN_NODES) # bias for hidden layer
        self.b2 =np.zeros((1, NUMBER_OF_OUTPUT_NODES))#np.random.randn(1, NUMBER_OF_OUTPUT_NODES)


