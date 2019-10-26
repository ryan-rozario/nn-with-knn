#import torch
#import model
import copy 
import random
import numpy as np
from scipy import stats
from sklearn.neighbors import NearestNeighbors
import csv


NEAREST_NEIGHBOURS=10

NUMBER_OF_INPUT_NODES = 22
NUMBER_OF_HIDDEN_NODES =10
NUMBER_OF_OUTPUT_NODES = 2

MAX_GENERATION = 10
POPULATION_SIZE =20
VMAX = 0.4
C1 = 1.8
C2 = 1.8


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





class Swarm:
    def __init__(self,size=POPULATION_SIZE,phi_1=C1,phi_2=C2,iter=MAX_GENERATION):

        self.size=size
        self.phi_p=phi_1
        self.phi_g=phi_2
        #self.learning_parameter=lp   #check paper and set this

        self.max_iter = iter

        self.velocity=[]
        self.group = []
        self.global_best=None
        self.local_best=[]

    def initialize_swarm(self):
        self.group = [Particle() for i in range(self.size)]    #todo: Set boundaries
        self.velocity = [Particle() for i in range(self.size)] #todo: Set boundaries
        self.local_best=copy.deepcopy(self.group)


        self.global_best=self.local_best[0]
        for i in self.local_best:
            if self.global_best.fitness<i.fitness:
                self.global_best=copy.deepcopy(i)


    def omptimize(self):
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

    def update(self):
        #random numbers have to be changed
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
            



            #update weights
            self.group[i].w1 = self.group[i].w1 + self.velocity[i].w1
            self.group[i].w2 = self.group[i].w2 + self.velocity[i].w2
            self.group[i].b1 = self.group[i].b1 + self.velocity[i].b1
            self.group[i].b2 = self.group[i].b2 + self.velocity[i].b2 

            if self.group[i].fitness > self.local_best[i].fitness:  #check conditions and sign later
                self.local_best[i] = copy.deepcopy(self.group[i])

                if self.group[i].fitness > self.global_best.fitness:  #check conditions and sign again
                    self.global_best = copy.deepcopy(self.group[i])
            '''
            print("Local Bests")
            for i in self.local_best:
                print(i.fitness)
            '''
            

class Particle:
    def  __init__(self):
        self.w1 =np.random.randn(NUMBER_OF_INPUT_NODES, NUMBER_OF_HIDDEN_NODES) # weight for hidden layer
        self.w2 =np.random.randn(NUMBER_OF_HIDDEN_NODES, NUMBER_OF_OUTPUT_NODES) # weight for output layer

        # initialize tensor variables for bias terms 
        self.b1 =np.random.randn(1, NUMBER_OF_HIDDEN_NODES) # bias for hidden layer
        self.b2 =np.random.randn(1, NUMBER_OF_OUTPUT_NODES)

        self.fitness=None  #this has to be done
        self.output=None   #this has to be done

        #this has to be set properly
        self.alpha=0.01
        self.weight_class=None
        
        '''
        data = loadTHEfile("SPECT.csv")

        data = np.array(data, dtype=np.float)

        x=data[:,:-1]
        y=data[:,-1]

        self.weight_class=self.frac_class_wt(y)
        self.calc_fitness(x,y)
        '''

        
    def frac_class_wt(self,arr):
        np.sort(arr)
        unique_elements, counts_elements = np.unique(arr, return_counts=True)
        return counts_elements/arr.size

    def forward(self,inp_x):
        ## activation of hidden layer 
        z1 = np.dot(inp_x, self.w1) + self.b1

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

        #print(indices)

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
        self.fitness=f
        return f



'''
    #gives similarity on inputing the h value of the respective points
    def similarity(self,h1,h2):
        return 2-np.linalg.norm(h1-h2)

'''


'''

s=Swarm()
s.initialize_swarm()
s.omptimize()
'''


        


