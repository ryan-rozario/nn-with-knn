import torch
import model
import copy 
import random
import scipy
import numpy as np

NEAREST_NEIGHBOURS=10

NUMBER_OF_INPUT_NODES = 50
NUMBER_OF_HIDDEN_NODES =20
NUMBER_OF_OUTPUT_NODES = 2

MAX_GENERATION = 10000
POPULATION_SIZE =20
VMAX = 0.4
C1 = 1.8
C2 = 1.8


class Swarm:
    def __init__(self,lp,size=POPULATION_SIZE,phi_1=C1,phi_2=C2,iter=MAX_GENERATION):

        self.size=size
        self.phi_p=phi_1
        self.phi_g=phi_2
        self.learning_parameter=lp   #check paper and set this

        self.max_iter = iter

        self.velocity=[]
        self.group = []
        self.global_best=None
        self.local_best=[]

    def initialize_swarm(self):
        self.group = [Particle() for i in range(self.size)]    #todo: Set boundaries
        self.velocity = [Particle() for i in range(self.size)] #todo: Set boundaries
        self.local_best=copy.deepcopy(self.group())


    def omptimize(self):
        iteration=0
        #untill termination condition
        while iteration<self.max_iter:
            self.update()
            iteration+=1

    def update(self):
        #random numbers have to be changed
        r_p=random.random()
        r_g=random.random()
        for i in range(self.size):

            #update velocities
            self.velocity[i].w1 = self.learning_parameter*self.velocity.w1  + self.phi_p*r_p*(self.local_best[i].w1-self.group[i].w1)+ self.phi_g*r_g*(self.global_best.w1-self.group[i].w1)
            self.velocity[i].w2 = self.learning_parameter*self.velocity.w2  + self.phi_p*r_p*(self.local_best[i].w2-self.group[i].w2)+ self.phi_g*r_g*(self.global_best.w2-self.group[i].w2)
            self.velocity[i].b1 = self.learning_parameter*self.velocity.b1  + self.phi_p*r_p*(self.local_best[i].b1-self.group[i].b1)+ self.phi_g*r_g*(self.global_best.b1-self.group[i].b1)
            self.velocity[i].b2 = self.learning_parameter*self.velocity.b2  + self.phi_p*r_p*(self.local_best[i].b2-self.group[i].b2)+ self.phi_g*r_g*(self.global_best.b2-self.group[i].b2)

            #update weights
            self.group[i].w1 = self.group[i].w1 + self.velocity[i].w1
            self.group[i].w2 = self.group[i].w2 + self.velocity[i].w2
            self.group[i].b1 = self.group[i].b1 + self.velocity[i].b1
            self.group[i].b2 = self.group[i].b2 + self.velocity[i].b2 

            if self.group[i].fitness > self.local_best[i].fitness:  #check conditions and sign later
                self.local_best[i] = copy.deepcopy(self.group[i])

                if self.group[i].fitness > self.global_best.fitness:  #check conditions and sign again
                    self.global_best = copy.deepcopy(self.group[i])

class Particle:
    def  __init__(self):
        self.w1 =torch.randn(NUMBER_OF_INPUT_NODES, NUMBER_OF_HIDDEN_NODES) # weight for hidden layer
        self.w2 =torch.randn(NUMBER_OF_HIDDEN_NODES, NUMBER_OF_OUTPUT_NODES) # weight for output layer

        ## initialize tensor variables for bias terms 
        self.b1 =torch.randn((1, NUMBER_OF_HIDDEN_NODES)) # bias for hidden layer
        self.b2 =torch.randn((1, NUMBER_OF_OUTPUT_NODES))

        self.fitness=None  #this had to be done
        self.output=None   #this has to be done

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

    def calc_fitness(self,inp_x,out_y):
        for i in range(len(inp_x)):
            self.output.append(model.Model(self.w1,self.w2,self.b1,self.b2).forward_propogation(inp_x[i]))
        self.output = scipy.stats.zscore(self.output)   #z-score function
        
        h=np.zeros((len(self.output),2))


        #normalized points constrained in hyperspace
        for i in range(len(self.output)):
            x_dist = np.linalg.norm(self.output[i])
            numerator=1-np.exp(-(x_dist/2))
            denominator= x_dist(1+np.exp(-(x_dist/2)))
            h[i]=self.output*(numerator/denominator)

        similarity_matrix = np.zeros((len(h),len(h)))
        distance_matric = np.zeros((len(self.output),len(self.output)))

        for i in range(len(h)):
            for j in range(i,len(h)):
                similarity = 2-(np.linalg.norm(h[i]-h[j]))
                similarity_matrix[i][j]=similarity
                similarity_matrix[j][i]=similarity

        






        



        s1 = similarity_self(self.output,out_y)
        s2 = similarity_nonself(self.output,out_y)

        #we have a weight for each class
        #return output as per equation 6 in the paper


    #gives similarity on inputing the h value of the respective points
    def similarity(self,h1,h2):
        return 2-np.linalg.norm(h1-h2)






        


