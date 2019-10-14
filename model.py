import torch
#import torch.nn as nn
#import torch.nn.functional as F

import numpy as np

NUMBER_OF_INPUT_NODES = 50
NUMBER_OF_OUTPUT_NODES = 2
NUMBER_OF_HIDDEN_NODES =20

## sigmoid activation function using pytorch
def sigmoid_activation(z):
    return 1 / (1 + torch.exp(-z))


class Model():
    def __init__(self):
        super().__init__(self,w1,w2,b1,b2)
        ## initialize tensor variables for weights 
        self.w1 = w1 #torch.randn(NUMBER_OF_INPUT_NODES, NUMBER_OF_HIDDEN_NODES) # weight for hidden layer
        self.w2 = w2 #torch.randn(NUMBER_OF_HIDDEN_NODES, NUMBER_OF_OUTPUT_NODES) # weight for output layer

        ## initialize tensor variables for bias terms 
        self.b1 = b1 #torch.randn((1, NUMBER_OF_HIDDEN_NODES)) # bias for hidden layer
        self.b2 = b2 #torch.randn((1, NUMBER_OF_OUTPUT_NODES)) # bias for output layer

    def forward_propogation(self,x):
        ## activation of hidden layer 
        z1 = torch.mm(x, self.w1) + self.b1
        a1 = sigmoid_activation(z1)  #replace with some other activation function if necessary

        ## activation (output) of final layer 
        z2 = torch.mm(a1, self.w2) + self.b2
        output = sigmoid_activation(z2)     #replace with some other activation function if necessary

        return output
    


def main():
    ## initialize tensor for inputs, and outputs 




if __name__ == "__main__":
    main()




'''
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # Inputs to hidden layer linear transformation
        self.hidden = nn.Linear(NUMBER_OF_INPUT_NODES, 20)

        #Output layer - one for each dimesion
        self.output = nn.Linear(20, NUMBER_OF_OUTPUT_NODES)



class Network(nn.Module):
    def __init__(self):
        super().__init__()



        # Define sigmoid activation
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.output(x)

        return x


'''