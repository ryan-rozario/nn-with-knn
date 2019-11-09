'''
This file contains the main code for the knn and checking accuracy
based on the paper L. Wang, B. Yang, Y. Chen, X. Zhang and J. Orchard, "Improving Neural-Network Classifiers Using Nearest Neighbor Partitioning," in IEEE Transactions on Neural Networks and Learning Systems, vol. 28, no. 10, pp. 2255-2267, Oct. 2017.
Tests the neural network obtained using test and training set by applying knn
'''

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


def sigmoid(Z):
    '''
    Sigmoid function
    '''
    return 1/(1+np.exp(-Z))


def forward(final_nn,inp_x,activation="sigmoid"):
    '''
    Map the points to reduced dimensionality partition space using the obtained neural network
    '''

    if activation is "sigmoid":
        activation = sigmoid
    else:
        raise Exception('Non-supported activation function')
    ## activation of hidden layer 
    z1 = np.dot(inp_x, final_nn.w1) + final_nn.b1
    #print(z1[0])
    #a1 = sigmoid(z1)
    #print(a1[0])
    ## activation (output) of final layer 
    z2 = np.dot(z1, final_nn.w2) + final_nn.b2
    #a2 = activation(z2)

    
    output=z2

    return output


def accuracy(final_nn,test,train):
    '''
    Obtain the accuracy for the neural network
    '''
    train_points=forward(final_nn,train[:,1:])
    test_points=forward(final_nn,test[:,1:])
    knn=KNeighborsClassifier(n_neighbors=10)
    #print(train[:,0])
    knn.fit(train_points,train[:,0])
    y_pred=knn.predict(test_points)
    score = metrics.accuracy_score(test[:,0],y_pred)

    return score


    #print(test_points)
    #print(train_points)