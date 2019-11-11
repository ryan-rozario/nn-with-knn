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



def accuracy(final_nn,test,train):
    '''
    Obtain the accuracy for the neural network
    '''

    final_nn.forward(train[:,1:])
    #final_nn.kmeans_eval(train[:,1:])
    train_points=final_nn.output
    final_nn.forward(test[:,1:])
    #final_nn.kmeans_eval(test[:,1:])
    test_points=final_nn.output
    knn=KNeighborsClassifier(n_neighbors=10)
    #print(train[:,0])
    knn.fit(train_points,train[:,0])
    y_pred=knn.predict(test_points)
    acc = metrics.accuracy_score(test[:,0],y_pred)
    fscr = metrics.f1_score(test[:,0],y_pred, average='macro',labels=np.unique(y_pred))

    
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    N = 4 # Number of labels

    # setup the plot
    fig, ax = plt.subplots(1,1, figsize=(6,6))
    # define the data
    x = train_points[:,0]
    y = train_points[:,1]
    tag = train[:,0] # Tag each point with a corresponding label    
    #print(train_points)
    # define the colormap
    cmap = plt.cm.jet
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # create the new map
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

    # define the bins and normalize
    bounds = np.linspace(0,N,N+1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    # make the scatter
    scat = ax.scatter(x,y,c=tag,cmap=cmap,     norm=norm)
    # create the colorbar
    cb = plt.colorbar(scat, spacing='proportional',ticks=bounds)
    cb.set_label('Custom cbar')
    plt.show()
    
    return acc , fscr


    #print(test_points)
    #print(train_points)