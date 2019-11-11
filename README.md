# Reducing Dimensionality of data using a Neural Network trained using PSO to apply KNN
## Paper
Base Paper: L. Wang, B. Yang, Y. Chen, X. Zhang and J. Orchard, "Improving Neural-Network Classifiers Using Nearest Neighbor Partitioning," in IEEE Transactions on Neural Networks and Learning Systems, vol. 28, no. 10, pp. 2255-2267, Oct. 2017. doi: 10.1109/TNNLS.2016.2580570

URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7502076&isnumber=8038919


Paper for Particle Swarm Optimization: R. Eberhart and J. Kennedy, “A new optimizer using particle swarm theory,” in Proc. 6th Int. Symp. Micro Mach. Human Sci., 1995 pp. 39–43.
## Implementation
A neural network is used to reduce the dimensionality of data, so as to perform KNN faster.
This neural network is trained using particle swarm optimization

* [pso](pso.py) contains code for the neural network and training it using pso
* [knn_code](knn_code.py) conains code for knn to test a neural network for accuracy
* [final_code](final_code.py) loads the dataset and does k-fold validation

### How the code works
* k-fold Validation
Dataset ->  Test Set and Training Set
* Training neural network
Adjust the weights of the neural network using Particle Swarm Optimization through the training set
* KNN Accuracy
Test your model for accuracy using the testing set

## Tasks
- [x] Code the neural network
- [x] Code the particle swarm optimizer
- [x] Code for k-fold validation and knn
- [x] Test for accuracy on different datasets

## Parameters
Parameters in [pso](pso.py) related to the neural network

* CLASS_NUM : number_of_classes in the dataset
* ALPHA: discriminant weight
* NEAREST_NEIGHBOURS: number of nearest neighbours
* NUMBER_OF_INPUT_NODES: input nodes of neural network should be equal to the number of features
* NUMBER_OF_HIDDEN_NODES: number of hidden nodes
* NUMBER_OF_OUTPUT_NODES: number of output nodes should be equal to  dimensions of partition space
* MAX_GENERATION : maximum number of iterations 
* POPULATION_SIZE: number of indivisuals in a population


Parameters in [final_code](final_code.py) related to the dataset input
* FILENAME: path to file
* class_flag: set the class_flag to -1 if the class label is the last column or set class_flag to 0 if class label is first column


By default you will be asked for number of classes, alpha, number of output nodes, filename, class_flag when you run the program


## Datasets Used
From the paper:
Well-known classification data sets were selected from the UCI machine learning repository (http://archive.ics.uci.edu/ml/) for the experiments, including 
* Ionosphere Data Set(Ionosphere),
* Wisconsin Breast Cancer Diagnostic Data Set (WBCD),
* Fertility Data Set (Fertility), 
* Haberman’s Survival Data Set (Haberman), 
* Parkinsons Data Set (Parkinsons), 
* Iris Data Set (IRIS), 
* Wine Data Set (Wine), 
* Contraceptive Method Choice Data Set (CMC),
* Seeds Data Set (Seeds), 
* Glass Identification Data Set (Glass), 
* Zoo Data Set (Zoo).

## Requirements
* python3
* numpy
* scikit-learn