import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

NUMBER_OF_INPUT_NODES = 50
NUMBER_OF_OUTPUT_NODES = 2
NUMBER_OF_HIDDEN_NODES =20



class Network(nn.Module):
    def __init__(self):
        super().__init__()

        # Inputs to hidden layer linear transformation
        self.hidden = nn.Linear(NUMBER_OF_INPUT_NODES, 20)

        #Output layer - one for each dimesion
        self.output = nn.Linear(20, NUMBER_OF_OUTPUT_NODES)

        # Define sigmoid activation
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.output(x)

        return x


