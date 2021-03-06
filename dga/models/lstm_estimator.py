# torch imports
import torch.nn.functional as F
import torch.nn as nn


## TODO: Complete this classifier
class LstmEstimator(nn.Module):
    """
    LSTM Estimator for generating sequential-based track target variables.
    """

    ## TODO: Define the init function, the input params are required (for loading code in train.py to work)
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Initialize the model by setting up linear layers.
        Use the input parameters to help define the layers of your model.
        :param input_features: the number of input features in your training/test data
        :param hidden_dim: helps define the number of nodes in the hidden layer(s)
        :param output_dim: the number of outputs you want to produce
        """
        super(LstmEstimator, self).__init__()
        
        # The LSTM takes track features as inputs, and outputs hidden states
        # with dimensionality hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        
        self.hidden2target = nn.Linear(hidden_dim, output_dim)
        
    
    ## TODO: Define the feedforward behavior of the network
    def forward(self, input_playlist):
        """
        Perform a forward pass of our model on input features, x.
        :param input_track: A batch of input features of size (batch_size, input_features)
        :return: A single, sigmoid-activated value as output
        """
        
        # define the feedforward behavior
        lstm_out, _ = self.lstm(embeds.view(len(input_playlist), 1, -1))
        target_feat = self.hidden2target(lstm_out.view(len(input_playlist), -1))
        return target_feat