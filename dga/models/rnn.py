import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch


class RNN(nn.Module):
    """
    RNN Classifier.
    """

    def __init__(self, input_features=65, hidden_dim=12, n_layers=2, output_dim=1, embedding_dim=5, batch_size=10):
        super(RNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.hidden_layers = n_layers
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(input_features, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, n_layers, dropout=0.3, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, x_lens):
        """
        Perform a forward pass of our model on batch of tracks.
        """

        # x: (batch_size, longest_sequence, embedding) i.e. 10, 32, 5
        # hidden size: (hidden_layers, batch_size, hidden_dim) i.e. 2, 10, 30
        batch_size, seq_len = x.size()

        # x_embed: (batch_size, longest_sequence, 1?, embedding_size)
        embed_x = self.embedding(x)

        x_packed = pack_padded_sequence(embed_x, x_lens, batch_first=True, enforce_sorted=False)

        # Passing in the input and hidden state into the model and obtaining outputs
        output_packed, hidden_state = self.rnn(x_packed)

        output_padded, lengths = pad_packed_sequence(output_packed, batch_first=False)

        output = output_padded.view(batch_size*seq_len, self.hidden_dim)

        adjusted_lengths = [(l-1)*batch_size + i for i, l in enumerate(lengths)]

        lengthTensor = torch.tensor(adjusted_lengths, dtype=torch.int64)

        output = output.index_select(0, lengthTensor)

        output = output.view(batch_size, self.hidden_dim)

        output = self.sigmoid(self.fc(output))

        return output
