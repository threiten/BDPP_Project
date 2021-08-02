import torch.nn as nn
import torch


class LSTMMultiClass(nn.Module):


    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.3, device=torch.device('cpu')):

        super().__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.device = device

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)

        self.dropout = nn.Dropout(0.3)

        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x, hidden):

        batch_size = x.size(0)

        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)

        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        out = self.dropout(lstm_out)
        out = self.fc(out)

        out = out.view(batch_size, self.output_size, -1)
        out = out[:, :, -1]

        return out, hidden

    def init_hidden(self, batch_size):

        weight = next(self.parameters()).data

        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device))

        return hidden

    
