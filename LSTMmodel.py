import torch.nn as nn
import torch
import numpy as np
from utils import accuracy


class LSTMMultiClass(nn.Module):

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.3, device=torch.device('cpu')):

        super().__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.device = device

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,
                            n_layers, dropout=drop_prob, batch_first=True)

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


class LSTMMultiClassWrapper:

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.3, device=torch.device('cpu')):

        self.net = LSTMMultiClass(
            vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob, device)
        self.propeties = {
            'vocab_size': vocab_size,
            'output_size': output_size,
            'embedding_dim': embedding_dim,
            'hidden_dim': hidden_dim,
            'n_layers': n_layers,
            'drop_prob': drop_prob,
            'device': device
        }
        self.device = device
        self.model_trained = False
        self.model_loaded = False

    def train(self, train_loader, valid_loader, lr=0.001, epochs=4, batch_size=50, weights=None):

        if weights is not None:
            print('Doing weighted training!')
            weights = weights.float().to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        val_losses_mean = []
        val_acc_mean = []

        counter = 0
        print_every = 50
        clip = 10

        self.net.to(self.device)

        self.net = self.net.train()

        for e in range(epochs):

            counter_epoch = 0
            h = self.net.init_hidden(batch_size)

            for inputs, labels in train_loader:
                counter += 1
                counter_epoch += 1

                h = tuple([each.data for each in h])
                h = tuple([each.to(self.device) for each in h])

                self.net.zero_grad()

                inputs = inputs.type(torch.LongTensor)
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                output, h = self.net(inputs, h)

                loss = self.criterion(output, labels)
                loss.backward()

                nn.utils.clip_grad_norm_(self.net.parameters(), clip)
                optimizer.step()

                if counter % print_every == 0:

                    val_h = self.net.init_hidden(batch_size)
                    val_losses = []
                    val_accs = []
                    self.net.eval()

                    for inputs_val, labels_val in valid_loader:

                        val_h = tuple([each.data for each in val_h])
                        val_h = tuple([each.to(self.device) for each in val_h])

                        inputs_val = inputs_val.type(torch.LongTensor)
                        inputs_val, labels_val = inputs_val.to(
                            self.device), labels_val.to(self.device)

                        output_val, val_h = self.net(inputs_val, val_h)
                        val_loss = self.criterion(output_val, labels_val)
                        val_acc = accuracy(output_val, labels_val)

                        val_losses.append(val_loss.item())
                        val_accs.append(val_acc.item())

                    self.net.train()
                    val_losses_mean.append(np.mean(val_losses))
                    val_acc_mean.append(np.mean(val_accs))

                    print("Epoch: {}/{}...".format(e+1, epochs),
                          "Step: {}...".format(counter),
                          "Loss: {:.6f}...".format(loss.item()),
                          "Val Loss: {:.6f}".format(np.mean(val_losses)),
                          "Val Acc: {:.6f}".format(np.mean(val_accs)))

                if e > 1 and counter_epoch > 10*print_every:
                    if np.amin(np.mean(val_losses_mean[:-5])) < np.amin(val_losses_mean[-5:]):
                        print(
                            'Early stopping, val loss did not improve over last 5 times it was evaluated!')
                        return

        self.model_trained = True

    def predict(self, inputs, return_probs=False):

        self.net.eval()

        h = self.net.init_hidden(inputs.size(0))
        inputs.to(self.device)

        output, h = self.net(inputs, h)
        y_pred_softmax = torch.log_softmax(output, dim=1)
        _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

        if return_probs:
            return torch.exp(y_pred_softmax)

        return y_pred_tags

    def loadModel(self, path, force=False, use_state_dict=False):

        if self.model_trained and not force:
            raise AttributeError(
                "The model is trained already. Set force to True if this should be overwritten.")
        elif self.model_loaded and not force:
            raise AttributeError(
                "Model already loaded, set force to True to overwrite.")
        elif use_state_dict:
            self.net.load_state_dict(torch.load(
                path, map_location=self.device))
            self.model_loaded = True
        else:
            self.net = torch.load(path, map_location=self.device)
            self.model_loaded = True

    def save(self, path, use_state_dict=False):
        if use_state_dict:
            torch.save(self.net.state_dict(), path)
        else:
            torch.save(self.net, path)
