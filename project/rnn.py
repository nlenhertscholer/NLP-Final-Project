# RNN.py
# by Nesta Lenhert-Scholer
#
# This file defines the character-level RNN model

import copy
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tud
import torch.nn.functional as F

START_CHAR = '<'
END_CHAR = '>'
PAD_CHAR = '#'


def load_data(filename: str) -> (torch.Tensor, list, dict):
    """
    Load the data from a given file. Assumes that file is split by lines
    Returns sentences and vocabulary (to index and from index)

    :param filename: string representing the path of formatted data
    :returns: (numerical sequence of characters as torch.Tensor, mapping from index to character, mapping from character to index)
    """

    with open(filename, "r", encoding="utf-8") as f:
        text = f.read()

    seq = [str(s+END_CHAR) for s in text.split(END_CHAR)][:-1]
    idx_to_char = list(set([t for s in seq for t in s]))
    char_to_idx = {token: idx_to_char.index(token) for token in idx_to_char}

    seq_idx = []
    for s in seq:
        seq_idx.append([char_to_idx[token] for token in s])
    sequences = [torch.LongTensor(x) for x in seq_idx]
    sequences = nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=char_to_idx[PAD_CHAR])

    return sequences, idx_to_char, char_to_idx


class CharNN(nn.Module):
    """
    Character Level Language Model Using RNN similar to
    'The Unreasonable Effectiveness of Recurrent Neural Networks'
    """

    def __init__(self, token_count, num_layers, emb_size, hid_size,
                 pad_idx, max_len, model="GRU", dropout=0.5, lr=0.001, sample_step=2,
                 num_samples=3, device="cuda"):
        """
        Initialization of CharNN

        :param token_count: Number of unique tokens
        :param num_layers: Number of layers for the GRU
        :param emb_size: Dimension of embedding layer
        :param hid_size: Dimension of hidden layer in GRU
        :param pad_idx: Index corresponding to padding layer to ignore
        :param max_len: Maximum length of a sequence
        :param model: Type of model, 'GRU' or 'LSTM'
        :param dropout: Amount of dropout to apply
        :param lr: Learning rate
        :param sample_step: How often to generate sample text during training
        :param num_samples: How many samples to generate during training
        :param device: Which device to use, 'cpu' or 'cuda'
        """

        super(CharNN, self).__init__()
        self.token_count = token_count
        self.num_layers = num_layers
        self.hid_size = hid_size
        self.model = model
        self.max_len = max_len
        self.sample_step = sample_step
        self.num_samples = num_samples
        self.device = device
        self.best_model = None

        self.emb = nn.Embedding(self.token_count, emb_size, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)

        if self.model is "GRU":
            self.rnn = nn.GRU(input_size=emb_size, hidden_size=self.hid_size,
                              num_layers=self.num_layers, batch_first=True)
        elif self.model is "LSTM":
            self.rnn = nn.LSTM(input_size=emb_size, hidden_size=self.hid_size,
                               num_layers=self.num_layers, batch_first=True)
        else:
            raise ValueError(f"Model must be one of GRU or LSTM, not {self.model}.")

        self.linear = nn.Linear(in_features=self.hid_size,
                                out_features=self.token_count)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)

        self.to(device)

    def forward(self, text, hidden):
        """
        Forward pass through the model

        :param text: numerical representation of character
        :param hidden: hidden vector output by previous timestep
        :return: (output vector, hidden vector)
        """

        emb = self.dropout(self.emb(text))
        out, hid = self.rnn(emb.view(text.shape[0], 1, -1), hidden)
        out = self.linear(out.view(-1, self.hid_size))
        return out, hid

    def train_model(self, train, val, char_to_idx, idx_to_char, epochs=10, batch_size=16):
        """
        Train the entire model

        :param train: training data
        :param val: validation data
        :param char_to_idx: char to index mapping
        :param idx_to_char: index to char mappoing
        :param epochs: Number of epochs to iterate over
        :param batch_size: batch size to input data for training
        """

        best_acc = 0

        for epoch in range(epochs):

            epoch_loss = self.train_epoch(train, batch_size)
            val_loss, acc = self.evaluate(val)

            print(f"Epoch {epoch + 1} out of {epochs}. "
                  f"Train Loss: {epoch_loss}, Val Loss: {val_loss}, Accuracy: {acc}")

            if acc > best_acc:
                # Save the best model
                self.best_model = copy.deepcopy(self)
                best_acc = acc

            if (epoch + 1) % self.sample_step == 0:
                # Generate samples after so many epochs
                for _ in range(self.num_samples):
                    gen_sent = self.generate_text(
                        char_to_idx, idx_to_char, self.max_len)
                    print(gen_sent)
                    print()
                # Uncomment this line to save the model as well
                # self.save_model(f"drive/My Drive/NLP_Final_Project_Data/models/wine_model_e{epoch}.pth")

    def train_epoch(self, data, batch_size):
        """
        Train a single epoch of data

        :param data: training data for the epoch
        :param batch_size: batch size to batch data
        :return: return total loss per word
        """

        dataloader = tud.DataLoader(data, batch_size=batch_size, shuffle=False)
        total_loss = 0
        self.train()

        for i, X in enumerate(dataloader):

            self.optimizer.zero_grad()

            output, target = self.fptt(X.to(self.device))

            loss = self.loss_fn(output.to(self.device), target.to(self.device))
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

            if i % ((len(dataloader)//10) - 1) == 0:
                print(f"Iter [{i + 1}/{len(dataloader)}] Loss/word = {total_loss / (i + 1)}")

        return total_loss / len(dataloader)

    def evaluate(self, data):
        """
        Evaluate the model based on given data

        :param data: validation data
        :return: (validation loss, training accuracy)
        """

        self.eval()
        dataloader = tud.DataLoader(data, batch_size=1, shuffle=False)
        val_loss = 0
        total = 0
        correct = 0

        with torch.no_grad():
            for i, X in enumerate(dataloader):

                output, target = self.fptt(X.to(self.device))

                loss = self.loss_fn(output.to(self.device), target.to(self.device))
                val_loss += loss.item()

                # Calculate the accuracy -> num_correct / total
                output = np.exp(output.cpu().numpy())
                output /= np.linalg.norm(output, axis=1, ord=1).reshape(-1, 1)

                for p, t in zip(output, target.cpu()):
                    if np.argmax(p) == t.item():
                        correct += 1
                    total += 1

        acc = correct / total

        return val_loss / len(dataloader), acc

    def fptt(self, X):
        """
        Forward propagation through time. Iterate through the tokens in the input and calculate the next character.

        :param X: Input sequence
        :return: (output of model, targets)
        """

        if self.model is "LSTM":
            hidden = (torch.rand(self.num_layers, X.shape[0], self.hid_size).to(self.device),
                      torch.rand(self.num_layers, X.shape[0], self.hid_size).to(self.device))
        else:
            hidden = torch.rand(self.num_layers, X.shape[0], self.hid_size).to(self.device)

        out = torch.Tensor(X.shape[0], self.max_len,  self.token_count)

        for t in range(self.max_len):
            in_t = X[:, t]
            out[:, t], hidden = self.forward(in_t, hidden)

        out = out[:, :-1, :].reshape(-1, self.token_count)      # Ignore the final character since it has no target
        trgt = X[:, 1:].reshape(-1)                             # Ignore the first character since it has no precedent

        return out, trgt

    def generate_text(self, ctoi, itoc, max_len, start_phrase=START_CHAR):
        """
        Generate text

        :param ctoi: char to index mapping
        :param itoc: index to char mapping
        :param max_len: maximum length of the generated sequence
        :param start_phrase: Seed phrase to generate the text off of
        :return: generated text
        """

        self.eval()

        if start_phrase[0] != START_CHAR:
            start_phrase = START_CHAR + start_phrase

        seq = [ctoi[t] for t in start_phrase]

        with torch.no_grad():
            input = torch.LongTensor([seq]).to(self.device)
            if self.model is "LSTM":
                hidden = (torch.rand(self.num_layers, 1, self.hid_size).to(self.device),
                          torch.rand(self.num_layers, 1, self.hid_size).to(self.device))
            else:
                hidden = torch.rand(self.num_layers, 1,
                                    self.hid_size).to(self.device)

            for t in range(len(seq) - 1):
                # Seed the model with the given sentence
                _, hidden = self.forward(input[:, t], hidden)

            for t in range(max_len - len(start_phrase)):
                input = torch.LongTensor([seq[-1]]).to(self.device)

                out, hidden = self.forward(input, hidden)

                probabilities = F.softmax(out.data[0].cpu(), dim=0).numpy()

                seq.append(np.random.choice(len(itoc), p=probabilities))

            # Convert the numerical representations back to characters
            string = ""
            for idx in seq:
                if itoc[idx] != START_CHAR and itoc[idx] != PAD_CHAR:
                    if itoc[idx] == END_CHAR:
                        break
                    else:
                        string += itoc[idx]

        return string

    def save_model(self, path):
        """
        Save the model to the given file

        :param path: file path to save model to
        """

        if self.best_model:
            torch.save(self.best_model, path)
        else:
            torch.save(self, path)
