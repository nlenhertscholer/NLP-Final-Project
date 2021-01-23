# Neural Network with bag of words
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from helper import bow_model

os.chdir("../")     # Change path to root project directory


class NNBoW(bow_model.BoWModel):

    def __init__(self, data):
        bow_model.BoWModel.__init__(self, data)

        self.l1 = nn.Linear(bow_model.VOCAB_SIZE, 32)
        self.l2 = nn.Linear(32, 2)

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def forward(self, bow):

        out = self.l1(bow)
        out = F.relu(out)
        out = self.l2(out)

        return out


# Read in the data
train = pd.read_csv("data/train.csv").to_numpy()
valid = pd.read_csv("data/validate.csv").to_numpy()
test = pd.read_csv("data/test.csv").to_numpy()

nn_bow_model = NNBoW(train)
if torch.cuda.is_available():
    nn_bow_model = nn_bow_model.cuda()
nn_bow_model.train_model(train, valid)
