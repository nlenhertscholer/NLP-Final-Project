# Base model to use for BOW implementations
import copy
from collections import Counter

import numpy as np
import torch
import torch.utils.data as tud
import torch.nn as nn
import sys

VOCAB_SIZE = 25_000
FAKE = "fake"
REAL = "real"


def word_tokenize(s):
    """Split the sentence into tokens"""
    return s.split()


class Model:

    def __init__(self, data):
        self.vocab = Counter([word for content, _ in data for word in word_tokenize(content)]).most_common(VOCAB_SIZE-1)
        self.wtoi = {k[0]: v+1 for v, k in enumerate(self.vocab)}
        self.wtoi["UNK"] = 0
        self.itow = {v: k for k, v in self.wtoi.items()}
        self.ltoi = {FAKE: 0, REAL: 1}
        self.itol = [FAKE, REAL]
        self.vocab = set(self.wtoi.keys())

    def train_model(self, train, dev):
        raise NotImplementedError

    def classify(self, data):
        raise NotImplementedError


class FakeNewsDataset(tud.Dataset):

    def __init__(self, word_to_idx, data):
        self.data = data
        self.wtoi = word_to_idx
        self.ltoi = {FAKE: 0, REAL: 1}
        self.vocab_size = VOCAB_SIZE

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        item = torch.from_numpy(np.zeros(self.vocab_size))

        for word in word_tokenize(self.data[i][0]):
            item[self.wtoi.get(word, 0)] += 1

        if len(self.data[i]) == 2:
            label = self.data[i][1]
        else:
            label = None

        return item, label


class BoWModel(nn.Module, Model):

    def __init__(self, data):
        nn.Module.__init__(self)
        Model.__init__(self, data)

        self.best_model = None
        self.num_epochs = 10

        # Implement these
        self.optimizer = None
        self.loss_fn = None

    def forward(self, bow):
        raise NotImplementedError

    def train_epoch(self, train_data):
        dataset = FakeNewsDataset(self.wtoi, train_data)
        dataloader = tud.DataLoader(dataset, batch_size=16, shuffle=True)
        self.train()

        for i, (X, y) in enumerate(dataloader):
            X = X.float()
            y = y.long()

            if torch.cuda.is_available():
                X = X.cuda()
                y = y.cuda()

            self.optimizer.zero_grad()
            predictions = self.forward(X)
            loss = self.loss_fn(predictions, y)
            loss.backward()

            if i % 250 == 0:
                print(f"Iter: [{i}/{len(dataloader)}] Loss: {loss.item()}")
            self.optimizer.step()

    def train_model(self, train_data, dev_data):
        best_acc = 0

        for epoch in range(self.num_epochs):
            self.train_epoch(train_data)
            dev_acc = self.evaluate(dev_data)
            print(f"Epoch [{epoch+1}/{self.num_epochs}] dev acc: {dev_acc})")

            if dev_acc > best_acc:
                self.best_model = copy.deepcopy(self)
                best_acc = dev_acc

    def classify(self, text):
        dataset = FakeNewsDataset(self.wtoi, text)
        dataloader = tud.DataLoader(dataset, batch_size=1, shuffle=False)
        results = []
        with torch.no_grad():
            for i, (X, _) in enumerate(dataloader):
                X = X.float()

                if torch.cuda.is_available():
                    X = X.cuda()

                preds = self.forward(X)
                results.append(preds.max(1)[1].cpu().numpy().reshape(-1))

        results = np.concatenate(results)
        # results = [self.itol[p] for p in results]
        return results

    def evaluate(self, data):
        self.eval()
        preds = self.classify(data)
        trgts = [d[1] for d in data]
        correct = 0.
        total = 0.

        for p, t in zip(preds, trgts):
            if p == t:
                correct += 1
            total += 1

        return correct/total
