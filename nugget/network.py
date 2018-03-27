# Copyright 2018 EPFL.

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from treenet.treelstm import TreeLSTM

import nugget.expressions as expressions

class Net(nn.Module):

    def __init__(self, n_atoms):
        super(Net, self).__init__()

        self.embedding_size = 64
        self.layer1_size = 128
        self.layer2_size = 64
        self.layer3_size = 32
        self.classes_size = len(expressions.transformations)

        self.treelstm = TreeLSTM(n_atoms+expressions.NON_ATOM_ENTRIES,
            self.embedding_size, 2)
        self.fc1 = nn.Linear(2 * self.embedding_size, self.layer1_size)
        self.fc2 = nn.Linear(self.layer1_size, self.layer2_size)
        self.fc3 = nn.Linear(self.layer2_size, self.layer3_size)
        self.classes = nn.Linear(self.layer3_size, self.classes_size)

    def forward(self, e1s, a1s, e2s, a2s):

        first_embeddings = self.embeddings(e1s, a1s)
        second_embeddings = self.embeddings(e2s, a2s)

        return (self.distances(first_embeddings, second_embeddings),
                self.classifications(first_embeddings, second_embeddings))

    def embeddings(self, exprs, arities):
        return self.treelstm(exprs, arities)

    def distances(self, first_embeddings, second_embeddings):
        return torch.sum(
            torch.abs(second_embeddings - first_embeddings), dim=1)

    def classifications(self, first_embeddings, second_embeddings):
        hidden = torch.cat([first_embeddings, second_embeddings], dim=1)
        hidden = F.relu(self.fc1(hidden))
        hidden = F.relu(self.fc2(hidden))
        hidden = F.relu(self.fc3(hidden))
        return self.classes(hidden)


class Discount(nn.Module):

    def __init__(self, loss):
        super(Discount, self).__init__()
        self.loss = loss

    def forward(self, pred, target, discount):
        return self.loss.forward(pred, target) / discount




