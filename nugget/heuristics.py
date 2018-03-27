# Copyright 2018 EPFL.

import torch

from nugget.encoder import ExpressionEncoder
from nugget.network import Net

class Heuristics(object):

    def __init__(self, atoms, model):
        self.model = Net(len(atoms))
        self.model.load_state_dict(torch.load(model))
        self.encoder = ExpressionEncoder(atoms)

    def with_target(self, target):
        e2s, a2s = self.encoder.encode_batch([target])
        e2s = torch.autograd.Variable(e2s)
        target_embeddings = self.model.embeddings(e2s, a2s)

        def apply(source):
            e1s, a1s = self.encoder.encode_batch([source])
            e1s = torch.autograd.Variable(e1s)
            source_embeddings = self.model.embeddings(e1s, a1s)
            distance = self.model.distances(source_embeddings,
                target_embeddings)
            classes = self.model.classifications(
                source_embeddings, target_embeddings)
            predicted_distance = distance.data.squeeze(0)[0]
            predicted_classes = classes.data.squeeze(0)
            return (predicted_distance, list(predicted_classes))

        return apply


