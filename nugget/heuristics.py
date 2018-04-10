# Copyright 2018 EPFL.

import torch

from nugget.encoder import ExpressionEncoder
from nugget.network import Net

class Heuristics(object):

    def __init__(self, atoms, model, embedding_size=None):
        if embedding_size is not None:
            self.model = Net(len(atoms), embedding_size)
        else:
            self.model = Net(len(atoms))
        self.model.load_state_dict(torch.load(model,
            map_location=lambda x, _: x.cpu()))
        self.encoder = ExpressionEncoder(atoms)
        self.is_cuda = False
        self.device = None

    def cuda(self, device=None):
        self.is_cuda = True
        self.device = device
        self.model.cuda(device)

    def with_target_batch(self, target):
        e2s, a2s = self.encoder.encode_batch([target])
        e2s = torch.autograd.Variable(e2s)
        if self.is_cuda:
            e2s = e2s.cuda(self.device)
            a2s = a2s.cuda(self.device)
        target_embeddings = self.model.embeddings(e2s, a2s)

        def apply(source):
            e1s, a1s = self.encoder.encode_batch(source)
            e1s = torch.autograd.Variable(e1s)
            if self.is_cuda:
                e1s = e1s.cuda(self.device)
                a1s = a1s.cuda(self.device)
            source_embeddings = self.model.embeddings(e1s, a1s)
            exp_target_embeddings = target_embeddings.expand(
                * source_embeddings.size())
            distance = self.model.distances(source_embeddings,
                exp_target_embeddings)
            return list(distance.data)

        return apply

    def with_target(self, target):
        e2s, a2s = self.encoder.encode_batch([target])
        e2s = torch.autograd.Variable(e2s)
        if self.is_cuda:
            e2s = e2s.cuda(self.device)
            a2s = a2s.cuda(self.device)
        target_embeddings = self.model.embeddings(e2s, a2s)

        def apply(source):
            single = False
            if not isinstance(source, (list, tuple)):
                source = [source]
                single = True

            e1s, a1s = self.encoder.encode_batch(source)
            e1s = torch.autograd.Variable(e1s)
            if self.is_cuda:
                e1s = e1s.cuda(self.device)
                a1s = a1s.cuda(self.device)
            source_embeddings = self.model.embeddings(e1s, a1s)
            exp_target_embeddings = target_embeddings.expand(
                * source_embeddings.size())
            distance = self.model.distances(source_embeddings,
                exp_target_embeddings)
            classes = self.model.classifications(
                source_embeddings, exp_target_embeddings)
            if single:
                predicted_distance = distance.data.squeeze(0)[0]
                predicted_classes = classes.data.squeeze(0)
                return (predicted_distance, list(predicted_classes))
            else:
                predicted_distance = list(distance.data)
                predicted_classes = [list(c) for c in classes.data]
                return (predicted_distance, predicted_classes)

        return apply


