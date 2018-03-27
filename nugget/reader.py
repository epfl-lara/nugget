# Copyright 2018 EPFL.

import torch
from torch import autograd

from nugget.expressions import *

class Reader(object):

    def __init__(self, file_name):
        self.file_name = file_name
        self.count = None

    def get_count(self):
        if self.count is None:
            self.count = sum(1 for _ in open(self.file_name))
        return self.count

    def entries(self):
        for line in open(self.file_name):
            [d, a, b, ms] = line[:-1].split(" ; ")
            ts = ms.split(" , ")
            t = transformations.index(ts[0])
            d = int(d)
            a = from_prefix_notation(a)
            b = from_prefix_notation(b)
            yield (a, b, d, t)

    def batch_entries(self, batch_size):
        batch_as = []
        batch_bs = []
        batch_ds = []
        batch_ts = []
        n = 0

        for (a, b, d, t) in self.entries():
            batch_as.append(a)
            batch_bs.append(b)
            batch_ds.append(d)
            batch_ts.append(t)
            n += 1

            if n == batch_size:
                yield (batch_as, batch_bs, batch_ds, batch_ts)

                batch_as = []
                batch_bs = []
                batch_ds = []
                batch_ts = []
                n = 0

        if n > 0:
            yield (batch_as, batch_bs, batch_ds, batch_ts)



