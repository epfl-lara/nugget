import argparse
import os
import sys

import torch
from torch import autograd
from torch import nn
from torch import optim

from nugget.encoder import ExpressionEncoder
from nugget.network import Net, Discount
from nugget.reader import Reader

DEFAULT_MODELS_DIR = "models/"
DEFAULT_ATOMS = list("abc")
DEFAULT_BATCH_SIZE = 32
DEFAULT_START = 0
DEFAULT_EPOCHS = 10

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Measure the MAE and accuracy of the models.")
    parser.add_argument("name", help="model name", type=str)
    parser.add_argument("data", help="data file", type=str)
    parser.add_argument("-s", "--start", help="epoch at which to start",
        type=int,
        default=DEFAULT_START)
    parser.add_argument("-e", "--epochs", help="number of epochs", type=int,
        default=DEFAULT_EPOCHS)
    parser.add_argument("-b", "--batch", help="batch size", type=int,
        default=DEFAULT_BATCH_SIZE)
    parser.add_argument("-a", "--atoms", help="atoms", nargs="+", type=str,
        default=DEFAULT_ATOMS)
    parser.add_argument("-m", "--models-dir", help="models directory", type=str,
        default=DEFAULT_MODELS_DIR)
    parser.add_argument("--no-cuda", help="disable CUDA", action="store_true")
    parser.add_argument("--device", type=int, help="GPU device")
    parser.add_argument("--size", help="embedding size", type=int)
    args = parser.parse_args()

    if args.size is not None:
        net = Net(len(args.atoms), args.size)
    else:
        net = Net(len(args.atoms))

    cuda = not args.no_cuda and torch.cuda.is_available()
    if cuda:
        net.cuda(args.device)

    reader = Reader(args.data)
    expr_encoder = ExpressionEncoder(args.atoms)

    distLoss = Discount(nn.MSELoss())
    classLoss = Discount(nn.CrossEntropyLoss())

    print(",".join(["epoch", "mae", "accuracy", "dmse", "dce"] +
        ["mae_distance_" + str(d) for d in range(1, 11)] +
        ["accuracy_distance_" + str(d) for d in range(1, 11)]))

    for i in range(args.start, args.start + args.epochs):
        model_file = os.path.join(args.models_dir, "{}-{}.model".format(args.name, i))
        net.load_state_dict(torch.load(model_file, map_location=lambda x, _: x.cpu()))

        n = 0
        distances_count = {}
        error_per_distances = {}
        correct_per_distances = {}
        for d in range(1, 11):
            distances_count[d] = 0
            correct_per_distances[d] = 0
            error_per_distances[d] = 0.0
        error_distance = 0.0
        absolute_error_distance = 0.0
        error_classes = 0.0
        classification_accuracy_classes = 0.0
        for (e1s, e2s, ds, ts) in reader.batch_entries(args.batch):
            step_size = len(ds)
            ds = autograd.Variable(torch.Tensor(ds))
            ts = autograd.Variable(torch.LongTensor(ts))

            e1s, a1s = expr_encoder.encode_batch(e1s)
            e2s, a2s = expr_encoder.encode_batch(e2s)

            e1s = autograd.Variable(e1s)
            e2s = autograd.Variable(e2s)
            if cuda:
                e1s = e1s.cuda(args.device)
                a1s = a1s.cuda(args.device)
                e2s = e2s.cuda(args.device)
                a2s = a2s.cuda(args.device)
                ds = ds.cuda(args.device)
                ts = ts.cuda(args.device)

            (out_ds, out_css) = net.forward(e1s, a1s, e2s, a2s)

            dl = torch.sum(distLoss(out_ds, ds, ds))
            cl = torch.sum(classLoss(out_css, ts, ds))

            error_distance += dl.data[0]
            error_classes += cl.data[0]

            abs_distance_diffs = torch.abs(out_ds - ds)
            correct_predictions = (torch.max(out_css, 1)[1] == ts).float()
            absolute_error_distance += torch.sum(abs_distance_diffs).data[0]
            classification_accuracy_classes += torch.sum(correct_predictions).data[0]

            for d in range(1, 11):
                at_distance = (ds == d).float()
                distances_count[d] += torch.sum(at_distance).data[0]
                error_per_distances[d] += torch.sum(at_distance * abs_distance_diffs).data[0]
                correct_per_distances[d] += torch.sum(at_distance * correct_predictions).data[0]

            n += step_size

        print(",".join([str(x) for x in [
            i,
            absolute_error_distance / n,
            classification_accuracy_classes / n,
            error_distance / n,
            error_classes / n,
        ] +
        [error_per_distances[d] / distances_count[d]\
            for d in range(1, 11)] +
        [correct_per_distances[d] / distances_count[d]\
            for d in range(1, 11)]]))


