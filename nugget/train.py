# Copyright 2018 EPFL.

import argparse
import os
import sys
import uuid

import torch
from torch import autograd
from torch import nn
from torch import optim

from nugget.encoder import ExpressionEncoder
from nugget.network import Net, Discount
from nugget.reader import Reader

DEFAULT_OUTPUT_DIR = "models/"
DEFAULT_TRAINING_FILE = "data/training.txt"
DEFAULT_VALIDATION_FILE = "data/validation.txt"
DEFAULT_ATOMS = list("abc")
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 10

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train the neural network.")
    parser.add_argument("-t", "--training", help="training file", type=str,
        default=DEFAULT_TRAINING_FILE)
    parser.add_argument("-v", "--validation", help="validation file", type=str,
        default=DEFAULT_VALIDATION_FILE)
    parser.add_argument("-m", "--model", help="initial model", type=str)
    parser.add_argument("-e", "--epochs", help="number of epochs", type=int,
        default=DEFAULT_EPOCHS)
    parser.add_argument("-b", "--batch", help="batch size", type=int,
        default=DEFAULT_BATCH_SIZE)
    parser.add_argument("-a", "--atoms", help="atoms", nargs="+", type=str,
        default=DEFAULT_ATOMS)
    parser.add_argument("-o", "--out", help="output directory", type=str,
        default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("-n", "--name", help="output model name", type=str)
    parser.add_argument("--no-cuda", help="disable CUDA", action="store_true")
    args = parser.parse_args()

    net = Net(len(args.atoms))
    if args.model:
        print("Loading weights from model: {}".format(args.model))
        net.load_state_dict(torch.load(args.model))


    name = args.name if args.name is not None else str(uuid.uuid4())
    if not args.name:
        print("Generated model name: {}".format(name))


    cuda = not args.no_cuda and torch.cuda.is_available()
    if cuda:
        print("CUDA enabled.")
        net.cuda()

    opt = optim.Adam(net.parameters())
    distLoss = Discount(nn.MSELoss())
    classLoss = Discount(nn.CrossEntropyLoss())

    expr_encoder = ExpressionEncoder(args.atoms)

    reader = Reader(args.training)
    total = reader.get_count()
    val_reader = Reader(args.validation)
    val_total = val_reader.get_count()

    for i in range(args.epochs):
        title = "EPOCH {}".format(i+1)
        print()
        print(title)
        print("=" * len(title))
        print()
        print("TRAINING")
        print("--------")

        print("\n\n\n\n\n")

        n = 0
        error_distance = 0.0
        absolute_error_distance = 0.0
        error_classes = 0.0
        classification_accuracy_classes = 0.0
        for (e1s, e2s, ds, ts) in reader.batch_entries(args.batch):
            step_size = len(ds)
            opt.zero_grad()
            ds = autograd.Variable(torch.Tensor(ds))
            ts = autograd.Variable(torch.LongTensor(ts))

            e1s, a1s = expr_encoder.encode_batch(e1s)
            e2s, a2s = expr_encoder.encode_batch(e2s)

            e1s = autograd.Variable(e1s)
            e2s = autograd.Variable(e2s)
            if cuda:
                e1s.cuda()
                a1s.cuda()
                e2s.cuda()
                a2s.cuda()
                ds.cuda()
                ts.cuda()

            (out_ds, out_css) = net.forward(e1s, a1s, e2s, a2s)

            dl = torch.sum(distLoss(out_ds, ds, ds))
            cl = torch.sum(classLoss(out_css, ts, ds))
            autograd.backward([dl, cl])
            opt.step()

            error_distance += dl.data[0]
            error_classes += cl.data[0]

            absolute_error_distance += torch.sum(torch.abs(out_ds - ds)).data[0]
            classification_accuracy_classes += torch.sum(
                (torch.max(out_css, 1)[1] == ts).float()).data[0]

            n += step_size

            sys.stdout.write("\033[F")
            sys.stdout.write("\033[F")
            sys.stdout.write("\033[F")
            sys.stdout.write("\033[F")
            sys.stdout.write("\033[F")
            sys.stdout.write("\033[K")
            print("{0:{1}d} / {2}".format(n, len(str(total)), total))
            sys.stdout.write("\033[K")
            print("Discounted MSE for distances: {}".format(error_distance / n))
            sys.stdout.write("\033[K")
            print("MAE for distances: {}".format(absolute_error_distance / n))
            sys.stdout.write("\033[K")
            print("Discounted CE for classes: {}".format(error_classes / n))
            sys.stdout.write("\033[K")
            print("Accuracy for classes: {}".format(
                classification_accuracy_classes / n))

        file_name = os.path.join(args.out, name + '-' + str(i) + '.model')
        print("Saving model under: {}\n".format(file_name))
        torch.save(net.state_dict(), file_name)

        print("VALIDATION")
        print("----------")

        print("\n\n\n")

        n = 0
        absolute_error_distance = 0.0
        classification_accuracy_classes = 0.0
        for (e1s, e2s, ds, ts) in val_reader.batch_entries(args.batch):
            step_size = len(ds)
            ds = autograd.Variable(torch.Tensor(ds))
            ts = autograd.Variable(torch.LongTensor(ts))

            e1s, a1s = expr_encoder.encode_batch(e1s)
            e2s, a2s = expr_encoder.encode_batch(e2s)

            e1s = autograd.Variable(e1s)
            e2s = autograd.Variable(e2s)
            if cuda:
                e1s.cuda()
                a1s.cuda()
                e2s.cuda()
                a2s.cuda()
                ds.cuda()
                ts.cuda()

            (out_ds, out_css) = net.forward(e1s, a1s, e2s, a2s)

            absolute_error_distance += torch.sum(torch.abs(out_ds - ds)).data[0]
            classification_accuracy_classes += torch.sum(
                (torch.max(out_css, 1)[1] == ts).float()).data[0]

            n += step_size

            sys.stdout.write("\033[F")
            sys.stdout.write("\033[F")
            sys.stdout.write("\033[F")
            sys.stdout.write("\033[K")
            print("{0:{1}d} / {2}".format(n, len(str(val_total)), val_total))
            sys.stdout.write("\033[K")
            print("MAE for distances: {}".format(absolute_error_distance / n))
            sys.stdout.write("\033[K")
            print("Accuracy for classes: {}".format(
                classification_accuracy_classes / n))


