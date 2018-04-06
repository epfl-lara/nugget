import argparse
import os
from timeit import default_timer

from nugget.heuristics import *
from nugget.search import *

DEFAULT_DATA = "data/testing.txt"
DEFAULT_LOGS_DIR = "logs/"
DEFAULT_ATOMS = list("abc")
DEFAULT_PENALTY = 0.0
DEFAULT_BATCH_SIZE = 64

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Evaluate the neural network on search problems.")
    parser.add_argument("model", type=str,
        help="trained model")
    parser.add_argument("-d", "--data", type=str,
        help="testing data file",
        default=DEFAULT_DATA)
    parser.add_argument("-l", "--log-dir", type=str,
        help="output directory for the history logs",
        default=DEFAULT_LOGS_DIR)
    parser.add_argument("--no-logs", action="store_true",
        help="disable logging of search history")
    parser.add_argument("-a", "--atoms", nargs="+", type=str,
        help="atoms",
        default=DEFAULT_ATOMS)
    parser.add_argument("-p", "--penalty", type=float,
        help="depth penalty",
        default=DEFAULT_PENALTY)
    parser.add_argument("-b", "--batch", help="batch size", type=int,
        default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--no-cuda", help="disable CUDA", action="store_true")
    parser.add_argument("--device", type=int, help="GPU device")
    args = parser.parse_args()

    h = Heuristics(args.atoms, args.model)
    cuda = not args.no_cuda and torch.cuda.is_available()
    if cuda:
        h_batch = Heuristics(args.atoms, args.model)
        h_batch.cuda(args.device)
        # We do not send h to CUDA, much slower on single queries.
    else:
        h_batch = h

    i = 0

    print(",".join([
        "id",
        "bfs_path_length",
        "bfs_visited_states",
        "bfs_time",
        "nngs_path_length",
        "nngs_visited_states",
        "nngs_time",
        "batched_nngs_path_length",
        "batched_nngs_visited_states",
        "batched_nngs_time"]))

    for line in open(args.data):
        [d, a, b, t] = line[:-1].split(" ; ")
        t = transformations.index(t)
        d = int(d)
        a = from_prefix_notation(a)
        b = from_prefix_notation(b)

        if d == 0:
            continue

        start_time = default_timer()
        (p0, a0, h0) = breadth_first_search(a, b)
        end_time = default_timer()
        d0 = end_time - start_time

        if not args.no_logs:
            outputFile = open(os.path.join(args.log_dir,
                "{}-bfs.csv".format(i)), 'w')
            outputFile.write(history_to_csv(h0))
            outputFile.close()

        start_time = default_timer()
        (p1, a1, h1) = best_first_search(a, b, h, args.penalty)
        end_time = default_timer()
        d1 = end_time - start_time

        if not args.no_logs:
            outputFile = open(os.path.join(args.log_dir,
                "{}-nngs.csv".format(i)), 'w')
            outputFile.write(history_to_csv(h1))
            outputFile.close()

        start_time = default_timer()
        (p2, a2, h2) = batch_best_first_search(a, b,
            h_batch, args.penalty, args.batch)
        end_time = default_timer()
        d2 = end_time - start_time

        if not args.no_logs:
            outputFile = open(os.path.join(args.log_dir,
                "{}-batch-nngs.csv".format(i)), 'w')
            outputFile.write(history_to_csv(h2))
            outputFile.close()

        print(','.join([str(x) for x in [
            i,
            len(a0),
            len(h0),
            d0,
            len(a1),
            len(h1),
            d1,
            len(a2),
            len(h2),
            d2]]))

        i += 1