import argparse
import os
import time

from nugget.heuristics import *
from nugget.search import *

DEFAULT_DATA = "data/testing.txt"
DEFAULT_LOGS_DIR = "logs/"
DEFAULT_ATOMS = list("abc")
DEFAULT_PENALTY = 0.0

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
    args = parser.parse_args()

    h = Heuristics(args.atoms, args.model)
    i = 0

    print(",".join([
        "id",
        "bfs_path_length",
        "bfs_visited_states",
        "bfs_time",
        "nngs_path_length",
        "nngs_visited_states",
        "nngs_time"]))

    for line in open(args.data):
        [d, a, b, t] = line[:-1].split(" ; ")
        t = transformations.index(t)
        d = int(d)
        a = from_prefix_notation(a)
        b = from_prefix_notation(b)

        if d == 0:
            continue

        start_time = time.clock()
        (p0, a0, h0) = breadth_first_search(a, b)
        end_time = time.clock()
        d0 = end_time - start_time

        if not args.no_logs:
            outputFile = open(os.path.join(args.log_dir,
                "{}-bfs.csv".format(i)), 'w')
            outputFile.write(history_to_csv(h0))
            outputFile.close()

        start_time = time.clock()
        (p1, a1, h1) = best_first_search(a, b, h, args.penalty)
        end_time = time.clock()
        d1 = end_time - start_time

        if not args.no_logs:
            outputFile = open(os.path.join(args.log_dir,
                "{}-nngs.csv".format(i)), 'w')
            outputFile.write(history_to_csv(h1))
            outputFile.close()

        print(','.join([str(x) for x in [
            i,
            len(a0),
            len(h0),
            d0,
            len(a1),
            len(h1),
            d1]]))

        i += 1