import argparse
import os
import random
from timeit import default_timer

from nugget.generate import random_pair
from nugget.heuristics import *
from nugget.search import *
from nugget.utils import timeout

DEFAULT_LOGS_DIR = "logs/"
DEFAULT_ATOMS = list("abc")
DEFAULT_PENALTY = 0.0
DEFAULT_BATCH_SIZE = 128
DEFAULT_APPROX_DISTANCE = 60
DEFAULT_MIN_DEPTH = 2
DEFAULT_MAX_DEPTH = 4
DEFAULT_TIMEOUT = 5 * 60


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=
            "Evaluate the neural network on search problems "
            "generated on the fly.")
    parser.add_argument("model", type=str,
        help="trained model")
    parser.add_argument("-l", "--log-dir", type=str,
        help="output directory for the history logs",
        default=DEFAULT_LOGS_DIR)
    parser.add_argument("-n", "--number", type=int,
        help="number of examples to generate and test")
    parser.add_argument("-d", "--distance", type=int,
        help="approximate distance for the generated expressions",
        default=DEFAULT_APPROX_DISTANCE)
    parser.add_argument("--timeout", type=int,
        help="timeout for the search, in seconds",
        default=DEFAULT_TIMEOUT)
    parser.add_argument("--no-logs", action="store_true",
        help="disable logging of search history")
    parser.add_argument("-a", "--atoms", nargs="+", type=str,
        help="atoms",
        default=DEFAULT_ATOMS)
    parser.add_argument("--min-depth", type=int,
        help="minimum depth of the expressions",
        default=DEFAULT_MIN_DEPTH)
    parser.add_argument("--max-depth", type=int,
        help="maximum depth of the expressions",
        default=DEFAULT_MAX_DEPTH)
    parser.add_argument("-p", "--penalty", type=float,
        help="depth penalty",
        default=DEFAULT_PENALTY)
    parser.add_argument("-b", "--batch", help="batch size", type=int,
        default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--no-cuda", help="disable CUDA", action="store_true")
    parser.add_argument("--device", type=int, help="GPU device")
    parser.add_argument("-s", "--size", help="embedding size", type=int)

    args = parser.parse_args()

    h = Heuristics(args.atoms, args.model, args.size)
    cuda = not args.no_cuda and torch.cuda.is_available()
    if cuda:
        h_batch = Heuristics(args.atoms, args.model, args.size)
        h_batch.cuda(args.device)
        # We do not send h to CUDA, much slower on single queries.
    else:
        h_batch = h

    i = 0

    print(",".join([
        "id",
        "source",
        "target",
        "bfs_path_length",
        "bfs_visited_states",
        "bfs_time",
        "nngs_path_length",
        "nngs_visited_states",
        "nngs_time",
        "batched_nngs_path_length",
        "batched_nngs_visited_states",
        "batched_nngs_time"]))

    while args.number is None or i < args.number:

        depth = random.randint(args.min_depth, args.max_depth)
        (a, b) = random_pair(depth, args.atoms, args.distance)

        if a == b:
            continue

        def exp0():
            start_time = default_timer()
            (p0, a0, h0) = breadth_first_search(a, b)
            end_time = default_timer()
            d0 = end_time - start_time
            return (p0, a0, h0, d0)

        res0 = timeout(exp0, args.timeout)
        if res0 is not None:
            (p0, a0, h0, d0) = res0

        if res0 is not None and not args.no_logs:
            outputFile = open(os.path.join(args.log_dir,
                "{}-bfs.csv".format(i)), 'w')
            outputFile.write(history_to_csv(h0))
            outputFile.close()


        def exp1():
            start_time = default_timer()
            (p1, a1, h1) = best_first_search(a, b, h, args.penalty)
            end_time = default_timer()
            d1 = end_time - start_time
            return (p1, a1, h1, d1)

        res1 = timeout(exp1, args.timeout)
        if res1 is not None:
            (p1, a1, h1, d1) = res1

        if res1 is not None and not args.no_logs:
            outputFile = open(os.path.join(args.log_dir,
                "{}-nngs.csv".format(i)), 'w')
            outputFile.write(history_to_csv(h1))
            outputFile.close()

        def exp2():
            start_time = default_timer()
            (p2, a2, h2) = batch_best_first_search(a, b,
                h_batch, args.penalty, args.batch)
            end_time = default_timer()
            d2 = end_time - start_time
            return (p2, a2, h2, d2)

        res2 = timeout(exp2, args.timeout)
        if res2 is not None:
            (p2, a2, h2, d2) = res2

        if res2 is not None and not args.no_logs:
            outputFile = open(os.path.join(args.log_dir,
                "{}-batch-nngs.csv".format(i)), 'w')
            outputFile.write(history_to_csv(h2))
            outputFile.close()

        print(','.join([
            str(i),
            str(a.to_prefix_notation()),
            str(b.to_prefix_notation()),
            str(len(a0)) if res0 else "",
            str(len(h0)) if res0 else "",
            str(d0) if res0 else "",
            str(len(a1)) if res1 else "",
            str(len(h1)) if res1 else "",
            str(d1) if res1 else "",
            str(len(a2)) if res2 else "",
            str(len(h2)) if res2 else "",
            str(d2) if res2 else ""]))

        i += 1