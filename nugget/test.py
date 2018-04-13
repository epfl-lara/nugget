import argparse
import os
from timeit import default_timer

from nugget.heuristics import *
from nugget.reader import *
from nugget.search import *
from nugget.utils import timeout

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
    parser.add_argument("--skip-bfs", help="disable BFS", action="store_true")
    parser.add_argument("-s", "--size", help="embedding size", type=int)
    parser.add_argument("--device", type=int, help="GPU device")
    parser.add_argument("--timeout", type=int, help="Timeout")
    args = parser.parse_args()

    h = Heuristics(args.atoms, args.model, args.size)
    cuda = not args.no_cuda and torch.cuda.is_available()
    if cuda:
        h_batch = Heuristics(args.atoms, args.model)
        h_batch.cuda(args.device)
        # We do not send h to CUDA, much slower on single queries.
    else:
        h_batch = h

    reader = Reader(args.data)

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

    for (a, b, _, _) in reader.entries():

        # BFS

        if not args.skip_bfs:
            def exp0():
                start_time = default_timer()
                (p0, a0, h0) = breadth_first_search(a, b)
                end_time = default_timer()
                d0 = end_time - start_time
                return (p0, a0, h0, d0)

            if args.timeout is None:
                (p0, a0, h0, d0) = exp0()
                valid0 = True
            else:
                res0 = timeout(exp0, args.timeout)
                valid0 = res0 is not None
                if valid0:
                    (p0, a0, h0, d0) = res0

            if not args.no_logs and valid0:
                outputFile = open(os.path.join(args.log_dir,
                    "{}-bfs.csv".format(i)), 'w')
                outputFile.write(history_to_csv(h0))
                outputFile.close()
        else:
            valid0 = False

        # NNGS

        def exp1():
            start_time = default_timer()
            (p1, a1, h1) = best_first_search(a, b, h, args.penalty)
            end_time = default_timer()
            d1 = end_time - start_time
            return (p1, a1, h1, d1)

        if args.timeout is None:
            (p1, a1, h1, d1) = exp1()
            valid1 = True
        else:
            res1 = timeout(exp1, args.timeout)
            valid1 = res1 is not None
            if valid1:
                (p1, a1, h1, d1) = res1

        if not args.no_logs and valid1:
            outputFile = open(os.path.join(args.log_dir,
                "{}-nngs.csv".format(i)), 'w')
            outputFile.write(history_to_csv(h1))
            outputFile.close()

        # Batch-NNGS

        def exp2():
            start_time = default_timer()
            (p2, a2, h2) = batch_best_first_search(a, b,
                h_batch, args.penalty, args.batch)
            end_time = default_timer()
            d2 = end_time - start_time
            return (p2, a2, h2, d2)

        if args.timeout is None:
            (p2, a2, h2, d2) = exp2()
            valid2 = True
        else:
            res2 = timeout(exp2, args.timeout)
            valid2 = res2 is not None
            if valid2:
                (p2, a2, h2, d2) = res2

        if not args.no_logs and valid2:
            outputFile = open(os.path.join(args.log_dir,
                "{}-batch-nngs.csv".format(i)), 'w')
            outputFile.write(history_to_csv(h2))
            outputFile.close()

        # Print results

        print(','.join([
            str(i),
            str(len(a0)) if valid0 else "",
            str(len(h0)) if valid0 else "",
            str(d0) if valid0 else "",
            str(len(a1)) if valid1 else "",
            str(len(h1)) if valid1 else "",
            str(d1) if valid1 else "",
            str(len(a2)) if valid2 else "",
            str(len(h2)) if valid2 else "",
            str(d2) if valid2 else ""]))

        i += 1