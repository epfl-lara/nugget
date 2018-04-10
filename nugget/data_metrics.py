
import argparse

from nugget.reader import Reader

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Compute data metrics.")
    parser.add_argument("file", type=str, help="data file")
    args = parser.parse_args()

    reader = Reader(args.file)

    n = 0

    min_length = float("inf")
    max_length = 0
    total_length = 0

    min_height = float("inf")
    max_height = 0
    total_height = 0

    for (a, b, _, _) in reader.entries():
        for e in [a, b]:
            height = e.height()
            min_height = min(min_height, height)
            max_height = max(max_height, height)
            total_height += height

            length = e.length()
            min_length = min(min_length, length)
            max_length = max(max_length, length)
            total_length += length

            n += 1

    print("Min height: {}".format(min_height))
    print("Max height: {}".format(max_height))
    print("Average height: {}".format(total_height / n))
    print("=" * 40)
    print("Min length: {}".format(min_length))
    print("Max length: {}".format(max_length))
    print("Average length: {}".format(total_length / n))