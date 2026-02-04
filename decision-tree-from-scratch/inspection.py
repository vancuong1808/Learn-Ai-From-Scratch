import argparse
import math
import csv


def MajorityVoteClassifier(data):
    count = dict()
    if not data:
        return None
    for val in data:
        if count.get(val) is None:
            count[val] = 0
        count[val] += 1
    result = None
    max_count = max(count.values())
    keys = list(key for key, val in count.items() if val == max_count)
    max_key = max(list(int(key) for key in keys))
    if len(keys) == 1:
        return keys[0]
    for key in keys:
        if int(key) == max_key:
            result = key
    return result


def filter_data_value(data):
    data_values = dict()
    if not data:
        return data_values
    for row in data:
        if row not in data_values:
            data_values[row] = 0
        data_values[row] += 1
    return data_values


def entropy_cal(data):
    total_data = len(data)
    if total_data == 0:
        return 0

    data_values = filter_data_value(data)
    p_values = list(value / total_data for value in data_values.values())
    entropy = -1 * sum(
        p_value * math.log2(p_value) for p_value in p_values if p_value > 0
    )
    return entropy


def inspect(infile, outfileLog, predicted, entropy):
    with open(infile, "r", newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter="\t")
        features = next(reader)
        listReader = list(reader)
        length = len(listReader)
        labels = list(row[-1] for row in listReader)
        count = 0
        for row in labels:
            if int(row) != int(predicted):
                count += 1
        error_rate = f"{(count / length):.6f}"
        entropy = f"{entropy:.6f}"
        with open(outfileLog, "a", newline="\n") as csvfile:
            writer = csv.writer(csvfile)
            result = f"entropy : {entropy}"
            entropy = f"error : {error_rate}"
            writer.writerow([result])
            writer.writerow([entropy])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "train_ip", type=str, help="path to training input extension .tsv"
    )
    parser.add_argument(
        "inspect", type=str, help="path to training inspect extension .txt"
    )
    args = parser.parse_args()
    with open(args.train_ip, "r", newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter="\t")
        features = next(reader)
        data = list(reader)
        labels = list(row[-1] for row in data)
        entropy = entropy_cal(list(row[-1] for row in data))
        predicted = MajorityVoteClassifier(list(row[-1] for row in data))
        inspect(args.train_ip, args.inspect, predicted, entropy)
