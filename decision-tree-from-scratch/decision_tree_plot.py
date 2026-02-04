import argparse
import math
import csv
import matplotlib.pyplot as plt


class Node:
    def __init__(self, attribute, vote):
        self.child_node = dict()
        self.attr = attribute
        self.vote = vote


def filter_data_value(data):
    data_values = dict()
    if not data:
        return data_values
    for row in data:
        if row not in data_values:
            data_values[row] = 0
        data_values[row] += 1
    return data_values


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


def entropy(data):
    total_data = len(data)
    if total_data == 0:
        return 0

    data_values = filter_data_value(data)
    p_values = list(value / total_data for value in data_values.values())
    entropy = -1 * sum(
        p_value * math.log2(p_value) for p_value in p_values if p_value > 0
    )
    return entropy


def mutual_information_cal(toData, fromData, features_index):
    label_entropy = entropy(toData)
    total_data = len(fromData)

    label_condition_feature = dict()
    for row in fromData:
        feature_value = row[features_index]
        label = row[-1]
        if feature_value not in label_condition_feature:
            label_condition_feature[feature_value] = list()
        label_condition_feature[feature_value].append(label)

    label_condition_feature_entropy = float(0)
    for value in label_condition_feature.values():
        p_value = len(value) / total_data
        label_condition_feature_entropy += p_value * entropy(value)

    ig = label_entropy - label_condition_feature_entropy
    return ig


def build_tree(node, max_depth, features, data, labels):
    if (max_depth == 0 and len(data) > 1) or len(
        filter_data_value(list(row[-1] for row in data))
    ) == 1:
        node.vote = MajorityVoteClassifier(list(row[-1] for row in data))
        return node

    max = 0
    features_index = -1
    for index in range(len(features) - 1):
        each_ig = mutual_information_cal(labels, data, index)
        if each_ig > max:
            max = each_ig
            features_index = index
            node.attr = features[index]
    new_features = list(filter(lambda x: x != node.attr, features))
    filter_data = filter_data_value(list(row[features_index] for row in data))
    for key in filter_data.keys():
        filter_data_features = list(
            filter(lambda data: data[features_index] == key, data)
        )
        new_data = list(
            [value for index, value in enumerate(row) if index != features_index]
            for row in filter_data_features
        )
        if max > 0:
            childNode = Node(None, None)
            childNode = build_tree(
                childNode, max_depth - 1, new_features, new_data, labels
            )
            node.child_node[key] = childNode
        else:
            node.vote = MajorityVoteClassifier(list(row[-1] for row in data))
            return node
    return node


def print_tree(node, data, features, all_label_value, writer, indent):
    if data:
        label_value_count = filter_data_value(list(row[-1] for row in data))
        if len(label_value_count) > 1:
            label_count_str = (
                "["
                + "/".join(
                    [f"{key} {value}" for key, value in label_value_count.items()]
                )
                + "]"
            )
            # print(f"{label_count_str}")
            writer.write(f" {label_count_str}\n")
        else:
            value_occur = list(
                f"{key} {value}" for key, value in label_value_count.items()
            )
            for value in all_label_value:
                if value not in label_value_count.keys():
                    value_occur.append(f"{value} {0}")
            label_count_str = "[" + "/".join(value_occur) + "]"
            # print(f"{indent} {label_count_str}")
            writer.write(f" {label_count_str}\n")

    if node.attr is None:
        return

    if node.child_node:
        feature_index = -1
        for index, value in enumerate(features):
            if node.attr and value == node.attr:
                feature_index = index
        for key, child in node.child_node.items():
            # print(f"{indent}| {node.attr}: {key}", end=" ")
            writer.write(f"{indent}| {node.attr} = {key}:")
            new_data = list(row for row in data if row[feature_index] == key)
            print_tree(
                child, new_data, features, all_label_value, writer, indent + "  |  "
            )


def process_predict(node, row, features):
    if node.attr is None:
        return node.vote

    feature_index = features.index(node.attr)
    for key, value in node.child_node.items():
        if row[feature_index] == key:
            next_node = value
            return process_predict(next_node, row, features)


def error_cal(input, node):
    with open(input, "r", newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter="\t")
        features = next(reader)
        data = list(reader)
        labels = list(row[-1] for row in data)
        list_predict = list()
        for row in data:
            predicted = process_predict(root, row, features)
            list_predict.append(predicted)
        error_count = 0
        for predicted, label in zip(list_predict, labels):
            if int(predicted) != int(label):
                error_count += 1
        error_rate = error_count / len(labels)
        # with open(output, "w", newline="\n") as csvfile:
        #     writer = csv.writer(csvfile)
        #     writer.writerows(list_predict)
        # with open(metrics, "a", newline="\n") as csvfile:
        #     writer = csv.writer(csvfile)
        #     writer.writerow([f"error({name}) : {error_rate:.6f}"])
        return error_rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "train_ip", type=str, help="path to training input extension with .tsv"
    )
    parser.add_argument(
        "test_ip", type=str, help="path to test input extension with .tsv"
    )
    args = parser.parse_args()
    with open(args.train_ip, "r", newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter="\t")
        features = next(reader)
        max_depth = len(features) - 1
        data = list(reader)
        labels = list(row[-1] for row in data)
        root_list: list = list()
        error_train_rate_list: list = list()
        error_test_rate_list: list = list()
        new_node = Node(None, None)
        label_values = filter_data_value(labels).keys()
        all_label_values = list(key for key in label_values)
        for i in range(max_depth):
            root = build_tree(new_node, i, features, data, labels)
            root_list.append(root)
            error_train_rate_list.append(error_cal(args.train_ip, root))
            error_test_rate_list.append(error_cal(args.test_ip, root))

        plt.figure(figsize=(10, 6))
        plt.plot(
            range(max_depth),
            error_train_rate_list,
            label="Training_errors",
            marker="+",
            linestyle="-",
            color="blue",
        )
        plt.plot(
            range(max_depth),
            error_test_rate_list,
            label="Testing_errors",
            marker="x",
            linestyle="--",
            color="red",
        )
        plt.title("Rate of error in decision tree")
        plt.xlabel("depth of attribute")
        plt.ylabel("error rate of decision tree")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig("heart.png")
        plt.show()
