from collections import Counter
from dataclasses import dataclass
from enum import IntEnum

import numpy as np

EPS = 1e-6

def most_common_class(labels: np.ndarray) -> int:
    assert len(labels) > 0, "Cannot find most common class in empty labels."
    ctr = Counter(labels.tolist())
    return ctr.most_common(1)[0][0]

def entropy(labels: np.ndarray) -> float:
    global EPS
    assert len(labels) > 0, "Cannot compute entropy of empty labels."
    ctr = Counter(labels.tolist())
    p0 = float(ctr[0]) / float(len(labels))
    p1 = float(ctr[1]) / float(len(labels))
    if p0 == 0. or p1 == 0.:
        return 0.
    return -p0 * np.log2(p0) - p1 * np.log2(p1)

def information_gain(parent_labels: np.ndarray, \
                     left_labels: np.ndarray, \
                     right_labels: np.ndarray) -> float:
    assert all([len(l) > 0 for l in [parent_labels, left_labels, right_labels]]), \
        "All input labels must be non-empty."
    pl = float(len(left_labels)) / float(len(parent_labels))
    pr = float(len(right_labels)) / float(len(parent_labels))
    return entropy(parent_labels) - \
           pl * entropy(left_labels) - \
           pr * entropy(right_labels)

class Direction(IntEnum):
    LEFT = 1,
    RIGHT = 2

@dataclass
class DecisionNode:
    feature: int
    threshold: float

    def direct(self, x: np.ndarray):
        if x[self.feature] <= self.threshold:
            return Direction.LEFT
        return Direction.RIGHT

@dataclass
class ClassNode:
    samples: list
    features: list 
    value: int

class DecisionTree:
    def __init__(self, data: np.ndarray, labels: np.ndarray, max_depth: int):
        self.data = data 
        self.labels = labels
        self.max_depth = max_depth
        no_samples, no_features = self.data.shape
        self.samples = np.arange(no_samples)
        self.features = np.arange(no_features)
        labels_mode = most_common_class(self.labels)
        self.nodes = [ClassNode(self.samples.copy(), \
                                self.features.copy(), \
                                labels_mode)]

    def split_node(self, node_idx: int) -> bool:
        assert node_idx >= 0, "node_idx must be nonnegative."
        if node_idx >= 2 ** (self.max_depth) - 1:
            """Max depth reached."""
            return False
        assert len(self.nodes) > node_idx, "node_idx does not yet exist."
        assert isinstance(self.nodes[node_idx], ClassNode), \
            "Cannot split a decision node."
        if len(self.nodes[node_idx].samples) < 2:
            """Cannot split a class node with fewer than two samples."""
            return False
        best_feature = -1
        best_pivot = 0.0
        best_info = 0.0
        samples = self.nodes[node_idx].samples
        first_label = self.labels[samples[0]]
        if all([l == first_label for l in self.labels[samples]]):
            """Constant label, leave as leaf."""
            return False
        for f in self.nodes[node_idx].features:
            column = self.data[samples, f].flatten()
            pivot = np.median(column.flatten())
            par_labels = self.labels[samples]
            left_labels = par_labels[column <= pivot]
            right_labels = par_labels[column > pivot]
            if not len(left_labels) or not len(right_labels):
                """Empty leaf or leaves."""
                continue
            info_gain = information_gain(par_labels, left_labels, right_labels)
            if info_gain > best_info:
                best_info = info_gain
                best_pivot = pivot
                best_feature = f
        if best_feature == -1:
            """No best feature found."""
            return False
        best_column = self.data[samples,best_feature].flatten()
        left_samples = samples[best_column <= best_pivot]
        right_samples = samples[best_column > best_pivot]
        left_value = most_common_class(self.labels[left_samples])
        right_value = most_common_class(self.labels[right_samples])
        subfeatures = self.nodes[node_idx].features.copy()
        subfeatures = subfeatures[[a != best_feature for a in self.nodes[node_idx].features]]
        self.nodes[node_idx] = DecisionNode(best_feature, best_pivot)
        if len(self.nodes) <= 2 * node_idx + 2:
            padding = 2 * node_idx + 3 - len(self.nodes)
            self.nodes += [None] * padding
        self.nodes[2 * node_idx + 1] = ClassNode(left_samples, subfeatures, left_value)
        self.nodes[2 * node_idx + 2] = ClassNode(right_samples, subfeatures, right_value)
        self.split_node(2 * node_idx + 1)
        self.split_node(2 * node_idx + 2)
        return True

    def fit(self):
        assert len(self.nodes) == 1, "Tree has already been fit."
        self.split_node(0)
    
    def predict(self, X: np.ndarray):
        preds = []
        for i in range(len(X)):
            datum = X[i]
            node_idx = 0
            cursor = self.nodes[node_idx]
            while not isinstance(cursor, ClassNode):
                node_idx = 2 * node_idx + cursor.direct(datum)
                cursor = self.nodes[node_idx]
            preds.append(cursor.value)
        return np.array(preds)

import time

if __name__ == "__main__":

    def generate_function(domain, func):
        x_list = []
        y_list = []
        bin_len = int(np.ceil(np.log2(max(domain))))
        for d in domain:
            x_list.append(list(str("{0:0" + str(bin_len) + "b}").format(d)))
            y_list.append(func(d))
        return np.array(x_list, dtype=float), np.array(y_list, dtype=float)

    def is_prime(n):
        if n % 2 == 0 and n > 2: 
            return False
        return int(all(n % i for i in range(3, int(np.sqrt(n)) + 1, 2)))


    data, labels = generate_function(range(2**12), is_prime)
    print(labels)
    t0 = time.time()
    dt = DecisionTree(data, labels, 10)
    dt.fit()
    t1 = time.time()
    print(f"Runtime = {t1 - t0}s")
    # print(f"Node array = \n{dt.nodes}")
    predictions = dt.predict(data)
    print(f"predictions = \n{predictions}")
