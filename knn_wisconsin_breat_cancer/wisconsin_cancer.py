from typing import NamedTuple, List
from collections import Counter
from chapter_4.wektory import Vector, distance


def majority_vote(labels: List[str]) -> str:
    # funkcja zaklada ze etykiety sa ustalone od najblizszej do najdalszej
    vote_counts = Counter(labels)
    winner, winner_count = vote_counts.most_common(1)[0]
    num_winners = len([count for count in vote_counts.values() if count == winner_count])

    if num_winners == 1:
        return winner
    else:
        return majority_vote(labels[:-1])


class LabelPoint(NamedTuple):
    point: Vector
    label: str


def knn_classify(k: int, labeled_points: List[LabelPoint], new_point: Vector) -> str:
    by_distance = sorted(labeled_points, key=lambda lp: distance(lp.point, new_point))

    k_nearest_neighbours = [lp.label for lp in by_distance[:k]]

    return majority_vote(k_nearest_neighbours)


from typing import Dict
import csv
from collections import defaultdict


def parse_cancer_row(row: List[str]) -> LabelPoint:
    measurements = [float(value) for value in row[2:]]
    label = row[1]

    return LabelPoint(measurements, label)


with open('wdbc.data') as f:
    reader = csv.reader(f)
    cancer_data = [parse_cancer_row(row) for row in reader if len(row) > 1]

points_by_malice: Dict[str, List[Vector]] = defaultdict(list)

for cancer in cancer_data:
    points_by_malice[cancer.label].append(cancer.point)

# print(points_by_malice)

import random
from chapter_11_Machine_learning.start import split_data

cancer_train, cancer_test = split_data(cancer_data, 0.7)

# print(len(cancer_test))
# print(len(cancer_train))

from typing import Tuple

confusion_matrix: Dict[Tuple[str, str], int] = defaultdict(int)

num_correct = 0

for cancer in cancer_test:
    predicted = knn_classify(10, cancer_train, cancer.point)
    actual = cancer.label

    if predicted == actual:
        num_correct += 1

    confusion_matrix[(predicted, actual)] += 1

pct_correct = num_correct / len(cancer_test)

print(f'Model got it right in {pct_correct * 100}% of occassions')
print(confusion_matrix)
