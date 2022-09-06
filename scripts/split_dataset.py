import csv
import sys
import numpy as np
from numpy.random import random_sample

sample_prob = float(sys.argv[1])
np.random.seed(2)
tsv_file = open('data/train.tsv', 'r')
new_tsv_file = open('data/train_split.tsv', 'w')
tsv_reader = csv.reader(tsv_file)
tsv_writer = csv.writer(new_tsv_file)
for row in tsv_reader:
    if random_sample() <= sample_prob:
        tsv_writer.writerow(row)

