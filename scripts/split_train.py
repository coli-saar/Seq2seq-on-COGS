import csv
import sys
import numpy as np
from numpy.random import random_sample

np.random.seed(0)

f = open(sys.argv[1], 'r')
lines = f.readlines()
if lines[-1][-1] != '\n':
    lines[-1] = lines[-1]+'\n'
np.random.shuffle(lines)
train_size = int(0.8 * len(lines))
train_lines, dev_lines = lines[:train_size], lines[train_size:]

train_f = open(sys.argv[2], 'w')
dev_f = open(sys.argv[3], 'w')
train_f.writelines(train_lines)
dev_f.writelines(dev_lines)