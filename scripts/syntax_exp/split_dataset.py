import os, sys
import numpy as np

def print_rand():
    print(np.random.rand(5))

def writefile(lines, w_file):
    f = open(w_file, 'w')
    f.writelines(lines)

def readfile(r_file):
    f = open(r_file, 'r')
    return f.readlines()

def remove_nonrec_lines(lines):
    return [line for line in lines if '_recursion' in line or 'obj_pp_to_subj_pp' in line]

def sample_subset(lines):
    # lines = remove_nonrec_lines(lines)
    num = int(len(lines)*0.1)
    np.random.shuffle(lines)
    return lines[:num]

def split_corpus(dirname):
    np.random.seed(0)
    gen_file = 'data/{}/gen.tsv'.format(dirname)
    dev_file = 'data/{}/dev_gen.tsv'.format(dirname)
    lines = readfile(gen_file)
    writefile(sample_subset(lines), dev_file)

# lines = readfile('data/pos_aug_debug/gen.tsv')
# writefile(sample_subset(lines), 'data/pos_aug_debug/dev.tsv')
# split_corpus('sem_aug_more')
# split_corpus('sem_debug_reformat')
# split_corpus('sem_debug_aug')
# split_corpus('sem_debug_aug_reformat')
# split_corpus('sem_debug_aug_reformat_reorder')
split_corpus(sys.argv[1])