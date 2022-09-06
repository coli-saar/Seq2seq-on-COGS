import os
import numpy as np
np.random.seed(0)
mapping = {}
frequency = {}

def extract_pos(line):
    src, mr, gen_type, syntax = line.rstrip().split('\t')
    src_tokens = src.split()
    src_tokens = src_tokens if len(src_tokens) == 1 else src_tokens[:-1]
    syntax_tokens = syntax.split()
    output = []
    src_idx = 0
    for i in range(len(syntax_tokens)-1):
        # print(i, src_idx, len(syntax_tokens), len(src_tokens))
        if syntax_tokens[i+1] == src_tokens[src_idx]:
            output.append(syntax_tokens[i])
            src_idx += 1
            if src_idx == len(src_tokens):
                break
        else:
            continue
    construct_mapping(src_tokens, output)
    pos = ' '.join(output)
    return "{}\t{}\t{}\n".format(src, pos, gen_type)

def construct_mapping(src_tokens, pos_tags):
    for token, pos, in zip(src_tokens, pos_tags):
        if token not in mapping:
            mapping[token] = pos
            frequency[token] = 1
        else:
            if token == 'to':
                continue
            if pos != mapping[token]:
                print(src_tokens, pos_tags)
                print(token, pos, mapping[token])
            assert pos == mapping[token]
            frequency[token] += 1

def writefile(lines, w_file):
    f = open(w_file, 'w')
    f.writelines(lines)

def generate_POS_lines(r_file):
    lines = open(r_file, 'r').readlines()
    output = []
    for line in lines:
        output.append(extract_pos(line))
    return output

def augment_train_with_longer_sentences():
    tokens = list(mapping.keys())
    token_freq = [frequency[t] for t in tokens]
    p = [f*1.0/sum(token_freq) for f in token_freq]
    num = 10000
    output = []
    for _ in range(num//50):
        for i in range(10, 60):
            src = np.random.choice(tokens, size=i,p=p)
            pos = [mapping[t] for t in src]
            line = "{}\t{}\t{}\n".format(' '.join(src), ' '.join(pos), 'augmented_length')
            output.append(line)

    return output

def augment_train_with_combined_sentences(lines, name):
    num = 2000 if name == 'train.tsv' else 100
    output = []
    for _ in range(num):
        for i in range(2, 10):
            sampled_ids = np.random.choice(list(range(len(lines))), size=i)
            sampled_sents = [lines[id] for id in sampled_ids]
            src_comb, pos_comb = [], []
            for line in sampled_sents:
                src, pos, _ = line.split('\t')
                src_comb += src.split()
                pos_comb += pos.split()
            line = "{}\t{}\t{}\n".format(' '.join(src_comb), ' '.join(pos_comb), 'augmented_length')
            output.append(line)
    return output

def augment_train_with_longlong_combined_sentences(lines, name):
    num = 100 if name == 'train.tsv' else 100
    output = []
    for _ in range(num):
        for i in range(2, 18):
            sampled_ids = np.random.choice(list(range(len(lines))), size=i)
            sampled_sents = [lines[id] for id in sampled_ids]
            src_comb, pos_comb = [], []
            for k, line in enumerate(sampled_sents):
                src, pos, _ = line.split('\t')
                src_toks = src.split()[:-1]
                if not src_toks:
                    continue
                if src_toks[0] == 'A' or src_toks[0] == 'The':
                    src_toks =[src_toks[0].lower()]+src_toks[1:]
                src_comb += src_toks
                pos_comb += pos.split()
            if src_comb[0] == 'a':
                src_comb[0] = 'A'
            if src_comb[0] == 'the':
                src_comb[0] = 'The'
            line = "{}\t{}\t{}\n".format(' '.join(src_comb+['.']), ' '.join(pos_comb), 'augmented_length')
            # line = "{}\t{}\t{}\n".format(' '.join(src_comb), ' '.join(pos_comb), 'augmented_length')
            output.append(line)
    return output

def augment_train_with_recur_sentences(lines, name):
    pp_lines = []
    cp_lines = []
    for line in lines:
        if ' in ' in line or ' on ' in line or ' beside ' in line:
            pp_lines.append(line)
        if ' that ' in line:
            cp_lines.append(line)
    num = 1000 if name == 'train.tsv' else 100
    output = []
    for _ in range(num):
        sampled_id = np.random.choice(list(range(len(pp_lines))))
        pp_line = pp_lines[sampled_id]
        sampled_id = np.random.choice(list(range(len(cp_lines))))
        cp_line = cp_lines[sampled_id]

        src, pos, _ = pp_line.split('\t')
        src_comb, pos_comb = [], []
        for i in range(2, 12):
            src_comb += src.split()
            pos_comb += pos.split()
        line = "{}\t{}\t{}\n".format(' '.join(src_comb), ' '.join(pos_comb), 'augmented_length')

        output.append(line)

        src, pos, _ = cp_line.split('\t')
        src_comb, pos_comb = [], []
        for i in range(2, 12):
            src_comb += src.split()
            pos_comb += pos.split()
        line = "{}\t{}\t{}\n".format(' '.join(src_comb), ' '.join(pos_comb), 'augmented_length')

        output.append(line)

    return output


def augment_train_with_recur_combine_sentences(lines, name):
    pp_lines = []
    cp_lines = []
    for line in lines:
        if ' in ' in line or ' on ' in line or ' beside ' in line:
            pp_lines.append(line)
        if ' that ' in line:
            cp_lines.append(line)
    num = 1000 if name == 'train.tsv' else 100
    output = []
    for _ in range(num):
        for i in range(2, 12):
            sampled_ids = np.random.choice(list(range(len(pp_lines))), size=i)
            sampled_sents = [pp_lines[id] for id in sampled_ids]
            src_comb, pos_comb = [], []
            for k, line in enumerate(sampled_sents):
                src, pos, _ = line.split('\t')
                src_toks = src.split()[:-1]
                if not src_toks:
                    continue
                if src_toks[0] == 'A' or src_toks[0] == 'The':
                    src_toks =[src_toks[0].lower()]+src_toks[1:]
                src_comb += src_toks
                pos_comb += pos.split()
            if src_comb[0] == 'a':
                src_comb[0] = 'A'
            if src_comb[0] == 'the':
                src_comb[0] = 'The'
            line = "{}\t{}\t{}\n".format(' '.join(src_comb+['.']), ' '.join(pos_comb), 'augmented_length')
            output.append(line)
    for _ in range(num):
        for i in range(2, 12):
            sampled_ids = np.random.choice(list(range(len(cp_lines))), size=i)
            sampled_sents = [cp_lines[id] for id in sampled_ids]
            src_comb, pos_comb = [], []
            for k, line in enumerate(sampled_sents):
                src, pos, _ = line.split('\t')
                src_toks = src.split()[:-1]
                if not src_toks:
                    continue
                if src_toks[0] == 'A' or src_toks[0] == 'The':
                    src_toks =[src_toks[0].lower()]+src_toks[1:]
                src_comb += src_toks
                pos_comb += pos.split()
            if src_comb[0] == 'a':
                src_comb[0] = 'A'
            if src_comb[0] == 'the':
                src_comb[0] = 'The'
            line = "{}\t{}\t{}\n".format(' '.join(src_comb+['.']), ' '.join(pos_comb), 'augmented_length')
            output.append(line)

    return output

def generate_POS_corpus(r_file, w_file):
    lines = generate_POS_lines(r_file)
    writefile(lines, w_file)

def generate_aug_POS_corpus(r_file, w_file, name):
    lines = generate_POS_lines(r_file)
    lines += augment_train_with_longer_sentences()
    writefile(lines, w_file)

def generate_aug_combine_POS_corpus(r_file, w_file, name):
    lines = generate_POS_lines(r_file)
    lines += augment_train_with_combined_sentences(lines, name)
    writefile(lines, w_file)

def generate_aug_longlong_combine_POS_corpus(r_file, w_file, name):
    lines = generate_POS_lines(r_file)
    lines += augment_train_with_longlong_combined_sentences(lines, name)
    writefile(lines, w_file)

def generate_aug_recur_POS_corpus(r_file, w_file, name):
    lines = generate_POS_lines(r_file)
    lines += augment_train_with_recur_sentences(lines, name)
    writefile(lines, w_file)

def generate_aug_recur_combine_POS_corpus(r_file, w_file, name):
    lines = generate_POS_lines(r_file)
    lines += augment_train_with_recur_combine_sentences(lines, name)
    writefile(lines, w_file)

def generate_aug_mix_combine_POS_corpus(r_file, w_file, name):
    lines = generate_POS_lines(r_file)
    recur_lines = augment_train_with_recur_combine_sentences(lines, name)
    len_lines = augment_train_with_longlong_combined_sentences(lines, name)
    lines = lines + recur_lines + len_lines
    writefile(lines, w_file)

def generate_POS_corpuses():
    src_dir = 'data/syntax'
    tgt_dir = 'data/pos'
    filename = ['train.tsv', 'dev.tsv', 'test.tsv', 'gen.tsv']
    for name in filename:
        writefile('{}/{}'.format(src_dir, name), '{}/{}'.format(tgt_dir, name))

def generate_aug_POS_corpuses(func, dirname):
    src_dir = 'data/syntax'
    tgt_dir = 'data/pos_{}'.format(dirname)
    filename = ['train.tsv', 'dev.tsv', 'test.tsv', 'gen.tsv']
    hold_files = ['train.tsv', 'dev.tsv']
    for name in filename:
        if name not in hold_files:
            generate_POS_corpus('{}/{}'.format(src_dir, name), '{}/{}'.format(tgt_dir, name))
        else:
            # generate_aug_POS_corpus('{}/{}'.format(src_dir, name), '{}/{}'.format(tgt_dir, name))
            func('{}/{}'.format(src_dir, name), '{}/{}'.format(tgt_dir, name), name)

# generate_aug_POS_corpuses(generate_aug_POS_corpus, 'aug')
# generate_aug_POS_corpuses(generate_aug_combine_POS_corpus, 'aug_combine')
# generate_aug_POS_corpuses(generate_aug_longlong_combine_POS_corpus, 'aug_longlong_combine')
# generate_aug_POS_corpuses(generate_aug_recur_POS_corpus, 'aug_recur')
generate_aug_POS_corpuses(generate_aug_recur_combine_POS_corpus, 'aug_recur_combine')
generate_aug_POS_corpuses(generate_aug_mix_combine_POS_corpus, 'aug_mix_combine')