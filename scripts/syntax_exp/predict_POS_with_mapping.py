import numpy as np
np.random.seed(0)
mapping = {}

def pred_pos():
    rf = open('data/pos/gen.tsv', 'r')
    lines = rf.readlines()
    for line in lines:
        src, pos, gen_type = line.rstrip().split('\t')
        src_tokens = src.split()
        src_tokens = src_tokens if len(src_tokens) == 1 else src_tokens[:-1]
        pos_tokens = pos.split()

        construct_mapping(src_tokens, pos_tokens)
    output = []
    for line in lines:
        src, pos, gen_type = line.rstrip().split('\t')
        src_tokens = src.split()
        src_tokens = src_tokens if len(src_tokens) == 1 else src_tokens[:-1]
        pos_tokens = pos.split()
        pred = pred_with_mapping(src_tokens, pos_tokens)
        newline = "{}\t{}\t{}\t{}\n".format(src, pos, gen_type, pred)
        output.append(newline)

    wf = open('data/pos/gen.map.pred', 'w')
    wf.writelines(output)


def construct_mapping(src_tokens, pos_tags):
    for token, pos, in zip(src_tokens, pos_tags):
        if token not in mapping:
            mapping[token] = [pos]
        else:
            # if token == 'to':
            #     continue
            mapping[token].append(pos)
            # if pos != mapping[token]:
            #     print(src_tokens, pos_tags)
            #     print(token, pos, mapping[token])
            # assert pos == mapping[token]

def pred_with_mapping(src_tokens, pos_tags):
    output = []
    for src_tok, pos_tok in zip(src_tokens, pos_tags):
        if len(mapping[src_tok]) == 1:
            output.append(pos_tok)
        else:
            output.append(mapping[src_tok][0])
    return ' '.join(output)

if __name__ == '__main__':
    pred_pos()