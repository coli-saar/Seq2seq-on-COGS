import sys
import csv

# Close the brackets of predicted syntax tree. Only used for the syntax task.
def close_brackets(item):
    src, gold, gen_type, pred = item
    pred_tokens = pred.split()
    left = pred_tokens.count('(')
    right = pred_tokens.count(')')
    pred_tokens += [')' for _ in range(left-right)]
    pred = ' '.join(pred_tokens)
    return src, gold, gen_type, pred

f = open(sys.argv[1], 'r')
tsv_reader = csv.reader(f, delimiter='\t')
acc_num = 0
num = 0
type2num = {}
close_brackets_for_syntax = False
for item in tsv_reader:
    if close_brackets_for_syntax:
        src, gold, gen_type, pred = close_brackets(item)
    # output for cogs contains 4 fields for each line
    if len(item) == 4:
        src, gold, gen_type, pred = item
    # qa tasks have more than 4 fields
    elif len(item) > 4:
        src, q, gold, gen_type, pred = item[:4]+item[-1:]
        # Postprocess for extractive model
        if pred[:4] == 'The ':
            pred = 'the '+pred[4:]
        if pred[:2] == 'A ':
            pred = 'a '+ pred[2:]
    else:
        raise NotImplementedError()

    if gen_type not in type2num:
        type2num[gen_type] = [0, 0]
    type2num[gen_type][1] += 1

    if gold == pred:
        acc_num += 1
        type2num[gen_type][0] += 1

    num += 1

print('Exact ACC: ', acc_num*1.0/num)
for type in list(type2num.keys()):
    print(type, ': ', type2num[type][0]*1.0/type2num[type][1], type2num[type][0], type2num[type][1])

lexical = []
struct_types = ['obj_pp_to_subj_pp', 'cp_recursion', 'pp_recursion']
for type in list(type2num.keys()):
    if type not in struct_types:
        lexical.append(type2num[type][0]*1.0/type2num[type][1])
print('LEX: {}'.format(sum(lexical) /18))