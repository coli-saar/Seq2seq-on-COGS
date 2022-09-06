import sys
import re
import nltk


def writefile(file, lines):
    wf = open(file, 'w')
    wf.writelines(lines)
    wf.close()

def replace(tree):
    for pos in tree.treepositions():
        if not isinstance(tree[pos][0], nltk.Tree):
            continue
        if len(tree[pos]) == 1 and tree[pos][0].label() == tree[pos].label():
            tree[pos] = tree[pos][0]
            return True
    return False

def clean_tree(tree_str, binary=True):
    tree_lines = re.split(' \(p=.*\)\n', tree_str)[:-1][0]
    output = []
    string = []
    # print(tree_str)
    # print(tree_lines)
    for line in tree_lines.split('\n'):
        line = line.replace('(', '( ')
        line = line.replace(')', ' )')
        string += line.split()
    for i, sym in enumerate(string):
        if '_targeted' in sym:
            string[i] = sym.replace('_targeted', '')
        if '_' in sym:
            string[i] = sym[:sym.index('_')]
    if string[string.index(')') - 1] == 'a':
        string[string.index(')') - 1] = 'A'
    if string[string.index(')') - 1] == 'the':
        string[string.index(')') - 1] = 'The'
    # if binary:
    #     string_tmp = ['(']
    #     for i in range(1, len(string)):
    #         if string[i-1] != '(':
    #             string_tmp.append(string[i])
    #     string = string_tmp
    tree = nltk.Tree.fromstring(' '.join(string))
    # print(string)
    while replace(tree):
        pass
    tree = tree._pformat_flat("", "()", False)
    tree_string = str(tree)
    tree_string = tree_string.replace('(', '( ')
    tree_string = tree_string.replace(')', ' )')

    if binary:
        string_tmp = ['(']
        linear_tokens = tree_string.split()
        for i in range(1, len(linear_tokens)):
            if linear_tokens[i - 1] != '(':
                string_tmp.append(linear_tokens[i])
        tree_string = ' '.join(string_tmp)
    # print(tree_string)
    # break
    return tree_string

def clean_tree_file(filepath):
    file = open(filepath, 'r')
    lines = file.read()
    split_lines = re.split(' \(p=.*\)\n', lines)[:-1]
    # print(len(split_lines))
    output = []
    binary = False
    for item in split_lines:
        string = []
        for line in item.split('\n'):
            line = line.replace('(', '( ')
            line = line.replace(')', ' )')
            string += line.split()
        for i, sym in enumerate(string):
            if '_targeted' in sym:
                string[i] = sym.replace('_targeted', '')
            if '_' in sym:
                string[i] = sym[:sym.index('_')]
        if string[string.index(')') - 1] == 'a':
            string[string.index(')') - 1] = 'A'
        if string[string.index(')') - 1] == 'the':
            string[string.index(')') - 1] = 'The'
        # if binary:
        #     string_tmp = ['(']
        #     for i in range(1, len(string)):
        #         if string[i-1] != '(':
        #             string_tmp.append(string[i])
        #     string = string_tmp
        tree = nltk.Tree.fromstring(' '.join(string))
        # print(string)
        while replace(tree):
            pass
        tree = tree._pformat_flat("", "()", False)
        tree_string = str(tree)
        tree_string = tree_string.replace('(', '( ')
        tree_string = tree_string.replace(')', ' )')

        if binary:
            string_tmp = ['(']
            linear_tokens = tree_string.split()
            for i in range(1, len(linear_tokens)):
                if linear_tokens[i - 1] != '(':
                    string_tmp.append(linear_tokens[i])
            tree_string = ' '.join(string_tmp)
        # print(tree_string)
        # break
        output.append(tree_string)
        output.append('\n')
    return output

def clean_dir_files(dirname):
    name = ['train.syntax', 'dev.syntax', 'test.syntax', 'gen.syntax']
    for n in name:
        filepath = '{}/{}'.format(dirname, n)
        out_filepath = '{}.clean'.format(filepath)
        # print(filepath)
        writefile(out_filepath, clean_tree_file(filepath))

if __name__ == '__main__':
    clean_dir_files('localview/qa_cogs_large/past_past')
    clean_dir_files('localview/qa_cogs_large/past_pres')
    clean_dir_files('localview/qa_cogs_large/pres_past')
    clean_dir_files('localview/qa_cogs_large/pres_pres')
    clean_dir_files('localview/qa_cogs_large_fine/past_past')
    clean_dir_files('localview/qa_cogs_large_fine/past_pres')
    clean_dir_files('localview/qa_cogs_large_fine/pres_past')
    clean_dir_files('localview/qa_cogs_large_fine/pres_pres')
#
# file = open(sys.argv[1], 'r')
# lines = file.read()
# split_lines = re.split(' \(p=.*\)\n', lines)[:-1]
# # print(len(split_lines))
# output = []
# binary = True
# for item in split_lines:
#     string = []
#     for line in item.split('\n'):
#         line = line.replace('(', '( ')
#         line = line.replace(')', ' )')
#         string += line.split()
#     for i, sym in enumerate(string):
#         if '_targeted' in sym:
#             string[i] = sym.replace('_targeted', '')
#         if '_' in sym:
#             string[i] = sym[:sym.index('_')]
#     if string[string.index(')')-1] == 'a':
#         string[string.index(')') - 1] = 'A'
#     if  string[string.index(')')-1] == 'the':
#         string[string.index(')') - 1] = 'The'
#     # if binary:
#     #     string_tmp = ['(']
#     #     for i in range(1, len(string)):
#     #         if string[i-1] != '(':
#     #             string_tmp.append(string[i])
#     #     string = string_tmp
#     tree = nltk.Tree.fromstring(' '.join(string))
#     # print(string)
#     while replace(tree):
#         pass
#     tree = tree._pformat_flat("", "()", False)
#     tree_string = str(tree)
#     tree_string = tree_string.replace('(', '( ')
#     tree_string = tree_string.replace(')', ' )')
#
#     if binary:
#         string_tmp = ['(']
#         linear_tokens = tree_string.split()
#         for i in range(1, len(linear_tokens)):
#             if linear_tokens[i-1] != '(':
#                 string_tmp.append(linear_tokens[i])
#         tree_string = ' '.join(string_tmp)
#     # print(tree_string)
#     # break
#     output.append(tree_string)
#     output.append('\n')
#
# outfile = open(sys.argv[2], 'w')
# outfile.writelines(output)




# tree = nltk.Tree.fromstring('(S(NP (NP (Det a) (N dog))))')
# def replace(tree):
#     for pos in tree.treepositions():
#         if not isinstance(tree[pos][0], nltk.Tree):
#             continue
#         if len(tree[pos]) == 1 and tree[pos][0].label() == tree[pos].label():
#             tree[pos] = tree[pos][0]
#             return True
#     return False
#
# print(tree)
# while replace(tree):
#     pass

# print(tree)