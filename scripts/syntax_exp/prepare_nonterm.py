
trainf = open('data/syntax_only/train.txt', 'r').readlines()
devf = open('data/syntax_only/dev.txt', 'r').readlines()
testf = open('data/syntax_only/test.txt', 'r').readlines()
genf = open('data/syntax_only/gen.txt', 'r').readlines()

train_vocab = []
for line in trainf:
    terms = line.split()
    for i in range(1, len(terms)):
        if terms[i-1] == '(' and terms[i] not in train_vocab:
            train_vocab.append(terms[i])

dev_vocab = []
for line in devf:
    terms = line.split()
    for i in range(1, len(terms)):
        if terms[i-1] == '(' and terms[i] not in dev_vocab:
            dev_vocab.append(terms[i])

test_vocab = []
for line in testf:
    terms = line.split()
    for i in range(1, len(terms)):
        if terms[i-1] == '(' and terms[i] not in test_vocab:
            test_vocab.append(terms[i])

gen_vocab = []
for line in genf:
    terms = line.split()
    for i in range(1, len(terms)):
        if terms[i-1] == '(' and terms[i] not in gen_vocab:
            gen_vocab.append(terms[i])
diff = []
print('train vocab: {}'.format(len(train_vocab)))
print('dev vocab: {}'.format(len(dev_vocab)))
print('test vocab: {}'.format(len(test_vocab)))
print('gen vocab: {}'.format(len(gen_vocab)))
for token in gen_vocab:
    if token not in train_vocab:
        diff.append(token)
print('Nonterms cannot be generalized: {}'.format(diff))

diff = []
for token in train_vocab:
    if token not in gen_vocab:
        diff.append(token)
print('Nonterms not in gen: {}'.format(diff))

vocabf = open('data/cogs_syntax/nonterms.txt', 'w')
vocab = train_vocab+dev_vocab+test_vocab+gen_vocab
vocab = [token+'\n' for token in vocab]
vocabf.writelines(list(set(vocab)))