trainf = open('data/cogs_syntax/train_100.txt', 'r').readlines()
# devf = open('data/cogs_syntax/dev.txt', 'r').readlines()
# testf = open('data/cogs_syntax/test.txt', 'r').readlines()
# genf = open('data/cogs_syntax/gen.txt', 'r').readlines()

trainf_out = open('data/cogs_mtl/train_100_sideanno.txt', 'w')
# devf_out = open('data/cogs_mtl/dev_sideanno_valsem.txt', 'w')
# testf_out = open('data/cogs_mtl/test_sideanno.txt', 'w')
# genf_out = open('data/cogs_mtl/gen_sideanno.txt', 'w')

train_output = []
for line in trainf:
    sent, semanno, type, synanno = line.split('\t')
    synanno = synanno.rstrip()
    train_output.append('\t'.join([sent+' @sem@', semanno, type])+'\n')
    train_output.append('\t'.join([sent+' @syn@', synanno, type])+'\n')
trainf_out.writelines(train_output)

# dev_output = []
# for line in devf:
#     sent, semanno, type, synanno = line.split('\t')
#     synanno = synanno.rstrip()
#     dev_output.append('\t'.join([sent+' @sem@', semanno, type])+'\n')
#     # dev_output.append('\t'.join([sent+' @syn@', synanno, type])+'\n')
# devf_out.writelines(dev_output)
#
# test_output = []
# for line in testf:
#     sent, semanno, type, synanno = line.split('\t')
#     synanno = synanno.rstrip()
#     test_output.append('\t'.join([sent+' @sem@', semanno, type])+'\n')
# testf_out.writelines(test_output)
#
# test_output = []
# for line in genf:
#     sent, semanno, type, synanno = line.split('\t')
#     synanno = synanno.rstrip()
#     test_output.append('\t'.join([sent+' @sem@', semanno, type])+'\n')
# genf_out.writelines(test_output)
