import json
import csv
import sys

json_file = open(sys.argv[1], 'r')
test_file = open(sys.argv[2], 'r')
new_file = open(sys.argv[3], 'w')

json_lines = json_file.readlines()
tsv_reader = csv.reader(test_file, delimiter='\t')
tsv_writer = csv.writer(new_file, delimiter='\t')

for json_line, tsv_line in zip(json_lines, tsv_reader):
    jsonitem = json.loads(json_line)
    # For pretrained transformer case:
    if 'predicted_text' in jsonitem:
        predicted_text = jsonitem['predicted_text']
    # Otherwise
    else:
        predicted_tokens = jsonitem['predicted_tokens']
        # The <s> and </s> here should be consistent with start and end symbol used in DataReader
        if predicted_tokens[0] == '<s>' and predicted_tokens[-1] == '</s>':
            predicted_text = ' '.join(predicted_tokens[1:-1])
    tsv_line.append(predicted_text)
    tsv_writer.writerow(tsv_line)