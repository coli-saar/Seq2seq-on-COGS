import json
import csv
import sys

json_file = open(sys.argv[1], 'r')
test_file = open(sys.argv[2], 'r')
new_file = open(sys.argv[3], 'w')

json_lines = json_file.readlines()
tsv_reader = test_file.readlines()

newlines = []

for json_line, tsv_line in zip(json_lines, tsv_reader):
    jsonitem = json.loads(json_line)
    if 'predicted_text' in jsonitem:
        predicted_text = jsonitem['predicted_text']
    else:
        predicted_text = ' '.join(jsonitem['predicted_tokens'])
    newline = '{}\t{}\n'.format(tsv_line[:-1], predicted_text)
    newlines.append(newline)
new_file.writelines(newlines)