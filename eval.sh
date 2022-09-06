cuda='0'

input_path=$1

model_path=$2

output_path=$3

allennlp predict \
          $model_path \
          $input_path \
          --batch-size 16 \
          --silent \
          --use-dataset-reader \
          --include-package modules \
          --cuda-device $cuda \
          --output-file $output_path

python scripts/json2tsv.py $output_path $input_path $output_path'.pred'

python scripts/eval.py $output_path'.pred'
