cuda='0'

lr='1e-4'

accum_steps='8'

seed=$1

config='configs/qa/T5_official.jsonnet'

config_filename="$(basename -s .jsonnet $config)"

batchsize='32'

data_path=$2

task="$(basename $data_path)"

traindata_name=$data_path'/train.tsv'

devdata_name=$data_path'/dev.tsv'

gendata_name=$data_path'/gen.tsv'

testdata_name=$data_path'/test.tsv'

model_name=$config_filename'_'$task'_'$seed

output_dirname='output/'$model_name

genoutput_filename=$output_dirname'/out.gen'

testoutput_filename=$output_dirname'/out.test'

archive_dirname='archive/'$model_name

mkdir -p $archive_dirname

allennlp train \
          $config \
          --serialization-dir  $archive_dirname\
          --include-package modules \
          -f --file-friendly-logging \
          -o '{"random_seed": '$seed', "numpy_seed": '$seed', "pytorch_seed": '$seed',
               "train_data_path": "'$traindata_name'",
               "validation_data_path": "'$devdata_name'",
               "test_data_path": "'$gendata_name'",
               "data_loader.batch_sampler.batch_size": '$batchsize',
               "trainer.cuda_device": '$cuda',
               "trainer.optimizer.lr": '$lr',
               "trainer.num_gradient_accumulation_steps": '$accum_steps'}'

mkdir -p $output_dirname

echo "Evaluating model on $testdata_name..."

./eval.sh  $testdata_name $archive_dirname $testoutput_filename  &> $output_dirname'/test.log'

echo "Evaluating model on $gendata_name..."

./eval.sh  $gendata_name $archive_dirname $genoutput_filename  &> $output_dirname'/gen.log'

echo "Done. Output saved in $output_dirname"