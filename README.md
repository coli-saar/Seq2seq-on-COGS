### Seq2seq on COGS

#### Install

```shell
conda create -n seq2seq_cogs python=3.7

conda activate seq2seq_cogs

pip install -r requirements.txt
```

#### Run scripts

```shell
# Train a T5 or BART model on semantic, syntactic or part-of-speech COGS and do evaluation on both test set and generalization set. 

./run_scripts/run_cogs_variants_T5.sh $RANDOM_SEED $DATADIR 

./run_scripts/run_cogs_variants_BART.sh $RANDOM_SEED $DATADIR 

# Example for training a BART model with random seed 0 on semantic (i.e. original) COGS task. This command creates an "archive" and "output" directory under the project directory, save the model in "archive" and prediction output in "output".

./run_scripts/run_cogs_variants_BART.sh 0 data/sem/ 

# Train a T5 or BART model on QA-COGS and do evaluation on both test set and generalization set. 

./run_scripts/run_cogs_qa_T5.sh $RANDOM_SEED $DATADIR 

./run_scripts/run_cogs_qa_BART.sh $RANDOM_SEED $DATADIR 

# Example for training a T5 model with random seed 0 on QA-COGS-base.

./run_scripts/run_cogs_qa_T5.sh 0 data/qa/ 
```

