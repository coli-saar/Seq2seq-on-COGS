# Seq2seq on COGS

This repository is used to train T5 and BART on [COGS](https://www.aclweb.org/anthology/2020.emnlp-main.731/) datasets.

### Install

```shell
conda create -n seq2seq_cogs python=3.7

conda activate seq2seq_cogs

pip install -r requirements.txt
```

### Datasets
For original COGS dataset, please download from <https://github.com/najoungkim/COGS> and put the data directory 
in `data/`. Data of Syntax-COGS, POS-COGS and QA-COGS is available in <https://github.com/coli-saar/Syntax-COGS>.

### Run scripts

```shell
# Train a T5 or BART model on semantic, syntactic or part-of-speech COGS 
# and do evaluation on both test set and generalization set. 

./run_scripts/run_cogs_variants_T5.sh $RANDOM_SEED $DATADIR 

./run_scripts/run_cogs_variants_BART.sh $RANDOM_SEED $DATADIR 

# Example for training a BART model with random seed 0 on syntactic 
#  COGS task. 
./run_scripts/run_cogs_variants_BART.sh 0 data/syntax/ 

# Train a T5 or BART model on QA-COGS and do evaluation on 
# both test set and generalization set. 

./run_scripts/run_cogs_qa_T5.sh $RANDOM_SEED $DATADIR 

./run_scripts/run_cogs_qa_BART.sh $RANDOM_SEED $DATADIR 

# Example for training a T5 model with random seed 0 on QA-COGS-base.

./run_scripts/run_cogs_qa_T5.sh 0 data/qa/ 
```

