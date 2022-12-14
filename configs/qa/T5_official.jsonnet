local epochs = 100;
local patience = 10;
local batch_size = 32;
local cuda_device = 1;

local file_prefix = "data/qa/";
local model_name = 't5-base';

{
    numpy_seed: 0,
    pytorch_seed: 0,
    random_seed: 0,
    dataset_reader: {
        type: "cogs_qa_reader",

        tokenizer: {
            type: "pretrained_transformer",
            model_name: model_name,
        },
        source_token_indexers: {
            tokens: {
                type: "pretrained_transformer",
                model_name: model_name,
                namespace: "source_tokens",
                max_length: 1024,
            }
        },
        target_token_indexers: {
            tokens: {
                type: "pretrained_transformer",
                model_name: model_name,
                namespace: "target_tokens",
            }
        },

        keep_metadata: true
    },
    train_data_path: file_prefix + "train.tsv",
    validation_data_path: file_prefix + "dev.tsv",
    test_data_path: file_prefix + "gen.tsv",
    model: {
        type: "pretrain_seq2seq",
        pretrained_seq2seq:{
            type: "cogs_t5",
            model_name: model_name,
            beam_search: {
                beam_size: 4,
                max_steps: 500
            }
        }
    },
    data_loader: {
        batch_sampler: {
            type: "bucket",
            batch_size: batch_size,
            padding_noise: 0.0,
            sorting_keys: ["source_tokens"]
        },
    },
    trainer: {
        num_epochs: epochs,
        patience: patience,
        num_gradient_accumulation_steps: 8,
        validation_metric: "+acc",
        // grad_clipping: 5.0,
        cuda_device: cuda_device,
        optimizer: {
            type: "adam", # default lr 0.001
            lr: 1e-4
        },
        callbacks: [{type: "tensorboard"}],
        enable_default_callbacks: false
    }
}