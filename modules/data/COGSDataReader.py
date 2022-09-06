# This class is modified from the code written by Pia Weißenhorn

import csv
import copy
from typing import Dict, List, Iterator, Optional
from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer, PretrainedTransformerIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer, PretrainedTransformerTokenizer


# inspiration from https://guide.allennlp.org/semantic-parsing-seq2seq#4
# and allennlp_models/generation/dataset_readers/cogs_oracle.py
@DatasetReader.register("cogs_reader")
class COGSDatasetReader(DatasetReader):
    """
    COGS (Kim & Linzen 2020) Dataset reader
    format of the dataset: TSV
    each line is one sample, consisting of 3 rows:
    1. The sentence (whitespace tokenized already)
    2. The logical form (whitespace tokenized as well)
    3. `in_distribution` or in gen.tsv which generalization type is required
    """

    def __init__(self,
                 tokenizer: Tokenizer = WhitespaceTokenizer(),
                 source_token_indexers: Dict[str, TokenIndexer] = None,
                 target_token_indexers: Dict[str, TokenIndexer] = None,
                 source_add_start_token: bool = False,
                 source_add_end_token: bool = False,
                 target_add_start_token: bool = False,
                 target_add_end_token: bool = False,
                 start_symbol: str = '<s>',
                 end_symbol: str = '</s>',
                 debug_max_train_size: int = None,
                 keep_metadata: bool = False,
                 **kwargs,) -> None:
        """
        Initializes the dataset reader, by default strips metadata
        By default, we add start and end symbols to source and target sequence
        :param tokenizer: should keep default white space tokenizer unless
        you're using some pretrained model to embed inputs that comes with it's
        own tokenization
        :param debug_max_train_size: debugging option for smaller training set
        :param keep_metadata: ideally we would like to keep metadata (i.e.
        the 3rd row, the input tokens), but if we use allennlp's default models,
        this will crash (extra parameter passed to forward)
        """
        super().__init__(**kwargs)
        self._tokenizer = tokenizer
        # same tokenizer used for source and target. For COGS, whitespace token-
        # ization is a good baseline, since data already comes pretokenized
        self._source_token_indexers = source_token_indexers or {
            "tokens": SingleIdTokenIndexer()
        }
        self._target_token_indexers = target_token_indexers or self._source_token_indexers

        # start and end tokens...
        self._source_add_start_token = source_add_start_token
        self._source_add_end_token = source_add_end_token
        self._target_add_start_token = target_add_start_token
        self._target_add_end_token = target_add_end_token
        self._start_token: Optional[Token] = None
        self._end_token: Optional[Token] = None
        if (source_add_start_token or source_add_end_token or
                target_add_start_token or target_add_end_token):
            # Check that the tokenizer correctly appends the start and end
            # tokens to the sequence without splitting them.
            tokens = self._tokenizer.tokenize(
                start_symbol + " " + end_symbol)
            err_msg = f"Bad start or end symbol ('{start_symbol}', " \
                      f"'{end_symbol}') for tokenizer {self._tokenizer}"
            try:
                start_token, end_token = tokens[0], tokens[-1]
            except IndexError:
                raise ValueError(err_msg)
            if start_token.text != start_symbol or end_token.text != end_symbol:
                raise ValueError(err_msg)
            self._start_token = start_token
            self._end_token = end_token

        self._keep_metadata = keep_metadata
        # max train size
        # debug only_ max train size, might delete later:
        assert(debug_max_train_size is None or debug_max_train_size > 0)
        self.debug_max_train_size = debug_max_train_size

    @overrides
    def _read(self, file_path: str) -> Iterator[Instance]:
        constrain_size = self.debug_max_train_size is not None and \
                         file_path.endswith("train.tsv")
        with open(cached_path(file_path), encoding='utf-8', mode='r') as infile:
            for line_num, row in enumerate(csv.reader(infile, delimiter="\t")):
                if constrain_size and line_num >= self.debug_max_train_size:
                    break
                if len(row) != 3 :
                    raise ConfigurationError(
                        f"Invalid line format: {row} "
                        f"(line number {line_num + 1})")
                sentence, representation, gen_type_required = row
                source_tokens = self._tokenizer.tokenize(sentence)
                if self._source_add_start_token:
                    source_tokens.insert(0, copy.deepcopy(self._start_token))
                if self._source_add_end_token:
                    source_tokens.append(copy.deepcopy(self._end_token))
                target_tokens = self._tokenizer.tokenize(representation)
                if self._target_add_start_token:
                    target_tokens.insert(0, copy.deepcopy(self._start_token))
                if self._target_add_end_token:
                    target_tokens.append(copy.deepcopy(self._end_token))
                type_of_sample = gen_type_required
                yield self.text_to_instance(source_tokens=source_tokens,
                                            target_tokens=target_tokens,
                                            type_of_sample=type_of_sample,
                                            target_text=representation)

    @overrides
    def text_to_instance(self,
                         source_tokens: List[Token],
                         target_tokens: List[Token] = None,
                         type_of_sample: str = None,
                         target_text: str = None) -> Instance:
        fields: Dict[str, Field] = {}
        # wrap each token in the file with a token object
        tokens = TextField([w for w in source_tokens],
                           self._source_token_indexers)
        # Instances in AllenNLP are created using Python dictionaries,
        # which map the token key to the Field type
        fields["source_tokens"] = tokens
        if self._keep_metadata:
            fields["metadata"] = MetadataField(
                {"type_of_sample": type_of_sample,
                 "source_tokens": source_tokens,
                 "target_text": target_text})
        if target_tokens is not None:
            tokenized_target = [t for t in target_tokens]
            tokenized_target.insert(0, Token(START_SYMBOL))
            tokenized_target.append(Token(END_SYMBOL))
            target_field = TextField(tokenized_target,
                                     self._target_token_indexers)
            fields["target_tokens"] = target_field
            return Instance(fields)


def main():
    """testing the dataset reader..."""
    testfile = "../COGS/data/sem/gen.tsv"
    pretrain_model =  't5-base'
    dataset_reader = COGSDatasetReader(
        tokenizer=PretrainedTransformerTokenizer(pretrain_model),
        source_token_indexers={
            "tokens": PretrainedTransformerIndexer(pretrain_model)},
        target_token_indexers={
            "tokens": PretrainedTransformerIndexer(pretrain_model)},
    )
    instances = list(dataset_reader.read(testfile))

    output_len = {}
    max_decode_len = 0
    for instance in instances:
        if len(instance['target_tokens']) not in output_len:
            output_len[len(instance['target_tokens'])] = 1
        else:
            output_len[len(instance['target_tokens'])] += 1
            if len(instance['target_tokens']) > max_decode_len:
                max_decode_len = len(instance['target_tokens'])

    print(max_decode_len)
    # sorted_list = sorted(list(output_len.items()), key=lambda x: x[0])
    # print(sorted_list)
    # x = sum([y if x > 250 else 0 for x, y in sorted_list])
    # print(x)
    # print("\n-Done!-")

    return


if __name__ == "__main__":
    main()