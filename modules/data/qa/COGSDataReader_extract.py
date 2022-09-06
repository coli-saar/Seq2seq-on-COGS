#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# cogs_reader.py
# Master LST
# author  piaw@coli
# OS      Ubuntu 20.04
# Python  3.7
# COGSDatasetReader class


import csv
import copy
from typing import Dict, List, Iterator, Optional
from overrides import overrides

import torch

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, MetadataField, TensorField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer, PretrainedTransformerTokenizer
from allennlp.data.token_indexers import PretrainedTransformerIndexer


# inspiration from https://guide.allennlp.org/semantic-parsing-seq2seq#4
# and allennlp_models/generation/dataset_readers/cogs_oracle.py
@DatasetReader.register("cogs_extract_qa_reader")
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
                if constrain_size and line_num > self.debug_max_train_size:
                    break
                if len(row) < 4:
                    raise ConfigurationError(
                        f"Invalid line format: {row} "
                        f"(line number {line_num + 1})")
                sentence, question, ans, gen_type_required = row[:4]
                # source_tokens = self._tokenizer.tokenize(sentence)
                # quest_tokens = self._tokenizer.tokenize(question)

                source_tokens = self._tokenizer.tokenize('{} {}'.format(sentence, question))

                if (ans[0] == 'a' or ans[0] == 't') and ans not in sentence:
                    ans = ans[0].upper()+ans[1:]

                if ans != sentence[:len(ans)]:
                    ans = ' '+ans

                ans_tokens = self._tokenizer.tokenize(ans)

                start_idx, end_idx, ans_tokens = format_ans_for_bart(ans_tokens, source_tokens)

                type_of_sample = gen_type_required
                yield self.text_to_instance(source_tokens=source_tokens,
                                            start_idx=start_idx,
                                            end_idx=end_idx,
                                            type_of_sample=type_of_sample,
                                            target_text=ans
                                            )

    @overrides
    def text_to_instance(self,
                         source_tokens: List[Token],
                         start_idx: int,
                         end_idx: int,
                         type_of_sample: str = None,
                         target_text:str = None) -> Instance:
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
        fields['span_start'] = TensorField(torch.LongTensor([start_idx]))
        fields['span_end'] = TensorField(torch.LongTensor([end_idx]))
        return Instance(fields)
        # if target_tokens is not None:
        #     tokenized_target = [t for t in target_tokens]
        #     # tokenized_target.insert(0, Token(START_SYMBOL))
        #     # tokenized_target.append(Token(END_SYMBOL))
        #     target_field = TextField(tokenized_target,
        #                              self._target_token_indexers)
        #     fields["target_tokens"] = target_field
        #     return Instance(fields)

def format_ans_for_bart(ans_tokens, source_tokens):
    # remove special symbols for pretrained LM tokenizers
    span_tokens = ans_tokens[1:-1]
    start_idx, end_idx = 0, 0
    match_cnt = 0
    for i in range(len(source_tokens)):
        if [token.text for token in source_tokens[i:i + len(span_tokens)]] == [token.text for token in span_tokens]:
            start_idx = i
            end_idx = start_idx + len(span_tokens)
            match_cnt += 1

    # if match_cnt == 0:
    #
    #     for i in range(len(source_tokens)):
    #         if source_tokens[i:i + len(span_tokens)] == span_tokens:
    #             start_idx = i
    #             end_idx = start_idx + len(span_tokens)
    #             match_cnt += 1

    if match_cnt > 1:
        print('multiple/zero spans matched. cnt={} {} {}'.format(match_cnt, source_tokens, span_tokens))
        # print(ans_tokens)
        # print(source_tokens)
        raise ConfigurationError

    return start_idx, end_idx, ans_tokens


def main():
    """testing the dataset reader..."""
    # x = [Token(text='I'), Token(text='eat')]
    # y = [Token(text='do'), Token(text='I'), Token(text='eat')]
    # print(x==y[1:])
    # return
    testfile = "../COGS/data/qa/all/dev.tsv"
    dataset_reader = COGSDatasetReader(
        # source_token_indexers={
        #     "tokens": SingleIdTokenIndexer(namespace="source_tokens")},
        # target_token_indexers={
        #     "tokens": SingleIdTokenIndexer(namespace="target_tokens")},
        tokenizer=PretrainedTransformerTokenizer('facebook/bart-base'),
        source_token_indexers={
            "tokens": PretrainedTransformerIndexer('facebook/bart-base')},
        target_token_indexers={
            "tokens": PretrainedTransformerIndexer('facebook/bart-base')},
        keep_metadata=True
    )
    instances = list(dataset_reader.read(testfile))

    for instance in instances[:10]:
        print(instance)
        print(instance['span_start'].tensor.numpy())

    print("\n-Done!-")
    return


if __name__ == "__main__":
    main()