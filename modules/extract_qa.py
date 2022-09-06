from typing import Dict

import numpy
import torch

from allennlp.models.model import Model
from allennlp.modules.feedforward import FeedForward
from allennlp.data.fields.text_field import TextFieldTensors
from allennlp.data import Vocabulary


from transformers import BartTokenizer, BartForQuestionAnswering
from allennlp.data.token_indexers.pretrained_transformer_indexer import PretrainedTransformerIndexer

from modules.training.metrics.exact_match import ExactMatchAcc

@Model.register("extractive_qa")
class Extractive_QA(Model):
    def __init__(self, model_name: str, vocab: Vocabulary):
        super(Extractive_QA, self).__init__(vocab=vocab)
        self._indexer = PretrainedTransformerIndexer(model_name, namespace="tokens")
        self.model = BartForQuestionAnswering.from_pretrained(model_name)

        self._acc = ExactMatchAcc()

    def forward(self,   source_tokens: TextFieldTensors,
                        span_start: torch.Tensor,
                        span_end: torch.Tensor,
                        metadata: Dict,) -> Dict[str, torch.Tensor]:
        # print(span_start.size())
        # raise NotImplementedError
        start_positions, end_positions = span_start, span_end
        input_ids, attention_mask = source_tokens['tokens']['token_ids'], source_tokens['tokens']['mask']
        outputs = self.model(input_ids=input_ids,
                             attention_mask=attention_mask,
                             start_positions=start_positions,
                             end_positions=end_positions)
        if self.training:
            return outputs
        else:
            start_logits, end_logits = outputs.start_logits, outputs.end_logits

            start_pos_pred = torch.argmax(start_logits, dim=-1).tolist()
            end_pos_pred = torch.argmax(end_logits, dim=-1).tolist()
            # bsz, seq_len
            input_ids_list = input_ids.tolist()
            ans_ids = [input_ids_list[i][start_pos_pred[i]:end_pos_pred[i]] for i in range(len(input_ids_list))]
            ans = self._indexer._tokenizer.batch_decode(ans_ids, skip_special_tokens=True)
            outputs['predicted_text'] = ans
            self._acc(ans, metadata)
            # return outputs
            return {'predicted_text': ans}
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {}
        if not self.training:
            metrics.update(self._acc.get_metric(reset=reset))
        return metrics