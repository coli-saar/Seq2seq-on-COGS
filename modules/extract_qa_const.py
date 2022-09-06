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
from modules.model.transformers.BartForConstQuestionAnswering import BartForConstQuestionAnswering

@Model.register("extractive_qa_const")
class Extractive_QA(Model):
    def __init__(self, model_name: str, vocab: Vocabulary,
                        train_with_mask:bool = False):
        super(Extractive_QA, self).__init__(vocab=vocab)
        self._indexer = PretrainedTransformerIndexer(model_name, namespace="tokens")
        self.train_with_mask = train_with_mask
        if not train_with_mask:
            self.model = BartForQuestionAnswering.from_pretrained(model_name)
        else:
            self.model = BartForConstQuestionAnswering.from_pretrained(model_name)

        self._acc = ExactMatchAcc()

    def forward(self,   source_tokens: TextFieldTensors,
                        span_start: torch.Tensor,
                        span_end: torch.Tensor,
                        metadata: Dict,
                        const_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        # print(span_start.size())
        # raise NotImplementedError
        start_positions, end_positions = span_start, span_end
        input_ids, attention_mask = source_tokens['tokens']['token_ids'], source_tokens['tokens']['mask']
        if self.train_with_mask:
            outputs = self.model(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 start_positions=start_positions,
                                 end_positions=end_positions,
                                 const_mask=const_mask)
        else:
            outputs = self.model(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 start_positions=start_positions,
                                 end_positions=end_positions,)
        if self.training:

            # start_logits, end_logits = outputs.start_logits, outputs.end_logits
            # start_logits = start_logits.unsqueeze(-1)
            # end_logits = end_logits.unsqueeze(-2)
            # # bsz, seq_len, seq_len
            # span_logits = torch.bmm(start_logits, end_logits)
            # mask = span_logits.masked_fill(~const_mask, torch.finfo(span_logits.dtype).min)
            # span_logits = span_logits + mask

            # print(span_logits.cpu().detach().numpy()[0])
            # raise NotImplementedError
            return outputs
        else:
            start_logits, end_logits = outputs.start_logits, outputs.end_logits
            # start_logits = start_logits.unsqueeze(-1)
            # end_logits = end_logits.unsqueeze(-2)
            # # bsz, seq_len, seq_len
            # span_logits = torch.bmm(start_logits, end_logits)
            # mask = span_logits.masked_fill(~const_mask, torch.finfo(span_logits.dtype).min)
            # span_logits = span_logits + mask
            #
            # start_pos_pred, end_pos_pred = [], []
            # for i in range(span_logits.size()[0]):
            #     # print(span_logits.size())
            #     x = span_logits[i]
            #     # print((x==torch.max(x)).nonzero().cpu().detach().numpy())
            #     start, end = (x==torch.max(x)).nonzero().tolist()[0]
            #     start_pos_pred.append(start)
            #     end_pos_pred.append(end)

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