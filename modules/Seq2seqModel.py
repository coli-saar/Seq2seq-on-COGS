from typing import Dict, Union, List

from overrides import overrides
import torch

from allennlp.common.checks import ConfigurationError
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.data.fields import MetadataField
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder
from allennlp.data.fields.text_field import TextFieldTensors
from allennlp.nn.util import get_text_field_mask
from allennlp.nn import util, InitializerApplicator
from allennlp.nn.util import get_text_field_mask
from modules.training.metrics.exact_match import ExactMatchAcc

@Model.register('pretrain_seq2seq')
class COGS_model(Model):
    def __init__(self, vocab: Vocabulary,
                    pretrained_seq2seq: Model,
                    ):
        super(COGS_model, self).__init__(vocab)
        self.pretrained_seq2seq = pretrained_seq2seq
        self._acc = ExactMatchAcc()

    def forward(self, source_tokens: TextFieldTensors,
                    target_tokens: TextFieldTensors = None,
                    metadata: List[Dict] = None) -> Dict[str, torch.Tensor]:

        output = self.pretrained_seq2seq(source_tokens, target_tokens)

        if not self.training:
            if "predicted_text" in output:
                self._acc(output["predicted_text"], metadata)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        if not self.training:
            metrics.update(self.pretrained_seq2seq.get_metrics(reset=reset))
            metrics.update(self._acc.get_metric(reset=reset))
        return metrics

