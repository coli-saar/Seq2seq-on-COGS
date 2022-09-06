from allennlp.training.metrics.metric import Metric
from typing import List, Dict, Any

@Metric.register("acc")
class ExactMatchAcc(Metric):
    def __init__(self, ignore_curly_brackets=False):
        self.match_num = 0
        self.total_num = 0
        self.ignore_curly_brackets = ignore_curly_brackets

    def reset(self) -> None:
        self.match_num = 0
        self.total_num = 0

    def __call__(self, predicted_text: List[str],
                        metadata: List[Dict]):
        count = 1
        for i in range(len(predicted_text)):
            predstr = predicted_text[i]
            goldstr = metadata[i]['target_text']
            # if predstr != goldstr and ''.join(predicted_text[i].split()) == ''.join(metadata[i]['target_text'].split()):
            #     print(predicted_text[i])
            #     print(metadata[i]['target_text'])

            if self.ignore_curly_brackets:
                goldstr = goldstr.replace('{', '')
                goldstr = goldstr.replace('}', '')

            if predstr == goldstr:
                self.match_num += 1

            self.total_num += 1

    def get_metric(self, reset: bool) -> Dict[str, Any]:
        acc = self.match_num * 1.0 / self.total_num if self.total_num != 0 else 0
        total_num = self.total_num
        match_num = self.match_num
        if reset:
            self.reset()
        return {'acc': acc}