import torch

from transformers import BartForQuestionAnswering
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import Seq2SeqQuestionAnsweringModelOutput

class BartForConstQuestionAnswering(BartForQuestionAnswering):
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        encoder_outputs=None,
        start_positions=None,
        end_positions=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        const_mask=None,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if start_positions is not None and end_positions is not None:
            use_cache = False

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)
            #
            # print(ignored_index)
            # print(start_logits.cpu().detach().numpy())
            # print(start_positions.cpu().detach().numpy())
            # raise NotImplementedError

            # obtain span logits based on independent start/end position scores
            # bsz, seq_len, seq_len
            span_logits = torch.bmm(start_logits.unsqueeze(-1), end_logits.unsqueeze(-2))
            mask = span_logits.masked_fill(~const_mask, torch.finfo(span_logits.dtype).min)
            span_logits = span_logits + mask

            target_logits = span_logits.new_zeros(span_logits.size(0), dtype=torch.int64)
            for i in range(start_positions.size(0)):
                target_logits[i] = start_positions[i]*ignored_index+end_positions[i]
            # print(target_logits[:2].cpu().detach().numpy())
            # print(start_positions[:2].cpu().detach().numpy())
            # print(end_positions[:2].cpu().detach().numpy())
            #
            # raise NotImplementedError

            # loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            # bsz, seq_len     bsz, => bsz, seq_len, seq_len     bsz, seq_len, seq_len
            # target indice   =>    probability
            # start_loss = loss_fct(start_logits, start_positions)
            # end_loss = loss_fct(end_logits, end_positions)
            # total_loss = (start_loss + end_loss) / 2
            #
            loss_fct = CrossEntropyLoss()
            total_loss = loss_fct(span_logits.flatten(start_dim=1), target_logits)

        if not return_dict:
            output = (
                start_logits,
                end_logits,
            ) + outputs[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        return Seq2SeqQuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )