from allennlp.modules.transformer.t5 import *
from typing import Optional, Tuple, List, Union, Dict, TYPE_CHECKING, NamedTuple, Callable

class fastT5(T5):

    def forward(
        self,
        input_ids: IntT,
        attention_mask: Optional[BoolT] = None,
        labels: Optional[IntT] = None,
        decoder_attention_mask: Optional[BoolT] = None,
        **kwargs,
    ) -> T5Output:
        """
        Run forward pass of the model.
        """
        if attention_mask is None:
            attention_mask = ~(input_ids == self.pad_token_id)

        # Encode inputs.
        encoder_outputs: T5StackOutput = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=self.output_attentions,
            output_all_hidden_states=self.output_all_hidden_states,
        )

        logits: Optional[FloatT] = None
        loss: Optional[FloatT] = None
        decoder_outputs: Optional[T5StackOutput] = None
        predictions: Optional[IntT] = None
        predicted_log_probs: Optional[FloatT] = None

        if labels is not None:
            # Calculate loss against targets.

            if decoder_attention_mask is None:
                decoder_attention_mask = ~(labels == self.pad_token_id)

            # Get decoder inputs from shifting lm labels to the right and pre-pending
            # the decoder start token ID.
            # Shape (both): (batch_size, target_length)
            decoder_input_ids = self._shift_right(labels, self.decoder_start_token_id)

            # Replace possible -100 values in labels by `pad_token_id`
            decoder_input_ids.masked_fill_(decoder_input_ids == -100, self.pad_token_id)

            # Decode.
            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=encoder_outputs.last_hidden_state,
                encoder_attention_mask=attention_mask,
                output_attentions=self.output_attentions,
                output_all_hidden_states=self.output_all_hidden_states,
            )

            # Shape: (batch_size, target_length, vocab_size)
            logits = self._get_lm_logits(decoder_outputs.last_hidden_state)  # type: ignore[union-attr]

            # Shape: (1,)
            loss = self.loss_fct(logits.view(-1, logits.size(-1)), labels.to(torch.long).view(-1))
        elif self.training:
            raise ValueError("'labels' required during training")

        do_inference = True if 'test_flag' in kwargs and kwargs['test_flag'] else False
        if not self.training and do_inference:
            # Use beam search to generate a sequence of predicted tokens.

            # Shape: (batch_size, 1)
            initial_decoder_ids = torch.tensor(
                [[self.decoder_start_token_id]],
                dtype=input_ids.dtype,
                device=input_ids.device,
            ).repeat(input_ids.shape[0], 1)

            initial_state = {
                "input_ids": input_ids,
                "encoder_hidden_states": encoder_outputs.last_hidden_state,
                "encoder_attention_mask": attention_mask,
            }

            # Run the beam search.
            # Shape (predictions): (batch_size, beam_size, max_decoding_steps)
            # Shape (predicted_log_probs):   (batch_size, beam_size)
            predictions, predicted_log_probs = self.beam_search.search(
                initial_decoder_ids, initial_state, self.take_search_step
            )

        return T5Output(
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_all_hidden_states=encoder_outputs.all_hidden_states,
            decoder_last_hidden_state=(
                None if decoder_outputs is None else decoder_outputs.last_hidden_state
            ),
            decoder_all_hidden_states=(
                None if decoder_outputs is None else decoder_outputs.all_hidden_states
            ),
            encoder_attentions=encoder_outputs.attentions,
            decoder_attentions=None if decoder_outputs is None else decoder_outputs.attentions,
            cross_attentions=None if decoder_outputs is None else decoder_outputs.cross_attentions,
            loss=loss,
            logits=logits,
            predictions=predictions,
            predicted_log_probs=predicted_log_probs,
        )