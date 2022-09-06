from typing import Tuple, Dict, Optional
from overrides import overrides

import torch
from torch.nn import LSTMCell, LSTM, Linear
from torch.nn import Dropout

from allennlp.modules import Attention
from allennlp.nn import util

from allennlp_models.generation.modules.decoder_nets.decoder_net import DecoderNet


@DecoderNet.register("customized_lstm_cell")
class CustomizedLstmCellDecoderNet(DecoderNet):
    """
    This decoder net implements simple decoding network with LSTMCell and Attention.
    # Parameters
    decoding_dim : `int`, required
        Defines dimensionality of output vectors.
    target_embedding_dim : `int`, required
        Defines dimensionality of input target embeddings.  Since this model takes it's output on a previous step
        as input of following step, this is also an input dimensionality.
    attention : `Attention`, optional (default = `None`)
        If you want to use attention to get a dynamic summary of the encoder outputs at each step
        of decoding, this is the function used to compute similarity between the decoder hidden
        state and encoder outputs.
    """

    def __init__(
        self,
        decoding_dim: int,
        target_embedding_dim: int,
        attention: Optional[Attention] = None,
        bidirectional_input: bool = False,
        dropout_input: float = 0.0,
        dropout_out: float = 0.0,
        num_layers: int = 1
    ) -> None:

        super().__init__(
            decoding_dim=decoding_dim,
            target_embedding_dim=target_embedding_dim,
            decodes_parallel=False,
        )

        # In this particular type of decoder output of previous step passes directly to the input of current step
        # We also assume that decoder output dimensionality is equal to the encoder output dimensionality
        decoder_input_dim = self.target_embedding_dim

        # Attention mechanism applied to the encoder output for each step.
        self._attention = attention

        if self._attention:
            # If using attention, a weighted average over encoder outputs will be concatenated
            # to the previous target embedding to form the input to the decoder at each
            # time step. encoder output dim will be same as decoding_dim
            decoder_input_dim += decoding_dim

        # We'll use an LSTM cell as the recurrent cell that produces a hidden state
        # for the decoder at each time step.
        self._num_layers = num_layers

        if num_layers == 1:
            self._decoder_cell = LSTMCell(decoder_input_dim, self.decoding_dim)
        else:
            self._decoder_cell = LSTM(decoder_input_dim, self.decoding_dim, num_layers, dropout=dropout_out)

        self._bidirectional_input = bidirectional_input

        self._dropout_in = Dropout(p=dropout_input)
        self._dropout_out = Dropout(p=dropout_out)

        self._combine_layer = Linear(self.target_embedding_dim+self.decoding_dim * 3, \
                                     self.decoding_dim)

    def _prepare_attended_input(
        self,
        decoder_hidden_state: torch.Tensor = None,
        encoder_outputs: torch.Tensor = None,
        encoder_outputs_mask: torch.BoolTensor = None,
    ) -> torch.Tensor:
        """Apply attention over encoder outputs and decoder state."""
        # shape: (batch_size, max_input_sequence_length)
        input_weights = self._attention(decoder_hidden_state, encoder_outputs, encoder_outputs_mask)

        # shape: (batch_size, encoder_output_dim)
        attended_input = util.weighted_sum(encoder_outputs, input_weights)

        return attended_input

    def init_decoder_state(
        self, encoder_out: Dict[str, torch.LongTensor]
    ) -> Dict[str, torch.Tensor]:

        batch_size, _ = encoder_out["source_mask"].size()

        # Initialize the decoder hidden state with the final output of the encoder,
        # and the decoder context with zeros.
        # shape: (batch_size, encoder_output_dim)
        final_encoder_output = util.get_final_encoder_states(
            encoder_out["encoder_outputs"],
            encoder_out["source_mask"],
            bidirectional=self._bidirectional_input,
        )
        if self._num_layers == 1:
            return {
                "decoder_hidden": final_encoder_output,  # shape: (batch_size, decoder_output_dim)
                "decoder_context": final_encoder_output.new_zeros(batch_size, self.decoding_dim)
                #                  shape: (batch_size, decoder_output_dim)
            }
        else:
            return {
                "decoder_hidden": final_encoder_output.unsqueeze(0).repeat(self._num_layers, 1, 1),
                "decoder_context": final_encoder_output.new_zeros(batch_size, self.decoding_dim).unsqueeze(0).\
                    repeat(self._num_layers, 1, 1)
            }
    #
    # @overrides
    # def forward(
    #     self,
    #     previous_state: Dict[str, torch.Tensor],
    #     encoder_outputs: torch.Tensor,
    #     source_mask: torch.BoolTensor,
    #     previous_steps_predictions: torch.Tensor,
    #     previous_steps_mask: Optional[torch.BoolTensor] = None,
    #     last_feed: torch.Tensor = None,
    # ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    #
    #     decoder_hidden = previous_state["decoder_hidden"]
    #     decoder_context = previous_state["decoder_context"]
    #
    #     # shape: (group_size, output_dim)
    #     last_predictions_embedding = previous_steps_predictions[:, -1]
    #
    #     if self._attention:
    #         # shape: (group_size, encoder_output_dim)
    #         attended_input = self._prepare_attended_input(
    #             decoder_hidden, encoder_outputs, source_mask
    #         )
    #
    #         # shape: (group_size, decoder_output_dim + target_embedding_dim)
    #         decoder_input = torch.cat((attended_input, last_predictions_embedding), -1)
    #     else:
    #         # shape: (group_size, target_embedding_dim)
    #         decoder_input = last_predictions_embedding
    #
    #     decoder_input = self._dropout_in(decoder_input)
    #
    #     # shape (decoder_hidden): (batch_size, decoder_output_dim)
    #     # shape (decoder_context): (batch_size, decoder_output_dim)
    #     decoder_hidden, decoder_context = self._decoder_cell(
    #         decoder_input.float(), (decoder_hidden.float(), decoder_context.float())
    #     )
    #
    #     return (
    #         {"decoder_hidden": decoder_hidden, "decoder_context": decoder_context},
    #         decoder_hidden, decoder_input
    #     )


    @overrides
    def forward(
        self,
        previous_state: Dict[str, torch.Tensor],
        encoder_outputs: torch.Tensor,
        source_mask: torch.BoolTensor,
        previous_steps_predictions: torch.Tensor,
        previous_steps_mask: Optional[torch.BoolTensor] = None,
        decoder_output_features: torch.Tensor = None
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:

        decoder_hidden = previous_state["decoder_hidden"]
        decoder_context = previous_state["decoder_context"]

        # shape: (group_size, output_dim)
        last_predictions_embedding = previous_steps_predictions[:, -1]

        # print('##################')
        # print(decoder_output_features.shape, last_predictions_embedding.shape)
        # raise NotImplementedError

        # shape: (group_size, feature_dim + target_embedding_dim)
        decoder_input = torch.cat((decoder_output_features, last_predictions_embedding), -1)

        # if self._attention:
        #     # shape: (group_size, encoder_output_dim)
        #     attended_input = self._prepare_attended_input(
        #         decoder_hidden, encoder_outputs, source_mask
        #     )
        #
        #     # shape: (group_size, decoder_output_dim + target_embedding_dim)
        #     decoder_input = torch.cat((attended_input, last_predictions_embedding), -1)
        # else:
        #     # shape: (group_size, target_embedding_dim)
        #     decoder_input = last_predictions_embedding

        # decoder_input = self._dropout_in(decoder_input)

        # shape (decoder_hidden): (batch_size, decoder_output_dim)
        # shape (decoder_context): (batch_size, decoder_output_dim)
        if self._num_layers == 1:
            decoder_hidden, decoder_context = self._decoder_cell(
                decoder_input.float(), (decoder_hidden.float(), decoder_context.float())
            )
        else:
            # shape: (1, batch_size, feature_dim + target_embedding_dim)
            decoder_input = decoder_input.unsqueeze(0)

            decoder_output, (decoder_hidden, decoder_context) = self._decoder_cell(
                decoder_input.float(), (decoder_hidden.contiguous().float(), decoder_context.contiguous().float())
            )

        if self._attention:
            if self._num_layers == 1:
                # shape: (group_size, encoder_output_dim)
                attended_input = self._prepare_attended_input(
                    decoder_hidden, encoder_outputs, source_mask
                )

                output_projection_feature = torch.cat([decoder_input, decoder_hidden, attended_input], dim=-1)

            else:
                # shape: (group_size, encoder_output_dim)
                attended_input = self._prepare_attended_input(
                    decoder_output[0], encoder_outputs, source_mask
                )

                output_projection_feature = torch.cat([decoder_input[0], decoder_output[0], attended_input], dim=-1)

        # output_projection_feature = torch.cat([decoder_input, decoder_hidden, attended_input], dim=-1)
        #
        output_projection_feature = self._dropout_out(self._combine_layer(output_projection_feature))

        return (
            {"decoder_hidden": decoder_hidden, "decoder_context": decoder_context, "decoder_output_features": output_projection_feature},
            decoder_hidden, output_projection_feature
        )