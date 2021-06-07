import torch.nn as nn

from attention import MultiHeadAttention
from module import PositionalEncoding, PositionwiseFeedForward
from utils_transformer import get_non_pad_mask, get_attn_pad_mask


class Encoder(nn.Module):
    """Encoder of Transformer including self-attention and feed forward.
    """

    def __init__(self, d_input, n_layers, n_head, d_k, d_v,
                 d_model, d_inner, dropout=0.1, pe_maxlen=5000):
        super(Encoder, self).__init__()
        # parameters
        self.d_input = d_input   #input dimension
        self.n_layers = n_layers
        self.n_head = n_head  
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model   ##hidden size
        self.d_inner = d_inner
        self.dropout_rate = dropout
        self.pe_maxlen = pe_maxlen
        self.batch = 64

        # use linear transformation with layer norm to replace input embedding
        self.linear_in = nn.Linear(d_input, d_model)
        self.layer_norm_in = nn.LayerNorm(d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=pe_maxlen)
        self.dropout = nn.Dropout(dropout)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, padded_input, input_lengths, return_attns=False):
        """
        Args:
            padded_input:   B x L x D_head  [64,25,256]
            input_lengths: L

        Returns:
            enc_output: B x L x Hidden  [64,25,256]
        """
        enc_slf_attn_list = []

        all_encoder_outputs = []

        # Prepare masks
        non_pad_mask = get_non_pad_mask(padded_input, input_lengths=input_lengths)
        length = padded_input.size(1)  #25
        slf_attn_mask = get_attn_pad_mask(padded_input, input_lengths, length)  #[64,25,25]

        # Forward
        enc_output = self.dropout(
            self.layer_norm_in(self.linear_in(padded_input)) +
            self.positional_encoding(padded_input))

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            #enc_slf_attn = enc_slf_attn.reshape(self.batch, 12, length, length)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]
                all_encoder_outputs += [enc_output]

        if return_attns:
            return all_encoder_outputs, enc_slf_attn_list
        return enc_output,


class EncoderLayer(nn.Module):
    """Compose with two sub-layers.
        1. A multi-head self-attention mechanism
        2. A simple, position-wise fully connected feed-forward network.
    """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask

        return enc_output, enc_slf_attn
