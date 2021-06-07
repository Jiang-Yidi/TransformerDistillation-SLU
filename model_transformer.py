import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import read_config
import numpy as np
import os, glob
from models import PretrainedModel, freeze_layer
from encoder import Encoder


class FinalPool(torch.nn.Module):
        def __init__(self):
                super(FinalPool, self).__init__()

        def forward(self, input):
                """
                input : Tensor of shape (batch size, T, Cin)

                Outputs a Tensor of shape (batch size, Cin).
                """

                return input.max(dim=1)[0]



class TransformerModelSpeech(nn.Module):
    def __init__(self, config, nIntent, embedding_size, batchsize, ninp, nhead, nhid, nlayers, d_k, d_v, d_model, d_inner, dropout=0.5):
        super(TransformerModelSpeech, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        pretrained_model = PretrainedModel(config)
        if config.pretraining_type != 0:
            pretrained_model_path = os.path.join(config.folder, "pretraining", "model_state.pth")
            if self.is_cuda:
                pretrained_model.load_state_dict(torch.load(pretrained_model_path))
            else:
                pretrained_model.load_state_dict(torch.load(pretrained_model_path, map_location="cpu"))
        self.pretrained_model = pretrained_model
        self.unfreezing_type = config.unfreezing_type
        self.unfreezing_index = config.starting_unfreezing_index
        self.intent_layers = []
        if config.pretraining_type != 0:
            self.freeze_all_layers()
        self.seq2seq = config.seq2seq
        out_dim = config.word_rnn_num_hidden[-1]
        if config.word_rnn_bidirectional:
            out_dim *= 2

        if not self.seq2seq:
            self.values_per_slot = config.values_per_slot
            self.num_values_total = sum(self.values_per_slot)
        
        self.ninp=ninp
        self.FinalPoollayer = FinalPool()
        self.encoder_model=Encoder(self.ninp, nlayers, nhead, d_k, d_v, d_model, d_inner) #d_input, n_layers, n_head, d_k, d_v, d_model, d_inner-----64, 64, 128, 512
        #from torch.nn import TransformerEncoder, TransformerEncoderLayer
        #self.model_type = 'Transformer'
        #self.src_mask = None
        #self.pos_encoder = PositionalEncoding(ninp, dropout)
        #encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        #self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        #self.encoder = nn.Embedding(ntoken, ninp)
        #self.encoder = nn.Embedding(ninp, embedding_size)
        #self.ninp = ninp
        #self.decoder = nn.Linear(ninp, ntoken)
        self.decoder = nn.Linear(d_model, self.num_values_total)
        #self.init_weights()
    def freeze_all_layers(self):
        for layer in self.pretrained_model.phoneme_layers:
            freeze_layer(layer)
        for layer in self.pretrained_model.word_layers:
            freeze_layer(layer)
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    def init_weights(self):
        initrange = 0.1
        self.encoder_model.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
    def forward(self, x, y_intent):
        src = self.pretrained_model.compute_features(x)  # pre_feat[64,L,256]  ##[B,12,256]
        #print("src.shape: {}".format(src.size()))
        device = src.device
        inp_lengths=np.repeat(self.ninp,src.shape[0])
        output_enc=self.encoder_model(src,inp_lengths,return_attns=True)
        hidden = output_enc[0]  #[64,25,256]
        attn = output_enc[1]  #[192,25,25]*6
        output = self.decoder(hidden[-1])
        output2 = self.FinalPoollayer(output)
        intent_logits = output2.to(device)  # shape: (batch size, num_values_total)
        intent_loss = 0.
        start_idx = 0
        predicted_intent = []
        score=[]
        y_intent=y_intent.to(device)
        for slot in range(len(self.values_per_slot)):
            end_idx = start_idx + self.values_per_slot[slot]
            subset = intent_logits[:, start_idx:end_idx]
            intent_loss += torch.nn.functional.cross_entropy(subset, y_intent[:, slot])
            predicted_intent.append(subset.max(1)[1])
            start_idx = end_idx
            b = torch.nn.functional.softmax(subset, 0)
            a=y_intent[:,slot]
        predicted_intent = torch.stack(predicted_intent, dim=1)
        #breakpoint()
        intent_acc = (predicted_intent == y_intent).prod(1).float().mean()  # all slots must be correct
        return output2, intent_loss, intent_acc, attn, hidden, src.size(1), score


class TransformerModelText(nn.Module):
    def __init__(self, nIntent,  embedding_size, batchsize, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModelText, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(embedding_size, dropout)
        encoder_layers = TransformerEncoderLayer(embedding_size, nhead, nhid, dropout)

        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        #self.encoder = nn.Embedding(ntoken, ninp)
        self.encoder = nn.Embedding(ninp,embedding_size)
        self.ninp = ninp
        self.decoder = nn.Linear(nhid, nIntent)
        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, pad_mask):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        #output1=torch.squeeze(output, 2)
        #output11=output[:, -1, :]
        output1=output.mul(pad_mask.transpose(1, 2)).sum(1)
        output1 = output1 / pad_mask.sum(2)
        output2 = self.decoder(output1)
        return output2

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class CrossAttention(nn.Module):
    r"""TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = decoder_layer(tgt, memory)
    """

    def __init__(self, d_model, nhead, dim_feedforward=128, dropout=0.1, activation="relu"):
        #d_model=input dimension of speech features
        super(CrossAttention, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

#    def __setstate__(self, state):
#        if 'activation' not in state:
#            state['activation'] = F.relu
#        super(nn.TransformerDecoderLayer, self).__setstate__(state)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # type: (Tensor, Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor]) -> Tensor
        r"""Pass the inputs (and mask) through the decoder layer.
        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        #tgt=SpeechFeature
        #memory=Text Feature
        torch.cuda.empty_cache()
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        #tgt=tgt.cpu()
        tgt2=tgt2.cpu()
        return tgt


#def _get_clones(module, N):
#    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))
