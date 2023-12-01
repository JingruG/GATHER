import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import math
import json
import os
import time
import argparse

from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

def bool_flag(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def check_path(path):
    d = os.path.dirname(path)
    if not os.path.exists(d):
        os.makedirs(d)


def check_file(file):
    return os.path.isfile(file)


def export_config(config, path):
    param_dict = dict(vars(config))
    check_path(path)
    with open(path, 'w', encoding='utf-8') as fout:
        json.dump(param_dict, fout, indent=4)


def freeze_net(module):
    for p in module.parameters():
        p.requires_grad = False


def unfreeze_net(module):
    for p in module.parameters():
        p.requires_grad = True


def test_data_loader_ms_per_batch(data_loader, max_steps=10000):
    start = time.time()
    n_batch = sum(1 for batch, _ in zip(data_loader, range(max_steps)))
    return (time.time() - start) * 1000 / n_batch

def gelu(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


import math
from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        # self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        # self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        # src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        # output = self.decoder(output)
        return output

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import Parameter

from .bert import BertConfig, BERTLayer, BERTEncoder
class TransformerMapping(nn.Module):
    """ Self-attention layer for image branch
    """
    def __init__(self, in_dim, out_dim):
        super(TransformerMapping, self).__init__()
        bert_config = BertConfig.from_json_file('config/path.json')
        self.layer = BERTLayer(bert_config)
        self.mapping = nn.Linear(in_dim, out_dim)
        #self.mapping2 = nn.Linear(opt.final_dims, opt.final_dims)

    def forward(self, x):
        # x: (batch_size, patch_num, img_dim)
        x = self.mapping(x) # x: (batch_size, patch_num, final_dims)
        attention_mask = torch.ones(x.size(0), x.size(1))
        if torch.cuda.is_available():
            attention_mask = attention_mask.cuda()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.float()
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        # print('x', x.shape)
        # print('extended_attention_mask', extended_attention_mask.shape)
        hidden_states = self.layer(x, extended_attention_mask)
        # hidden_states = self.mapping2(hidden_states)
        embed = torch.mean(hidden_states, 1) # (batch_size, final_dims)
        codes = F.normalize(embed, p=2, dim=1)  # (N, C)

        # print('embed', embed.shape)
        # print('codes', codes.shape)
        return codes




class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return gelu(x)


class TypedLinear(nn.Linear):
    def __init__(self, in_features, out_features, n_type):
        super().__init__(in_features, n_type * out_features)
        self.in_features = in_features
        self.out_features = out_features
        self.n_type = n_type

    def forward(self, X, type_ids=None):
        """
        X: tensor of shape (*, in_features)
        type_ids: long tensor of shape (*)
        """
        output = super().forward(X)
        if type_ids is None:
            return output
        output_shape = output.size()[:-1] + (self.out_features,)
        output = output.view(-1, self.n_type, self.out_features)
        idx = torch.arange(output.size(0), dtype=torch.long, device=type_ids.device)
        output = output[idx, type_ids.view(-1)].view(*output_shape)
        return output


class MLP(nn.Module):
    """
    Multi-layer perceptron

    Parameters
    ----------
    num_layers: number of hidden layers
    """
    activation_classes = {'gelu': GELU, 'relu': nn.ReLU, 'tanh': nn.Tanh}

    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout, batch_norm=False,
                 init_last_layer_bias_to_zero=False, layer_norm=False, activation='gelu'):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm

        assert not (self.batch_norm and self.layer_norm)

        self.layers = nn.Sequential()
        for i in range(self.num_layers + 1):
            n_in = self.input_size if i == 0 else self.hidden_size
            n_out = self.hidden_size if i < self.num_layers else self.output_size
            self.layers.add_module(f'{i}-Linear', nn.Linear(n_in, n_out))
            if i < self.num_layers:
                self.layers.add_module(f'{i}-Dropout', nn.Dropout(self.dropout))
                if self.batch_norm:
                    self.layers.add_module(f'{i}-BatchNorm1d', nn.BatchNorm1d(self.hidden_size))
                if self.layer_norm:
                    self.layers.add_module(f'{i}-LayerNorm', nn.LayerNorm(self.hidden_size))
                self.layers.add_module(f'{i}-{activation}', self.activation_classes[activation.lower()]())
        if init_last_layer_bias_to_zero:
            self.layers[-1].bias.data.fill_(0)

    def forward(self, input):
        return self.layers(input)


class MaxPoolLayer(nn.Module):
    """
    A layer that performs max pooling along the sequence dimension
    """

    def __init__(self):
        super().__init__()

    def forward(self, inputs, mask_or_lengths):
        """
        inputs: tensor of shape (batch_size, seq_len, hidden_size)
        mask_or_lengths: tensor of shape (batch_size) or (batch_size, seq_len)

        returns: tensor of shape (batch_size, hidden_size)
        """
        bs, sl, _ = inputs.size()
        if len(mask_or_lengths.size()) == 1:
            mask = (torch.arange(sl, device=inputs.device).unsqueeze(0).expand(bs, sl) >= mask_or_lengths.unsqueeze(1))
        else:
            mask = mask_or_lengths
        masked_inputs = inputs.masked_fill(mask.unsqueeze(-1).expand_as(inputs), float('-inf'))
        max_pooled = masked_inputs.max(1)[0]
        return max_pooled


class MeanPoolLayer(nn.Module):
    """
    A layer that performs mean pooling along the sequence dimension
    """

    def __init__(self):
        super().__init__()

    def forward(self, inputs, mask_or_lengths):
        """
        inputs: tensor of shape (batch_size, seq_len, hidden_size)
        mask_or_lengths: tensor of shape (batch_size) or (batch_size, seq_len)

        returns: tensor of shape (batch_size, hidden_size)
        """
        bs, sl, _ = inputs.size()
        if len(mask_or_lengths.size()) == 1:
            mask = (torch.arange(sl, device=inputs.device).unsqueeze(0).expand(bs, sl) >= mask_or_lengths.unsqueeze(1))
            lengths = mask_or_lengths.float()
        else:
            mask, lengths = mask_or_lengths, (1 - mask_or_lengths.float()).sum(1)
        masked_inputs = inputs.masked_fill(mask.unsqueeze(-1).expand_as(inputs), 0.0)
        mean_pooled = masked_inputs.sum(1) / lengths.unsqueeze(-1)
        return mean_pooled


def dropout_mask(x, sz, p: float):
    """
    Return a dropout mask of the same type as `x`, size `sz`, with probability `p` to cancel an element.

    (adapted from https://github.com/fastai/fastai/blob/1.0.42/fastai/text/models/awd_lstm.py)
    """
    return x.new(*sz).bernoulli_(1 - p).div_(1 - p)


class EmbeddingDropout(nn.Module):
    """
    Apply dropout with probabily `embed_p` to an embedding layer `emb`.

    (adapted from https://github.com/fastai/fastai/blob/1.0.42/fastai/text/models/awd_lstm.py)
    """

    def __init__(self, emb: nn.Module, embed_p: float):
        super().__init__()
        self.emb, self.embed_p = emb, embed_p
        self.pad_idx = self.emb.padding_idx
        if self.pad_idx is None:
            self.pad_idx = -1

    def forward(self, words):
        if self.training and self.embed_p != 0:
            size = (self.emb.weight.size(0), 1)
            mask = dropout_mask(self.emb.weight.data, size, self.embed_p)
            masked_embed = self.emb.weight * mask
        else:
            masked_embed = self.emb.weight
        return F.embedding(words, masked_embed, self.pad_idx, self.emb.max_norm,
                           self.emb.norm_type, self.emb.scale_grad_by_freq, self.emb.sparse)


class RNNDropout(nn.Module):
    "Dropout with probability `p` that is consistent on the seq_len dimension."

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.:
            return x
        m = dropout_mask(x.data, (x.size(0), 1, x.size(2)), self.p)
        return x * m


class LSTMEncoder(nn.Module):

    def __init__(self, vocab_size=300, emb_size=300, hidden_size=300, num_layers=2, bidirectional=True,
                 emb_p=0, input_p=0, hidden_p=0, output_p=0, pretrained_emb=None, pooling=True, pad=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.emb_p = emb_p
        self.input_p = input_p
        self.hidden_p = hidden_p
        self.output_p = output_p
        self.pooling = pooling

        self.emb = EmbeddingDropout(nn.Embedding(vocab_size, emb_size), emb_p)
        if pretrained_emb is not None:
            self.emb.emb.weight.data.copy_(pretrained_emb)
        else:
            bias = np.sqrt(6.0 / emb_size)
            nn.init.uniform_(self.emb.emb.weight, -bias, bias)
        self.input_dropout = nn.Dropout(input_p)
        self.output_dropout = nn.Dropout(output_p)
        self.rnn = nn.LSTM(input_size=emb_size, hidden_size=(hidden_size // 2 if self.bidirectional else hidden_size),
                           num_layers=num_layers, dropout=hidden_p, bidirectional=bidirectional,
                           batch_first=True)
        self.max_pool = MaxPoolLayer()

    def forward(self, inputs, lengths):
        """
        inputs: tensor of shape (batch_size, seq_len)
        lengths: tensor of shape (batch_size)

        returns: tensor of shape (batch_size, hidden_size)
        """
        bz, full_length = inputs.size()
        embed = self.emb(inputs)
        embed = self.input_dropout(embed)
        lstm_inputs = pack_padded_sequence(embed, lengths, batch_first=True, enforce_sorted=False)
        rnn_outputs, _ = self.rnn(lstm_inputs)
        rnn_outputs, _ = pad_packed_sequence(rnn_outputs, batch_first=True, total_length=full_length)
        rnn_outputs = self.output_dropout(rnn_outputs)
        return self.max_pool(rnn_outputs, lengths) if self.pooling else rnn_outputs


class TripleEncoder(nn.Module):
    def __init__(self, emb_dim, hidden_dim, input_p, output_p, hidden_p, num_layers, bidirectional=True, pad=False,
                 concept_emb=None, relation_emb=None
                 ):
        super().__init__()
        if pad:
            raise NotImplementedError
        self.input_p = input_p
        self.output_p = output_p
        self.hidden_p = hidden_p
        self.cpt_emb = concept_emb
        self.rel_emb = relation_emb
        self.input_dropout = nn.Dropout(input_p)
        self.output_dropout = nn.Dropout(output_p)
        self.bidirectional = bidirectional
        self.rnn = nn.GRU(input_size=emb_dim, hidden_size=(hidden_dim // 2 if self.bidirectional else hidden_dim),
                          num_layers=num_layers, dropout=hidden_p, bidirectional=bidirectional,
                          batch_first=True)

    def forward(self, inputs):
        '''
        inputs: (batch_size, seq_len)

        returns: (batch_size, h_dim(*2))
        '''
        bz, sl = inputs.size()
        h, r, t = torch.chunk(inputs, 3, dim=1)  # (bz, 1)

        h, t = self.input_dropout(self.cpt_emb(h)), self.input_dropout(self.cpt_emb(t))  # (bz, 1, dim)
        r = self.input_dropout(self.rel_emb(r))
        inputs = torch.cat((h, r, t), dim=1)  # (bz, 3, dim)
        rnn_outputs, _ = self.rnn(inputs)  # (bz, 3, dim)
        if self.bidirectional:
            outputs_f, outputs_b = torch.chunk(rnn_outputs, 2, dim=2)
            outputs = torch.cat((outputs_f[:, -1, :], outputs_b[:, 0, :]), 1)  # (bz, 2 * h_dim)
        else:
            outputs = rnn_outputs[:, -1, :]

        return self.output_dropout(outputs)


class MatrixVectorScaledDotProductAttention(nn.Module):

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, q, k, v, mask=None):
        """
        q: tensor of shape (n*b, d_k)
        k: tensor of shape (n*b, l, d_k)
        v: tensor of shape (n*b, l, d_v)

        returns: tensor of shape (n*b, d_v), tensor of shape(n*b, l)
        """
        attn = (q.unsqueeze(1) * k).sum(2)  # (n*b, l)
        attn = attn / self.temperature
        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = (attn.unsqueeze(2) * v).sum(1)
        return output, attn


class AttPoolLayer(nn.Module):

    def __init__(self, d_q, d_k, dropout=0.1):
        super().__init__()
        self.w_qs = nn.Linear(d_q, d_k)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_q + d_k)))
        self.attention = MatrixVectorScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, mask=None):
        """
        q: tensor of shape (b, d_q)
        k: tensor of shape (b, l, d_k)
        mask: tensor of shape (b, l) (optional, default None)
        returns: tensor of shape (b, d_k)
        """
        qs = self.w_qs(q)  # (b, d_k)
        output, attn = self.attention(qs, k, k, mask=mask)
        output = self.dropout(output)
        return output, attn


class MultiheadAttPoolLayer(nn.Module):

    def __init__(self, n_head, d_q_original, d_k_original, dropout=0.1):
        super().__init__()
        assert d_k_original % n_head == 0  # make sure the outpute dimension equals to d_k_origin
        self.n_head = n_head
        self.d_k = d_k_original // n_head
        self.d_v = d_k_original // n_head

        self.w_qs = nn.Linear(d_q_original, n_head * self.d_k)
        self.w_ks = nn.Linear(d_k_original, n_head * self.d_k)
        self.w_vs = nn.Linear(d_k_original, n_head * self.d_v)

        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_q_original + self.d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_k_original + self.d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_k_original + self.d_v)))

        self.attention = MatrixVectorScaledDotProductAttention(temperature=np.power(self.d_k, 0.5))
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, mask=None):
        """
        q: tensor of shape (b, d_q_original)
        k: tensor of shape (b, l, d_k_original)
        mask: tensor of shape (b, l) (optional, default None)
        returns: tensor of shape (b, n*d_v)
        """
        n_head, d_k, d_v = self.n_head, self.d_k, self.d_v

        bs, _ = q.size()
        bs, len_k, _ = k.size()

        qs = self.w_qs(q).view(bs, n_head, d_k)  # (b, n, dk)
        ks = self.w_ks(k).view(bs, len_k, n_head, d_k)  # (b, l, n, dk)
        vs = self.w_vs(k).view(bs, len_k, n_head, d_v)  # (b, l, n, dv)

        qs = qs.permute(1, 0, 2).contiguous().view(n_head * bs, d_k)
        ks = ks.permute(2, 0, 1, 3).contiguous().view(n_head * bs, len_k, d_k)
        vs = vs.permute(2, 0, 1, 3).contiguous().view(n_head * bs, len_k, d_v)

        if mask is not None:
            mask = mask.repeat(n_head, 1)
        output, attn = self.attention(qs, ks, vs, mask=mask)

        output = output.view(n_head, bs, d_v)
        output = output.permute(1, 0, 2).contiguous().view(bs, n_head * d_v)  # (b, n*dv)
        output = self.dropout(output)
        return output, attn


class TypedMultiheadAttPoolLayer(nn.Module):

    def __init__(self, n_head, d_q_original, d_k_original, dropout=0.1, n_type=1):
        super().__init__()
        assert d_k_original % n_head == 0  # make sure the outpute dimension equals to d_k_origin
        self.n_head = n_head
        self.d_k = d_k_original // n_head
        self.d_v = d_k_original // n_head

        self.w_qs = nn.Linear(d_q_original, n_head * self.d_k)
        self.w_ks = TypedLinear(d_k_original, n_head * self.d_k, n_type)
        self.w_vs = TypedLinear(d_k_original, n_head * self.d_v, n_type)

        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_q_original + self.d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_k_original + self.d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_k_original + self.d_v)))

        self.attention = MatrixVectorScaledDotProductAttention(temperature=np.power(self.d_k, 0.5))
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, mask=None, type_ids=None):
        """
        q: tensor of shape (b, d_q_original)
        k: tensor of shape (b, l, d_k_original)
        mask: bool tensor of shape (b, l) (optional, default None)
        type_ids: long tensor of shape (b, l) (optional, default None)
        returns: tensor of shape (b, n*d_v)
        """
        n_head, d_k, d_v = self.n_head, self.d_k, self.d_v

        bs, _ = q.size()
        bs, len_k, _ = k.size()

        qs = self.w_qs(q).view(bs, n_head, d_k)  # (b, n, dk)
        ks = self.w_ks(k, type_ids=type_ids).view(bs, len_k, n_head, d_k)  # (b, l, n, dk)
        vs = self.w_vs(k, type_ids=type_ids).view(bs, len_k, n_head, d_v)  # (b, l, n, dv)

        qs = qs.permute(1, 0, 2).contiguous().view(n_head * bs, d_k)
        ks = ks.permute(2, 0, 1, 3).contiguous().view(n_head * bs, len_k, d_k)
        vs = vs.permute(2, 0, 1, 3).contiguous().view(n_head * bs, len_k, d_v)

        if mask is not None:
            mask = mask.repeat(n_head, 1)
        output, attn = self.attention(qs, ks, vs, mask=mask)

        output = output.view(n_head, bs, d_v)
        output = output.permute(1, 0, 2).contiguous().view(bs, n_head * d_v)  # (b, n*dv)
        output = self.dropout(output)
        return output, attn


class BilinearAttentionLayer(nn.Module):

    def __init__(self, query_dim, value_dim):
        super().__init__()
        self.linear = nn.Linear(value_dim, query_dim, bias=False)
        self.softmax = nn.Softmax(1)

    def forward(self, query, value, node_mask=None):
        """
        query: tensor of shape (batch_size, query_dim)
        value: tensor of shape (batch_size, seq_len, value_dim)
        node_mask: tensor of shape (batch_size, seq_len)

        returns: tensor of shape (batch_size, value_dim)
        """
        attn = self.linear(value).bmm(query.unsqueeze(-1))
        attn = self.softmax(attn.squeeze(-1))
        if node_mask is not None:
            attn = attn * node_mask
            attn = attn / attn.sum(1, keepdim=True)
        pooled = attn.unsqueeze(1).bmm(value).squeeze(1)
        return pooled, attn


def masked_softmax(vector: torch.Tensor,
                   mask: torch.Tensor,
                   dim: int = -1,
                   memory_efficient: bool = True,
                   mask_fill_value: float = -1e32) -> torch.Tensor:
    """
    ``torch.nn.functional.softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular softmax.
    ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
    broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.
    If ``memory_efficient`` is set to true, we will simply use a very large negative number for those
    masked positions so that the probabilities of those positions would be approximately 0.
    This is not accurate in math, but works for most cases and consumes less memory.
    In the case that the input vector is completely masked and ``memory_efficient`` is false, this function
    returns an array of ``0.0``. This behavior may cause ``NaN`` if this is used as the last layer of
    a model that uses categorical cross-entropy loss. Instead, if ``memory_efficient`` is true, this function
    will treat every element as equal, and do softmax over equal numbers.
    """
    if mask is None:
        result = nn.functional.softmax(vector, dim=dim)
    else:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        if not memory_efficient:
            # # To limit numerical errors from large vector elements outside the mask, we zero these out.
            # result = nn.functional.softmax(vector * mask, dim=dim)
            # result = result * mask
            # result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
            raise NotImplementedError
        else:
            masked_vector = vector.masked_fill(mask.to(dtype=torch.uint8), mask_fill_value)
            result = nn.functional.softmax(masked_vector, dim=dim)
            result = result * (1 - mask)
    return result


class DiffTopK(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, k):
        """
        x: tensor of shape (batch_size, n_node)
        k: int
        returns: tensor of shape (batch_size, n_node)
        """
        bs, _ = x.size()
        _, topk_indexes = x.topk(k, 1)  # (batch_size, k)
        output = x.new_zeros(x.size())
        ri = torch.arange(bs).unsqueeze(1).expand(bs, k).contiguous().view(-1)
        output[ri, topk_indexes.view(-1)] = 1
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone(), None


class SimilarityFunction(nn.Module):
    """
    A ``SimilarityFunction`` takes a pair of tensors with the same shape, and computes a similarity
    function on the vectors in the last dimension.  For example, the tensors might both have shape
    `(batch_size, sentence_length, embedding_dim)`, and we will compute some function of the two
    vectors of length `embedding_dim` for each position `(batch_size, sentence_length)`, returning a
    tensor of shape `(batch_size, sentence_length)`.
    The similarity function could be as simple as a dot product, or it could be a more complex,
    parameterized function.
    """
    default_implementation = 'dot_product'

    def forward(self, tensor_1: torch.Tensor, tensor_2: torch.Tensor) -> torch.Tensor:
        """
        Takes two tensors of the same shape, such as ``(batch_size, length_1, length_2,
        embedding_dim)``.  Computes a (possibly parameterized) similarity on the final dimension
        and returns a tensor with one less dimension, such as ``(batch_size, length_1, length_2)``.
        """
        raise NotImplementedError


class DotProductSimilarity(SimilarityFunction):
    """
    This similarity function simply computes the dot product between each pair of vectors, with an
    optional scaling to reduce the variance of the output elements.
    Parameters
    ----------
    scale_output : ``bool``, optional
        If ``True``, we will scale the output by ``math.sqrt(tensor.size(-1))``, to reduce the
        variance in the result.
    """

    def __init__(self, scale_output: bool = False) -> None:
        super(DotProductSimilarity, self).__init__()
        self._scale_output = scale_output

    def forward(self, tensor_1: torch.Tensor, tensor_2: torch.Tensor) -> torch.Tensor:
        result = (tensor_1 * tensor_2).sum(dim=-1)
        if self._scale_output:
            result *= math.sqrt(tensor_1.size(-1))
        return result


class MatrixAttention(nn.Module):
    def __init__(self, similarity_function: SimilarityFunction = None) -> None:
        super().__init__()
        self._similarity_function = similarity_function or DotProductSimilarity()

    def forward(self, matrix_1: torch.Tensor, matrix_2: torch.Tensor) -> torch.Tensor:
        tiled_matrix_1 = matrix_1.unsqueeze(2).expand(matrix_1.size()[0],
                                                      matrix_1.size()[1],
                                                      matrix_2.size()[1],
                                                      matrix_1.size()[2])
        tiled_matrix_2 = matrix_2.unsqueeze(1).expand(matrix_2.size()[0],
                                                      matrix_1.size()[1],
                                                      matrix_2.size()[1],
                                                      matrix_2.size()[2])

        return self._similarity_function(tiled_matrix_1, tiled_matrix_2)


class CustomizedEmbedding(nn.Module):
    def __init__(self, concept_num, concept_in_dim, concept_out_dim, use_contextualized,
                 pretrained_concept_emb=None, freeze_ent_emb=True, scale=1.0, init_range=0.02):
        super().__init__()
        self.scale = scale
        self.use_contextualized = use_contextualized
        if not use_contextualized:
            self.emb = nn.Embedding(concept_num, concept_in_dim)
            if pretrained_concept_emb is not None:
                self.emb.weight.data.copy_(pretrained_concept_emb)
            else:
                self.emb.weight.data.normal_(mean=0.0, std=init_range)
            if freeze_ent_emb:
                freeze_net(self.emb)

        if concept_in_dim != concept_out_dim:
            self.cpt_transform = nn.Linear(concept_in_dim, concept_out_dim)
            self.activation = GELU()

    def forward(self, index, contextualized_emb=None):
        """
        index: size (bz, a)
        contextualized_emb: size (bz, b, emb_size) (optional)
        """
        if contextualized_emb is not None:
            assert index.size(0) == contextualized_emb.size(0)
            if hasattr(self, 'cpt_transform'):
                contextualized_emb = self.activation(self.cpt_transform(contextualized_emb * self.scale))
            else:
                contextualized_emb = contextualized_emb * self.scale
            emb_dim = contextualized_emb.size(-1)
            return contextualized_emb.gather(1, index.unsqueeze(-1).expand(-1, -1, emb_dim))
        else:
            if hasattr(self, 'cpt_transform'):
                return self.activation(self.cpt_transform(self.emb(index) * self.scale))
            else:
                return self.emb(index) * self.scale


def run_test():
    print('testing BilinearAttentionLayer...')
    att = BilinearAttentionLayer(100, 20)
    mask = (torch.randn(70, 30) > 0).float()
    mask.requires_grad_()
    v = torch.randn(70, 30, 20)
    q = torch.randn(70, 100)
    o, _ = att(q, v, mask)
    o.sum().backward()
    print(mask.grad)

    print('testing DiffTopK...')
    x = torch.randn(5, 3)
    x.requires_grad_()
    k = 2
    r = DiffTopK.apply(x, k)
    loss = (r ** 2).sum()
    loss.backward()
    assert (x.grad == r * 2).all()
    print('pass')

    a = TripleEncoder()

    triple_input = torch.tensor([[1, 2, 3], [4, 5, 6]])
    res = a(triple_input)
    print(res.size())

    b = LSTMEncoder(pooling=False)
    lstm_inputs = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    lengths = torch.tensor([3, 2])
    res = b(lstm_inputs, lengths)
    print(res.size())



class PathAttentionLayer(nn.Module):
    def __init__(self, n_type, n_head, sent_dim, att_dim, att_layer_num, dropout=0.1, ablation=[]):
        super().__init__()
        self.n_head = n_head
        self.ablation = ablation
        if 'no_att' not in self.ablation:
            if 'no_type_att' not in self.ablation:
                self.start_attention = MLP(sent_dim, att_dim, n_type, att_layer_num, dropout, layer_norm=True)
                self.end_attention = MLP(sent_dim, att_dim, n_type, att_layer_num, dropout, layer_norm=True)

            if 'no_unary' not in self.ablation and 'no_rel_att' not in self.ablation:
                self.path_uni_attention = MLP(sent_dim, att_dim, n_head, att_layer_num, dropout, layer_norm=True)

            if 'no_trans' not in self.ablation and 'no_rel_att' not in self.ablation:
                if 'ctx_trans' in self.ablation:
                    self.path_pair_attention = MLP(sent_dim, att_dim, n_head ** 2, 1, dropout, layer_norm=True)
                self.trans_scores = nn.Parameter(torch.zeros(n_head ** 2))

    def forward(self, S, node_type):
        """
        S: tensor of shape (batch_size, d_sent)
        node_type: tensor of shape (batch_size, n_node)
        returns: tensors of shapes (batch_size, n_node) (batch_size, n_node) (batch_size, n_head) (batch_size, n_head, n_head)
        """
        n_head = self.n_head
        bs, n_node = node_type.size()

        if 'detach_s_all' in self.ablation:
            S = S.detach()

        if 'no_att' not in self.ablation and 'no_type_att' not in self.ablation:
            bi = torch.arange(bs).unsqueeze(-1).expand(bs, n_node).contiguous().view(-1)  # [0 ... 0 1 ... 1 ...]
            start_attn = self.start_attention(S)
            if 'q2a_only' in self.ablation:
                start_attn[:, 1] = -np.inf
                start_attn[:, 2] = -np.inf
            start_attn = torch.exp(start_attn - start_attn.max(1, keepdim=True)[0])  # softmax trick to avoid numeric overflow
            start_attn = start_attn[bi, node_type.view(-1)].view(bs, n_node)
            end_attn = self.end_attention(S)
            if 'q2a_only' in self.ablation:
                end_attn[:, 0] = -np.inf
                end_attn[:, 2] = -np.inf
            end_attn = torch.exp(end_attn - end_attn.max(1, keepdim=True)[0])
            end_attn = end_attn[bi, node_type.view(-1)].view(bs, n_node)
        else:
            start_attn = torch.ones((bs, n_node), device=S.device)
            end_attn = torch.ones((bs, n_node), device=S.device)

        if 'no_att' not in self.ablation and 'no_unary' not in self.ablation and 'no_rel_att' not in self.ablation:
            uni_attn = self.path_uni_attention(S).view(bs, n_head)  # (bs, n_head)
            uni_attn = torch.exp(uni_attn - uni_attn.max(1, keepdim=True)[0]).view(bs, n_head)
        else:
            uni_attn = torch.ones((bs, n_head), device=S.device)

        if 'no_att' not in self.ablation and 'no_trans' not in self.ablation and 'no_rel_att' not in self.ablation:
            if 'ctx_trans' in self.ablation:
                trans_attn = self.path_pair_attention(S) + self.trans_scores
            else:
                trans_attn = self.trans_scores.unsqueeze(0).expand(bs, n_head ** 2)
            trans_attn = torch.exp(trans_attn - trans_attn.max(1, keepdim=True)[0])
            trans_attn = trans_attn.view(bs, n_head, n_head)
        else:
            trans_attn = torch.ones((bs, n_head, n_head), device=S.device)
        return start_attn, end_attn, uni_attn, trans_attn


class MultiHopMessagePassingLayer(nn.Module):
    def __init__(self, k, n_head, hidden_size, diag_decompose, n_basis, eps=1e-20, init_range=0.01, ablation=[]):
        super().__init__()
        self.diag_decompose = diag_decompose
        self.k = k
        self.n_head = n_head
        self.hidden_size = hidden_size
        self.n_basis = n_basis
        self.eps = eps
        self.ablation = ablation

        if diag_decompose and n_basis > 0:
            raise ValueError('diag_decompose and n_basis > 0 cannot be True at the same time')

        if diag_decompose:
            self.w_vs = nn.Parameter(torch.zeros(k, hidden_size, n_head + 1))  # the additional head is used for the self-loop
            self.w_vs.data.uniform_(-init_range, init_range)
        elif n_basis == 0:
            self.w_vs = nn.Parameter(torch.zeros(k, n_head + 1, hidden_size, hidden_size))
            self.w_vs.data.uniform_(-init_range, init_range)
        else:
            self.w_vs = nn.Parameter(torch.zeros(k, n_basis, hidden_size * hidden_size))
            self.w_vs.data.uniform_(-init_range, init_range)
            self.w_vs_co = nn.Parameter(torch.zeros(k, n_head + 1, n_basis))
            self.w_vs_co.data.uniform_(-init_range, init_range)

    def init_from_old(self, w_vs, w_vs_co=None):
        """
        w_vs: tensor of shape (k, h_size, n_head) or (k, h_size, h_size * n_head) or (k, h_size*h_size, n_basis)
        w_vs_co: tensor of shape (k, n_basis, n_head)
        """
        raise NotImplementedError()
        k, n_head, h_size = self.k, self.n_head, self.hidden_size
        if self.diag_decompose:
            self.w_vs.data.copy_(w_vs)
        elif self.n_basis == 0:
            self.w_vs.data.copy_(w_vs.view(k, h_size, h_size, n_head).permute(0, 3, 1, 2))
        else:
            self.w_vs.data.copy_(w_vs.permute(0, 2, 1))
            self.w_vs_co.data.copy_(w_vs_co.permute(0, 2, 1))

    def _get_weights(self):
        if self.diag_decompose:
            W, Wi = self.w_vs[:, :, :-1], self.w_vs[:, :, -1]
        elif self.n_basis == 0:
            W, Wi = self.w_vs[:, :-1, :, :], self.w_vs[:, -1, :, :]
        else:
            W = self.w_vs_co.bmm(self.w_vs).view(self.k, self.n_head, self.hidden_size, self.hidden_size)
            W, Wi = W[:, :-1, :, :], W[:, -1, :, :]

        k, h_size = self.k, self.hidden_size
        W_pad = [W.new_ones((h_size,)) if self.diag_decompose else torch.eye(h_size, device=W.device)]
        for t in range(k - 1):
            if self.diag_decompose:
                W_pad = [Wi[k - 1 - t] * W_pad[0]] + W_pad
            else:
                W_pad = [Wi[k - 1 - t].mm(W_pad[0])] + W_pad
        assert len(W_pad) == k
        return W, W_pad

    def decode(self, end_ids, ks, A, start_attn, uni_attn, trans_attn):
        """
        end_ids: tensor of shape (batch_size,)
        ks: tensor of shape (batch_size,)
        A: tensor of shape (batch_size, n_head, n_node, n_node)
        start_attn: tensor of shape (batch_size, n_node)
        end_attn: tensor of shape (batch_size, n_node)
        uni_attn: tensor of shape (batch_size, n_head)
        trans_attn: tensor of shape (batch_size, n_head, n_head)
        returns: list[tensor of shape (path_len,)]
        """
        bs, n_head, n_node, n_node = A.size()
        assert ((A == 0) | (A == 1)).all()

        path_ids = end_ids.new_zeros((bs, self.k * 2 + 1))
        path_lengths = end_ids.new_zeros((bs,))

        for idx in range(bs):
            back_trace = []
            end_id, k, adj = end_ids[idx], ks[idx], A[idx]
            uni_a, trans_a, start_a = uni_attn[idx], trans_attn[idx], start_attn[idx]

            if (adj[:, end_id, :] == 0).all():  # end_id is not connected to any other node
                path_ids[idx, 0] = end_id
                path_lengths[idx] = 1
                continue

            dp = F.one_hot(end_id, num_classes=n_node).float()  # (n_node,)
            assert 1 <= k <= self.k
            for t in range(k):
                if t == 0:
                    dp = dp.unsqueeze(0).expand(n_head, n_node)
                else:
                    dp = dp.unsqueeze(0) * trans_a.unsqueeze(-1)  # (n_head, n_head, n_node)
                    dp, ptr = dp.max(1)
                    back_trace.append(ptr)  # (n_head, n_node)
                dp = dp.unsqueeze(-1) * adj  # (n_head, n_node, n_node)
                dp, ptr = dp.max(1)
                back_trace.append(ptr)  # (n_head, n_node)
                dp = dp * uni_a.unsqueeze(-1)  # (n_head, n_node)
            dp, ptr = dp.max(0)
            back_trace.append(ptr)  # (n_node,)
            dp = dp * start_a
            dp, ptr = dp.max(0)
            back_trace.append(ptr)  # （)
            assert dp.dim() == 0
            assert len(back_trace) == k + (k - 1) + 2

            # re-construct path from back_trace
            path = end_id.new_zeros((2 * k + 1,))  # (k + 1) entities and k relations
            path[0] = back_trace.pop(-1)
            path[1] = back_trace.pop(-1)[path[0]]
            for p in range(2, 2 * k + 1):
                if p % 2 == 0:  # need to fill a entity id
                    path[p] = back_trace.pop(-1)[path[p - 1], path[p - 2]]
                else:  # need to fill a relation id
                    path[p] = back_trace.pop(-1)[path[p - 2], path[p - 1]]
            assert len(back_trace) == 0
            assert path[-1] == end_id
            path_ids[idx, :2 * k + 1] = path
            path_lengths[idx] = 2 * k + 1

        return path_ids, path_lengths

    def forward(self, X, A, start_attn, end_attn, uni_attn, trans_attn):
        """
        X: tensor of shape (batch_size, n_node, h_size)
        A: tensor of shape (batch_size, n_head, n_node, n_node)
        start_attn: tensor of shape (batch_size, n_node)
        end_attn: tensor of shape (batch_size, n_node)
        uni_attn: tensor of shape (batch_size, n_head)
        trans_attn: tensor of shape (batch_size, n_head, n_head)
        """
        k, n_head = self.k, self.n_head
        bs, n_node, h_size = X.size()

        W, W_pad = self._get_weights()  # (k, h_size, n_head) or (k, n_head, h_size h_size)

        A = A.view(bs * n_head, n_node, n_node)
        uni_attn = uni_attn.view(bs * n_head)

        Z_all = []
        Z = X * start_attn.unsqueeze(2)  # (bs, n_node, h_size)
        for t in range(k):
            if t == 0:  # Z.size() == (bs, n_node, h_size)
                Z = Z.unsqueeze(-1).expand(bs, n_node, h_size, n_head)
            else:  # Z.size() == (bs, n_head, n_node, h_size)
                Z = Z.permute(0, 2, 3, 1).view(bs, n_node * h_size, n_head)
                Z = Z.bmm(trans_attn).view(bs, n_node, h_size, n_head)
            if self.diag_decompose:
                Z = Z * W[t]  # (bs, n_node, h_size, n_head)
                Z = Z.permute(0, 3, 1, 2).contiguous().view(bs * n_head, n_node, h_size)
            else:
                Z = Z.permute(3, 0, 1, 2).view(n_head, bs * n_node, h_size)
                Z = Z.bmm(W[t]).view(n_head, bs, n_node, h_size)
                Z = Z.permute(1, 0, 2, 3).contiguous().view(bs * n_head, n_node, h_size)
            Z = Z * uni_attn[:, None, None]
            Z = A.bmm(Z)
            Z = Z.view(bs, n_head, n_node, h_size)
            Zt = Z.sum(1) * W_pad[t] if self.diag_decompose else Z.sum(1).matmul(W_pad[t])
            Zt = Zt * end_attn.unsqueeze(2)
            Z_all.append(Zt)

        # compute the normalization factor
        D_all = []
        D = start_attn
        for t in range(k):
            if t == 0:  # D.size() == (bs, n_node)
                D = D.unsqueeze(1).expand(bs, n_head, n_node)
            else:  # D.size() == (bs, n_head, n_node)
                D = D.permute(0, 2, 1).bmm(trans_attn)  # (bs, n_node, n_head)
                D = D.permute(0, 2, 1)
            D = D.contiguous().view(bs * n_head, n_node, 1)
            D = D * uni_attn[:, None, None]
            D = A.bmm(D)
            D = D.view(bs, n_head, n_node)
            Dt = D.sum(1) * end_attn
            D_all.append(Dt)

        Z_all = [Z / (D.unsqueeze(2) + self.eps) for Z, D in zip(Z_all, D_all)]
        assert len(Z_all) == k
        if 'agg_self_loop' in self.ablation:
            Z_all = [X] + Z_all
        return Z_all


class Aggregator(nn.Module):

    def __init__(self, sent_dim, hidden_size, ablation=[]):
        super().__init__()
        self.ablation = ablation
        self.w_qs = nn.Linear(sent_dim, hidden_size)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (sent_dim + hidden_size)))
        self.temperature = np.power(hidden_size, 0.5)
        self.softmax = nn.Softmax(2)

    def forward(self, S, Z_all):
        """
        S: tensor of shape (batch_size, d_sent)
        Z_all: tensor of shape (batch_size, n_node, k, d_node)
        returns: tensor of shape (batch_size, n_node, d_node)
        """
        if 'detach_s_agg' in self.ablation or 'detach_s_all' in self.ablation:
            S = S.detach()
        S = self.w_qs(S)  # (bs, d_node)
        attn = (S[:, None, None, :] * Z_all).sum(-1)  # (bs, n_node, k)
        if 'no_1hop' in self.ablation:
            if 'agg_self_loop' in self.ablation:
                attn[:, :, 1] = -np.inf
            else:
                attn[:, :, 0] = -np.inf

        attn = self.softmax(attn / self.temperature)
        Z = (attn.unsqueeze(-1) * Z_all).sum(2)
        return Z, attn



class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class CrossAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_k = nn.Linear(dim, inner_dim , bias=False)
        self.to_v = nn.Linear(dim, inner_dim , bias = False)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x_qkv):
        b, n, _, h = *x_qkv.shape, self.heads

        k = self.to_k(x_qkv)
        k = rearrange(k, 'b n (h d) -> b h n d', h = h)

        v = self.to_v(x_qkv)
        v = rearrange(v, 'b n (h d) -> b h n d', h = h)

        q = self.to_q(x_qkv[:, 0].unsqueeze(1))
        q = rearrange(q, 'b n (h d) -> b h n d', h = h)



        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out
        

class CrossAttentionEncoder(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, context_dim = 256, cross_attn_depth=1, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        small_dim = dim
        large_dim = context_dim
        cross_attn_heads= heads
        small_dim_head = dim_head
        large_dim_head = dim_head

        self.cross_attn_layers = nn.ModuleList([])
        for _ in range(cross_attn_depth):
            self.cross_attn_layers.append(nn.ModuleList([
                nn.Linear(small_dim, large_dim),
                nn.Linear(large_dim, small_dim),
                PreNorm(large_dim, CrossAttention(large_dim, heads = cross_attn_heads, dim_head = large_dim_head, dropout = dropout)),
                nn.Linear(large_dim, small_dim),
                nn.Linear(small_dim, large_dim),
                PreNorm(small_dim, CrossAttention(small_dim, heads = cross_attn_heads, dim_head = small_dim_head, dropout = dropout)),
            ]))

    def forward(self, xs,  xl):
        for f_sl, g_ls, cross_attn_s, f_ls, g_sl, cross_attn_l in self.cross_attn_layers:
            small_class = xs[:, 0]
            x_small = xs[:, 1:]
            large_class = xl[:, 0]
            x_large = xl[:, 1:]

            # Cross Attn for Large Patch

            cal_q = f_ls(large_class.unsqueeze(1))
            cal_qkv = torch.cat((cal_q, x_small), dim=1)
            cal_out = cal_q + cross_attn_l(cal_qkv)
            cal_out = g_sl(cal_out)
            xl = torch.cat((cal_out, x_large), dim=1)

            # Cross Attn for Smaller Patch
            cal_q = f_sl(small_class.unsqueeze(1))
            cal_qkv = torch.cat((cal_q, x_large), dim=1)
            cal_out = cal_q + cross_attn_s(cal_qkv)
            cal_out = g_ls(cal_out)
            xs = torch.cat((cal_out, x_small), dim=1)
        return xs, xl
            


import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F

class SAGE(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        self.conv1 = dglnn.SAGEConv(
            in_feats=in_feats, out_feats=hid_feats, aggregator_type='gcn')
        self.conv2 = dglnn.SAGEConv(
            in_feats=hid_feats, out_feats=out_feats, aggregator_type='gcn')

    def forward(self, graph, inputs):
        # inputs are features of nodes
        h = self.conv1(graph, inputs)
        h = F.relu(h)
        h = self.conv2(graph, h)
        return h

class GCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super(GCN, self).__init__()
        self.conv1 = dglnn.GraphConv(in_feats, hid_feats, allow_zero_in_degree=True)
        self.conv2 = dglnn.GraphConv(hid_feats, out_feats, allow_zero_in_degree=True)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h
        
from dgl.nn.pytorch.conv import GATConv
# from .layers import GatedGraphConv, EdgeConv
class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, heads=4, dropout=0.3):
        super(GAT, self).__init__()

        self.layers = nn.ModuleList([
            GATConv(in_dim, hidden_dim, heads, residual=True, activation=F.leaky_relu),
            GATConv(heads*hidden_dim, hidden_dim, heads, feat_drop=dropout, residual=True, activation=F.leaky_relu),])

        self.bn = nn.ModuleList([
            nn.BatchNorm1d(heads*hidden_dim),
            nn.BatchNorm1d(heads*hidden_dim),])

        self.last_layer = GATConv(heads*hidden_dim, out_dim, heads, residual=True)

    def forward(self, g, in_feat):
        h = in_feat

        for i, conv in enumerate(self.layers):
            h = conv(g, h)
            h = h.view(h.shape[0], -1)
            h = self.bn[i](h)

        h = self.last_layer(g, h)
        h = h.mean(1)

        # g.ndata['h'] = h
        return h


# class GNN_GRU(nn.Module):
#     def __init__(self, in_dim, hidden_dim, out_dim, heads=4, dropout=0.3):
#         super(GNN_GRU, self).__init__()

#         self.embedding = nn.Linear(in_dim, out_dim)
#         self.edge_embedding = EdgeConv(hidden_dim, hidden_dim, activation=torch.relu)
#         self.edge_func = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(True), nn.Linear(64, hidden_dim*hidden_dim))
# #        self.layers = GatedGraphConv(hidden_dim, hidden_dim, 3, self.edge_func, dropout=dropout)
#         self.layers = GatedGraphConv(hidden_dim, out_dim, 3, self.edge_func, 1, dropout=dropout)
# #        self.last_layer = nn.Linear(hidden_dim, out_dim)
# #        self.last_edge_layer = EdgeConv(hidden_dim, out_dim, activation=None)

#     def forward(self, g):
#         h = g.ndata['pos']

#         h = self.embedding(h)
#         he = self.edge_embedding(g, h)
#         h = self.layers(g, h, he)
# #        h = self.layers(g, h, torch.zeros(g.edges()[0].shape[0]))
# #        he = self.last_edge_layer(g, h)
# #        h = self.last_layer(h)

#         g.ndata['h'] = h
# #        g.edata['h'] = he

#         return g

