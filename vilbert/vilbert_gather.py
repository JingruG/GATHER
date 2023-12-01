# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# import tensorflow as tf
import copy
from filecmp import BUFSIZE
import json, pickle
from locale import normalize
import logging
import math
import os
import shutil
import tarfile
import tempfile
import sys
from io import open
import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
from .utils import PreTrainedModel
from .layers import *
import pdb
from torch.nn.utils.weight_norm import weight_norm
import time

from transformers import AutoTokenizer, AutoModel, AutoModelForQuestionAnswering
from transformers import LxmertModel, LxmertPreTrainedModel
# from transformers import pipeline

from collections import OrderedDict


import networkx as nx
import pandas as pd
import numpy as np
import itertools
import os
import random
import matplotlib.pyplot as plt
import dgl
# from dgl.nn import APPNPConv, SGConv
import networkx as nx
from itertools import groupby

# from ppnp.pytorch import PPNP
# from ppnp.pytorch.training import train_model
# from ppnp.pytorch.earlystopping import stopping_args
# from ppnp.pytorch.propagation import PPRExact, PPRPowerIteration
# from .ppnp.ppnp.data.io import load_dataset, networkx_to_sparsegraph

# from .SSGC.utils import sgc_precompute, load_citation, set_seed
# from .SSGC.models import SGC
from .my_pagerank import pagerank as PageRank
from .datasets.conceptnet import id2concept, relation_text
# from .paths import find_paths
# from bidirectional_cross_attention import BidirectionalCrossAttention
from .modeling.modeling_rgcn import RGCN
from .modeling.modeling_qagnn import QAGNN_Message_Passing

from .task_utils import TripletLoss
from .gpt2 import load_generator

from .MAGNA.MAGNAConv import MAGNALayer
from .random_walks import RandomWalk


logger = logging.getLogger(__name__)

BERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "bert-base-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin",
    "bert-large-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-pytorch_model.bin",
    "bert-base-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-pytorch_model.bin",
    "bert-large-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-pytorch_model.bin",
    "bert-base-multilingual-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-pytorch_model.bin",
    "bert-base-multilingual-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-pytorch_model.bin",
    "bert-base-chinese": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-pytorch_model.bin",
    "bert-base-german-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-cased-pytorch_model.bin",
    "bert-large-uncased-whole-word-masking": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-pytorch_model.bin",
    "bert-large-cased-whole-word-masking": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-pytorch_model.bin",
    "bert-large-uncased-whole-word-masking-finetuned-squad": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-pytorch_model.bin",
    "bert-large-cased-whole-word-masking-finetuned-squad": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-finetuned-squad-pytorch_model.bin",
    "bert-base-cased-finetuned-mrpc": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-finetuned-mrpc-pytorch_model.bin",
    "roberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-pytorch_model.bin",
    "roberta-large": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-pytorch_model.bin",
    "roberta-large-mnli": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-pytorch_model.bin",
}

# class QASystemWithBERT:
#     def __init__(self, pretrained_model_name_or_path='/data/gjr/huggingface/roberta-base-squad2'):
#         self.READER_PATH = pretrained_model_name_or_path
#         self.tokenizer = AutoTokenizer.from_pretrained(self.READER_PATH
#                                                       )
#         self.model = AutoModelForQuestionAnswering.from_pretrained(self.READER_PATH
#                                                                   )
#         self.max_len = self.model.config.max_position_embeddings
#         self.chunked = False

#     def tokenize(self, question, text):
#         self.inputs = self.tokenizer.encode_plus(question, text, add_special_tokens=True, return_tensors="pt", return_token_type_ids=True)
#         self.input_ids = self.inputs["input_ids"].tolist()[0]

#         if len(self.input_ids) > self.max_len:
#             self.inputs = self.chunkify()
#             self.chunked = True

#     def chunkify(self):
#         """ 
#         Break up a long article into chunks that fit within the max token
#         requirement for that Transformer model. 
#         """
#         qmask = self.inputs['token_type_ids'].lt(1)
        
#         qt = torch.masked_select(self.inputs['input_ids'], qmask)
#         chunk_size = self.max_len - qt.size()[0] - 1 
        
#         chunked_input = OrderedDict()
#         for k,v in self.inputs.items():
#             q = torch.masked_select(v, qmask)
#             c = torch.masked_select(v, ~qmask)
#             chunks = torch.split(c, chunk_size)
            
#             for i, chunk in enumerate(chunks):
#                 if i not in chunked_input:
#                     chunked_input[i] = {}

#                 thing = torch.cat((q, chunk))
#                 if i != len(chunks)-1:
#                     if k == 'input_ids':
#                         thing = torch.cat((thing, torch.tensor([102])))
#                     else:
#                         thing = torch.cat((thing, torch.tensor([1])))

#                 chunked_input[i][k] = torch.unsqueeze(thing, dim=0)
#         return chunked_input

#     def get_answer(self):
#         if self.chunked:
#             answer = ''
#             for k, chunk in self.inputs.items():
#                 answer_start_scores, answer_end_scores = self.model(**chunk)[:2]

#                 answer_start = torch.argmax(answer_start_scores)
#                 answer_end = torch.argmax(answer_end_scores) + 1

#                 ans = self.convert_ids_to_string(chunk['input_ids'][0][answer_start:answer_end])
#                 if ans != '[CLS]':
#                     answer += ans + " / "
#             return answer
#         else:
#             answer_start_scores, answer_end_scores = self.model(**self.inputs)[:2]

#             answer_start = torch.argmax(answer_start_scores)  
#             answer_end = torch.argmax(answer_end_scores) + 1  
        
#             return self.convert_ids_to_string(self.inputs['input_ids'][0][
#                                               answer_start:answer_end])

#     def convert_ids_to_string(self, input_ids):
#         return self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(input_ids))


# class FC(nn.Module):
#     def __init__(self, in_size, out_size, dropout_r=0., use_relu=True):
#         super(FC, self).__init__()
#         self.dropout_r = dropout_r
#         self.use_relu = use_relu

#         self.linear = nn.Linear(in_size, out_size)

#         if use_relu:
#             self.relu = GeLU()

#         if dropout_r > 0:
#             self.dropout = nn.Dropout(dropout_r)

#     def forward(self, x):
#         x = self.linear(x)

#         if self.use_relu:
#             x = self.relu(x)

#         if self.dropout_r > 0:
#             x = self.dropout(x)

#         return x


class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps

        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)


        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class FCNet(nn.Module):
    """Simple class for non-linear fully connect network
    """

    def __init__(self, dims, dropout):
        super(FCNet, self).__init__()
        layers = []
        for i in range(len(dims) - 2):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(LayerNorm(out_dim))
            layers.append(GeLU())
            layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        if dims[-1] != 1:
            layers.append(LayerNorm(dims[-1]))
        layers.append(GeLU())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


# class MLP_(nn.Module):
#     def __init__(self, in_size, mid_size, out_size, dropout_r=0., use_relu=True):
#         super(MLP_, self).__init__()

#         self.fc = FC(in_size, mid_size, dropout_r=dropout_r, use_relu=use_relu)
#         self.linear = nn.Linear(mid_size, out_size)

#     def forward(self, x):
#         return self.linear(self.fc(x))


# class MHAtt(nn.Module):
#     def __init__(self, num_head, num_hid, dropout):
#         super(MHAtt, self).__init__()
#         self.num_hid = num_hid
#         self.num_head = num_head
#         self.linear_v = nn.Linear(num_hid, num_hid)
#         self.linear_k = nn.Linear(num_hid, num_hid)
#         self.linear_q = nn.Linear(num_hid, num_hid)
#         self.linear_merge = nn.Linear(num_hid, num_hid)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, v, k, q, mask=None):
#         n_batches = q.size(0)

#         v = self.linear_v(v).view(
#             n_batches,
#             -1,
#             self.num_head,
#             int(self.num_hid // self.num_head)
#         ).transpose(1, 2)  # batch, n_items, num_head, num_feats --> batch, num_head, n_items, num_feats

#         k = self.linear_k(k).view(
#             n_batches,
#             -1,
#             self.num_head,
#             int(self.num_hid // self.num_head)
#         ).transpose(1, 2)  # batch, n_items, num_head, num_feats --> batch, num_head, n_items, num_feats

#         q = self.linear_q(q).view(
#             n_batches,
#             -1,
#             self.num_head,
#             int(self.num_hid // self.num_head)
#         ).transpose(1, 2)  # batch, n_items, num_head, num_feats --> batch, num_head, n_items, num_feats
#         atted, att_map = self.att(v, k, q,
#                                   mask)  # batch, num_head, q_items, num_feats |||batch, num_head, q_items, k_items
#         atted = atted.transpose(1, 2).contiguous().view(
#             n_batches,
#             -1,
#             self.num_hid
#         )  # batch, q_items, num_head * num_feats
#         atted = self.linear_merge(atted)

#         return atted, att_map

#     def att(self, value, key, query, mask=None):
#         d_k = query.size(-1)

#         scores = torch.matmul(
#             query, key.transpose(-2, -1)
#             # batch, num_head, n_items, num_feats ** batch, num_head, num_feats, n_items,
#         ) / math.sqrt(d_k)  # batch, num_head, n_items, n_items

#         if mask is not None:
#             scores = scores.masked_fill(mask, -1e9)

#         att_map = F.softmax(scores, dim=-1)
#         if query.size(1) > 1:
#             att_map = self.dropout(att_map)

#         return torch.matmul(att_map, value), att_map


# # ---------------------------
# # ---- Feed Forward Nets ----
# # ---------------------------

# class FFN(nn.Module):
#     def __init__(self, num_hid, drop_out):
#         super(FFN, self).__init__()

#         self.mlp = MLP_(
#             in_size=num_hid,
#             mid_size=num_hid,
#             out_size=num_hid,
#             dropout_r=drop_out,
#             use_relu=True
#         )

#     def forward(self, x):
#         return self.mlp(x)


# # -------------------------------
# # ---- Self Guided Attention ----
# # -------------------------------


# class SGASimple(nn.Module):
#     def __init__(self, num_head, num_hid, num_out, dropout, input_dim1=None, input_dim2=None):
#         super(SGASimple, self).__init__()
#         self.input1 = None
#         if input_dim1 is not None:
#             self.input1 = FC(input_dim1, num_hid, dropout_r=dropout, use_relu=True)
#         self.input2 = None
#         if input_dim2 is not None:
#             self.input2 = FC(input_dim2, num_hid, dropout_r=dropout, use_relu=True)

#         self.mhatt2 = MHAtt(num_head, num_hid, dropout)
#         self.ffn = FFN(num_hid, dropout)
#         self.out = FC(num_hid, num_out)

#         self.dropout2 = nn.Dropout(dropout)
#         self.norm2 = LayerNorm(num_hid)

#         self.dropout3 = nn.Dropout(dropout)
#         self.norm3 = LayerNorm(num_hid)

#     def forward(self, x, y, x_mask=None, y_mask=None):
#         if self.input1 is not None:
#             x = self.input1(x)
#         if self.input2 is not None:
#             y = self.input2(y)
#         mhatt2, att_map = self.mhatt2(y, y, x, y_mask)
#         x = self.norm2(x + self.dropout2(mhatt2))

#         x = self.norm3(x + self.dropout3(
#             self.ffn(x)
#         ))
#         x = self.out(x)
#         return x


def load_tf_weights_in_bert(model, tf_checkpoint_path):
    """ Load tf checkpoints in a pytorch model
    """
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        print(
            "Loading a TensorFlow models in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    print("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        print("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split("/")
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(n in ["adam_v", "adam_m"] for n in name):
            print("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                l = re.split(r"_(\d+)", m_name)
            else:
                l = [m_name]
            if l[0] == "kernel" or l[0] == "gamma":
                pointer = getattr(pointer, "weight")
            elif l[0] == "output_bias" or l[0] == "beta":
                pointer = getattr(pointer, "bias")
            elif l[0] == "output_weights":
                pointer = getattr(pointer, "weight")
            else:
                pointer = getattr(pointer, l[0])
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        elif m_name == "kernel":
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        print("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)
    return model


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class GeLU(nn.Module):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return gelu(x)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """

    def __init__(
            self,
            vocab_size_or_config_json_file,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=512,
            type_vocab_size=2,
            initializer_range=0.02,
            v_feature_size=2048,
            v_target_size=1601,
            v_hidden_size=768,
            v_num_hidden_layers=3,
            v_num_attention_heads=12,
            v_intermediate_size=3072,
            bi_hidden_size=1024,
            bi_num_attention_heads=16,
            v_attention_probs_dropout_prob=0.1,
            v_hidden_act="gelu",
            v_hidden_dropout_prob=0.1,
            v_initializer_range=0.2,
            v_biattention_id=[0, 1],
            t_biattention_id=[10, 11],
            visual_target=0,
            fast_mode=False,
            fixed_v_layer=0,
            fixed_t_layer=0,
            in_batch_pairs=False,
            fusion_method="mul",
            dynamic_attention=False,
            with_coattention=True,
            objective=0,
            num_negative=128,
            model="bert",
            task_specific_tokens=False,
            visualization=False,
    ):

        """Constructs BertConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        assert len(v_biattention_id) == len(t_biattention_id)
        assert max(v_biattention_id) < v_num_hidden_layers
        assert max(t_biattention_id) < num_hidden_layers

        if isinstance(vocab_size_or_config_json_file, str) or (
                sys.version_info[0] == 2
                and isinstance(vocab_size_or_config_json_file, unicode)
        ):
            with open(vocab_size_or_config_json_file, "r", encoding="utf-8") as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
            self.v_feature_size = v_feature_size
            self.v_hidden_size = v_hidden_size
            self.v_num_hidden_layers = v_num_hidden_layers
            self.v_num_attention_heads = v_num_attention_heads
            self.v_intermediate_size = v_intermediate_size
            self.v_attention_probs_dropout_prob = v_attention_probs_dropout_prob
            self.v_hidden_act = v_hidden_act
            self.v_hidden_dropout_prob = v_hidden_dropout_prob
            self.v_initializer_range = v_initializer_range
            self.v_biattention_id = v_biattention_id
            self.t_biattention_id = t_biattention_id
            self.v_target_size = v_target_size
            self.bi_hidden_size = bi_hidden_size
            self.bi_num_attention_heads = bi_num_attention_heads
            self.visual_target = visual_target
            self.fast_mode = fast_mode
            self.fixed_v_layer = fixed_v_layer
            self.fixed_t_layer = fixed_t_layer

            self.model = model
            self.in_batch_pairs = in_batch_pairs
            self.fusion_method = fusion_method
            self.dynamic_attention = dynamic_attention
            self.with_coattention = with_coattention
            self.objective = objective
            self.num_negative = num_negative
            self.task_specific_tokens = task_specific_tokens
            self.visualization = visualization
        else:
            raise ValueError(
                "First argument must be either a vocabulary size (int)"
                "or the path to a pretrained model config file (str)"
            )

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
except ImportError:
    logger.info(
        "Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex ."
    )


    class BertLayerNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-12):
            """Construct a layernorm module in the TF style (epsilon inside the square root).
            """
            super(BertLayerNorm, self).__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

        def forward(self, x):
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()

        self.task_specific_tokens = config.task_specific_tokens
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=0
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if self.task_specific_tokens:
            self.task_embeddings = nn.Embedding(20, config.hidden_size)

    def forward(self, input_ids, token_type_ids=None, task_ids=None, position_ids=None):

        seq_length = input_ids.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_ids.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = words_embeddings + position_embeddings + token_type_embeddings

        if self.task_specific_tokens:
            task_embeddings = self.task_embeddings(task_ids)
            embeddings = torch.cat(
                [embeddings[:, 0:1], task_embeddings, embeddings[:, 1:]], dim=1
            )

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class RobertaEmbeddings(BertEmbeddings):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    def __init__(self, config):
        super(RobertaEmbeddings, self).__init__(config)
        self.padding_idx = 1

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            # Position numbers begin at padding_idx+1. Padding symbols are ignored.
            # cf. fairseq's `utils.make_positions`
            position_ids = torch.arange(
                self.padding_idx + 1,
                seq_length + self.padding_idx + 1,
                dtype=torch.long,
                device=input_ids.device,
            )
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        return super(RobertaEmbeddings, self).forward(
            input_ids, token_type_ids=token_type_ids, position_ids=position_ids
        )


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.visualization = config.visualization

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        if self.visualization:
            attn_data = {
                "attn": attention_probs,
                "queries": query_layer,
                "keys": key_layer,
            }
        else:
            attn_data = None

        return context_layer, attn_data


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output, attention_probs = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output, attention_probs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str) or (
                sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)
        ):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output, attention_probs = self.attention(
            hidden_states, attention_mask
        )
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, attention_probs


class BertImageSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertImageSelfAttention, self).__init__()
        if config.v_hidden_size % config.v_num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.v_hidden_size, config.v_num_attention_heads)
            )
        self.dynamic_attention = config.dynamic_attention
        self.num_attention_heads = config.v_num_attention_heads
        self.attention_head_size = int(
            config.v_hidden_size / config.v_num_attention_heads
        )

        self.visualization = config.visualization

        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.v_hidden_size, self.all_head_size)
        self.key = nn.Linear(config.v_hidden_size, self.all_head_size)
        self.value = nn.Linear(config.v_hidden_size, self.all_head_size)

        if self.dynamic_attention:
            self.dyLinear_q = nn.Linear(config.hidden_size, self.all_head_size)
            self.dyLinear_k = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.v_attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, txt_embedding, txt_attention_mask):

        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        if self.dynamic_attention:
            pool_embedding = (txt_embedding * txt_attention_mask).sum(1)
            pool_embedding = pool_embedding / txt_attention_mask.sum(1)

            # given pool embedding, Linear and Sigmoid layer.
            gate_q = 1 + torch.sigmoid(self.dyLinear_q(pool_embedding))
            gate_k = 1 + torch.sigmoid(self.dyLinear_k(pool_embedding))

            mixed_query_layer = mixed_query_layer * gate_q.unsqueeze(1)
            mixed_key_layer = mixed_key_layer * gate_k.unsqueeze(1)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        if self.visualization:
            attn_data = {
                "attn": attention_probs,
                "queries": query_layer,
                "keys": key_layer,
            }
        else:
            attn_data = None

        return context_layer, attn_data


class BertImageSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertImageSelfOutput, self).__init__()
        self.dense = nn.Linear(config.v_hidden_size, config.v_hidden_size)
        self.LayerNorm = BertLayerNorm(config.v_hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.v_hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertImageAttention(nn.Module):
    def __init__(self, config):
        super(BertImageAttention, self).__init__()
        self.self = BertImageSelfAttention(config)
        self.output = BertImageSelfOutput(config)

    def forward(self, input_tensor, attention_mask, txt_embedding, txt_attention_mask):
        self_output, attention_probs = self.self(
            input_tensor, attention_mask, txt_embedding, txt_attention_mask
        )
        attention_output = self.output(self_output, input_tensor)
        return attention_output, attention_probs


class BertImageIntermediate(nn.Module):
    def __init__(self, config):
        super(BertImageIntermediate, self).__init__()
        self.dense = nn.Linear(config.v_hidden_size, config.v_intermediate_size)
        if isinstance(config.v_hidden_act, str) or (
                sys.version_info[0] == 2 and isinstance(config.v_hidden_act, unicode)
        ):
            self.intermediate_act_fn = ACT2FN[config.v_hidden_act]
        else:
            self.intermediate_act_fn = config.v_hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertImageOutput(nn.Module):
    def __init__(self, config):
        super(BertImageOutput, self).__init__()
        self.dense = nn.Linear(config.v_intermediate_size, config.v_hidden_size)
        self.LayerNorm = BertLayerNorm(config.v_hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.v_hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertImageLayer(nn.Module):
    def __init__(self, config):
        super(BertImageLayer, self).__init__()
        self.attention = BertImageAttention(config)
        self.intermediate = BertImageIntermediate(config)
        self.output = BertImageOutput(config)

    def forward(self, hidden_states, attention_mask, txt_embedding, txt_attention_mask):
        attention_output, attention_probs = self.attention(
            hidden_states, attention_mask, txt_embedding, txt_attention_mask
        )
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, attention_probs


class BertBiAttention(nn.Module):
    def __init__(self, config):
        super(BertBiAttention, self).__init__()
        if config.bi_hidden_size % config.bi_num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.bi_hidden_size, config.bi_num_attention_heads)
            )

        self.visualization = config.visualization
        self.num_attention_heads = config.bi_num_attention_heads
        self.attention_head_size = int(
            config.bi_hidden_size / config.bi_num_attention_heads
        )
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # self.scale = nn.Linear(1, self.num_attention_heads, bias=False)
        # self.scale_act_fn = ACT2FN['relu']

        self.query1 = nn.Linear(config.v_hidden_size, self.all_head_size)
        self.key1 = nn.Linear(config.v_hidden_size, self.all_head_size)
        self.value1 = nn.Linear(config.v_hidden_size, self.all_head_size)
        # self.logit1 = nn.Linear(config.hidden_size, self.num_attention_heads)

        self.dropout1 = nn.Dropout(config.v_attention_probs_dropout_prob)

        self.query2 = nn.Linear(config.hidden_size, self.all_head_size)
        self.key2 = nn.Linear(config.hidden_size, self.all_head_size)
        self.value2 = nn.Linear(config.hidden_size, self.all_head_size)
        # self.logit2 = nn.Linear(config.hidden_size, self.num_attention_heads)

        self.dropout2 = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
            self,
            input_tensor1,
            attention_mask1,
            input_tensor2,
            attention_mask2,
            co_attention_mask=None,
            use_co_attention_mask=False,
    ):

        # for vision input.
        mixed_query_layer1 = self.query1(input_tensor1)
        mixed_key_layer1 = self.key1(input_tensor1)
        mixed_value_layer1 = self.value1(input_tensor1)
        # mixed_logit_layer1 = self.logit1(input_tensor1)

        query_layer1 = self.transpose_for_scores(mixed_query_layer1)
        key_layer1 = self.transpose_for_scores(mixed_key_layer1)
        value_layer1 = self.transpose_for_scores(mixed_value_layer1)
        # logit_layer1 = self.transpose_for_logits(mixed_logit_layer1)

        # for text input:
        mixed_query_layer2 = self.query2(input_tensor2)
        mixed_key_layer2 = self.key2(input_tensor2)
        mixed_value_layer2 = self.value2(input_tensor2)
        # mixed_logit_layer2 = self.logit2(input_tensor2)

        query_layer2 = self.transpose_for_scores(mixed_query_layer2)
        key_layer2 = self.transpose_for_scores(mixed_key_layer2)
        value_layer2 = self.transpose_for_scores(mixed_value_layer2)
        # logit_layer2 = self.transpose_for_logits(mixed_logit_layer2)

        # Take the dot product between "query2" and "key1" to get the raw attention scores for value 1.
        attention_scores1 = torch.matmul(query_layer2, key_layer1.transpose(-1, -2))
        attention_scores1 = attention_scores1 / math.sqrt(self.attention_head_size)
        attention_scores1 = attention_scores1 + attention_mask1
        # if use_co_attention_mask:
        # attention_scores1 = attention_scores1 + co_attention_mask.permute(0,1,3,2)

        # Normalize the attention scores to probabilities.
        attention_probs1 = nn.Softmax(dim=-1)(attention_scores1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs1 = self.dropout1(attention_probs1)

        context_layer1 = torch.matmul(attention_probs1, value_layer1)
        context_layer1 = context_layer1.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape1 = context_layer1.size()[:-2] + (self.all_head_size,)
        context_layer1 = context_layer1.view(*new_context_layer_shape1)

        # Take the dot product between "query1" and "key2" to get the raw attention scores for value 2.
        attention_scores2 = torch.matmul(query_layer1, key_layer2.transpose(-1, -2))
        attention_scores2 = attention_scores2 / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)

        # we can comment this line for single flow.
        attention_scores2 = attention_scores2 + attention_mask2
        # if use_co_attention_mask:
        # attention_scores2 = attention_scores2 + co_attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs2 = nn.Softmax(dim=-1)(attention_scores2)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs2 = self.dropout2(attention_probs2)

        context_layer2 = torch.matmul(attention_probs2, value_layer2)
        context_layer2 = context_layer2.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape2 = context_layer2.size()[:-2] + (self.all_head_size,)
        context_layer2 = context_layer2.view(*new_context_layer_shape2)

        attn_data = None

        if self.visualization:
            attn_data = {
                "attn1": attention_probs1,
                "queries1": query_layer2,
                "keys1": key_layer1,
                "attn2": attention_probs2,
                "querues2": query_layer1,
                "keys2": key_layer2,
            }

        return context_layer1, context_layer2, attn_data


class BertBiOutput(nn.Module):
    def __init__(self, config):
        super(BertBiOutput, self).__init__()

        self.dense1 = nn.Linear(config.bi_hidden_size, config.v_hidden_size)
        self.LayerNorm1 = BertLayerNorm(config.v_hidden_size, eps=1e-12)
        self.dropout1 = nn.Dropout(config.v_hidden_dropout_prob)

        self.q_dense1 = nn.Linear(config.bi_hidden_size, config.v_hidden_size)
        self.q_dropout1 = nn.Dropout(config.v_hidden_dropout_prob)

        self.dense2 = nn.Linear(config.bi_hidden_size, config.hidden_size)
        self.LayerNorm2 = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout2 = nn.Dropout(config.hidden_dropout_prob)

        self.q_dense2 = nn.Linear(config.bi_hidden_size, config.hidden_size)
        self.q_dropout2 = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states1, input_tensor1, hidden_states2, input_tensor2):
        context_state1 = self.dense1(hidden_states1)
        context_state1 = self.dropout1(context_state1)

        context_state2 = self.dense2(hidden_states2)
        context_state2 = self.dropout2(context_state2)

        hidden_states1 = self.LayerNorm1(context_state1 + input_tensor1)
        hidden_states2 = self.LayerNorm2(context_state2 + input_tensor2)

        return hidden_states1, hidden_states2


class BertConnectionLayer(nn.Module):
    def __init__(self, config):
        super(BertConnectionLayer, self).__init__()
        self.biattention = BertBiAttention(config)

        self.biOutput = BertBiOutput(config)

        self.v_intermediate = BertImageIntermediate(config)
        self.v_output = BertImageOutput(config)

        self.t_intermediate = BertIntermediate(config)
        self.t_output = BertOutput(config)

    def forward(
            self,
            input_tensor1,
            attention_mask1,
            input_tensor2,
            attention_mask2,
            co_attention_mask=None,
            use_co_attention_mask=False,
    ):
        bi_output1, bi_output2, co_attention_probs = self.biattention(
            input_tensor1,
            attention_mask1,
            input_tensor2,
            attention_mask2,
            co_attention_mask,
            use_co_attention_mask,
        )

        attention_output1, attention_output2 = self.biOutput(
            bi_output2, input_tensor1, bi_output1, input_tensor2
        )

        intermediate_output1 = self.v_intermediate(attention_output1)
        layer_output1 = self.v_output(intermediate_output1, attention_output1)

        intermediate_output2 = self.t_intermediate(attention_output2)
        layer_output2 = self.t_output(intermediate_output2, attention_output2)

        return layer_output1, layer_output2, co_attention_probs


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()

        # in the bert encoder, we need to extract three things here.
        # text bert layer: BertLayer
        # vision bert layer: BertImageLayer
        # Bi-Attention: Given the output of two bertlayer, perform bi-directional
        # attention and add on two layers.

        self.FAST_MODE = config.fast_mode
        self.with_coattention = config.with_coattention
        self.v_biattention_id = config.v_biattention_id
        self.t_biattention_id = config.t_biattention_id
        self.in_batch_pairs = config.in_batch_pairs
        self.fixed_t_layer = config.fixed_t_layer
        self.fixed_v_layer = config.fixed_v_layer
        layer = BertLayer(config)
        v_layer = BertImageLayer(config)
        connect_layer = BertConnectionLayer(config)

        self.layer = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(config.num_hidden_layers)]
        )
        self.v_layer = nn.ModuleList(
            [copy.deepcopy(v_layer) for _ in range(config.v_num_hidden_layers)]
        )
        self.c_layer = nn.ModuleList(
            [copy.deepcopy(connect_layer) for _ in range(len(config.v_biattention_id))]
        )

    def forward(
            self,
            txt_embedding,
            image_embedding,
            txt_attention_mask,
            txt_attention_mask2,
            image_attention_mask,
            co_attention_mask=None,
            output_all_encoded_layers=True,
            output_all_attention_masks=False,
    ):

        v_start = 0
        t_start = 0
        count = 0
        all_encoder_layers_t = []
        all_encoder_layers_v = []

        all_attention_mask_t = []
        all_attnetion_mask_v = []
        all_attention_mask_c = []

        batch_size, num_words, t_hidden_size = txt_embedding.size()
        _, num_regions, v_hidden_size = image_embedding.size()

        use_co_attention_mask = False
        for v_layer_id, t_layer_id in zip(self.v_biattention_id, self.t_biattention_id):

            v_end = v_layer_id
            t_end = t_layer_id

            assert self.fixed_t_layer <= t_end
            assert self.fixed_v_layer <= v_end

            for idx in range(t_start, self.fixed_t_layer):
                with torch.no_grad():
                    txt_embedding, txt_attention_probs = self.layer[idx](
                        txt_embedding, txt_attention_mask
                    )
                    t_start = self.fixed_t_layer
                    if output_all_attention_masks:
                        all_attention_mask_t.append(txt_attention_probs)

            for idx in range(t_start, t_end):
                txt_embedding, txt_attention_probs = self.layer[idx](
                    txt_embedding, txt_attention_mask
                )
                if output_all_attention_masks:
                    all_attention_mask_t.append(txt_attention_probs)

            for idx in range(v_start, self.fixed_v_layer):
                with torch.no_grad():
                    image_embedding, image_attention_probs = self.v_layer[idx](
                        image_embedding,
                        image_attention_mask,
                        txt_embedding,
                        txt_attention_mask2,
                    )
                    v_start = self.fixed_v_layer

                    if output_all_attention_masks:
                        all_attnetion_mask_v.append(image_attention_probs)

            for idx in range(v_start, v_end):
                image_embedding, image_attention_probs = self.v_layer[idx](
                    image_embedding,
                    image_attention_mask,
                    txt_embedding,
                    txt_attention_mask2,
                )

                if output_all_attention_masks:
                    all_attnetion_mask_v.append(image_attention_probs)

            if count == 0 and self.in_batch_pairs:
                # new batch size is the batch_size ^2
                image_embedding = (
                    image_embedding.unsqueeze(0)
                        .expand(batch_size, batch_size, num_regions, v_hidden_size)
                        .contiguous()
                        .view(batch_size * batch_size, num_regions, v_hidden_size)
                )
                image_attention_mask = (
                    image_attention_mask.unsqueeze(0)
                        .expand(batch_size, batch_size, 1, 1, num_regions)
                        .contiguous()
                        .view(batch_size * batch_size, 1, 1, num_regions)
                )

                txt_embedding = (
                    txt_embedding.unsqueeze(1)
                        .expand(batch_size, batch_size, num_words, t_hidden_size)
                        .contiguous()
                        .view(batch_size * batch_size, num_words, t_hidden_size)
                )
                txt_attention_mask = (
                    txt_attention_mask.unsqueeze(1)
                        .expand(batch_size, batch_size, 1, 1, num_words)
                        .contiguous()
                        .view(batch_size * batch_size, 1, 1, num_words)
                )
                co_attention_mask = (
                    co_attention_mask.unsqueeze(1)
                        .expand(batch_size, batch_size, 1, num_regions, num_words)
                        .contiguous()
                        .view(batch_size * batch_size, 1, num_regions, num_words)
                )

            if count == 0 and self.FAST_MODE:
                txt_embedding = txt_embedding.expand(
                    image_embedding.size(0),
                    txt_embedding.size(1),
                    txt_embedding.size(2),
                )
                txt_attention_mask = txt_attention_mask.expand(
                    image_embedding.size(0),
                    txt_attention_mask.size(1),
                    txt_attention_mask.size(2),
                    txt_attention_mask.size(3),
                )

            if self.with_coattention:
                # do the bi attention.
                image_embedding, txt_embedding, co_attention_probs = self.c_layer[
                    count
                ](
                    image_embedding,
                    image_attention_mask,
                    txt_embedding,
                    txt_attention_mask,
                    co_attention_mask,
                    use_co_attention_mask,
                )

                if output_all_attention_masks:
                    all_attention_mask_c.append(co_attention_probs)

            v_start = v_end
            t_start = t_end
            count += 1

            if output_all_encoded_layers:
                all_encoder_layers_t.append(txt_embedding)
                all_encoder_layers_v.append(image_embedding)

        for idx in range(v_start, len(self.v_layer)):
            image_embedding, image_attention_probs = self.v_layer[idx](
                image_embedding,
                image_attention_mask,
                txt_embedding,
                txt_attention_mask2,
            )

            if output_all_attention_masks:
                all_attnetion_mask_v.append(image_attention_probs)

        for idx in range(t_start, len(self.layer)):
            txt_embedding, txt_attention_probs = self.layer[idx](
                txt_embedding, txt_attention_mask
            )

            if output_all_attention_masks:
                all_attention_mask_t.append(txt_attention_probs)

        # add the end part to finish.
        if not output_all_encoded_layers:
            all_encoder_layers_t.append(txt_embedding)
            all_encoder_layers_v.append(image_embedding)

        return (
            all_encoder_layers_t,
            all_encoder_layers_v,
            (all_attention_mask_t, all_attnetion_mask_v, all_attention_mask_c),
        )


class BertTextPooler(nn.Module):
    def __init__(self, config):
        super(BertTextPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.bi_hidden_size)
        self.activation = nn.ReLU()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertImagePooler(nn.Module):
    def __init__(self, config):
        super(BertImagePooler, self).__init__()
        self.dense = nn.Linear(config.v_hidden_size, config.bi_hidden_size)
        self.activation = nn.ReLU()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str) or (
                sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)
        ):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertImgPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertImgPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.v_hidden_size, config.v_hidden_size)
        if isinstance(config.hidden_act, str) or (
                sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)
        ):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.v_hidden_act
        self.LayerNorm = BertLayerNorm(config.v_hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(
            bert_model_embedding_weights.size(1),
            bert_model_embedding_weights.size(0),
            bias=False,
        )
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super(BertOnlyNSPHead, self).__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)
        self.bi_seq_relationship = nn.Linear(config.bi_hidden_size, 2)
        self.imagePredictions = BertImagePredictionHead(config)
        self.fusion_method = config.fusion_method
        self.dropout = nn.Dropout(0.1)

    def forward(
            self, sequence_output_t, sequence_output_v, pooled_output_t, pooled_output_v
    ):

        if self.fusion_method == "sum":
            pooled_output = self.dropout(pooled_output_t + pooled_output_v)
        elif self.fusion_method == "mul":
            pooled_output = self.dropout(pooled_output_t * pooled_output_v)
        else:
            assert False

        prediction_scores_t = self.predictions(sequence_output_t)
        seq_relationship_score = self.bi_seq_relationship(pooled_output)
        prediction_scores_v = self.imagePredictions(sequence_output_v)

        return prediction_scores_t, prediction_scores_v, seq_relationship_score


class BertImagePredictionHead(nn.Module):
    def __init__(self, config):
        super(BertImagePredictionHead, self).__init__()
        self.transform = BertImgPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.v_hidden_size, config.v_target_size)

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertPreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """

    config_class = BertConfig
    pretrained_model_archive_map = BERT_PRETRAINED_MODEL_ARCHIVE_MAP
    load_tf_weights = load_tf_weights_in_bert
    base_model_prefix = "bert"

    def __init__(self, *inputs, **kwargs):
        super(BertPreTrainedModel, self).__init__(*inputs, **kwargs)

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # torch.nn.init.kaiming_normal_(module.weight.data, a=0.001)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class BertModel(BertPreTrainedModel):
    def __init__(self, config):
        super(BertModel, self).__init__(config)

        # initilize word embedding
        if config.model == "bert":
            self.embeddings = BertEmbeddings(config)
        elif config.model == "roberta":
            self.embeddings = RobertaEmbeddings(config)

        self.task_specific_tokens = config.task_specific_tokens

        # initlize the vision embedding
        self.v_embeddings = BertImageEmbeddings(config)

        self.encoder = BertEncoder(config)
        self.t_pooler = BertTextPooler(config)
        self.v_pooler = BertImagePooler(config)

        self.apply(self.init_weights)

    def forward(
            self,
            input_txt,
            input_imgs,
            image_loc,
            token_type_ids=None,
            attention_mask=None,
            image_attention_mask=None,
            co_attention_mask=None,
            task_ids=None,
            output_all_encoded_layers=False,
            output_all_attention_masks=False,
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_txt)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_txt)
        if image_attention_mask is None:
            image_attention_mask = torch.ones(
                input_imgs.size(0), input_imgs.size(1)
            ).type_as(input_txt)

        if self.task_specific_tokens:
            # extend the mask
            mask_tokens = input_txt.new().resize_(input_txt.size(0), 1).fill_(1)
            attention_mask = torch.cat([mask_tokens, attention_mask], dim=1)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_image_attention_mask = image_attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask2 = attention_mask.unsqueeze(2)
        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            # dtype=torch.float32
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        extended_attention_mask2 = extended_attention_mask2.to(
            # dtype=torch.float32
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility

        extended_image_attention_mask = extended_image_attention_mask.to(
            # dtype=torch.float32
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_image_attention_mask = (1.0 - extended_image_attention_mask) * -10000.0

        if co_attention_mask is None:
            co_attention_mask = torch.zeros(
                input_txt.size(0), input_imgs.size(1), input_txt.size(1)
            ).type_as(extended_image_attention_mask)

        extended_co_attention_mask = co_attention_mask.unsqueeze(1)

        # extended_co_attention_mask = co_attention_mask.unsqueeze(-1)
        extended_co_attention_mask = extended_co_attention_mask * 5.0
        extended_co_attention_mask = extended_co_attention_mask.to(
            # dtype=torch.float32
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility

        embedding_output = self.embeddings(input_txt, token_type_ids, task_ids)
        v_embedding_output = self.v_embeddings(input_imgs, image_loc)
        encoded_layers_t, encoded_layers_v, all_attention_mask = self.encoder(
            embedding_output,
            v_embedding_output,
            extended_attention_mask,
            extended_attention_mask2,
            extended_image_attention_mask,
            extended_co_attention_mask,
            output_all_encoded_layers=output_all_encoded_layers,
            output_all_attention_masks=output_all_attention_masks,
        )

        sequence_output_t = encoded_layers_t[-1]
        sequence_output_v = encoded_layers_v[-1]

        pooled_output_t = self.t_pooler(sequence_output_t)
        pooled_output_v = self.v_pooler(sequence_output_v)

        if not output_all_encoded_layers:
            encoded_layers_t = encoded_layers_t[-1]
            encoded_layers_v = encoded_layers_v[-1]

        return (
            encoded_layers_t,
            encoded_layers_v,
            pooled_output_t,
            pooled_output_v,
            all_attention_mask,
        )


class BertImageEmbeddings(nn.Module):
    """Construct the embeddings from image, spatial location (omit now) and token_type embeddings.
    """

    def __init__(self, config):
        super(BertImageEmbeddings, self).__init__()

        self.image_embeddings = nn.Linear(config.v_feature_size, config.v_hidden_size)
        self.image_location_embeddings = nn.Linear(5, config.v_hidden_size)
        self.LayerNorm = BertLayerNorm(config.v_hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, input_loc):
        img_embeddings = self.image_embeddings(input_ids)
        loc_embeddings = self.image_location_embeddings(input_loc)

        # TODO: we want to make the padding_idx == 0, however, with custom initilization, it seems it will have a bias.
        # Let's do masking for now
        embeddings = self.LayerNorm(img_embeddings + loc_embeddings)
        # embeddings = self.LayerNorm(img_embeddings+loc_embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings



class SequenceEncoder(nn.Module):
    def __init__(self, use_search_word, bert_size, output_dim = 512, dropout=0.1):
        super(SequenceEncoder, self).__init__()
        # print(bert_size)
        self.text_bert = AutoModel.from_pretrained("/data/gjr/huggingface/bert-%s" % bert_size)
        if bert_size == 'tiny':
            self.feat_dim = 128
        elif bert_size == 'mini':
            self.feat_dim = 256
        else:
            self.feat_dim = 512
        self.use_search_word = use_search_word
        self.output_dim = output_dim
        self.proj = FCNet([self.feat_dim, output_dim], dropout)

    def forward(self, seqs):
        # the last dim is seq length  seqs: [batch, num_sents]
        original_shape = list(seqs.size())
        flattened_seqs = seqs.reshape(-1, original_shape[-1])
        flattened_masks = (flattened_seqs.sum(1) > 0).long().detach().cpu().numpy()
        gather_idxs = []

        for i in range(len(flattened_masks)):
            if flattened_masks[i]:
                gather_idxs.append(i)

        if len(gather_idxs):
            valid_seqs = flattened_seqs[gather_idxs]
            processed_valid_seqs = self.proj(self.text_bert(valid_seqs).pooler_output)
            processed_valid_seqs = torch.cat([processed_valid_seqs, torch.zeros((1, processed_valid_seqs.size(1))).cuda()])
        else:
            processed_valid_seqs =  torch.zeros((1, self.feat_dim)).cuda()

        post_gather_idxs = []
        ct = 0
        for i in range(len(flattened_masks)):
            if flattened_masks[i]:
                post_gather_idxs.append(ct)
                ct += 1
            else:
                post_gather_idxs.append(0)
        assert ct == len(gather_idxs)
        processed_flattened_seqs = processed_valid_seqs[post_gather_idxs]
        processed_seqs = processed_flattened_seqs.reshape(original_shape[:-1] + [self.output_dim])
        return processed_seqs


def SoftPlus(val):
    return (torch.log(1 + torch.exp(val)) - torch.log(1 + torch.exp((-1)*val)))


class VILBertForVLTasks(BertPreTrainedModel):
    def __init__(self, config, num_labels, dropout_prob=0.1, ablation=[], graph_size=1000):
        super(VILBertForVLTasks, self).__init__(config)
        self.num_labels = num_labels
        self.backbone_model = config.backbone_model
        self.dropout = nn.Dropout(0.5)
        self.ablation = ablation
        self.graph_size = graph_size
        vil_dim = num_labels # 512 # 5020 #fixed

        # LXMERT 
        if self.backbone_model == 'bert':
            self.bert = BertModel(config)
            self.vil_prediction_backbone = nn.Linear(config.bi_hidden_size, vil_dim)
            # self.vil_prediction_t = SimpleClassifier(config.bi_hidden_size, config.bi_hidden_size * 2, vil_dim, 0.5)
            # self.vil_prediction_v = SimpleClassifier(config.bi_hidden_size, config.bi_hidden_size * 2, vil_dim, 0.5)
        elif self.backbone_model == 'lxmert':
            self.lxmert = LxmertModel(config)
            # self.vil_prediction_backbone = SimpleClassifier(config.hidden_size, config.hidden_size * 2, vil_dim, 0.5)
        
        self.num_sentences = config.num_wiki_sentences
        self.num_concepts = config.num_concepts

        self.num_head = config.num_head
        self.use_wiki = config.use_wiki
        self.use_concept = config.use_concept
        self.use_image = config.use_image
        self.segment_dim = config.segment_dim
        self.num_hid = config.num_hid
        self.qid2knowledge = pickle.load(open('/data/gjr/OK-VQA/mavex/qid2knowledge_cleaned_mavex.pkl', 'rb'))
        self.qid2statements = pickle.load(open('qid2statements.pkl', 'rb'))
        self.label2ans = pickle.load(open('data/okvqa_new/cache/trainval_label2ans_729.pkl', 'rb'))
        self.add_answer_emb = config.add_answer_emb

        self.sigmoid = nn.Sigmoid()
        self.small_bert_tokenizer = AutoTokenizer.from_pretrained("/data/gjr/huggingface/bert-small")
        self.small_bert_model = AutoModel.from_pretrained("/data/gjr/huggingface/bert-small",output_hidden_states=True)


        concept_emb = np.load('/data/gjr/ConceptNet/transe/concept.nb.npy')
        self.concept_emb = torch.tensor(concept_emb).float().cuda()
        self.concept2label = pickle.load(open('/data/gjr/OK-VQA/myAnnotations/trainval_concept2label_729.pkl', "rb"))
        self.candidate_emb = self.concept_emb[torch.tensor(list(self.concept2label.keys()))]

        concept_num, concept_dim = self.concept_emb.size(0), self.concept_emb.size(1) # 516782, 
        print('num_concepts: {}, concept_dim: {}'.format(concept_num, concept_dim))

        self.concept2id = {}
        self.id2concept = id2concept
        self.concept2id = {w: i for i, w in enumerate(self.id2concept)}
        self.relation_text = relation_text
        self.relation_text.extend([x + ' reverse' for x in relation_text])
        self.SOS_token = 0
        self.EOS_token = 1


        

        self.qa_statement_encoder = SequenceEncoder(False, config.bert_size, vil_dim)
        n_type = 4
        n_head = 1
        n_layer = 1
        n_basis = 0
        # self.pooler = TypedMultiheadAttPoolLayer(12, lstm_hid_size*2, concept_dim, n_type=n_type)
        self.dropout_fc = nn.Dropout(0.5)
        self.dropout_e = nn.Dropout(0.5)

        self.node_mlp = MLP(concept_dim + vil_dim, 1024, vil_dim, 2, 0.5, layer_norm=True, batch_norm=False)
        self.cosinesim = nn.CosineSimilarity(dim=2, eps=1e-6)

        if 'close' in self.ablation:
            self.schema_node_num = self.graph_size  # schema graph size
            self.sub_node_num = 20  # subgraph size
            self.qv_path_num, self.q_path_num =  50, 50
        else:
            self.schema_node_num = self.graph_size  # schema graph size
            self.sub_node_num = 100  # subgraph size
            self.qv_path_num, self.q_path_num =  50, 50

        if 'prune' in self.ablation:
            self.prune_network = PruneNetwork(node_in=self.schema_node_num, node_out=self.sub_node_num, v_dim=vil_dim)
            # self.ptm_edge_mlp = MLP( vil_dim , 1024, vil_dim, 2, 0.5, layer_norm=True, batch_norm=False)
        if 'ptm' in self.ablation:
            self.ptm_mlp = MLP(vil_dim + vil_dim + vil_dim//4 , 1024, vil_dim, 2, 0.5, layer_norm=True, batch_norm=False)
        if 'node_type' in self.ablation:
            self.n_ntype = 4
            self.emb_node_type = nn.Linear(self.n_ntype, vil_dim//4)
            self.activation = GELU()

        if 'path' in self.ablation:
            self.path_network = PathNet(vil_dim, self.ablation, self.qv_path_num, self.q_path_num)
        else:
            self.fc = nn.Linear(vil_dim, 1)

        # if 'classification' in self.ablation:
        #     # self.vil_classification = SimpleClassifier(config.bi_hidden_size, config.bi_hidden_size * 2, 4025, 0.5)
        #     self.vil_classification = SimpleClassifier(2*vil_dim, config.bi_hidden_size * 2, 4025, 0.5)
        #     # no parameter sharing
        #     # self.bert2 = BertModel(config)
        #     # self.bert2.load_state_dict(self.bert.state_dict())
        #     # self.vil_prediction_backbone2 = SimpleClassifier(config.bi_hidden_size, config.bi_hidden_size * 2, num_labels, 0.5)
        #     self.candidate_net = MLP(concept_dim, 512, vil_dim, 2, 0.5, layer_norm=True, batch_norm=False)
        #     # self.answer_candidate_tensor = torch.arange(0, 4025).view(-1, 1).long().cuda()
        #     # self.cand_decode = nn.Embedding(4025, vil_dim)
        #     self.anchor_fc = nn.Linear(vil_dim*2, vil_dim)
        #     self.graph_pool_mlp = MLP(self.graph_size, 512, 1, 2, 0.5, layer_norm=True, batch_norm=False)
        #     # self.graph_model = GCN(in_feats=vil_dim, hid_feats=256, out_feats=vil_dim)
        #     self.graph_model = GAT(in_dim=vil_dim, hidden_dim=512, out_dim=vil_dim, heads=4, dropout=0.3)
        self.apply(self.init_weights)

    def PTM_statements(self, batch_size, q_tokens, nodes, batch_num_nodes, statement_max_len = 14):
        # q_tokens: batch_size x 14
        q_tokens = q_tokens.unsqueeze(1).expand(-1, batch_num_nodes[0], -1).reshape(batch_num_nodes[0] * batch_size, -1) # batch_size x 14 -> (batch_size * node_num) x 14
        cand_txts = [self.id2concept[node].replace('_', ' ') for node in nodes.tolist()]
        cand_tokens = [self.small_bert_tokenizer.convert_tokens_to_ids(self.small_bert_tokenizer.tokenize(txt)) for txt in cand_txts]
        # cand_tokens = torch.nn.utils.rnn.pad_sequence(torch.split(cand_tokens, batch_num_nodes.tolist()), batch_first=True) # batch_size x 100 x 1
        statement_tokens =[c + [101] + q[:q.index(102) - 1] for q, c in zip(q_tokens.tolist(), cand_tokens)]
        statement_tokens = [s[:statement_max_len - 1] + [102] for s in statement_tokens]
        statement_tokens = [s + [0] * (statement_max_len - len(s)) if len(s) < statement_max_len else s for s in statement_tokens]
        if len(statement_tokens) < statement_max_len:
            padding = [0] * (statement_max_len - len(statement_tokens))
            statement_tokens = statement_tokens + padding
        return torch.tensor(statement_tokens)

    def PTM_statements_edge(self, batch_size, q_tokens, edge_txts, batch_num_nodes, statement_max_len = 14):
        # q_tokens: batch_size x 14
        q_tokens = q_tokens.unsqueeze(1).expand(-1, batch_num_nodes[0], -1).reshape(batch_num_nodes[0] * batch_size, -1) # batch_size x 14 -> (batch_size * node_num) x 14
        cand_txts = edge_txts
        cand_tokens = [self.small_bert_tokenizer.convert_tokens_to_ids(self.small_bert_tokenizer.tokenize(txt)) for txt in cand_txts]
        # cand_tokens = torch.nn.utils.rnn.pad_sequence(torch.split(cand_tokens, batch_num_nodes.tolist()), batch_first=True) # batch_size x 100 x 1
        statement_tokens =[c  for q, c in zip(q_tokens.tolist(), cand_tokens)]
        statement_tokens = [s[:statement_max_len - 1] + [102] for s in statement_tokens]
        statement_tokens = [s + [0] * (statement_max_len - len(s)) if len(s) < statement_max_len else s for s in statement_tokens]
        if len(statement_tokens) < statement_max_len:
            padding = [0] * (statement_max_len - len(statement_tokens))
            statement_tokens = statement_tokens + padding
        return torch.tensor(statement_tokens)

    def forward(
            self,
            input_txt,
            input_imgs,
            image_loc,
            token_type_ids=None,
            attention_mask=None,
            question_ids=None,
            image_attention_mask=None,
            co_attention_mask=None,
            task_ids=None,
            output_all_encoded_layers=False,
            output_all_attention_masks=False,
            verif_inds=None,
            question_txt=None,
            target_txt=None,
            kg_sents=None,
            graph=None,
            target_mask=None,            
            epoch_id=0,
            ):

        # GJRTIME 
        checktime = False  
        if checktime:
            torch.cuda.synchronize()
            start = time.time()

        # LXMERT
        if self.backbone_model == 'bert':
            sequence_output_t, sequence_output_v, pooled_output_t, pooled_output_v, all_attention_mask = self.bert(
                input_txt,
                input_imgs,
                image_loc,
                token_type_ids,
                attention_mask,
                image_attention_mask,
                co_attention_mask,
                task_ids,
                output_all_encoded_layers=output_all_encoded_layers,
                output_all_attention_masks=output_all_attention_masks,
            )
            batch_size = input_txt.size(0)
            pooled_output = self.dropout(pooled_output_t * pooled_output_v)
            vil_prediction = self.vil_prediction_backbone(pooled_output)
            # vil_pred_t = self.vil_prediction_t(pooled_output_t)
            # vil_pred_v = self.vil_prediction_v(pooled_output_v)

        elif self.backbone_model == 'lxmert':
            # print(input_txt, 'input_txt', input_txt.shape, 'input_imgs', input_imgs.shape, 'image_loc', image_loc.shape, 
            #     'token_type_ids', token_type_ids.shape,
            #     attention_mask.shape,
            #     image_attention_mask.shape,
            #     co_attention_mask.shape)
            outputs = self.lxmert(
                input_ids=input_txt, # 64, 23
                attention_mask=attention_mask, # 64, 23
                token_type_ids=token_type_ids, # 64, 23
                visual_feats=input_imgs, # 64, 101, 2048
                visual_pos=image_loc) # 64, 101, 5
            batch_size = input_txt.size(0)
            # print(outputs.shape)
            # pooled_output = self.dropout(pooled_output_t * pooled_output_v)
            pooled_output = outputs.pooled_output
            # vil_prediction = self.vil_prediction_backbone(outputs.pooled_output)


        # GJRTIME
        if checktime:
            torch.cuda.synchronize()
            end = time.time()
            print('bert', end-start)
            start = end

        # **********************************************************************************
        # Construct Graph
        # batch_num_nodes = graph.batch_num_nodes()
        # batch_num_nodes_ = torch.cat((torch.tensor([0]).cuda(), batch_num_nodes)) 
        # batch_sum_nodes = torch.cumsum(batch_num_nodes_, dim=0) # [2,3,2,4] -> [0,2,5,7,11]

        node_ids = graph.ndata['id'] # entity ids in numberbatch
        node_num = int(len(node_ids) / batch_size)
        node_idxs = torch.arange(0,len(node_ids)).cuda()
 
        # # Concatenate ViL prediction: BERT / LXMERT Embedding
        # batch_vil_pred_t = vil_pred_t.unsqueeze(1).expand(-1, 1000, vil_pred_t.shape[1])
        # batch_vil_pred_v = vil_pred_v.unsqueeze(1).expand(-1, 100, vil_pred_v.shape[1])
        batch_vil = vil_prediction.unsqueeze(1).expand(batch_size, node_num, vil_prediction.shape[1])
        fullgraph_node_features = self.concept_emb[node_ids]
        fullgraph_node_features = torch.cat((fullgraph_node_features, batch_vil.reshape(-1, vil_prediction.shape[1])), 1)
        fullgraph_node_features = self.node_mlp(fullgraph_node_features)
        fullgraph_node_features = fullgraph_node_features.reshape(batch_size, -1, fullgraph_node_features.shape[1])
        node_features = fullgraph_node_features
        fullgraph_anchors = vil_prediction 

        class_logits = self.cosinesim(batch_vil, fullgraph_node_features)

        # GJRTIME
        if checktime:
            torch.cuda.synchronize()
            end = time.time()
            print('batch vil', end-start)
            start = end

        if 'classification' in self.ablation:
            anchors = None
            # CASE 1 ViLBERT 4025 classification
            # class_logits = self.vil_classification(pooled_output)
            # CASE 2 concat gcn pool feature and vil prediction
            # fullgraph_node_features = self.graph_model(graph, fullgraph_node_features) # nodes x f_dim
            # graph_emb_pool = self.graph_pool_mlp(fullgraph_node_features.reshape(batch_size, fullgraph_node_features.shape[1], -1)).squeeze()
            # class_logits = self.vil_classification(torch.cat((vil_prediction, graph_emb_pool), 1))

            # CASE 3 prediction as anchor, anchor x candidates as score
            fullgraph_node_features = self.graph_model(graph, fullgraph_node_features) # nodes x f_dim
            graph_emb_pool = self.graph_pool_mlp(fullgraph_node_features.reshape(batch_size, fullgraph_node_features.shape[1], -1)).squeeze()
            # class_logits = self.vil_classification(torch.cat((vil_prediction, graph_emb_pool), 1))
            anchors = self.anchor_fc(torch.cat((pooled_output, graph_emb_pool), 1))
            # cand_feature = self.cand_decode(self.answer_candidate_tensor).squeeze()
            cand_feature = self.candidate_net(self.candidate_emb)
            class_logits = anchors.mm(cand_feature.t())
            fullgraph_node_features = cand_feature.unsqueeze(0).expand(batch_size, cand_feature.shape[0], -1) # for triplet loss

        # if 'path' not in self.ablation and 'classification' not in self.ablation:
        #     if len(node_features.shape) == 2:
        #         node_features = node_features.reshape(batch_size, -1, node_features.shape[1])
        #     # node_features = torch.nn.utils.rnn.pad_sequence(torch.split(node_features, batch_num_nodes.tolist()), batch_first=True) # batch x 50 x 2048
        #     # case 1: cosine similarity
        #     class_logits = self.cosinesim(batch_vil, node_features)
        #     # case 2: fc + softmax dim=1 
        #     # class_logits = self.fc(node_features).squeeze() # batch x node_num
        #     # class_logits = nn.Softmax(dim=1)(class_logits)
        #     # case 3: fc + softmax dim=2 fc 2
        #     # class_logits = self.fc(node_features)
        #     # class_logits = nn.Softmax(dim=2)(class_logits)
        #     # class_logits = class_logits[:,:,0].squeeze()
        #     # case 4: fc + sigmoid
        #     # class_logits = torch.sigmoid(self.fc(node_features)).squeeze()
        #     # case 5: fc  
        #     # class_logits = self.fc(node_features).squeeze()
        #     anchors = vil_prediction # vil_pred_t  # 

        pred_features, keep_mask, anchors = None, None, None
        batch_end_point_idxs, batch_sampled_paths = None, None
        if 'e2e' in self.ablation and epoch_id < 40:
            return vil_prediction, fullgraph_anchors, anchors, fullgraph_node_features, pred_features, \
                batch_end_point_idxs, class_logits, keep_mask, node_ids, None
        # **********************************************************************************
        # Graph Prune 

        # GJRTIME
        if checktime:
            torch.cuda.synchronize()
            end = time.time()
            print('ptm', end-start)
            start = end
        if 'prune' in self.ablation:
            # fullgraph = copy.deepcopy(graph)
            graph, node_features, keep_mask = self.prune_network(graph, batch_size, batch_vil, fullgraph_node_features)
            # node_features = node_features.reshape(-1, node_features.shape[2])
            # Construct Graph
            batch_num_nodes = graph.batch_num_nodes()
            batch_num_nodes_ = torch.cat((torch.tensor([0]).cuda(), batch_num_nodes)) 
            batch_sum_nodes = torch.cumsum(batch_num_nodes_, dim=0) # [2,3,2,4] -> [0,2,5,7,11]

            node_ids = graph.ndata['id'] # entity ids in numberbatch
            node_idxs = torch.arange(0,len(node_ids)).cuda()
            batch_vil = vil_prediction.unsqueeze(1).expand(-1, batch_num_nodes[0], vil_prediction.shape[1])
            anchors = vil_prediction # vil_pred_t  # 

            # **********************************************************************************
            if 'node_type' in self.ablation:
                node_type_ids = torch.stack((graph.ndata['a'], graph.ndata['v']*2, graph.ndata['q']*3)) # n,a,v,q -> 0,1,2,3
                node_type_ids = node_type_ids.max(0)[0]
                # edge_types = graph.edata['rel'].cuda()
                # node_type_ids = torch.cat((node_type_ids, pad_x),1)
                T = make_one_hot(node_type_ids.view(-1).contiguous(), self.n_ntype)#.view(batch_size, -1, self.n_ntype)
                node_type_emb = self.activation(self.emb_node_type(T)).view(batch_size, node_features.shape[1], -1) #[batch_size x n_node, dim//4]
                node_features = torch.cat((node_type_emb, node_features), 2)

                # GJRTIME
                if checktime:
                    torch.cuda.synchronize()
                    end = time.time()
                    print('node type', end-start)
                    start = end

            # **********************************************************************************
            if 'ptm' in self.ablation:
                ptm_statements = self.PTM_statements(batch_size, input_txt, node_ids, batch_num_nodes)
                ptm_statements = ptm_statements.cuda()
                qa_features = self.qa_statement_encoder(ptm_statements).view(batch_size, node_features.shape[1], -1) #.view(batch_size, 5, -1)
                node_features = torch.cat((node_features, qa_features), 2)
                # ptm_statements_edge = self.PTM_statements_edge(batch_size, input_txt, self.relation_text, batch_num_nodes)
                # ptm_statements_edge = ptm_statements_edge.cuda()
                # edge_features = self.qa_statement_encoder(ptm_statements_edge)
                # edge_features = self.ptm_edge_mlp(edge_features)
                # edge_features = edge_features[edge_types]
                # node_features = self.graph_model(graph, node_features) # nodes x f_dim
                node_features = self.ptm_mlp(node_features)
            
            # **********************************************************************************
            if 'path' in self.ablation:
                class_logits, pred_features, batch_sampled_paths, batch_end_point_idxs = self.path_network(graph, vil_prediction, node_features)
            else:
                pred_features = node_features #.reshape(batch_size, -1, node_features.shape[1])
                class_logits = self.fc(pred_features).squeeze()
        return vil_prediction, fullgraph_anchors, anchors, fullgraph_node_features, pred_features, \
            batch_end_point_idxs, class_logits, keep_mask, node_ids, batch_sampled_paths


class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super().__init__()
        self.logit_fc = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            GeLU(),
            BertLayerNorm(hid_dim, eps=1e-12),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, out_dim),
        )

    def forward(self, hidden_states):
        return self.logit_fc(hidden_states) 

from torch.autograd import Variable
def make_one_hot(labels, C):
    '''
    Converts an integer label torch.autograd.Variable to a one-hot Variable.
    labels : torch.autograd.Variable of torch.cuda.LongTensor
        (N, ), where N is batch size.
        Each value is an integer representing correct classification.
    C : integer.
        number of classes in labels.
    Returns : torch.autograd.Variable of torch.cuda.FloatTensor
        N x C, where C is class number. One-hot encoded.
    '''
    labels = labels.unsqueeze(1)
    one_hot = torch.FloatTensor(labels.size(0), C).zero_().to(labels.device)
    target = one_hot.scatter_(1, labels.data, 1)
    target = Variable(target)
    return target

from math import pi, acos
class PruneNetwork(nn.Module):
    def __init__(self, node_in, node_out, v_dim):
        super().__init__()
        self.cosinesim = nn.CosineSimilarity(dim=2, eps=1e-6)
        self.node_in = node_in
        self.node_out = node_out

    def forward(self, graph, batch_size, vil_pred, node_features):
        node_idxs = torch.arange(0,self.node_in).unsqueeze(0).expand(batch_size, -1).cuda()

        batch_num_nodes = torch.cat((torch.tensor([0]).cuda(), graph.batch_num_nodes()))
        batch_sum_nodes = torch.cumsum(batch_num_nodes, dim=0) # [2,3,2,4] -> [0,2,5,7,11]
        # isolated_nodes = ((g.in_degrees() == 0) & (g.out_degrees() == 0)).nonzero().squeeze(1)
        # TODO: keep v and q nodes
        sims = 1 + self.cosinesim(vil_pred, node_features).squeeze()

        graphs = dgl.unbatch(graph)
        for i, g in enumerate(graphs):
            node_idx = torch.arange(0, self.node_in)
            qv_ids = (g.ndata['q'] if sum(g.ndata['q']) >= 1 else torch.logical_or(g.ndata['v'], g.ndata['q'])).int().nonzero().squeeze(1)
            # print(qv_ids, [g.ndata['id'][x] for x in qv_ids])
            bfs = dgl.bfs_nodes_generator(g, qv_ids)
            # print(qv_ids, g.edges())
            # print('bfs', [len(set(x.tolist())) for x in bfs])
            # if len(bfs)<2:
            #     print(qv_ids, g.edges())
            bfs_dict = {n:1 for n in node_idx.tolist()}
            for k,nodes in enumerate(bfs):
                for n in nodes.tolist():
                    bfs_dict.update({n:(10-k)/5})
            node_bfs = torch.tensor([bfs_dict[n] for n in node_idx.tolist()]).cuda()
            th_bfs = 0.3
            sims[i] = sims[i][:] * (1 - th_bfs) + th_bfs * node_bfs 

            # sims[i] = sims[i][:] * node_bfs #* 0.7 + 0.3 
            # delete
            # in_node_ids = torch.tensor(list(set(g.in_edges(keep_ids[i])[0].tolist()))).cuda()
            # keep_node_ids = in_node_ids[torch.tensor([a not in keep_ids[i] for a in in_node_ids])]
            # keep_node_ids = torch.cat((keep_node_ids, keep_ids[i]),0)
            # qv_ids = torch.logical_or(g.ndata['v'], g.ndata['q']).int().nonzero().squeeze(1)

            # print('bfs_dict', bfs_dict)
            # sg = g.subgraph(keep_ids[i])

        # random
        # keep_ids = torch.stack([torch.randperm(self.node_in)[:self.node_out] for i in range(batch_size)]).cuda()
        keep_ids = torch.topk(sims, self.node_out).indices.sort()[0] # no grad # batch_size x node_out
        # innout = g.in_degrees() + g.out_degrees()


        mask = torch.zeros(node_features.shape[:-1]).cuda()
        mask.scatter_(1, keep_ids, 1.)
        prune_ids = node_idxs[mask==0.].reshape(batch_size, -1)
        prune_ids += batch_sum_nodes[:-1].unsqueeze(1).expand(prune_ids.shape)
        graph.remove_nodes(prune_ids.reshape(-1))
        node_features = torch.stack([x[y] for x,y in zip(node_features, keep_ids)])

        # qv_ids = (graph.ndata['q'] if sum(graph.ndata['q']) >= 1 else torch.logical_or(graph.ndata['v'], graph.ndata['q'])).int().nonzero().squeeze(1)
        # bfs = dgl.bfs_nodes_generator(graph, qv_ids)
        # isolated_nodes = ((graph.in_degrees() == 0) & (graph.out_degrees() == 0)).nonzero().squeeze(1)
        # in_isolated_nodes = ((graph.in_degrees() == 0) ).nonzero().squeeze(1)
        # out_isolated_nodes = ((graph.out_degrees() == 0) ).nonzero().squeeze(1)
        # print('isolated_nodes', isolated_nodes.shape[0], '/', node_features.shape[0]*self.node_out, in_isolated_nodes.shape[0], out_isolated_nodes.shape[0])
        # print('bfs', [len(set(x.tolist())) for x in bfs])
        # iso_nodes = isolated_nodes.shape[0]*100 / (node_features.shape[0]*self.node_out)
        # print(iso_nodes)

        return graph, node_features, mask

import dgl.function as fn
class PathNet(nn.Module):
    def __init__(self, vil_dim, ablation, qv_path_num=50, q_path_num=50):
        super(PathNet, self).__init__()
        self.ablation = ablation
        self.qv_path_num = qv_path_num
        self.q_path_num = q_path_num
        # self.vil2lstm = nn.Linear(vil_dim, lstm_hid_size*2) #MLP(vil_dim, vil_dim*2, lstm_hid_size*2, 2, 0.5, layer_norm=True)
        self.path_mlp = MLP(vil_dim*(4) , 1024, vil_dim, 2, 0.5, layer_norm=True, batch_norm=False)
        # self.lstm = nn.LSTM(input_size=vil_dim, hidden_size=vil_dim // 2, num_layers=2, bidirectional=True, dropout=0.5 )
        # self.transformer = TransformerMapping(in_dim = vil_dim, out_dim = vil_dim)
        self.bilin_func = nn.Bilinear(vil_dim, vil_dim, 1)
        self.path_length = 3
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
    
    def edge_attention(self, edges):
        # 公式 (2) 所需，边上的用户定义函数
        # a = edges.data['weight'] * 2
        a = edges.src['h'] + edges.dst['h']
        return {'weight': a}
        # a = self.attn_fc(z2)
        # return {'weight' : F.leaky_relu(a)}

    def path_sampler(self, graph, node_features, batch_size, node_num):
        # graph = copy.deepcopy(G)
        node_ids = graph.ndata['id'] # entity ids in numberbatch
        node_idxs = torch.arange(0,len(node_ids)).cuda()
        # node_num = int(len(node_idxs) / batch_size)

        if 'biased_rw' in self.ablation:
            qv_mask = torch.logical_or(graph.ndata['v'], graph.ndata['q']).view(batch_size,-1)
            qv_node_idxs = torch.arange(node_num).expand(batch_size,-1)
            qv_node_idxs_list = [b[a].tolist() if a.sum()>0 else b.tolist() for a,b in zip(qv_mask, qv_node_idxs)]
            sampled_qv_node_idxs = [random.choices(qv, k=100) + random.choices(all.tolist(), k=200) for qv, all in zip(qv_node_idxs_list, qv_node_idxs)]
            # sampled_qv_node_idxs = [random.choices(qv, k=200) for qv in qv_node_idxs_list]


            # graph.ndata['h'] = torch.mean(node_features,1).data
            # print('graph.ndata ', graph.ndata['h'][0])
            # graph.update_all(fn.u_mul_e('h', 'weight', 'm'), fn.sum('m', 'h'))
            # graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h_sum'))
            # print('h_sum', graph.ndata['h_sum'].shape, graph.ndata['h'].shape)
            # graph.apply_edges(func=fn.u_add_v('h','h','weight'))

            graph.ndata['h'] = node_features.data
            graph.apply_edges(func=fn.u_mul_v('h','h','he'))
            graph.edata['weight'] *= torch.mean(graph.edata['he'], 1)
            graph.edata['weight'] = torch.clamp(graph.edata['weight'], min=0.01)
            batch_sampled_paths = []
            graphs = dgl.unbatch(graph)
            for i, g in enumerate(graphs):
                nx_g = dgl.to_networkx(g.cpu(), node_attrs=['id'], edge_attrs=['weight'])
                random_walk = RandomWalk(nx_g, walk_length=self.path_length, num_walks=1, p=2, q=1, workers=1, quiet=True, start_ids=sampled_qv_node_idxs[i])
                walklist = random_walk.walks
                walklist = [wl + [-1]*(self.path_length - len(wl)) for wl in walklist]
                batch_sampled_paths.extend(walklist)
                # print('-------------------')
                # print('qv_node_idxs', qv_node_idxs_list[i])
                # for w in walklist[:10]:
                #     print(w)
            path_num = int(len(batch_sampled_paths) / batch_size)
            batch_sampled_paths = torch.tensor(batch_sampled_paths).reshape(batch_size, path_num, -1).cuda()
            batch_sampled_paths += node_num * torch.arange(batch_size).unsqueeze(1).unsqueeze(2).expand(batch_size, path_num, self.path_length).cuda()
        else:
            # CASE 1: use any start point            
            # all_node_idxs_list = torch.split(node_idxs, batch_num_nodes.tolist())
            # all_node_idxs_list = [l[l>=0] for l in all_node_idxs_list]
            # sampled_qv_node_idxs = [torch.tensor(random.choices(qv, k=qv_path_num + q_path_num)) for qv in all_node_idxs_list]
            # sampled_qv_node_idxs = torch.cat(sampled_qv_node_idxs).cuda().tolist()
            # assert len(sampled_qv_node_idxs) == batch_size * (qv_path_num + q_path_num)

            # CASE 2: use q and v nodes as start point
            qv_node_idxs = torch.logical_or(graph.ndata['v'], graph.ndata['q']).int() * 2 - 1
            qv_node_idxs_list = (node_idxs * qv_node_idxs.int()).view(batch_size, -1)
            # qv_node_idxs_list = torch.split(node_idxs * qv_node_idxs.int(), batch_num_nodes.tolist())
            qv_node_idxs_list = [l[l>=0] for l in qv_node_idxs_list]
            qv_node_idxs_list = [qv if len(qv)>0 else all_n for qv, all_n in zip(qv_node_idxs_list, node_idxs.view(batch_size, -1))]
            q_node_idxs = graph.ndata['q'].int() * 2 - 1
            # q_node_idxs_list = torch.split(node_idxs * q_node_idxs.int(), batch_num_nodes.tolist())
            q_node_idxs_list = (node_idxs * q_node_idxs.int()).view(batch_size, -1)
            q_node_idxs_list = [l[l>=0] for l in q_node_idxs_list]
            q_node_idxs_list = [q if len(q)>0 else qv for q, qv in zip(q_node_idxs_list, qv_node_idxs_list)]
            sampled_qv_node_idxs = [torch.tensor(random.choices(qv, k=self.qv_path_num) + random.choices(q, k=self.q_path_num)) for qv, q in zip(qv_node_idxs_list, q_node_idxs_list)]
            sampled_qv_node_idxs = torch.cat(sampled_qv_node_idxs).cuda().tolist()
            assert len(sampled_qv_node_idxs) == batch_size * (self.qv_path_num + self.q_path_num)
            qv_sampled_paths, qv_sampled_edges, _  = dgl.sampling.random_walk(graph, sampled_qv_node_idxs, length=2, return_eids=True)
            sampled_all_node_idxs = node_idxs.tolist()
            qv_sampled_paths = qv_sampled_paths.reshape(batch_size, self.q_path_num+self.qv_path_num, -1).clone().detach()
            # qv_sampled_edges = qv_sampled_edges.reshape(batch_size, q_path_num+qv_path_num, -1).clone().detach()

            # # CASE 3: Remove duplicate paths
            # # # filter current paths
            # qv_sampled_paths_, qv_sampled_edges_= [], []
            # for i in torch.arange(batch_size):
            #     x = qv_sampled_paths[i,:,:]
            #     rand_perm = torch.randperm(x.shape[0])
            #     x_1 = x[rand_perm,:] 
            #     # e = qv_sampled_edges[i,:,:][rand_perm,:] 
            #     x, x_indices = torch.unique(x_1, dim=0, return_inverse=True)
                
            #     x_ = [y for y in x if [yy for yy in y if yy!=-1][-1] in a_node_idxs_list[i]]
            #     x = torch.stack(x_) if len(x_) > 0 else x 
            #     x = torch.tensor(x[:final_qv_num,:]).clone().detach()
            #     if x.shape[0] < final_qv_num:
            #         x = torch.cat((x, x_1[:final_qv_num - x.shape[0], :]), 0)
            #     assert x.shape[0] == final_qv_num
            #     # print(a_node_idxs_list[i], x[:10])
            #     qv_sampled_paths_.append(x)
            #     # change
            #     # qv_sampled_edges_.append(e[:final_qv_num,:])
            # qv_sampled_paths = torch.stack(qv_sampled_paths_)            
            # # qv_sampled_edges = torch.stack(qv_sampled_edges_)          
                
            # CASE 4: sample paths start with every node
            all_sampled_paths, all_sampled_edges, _ = dgl.sampling.random_walk(graph, sampled_all_node_idxs, length=2, return_eids=True)
            all_sampled_paths = torch.flip(all_sampled_paths,[1]).reshape(batch_size, -1, all_sampled_paths.shape[1])
            # all_sampled_edges = torch.flip(all_sampled_edges,[1]).reshape(batch_size, -1, all_sampled_edges.shape[1])
            batch_sampled_paths = torch.cat((qv_sampled_paths, all_sampled_paths), 1)
            # batch_sampled_edges = torch.cat((qv_sampled_edges, all_sampled_edges), 1)
            path_num = self.q_path_num + self.qv_path_num + node_num 

        # CASE 5: use a as end point, do not specify start point
        if 'close' in self.ablation or 'close_end' in self.ablation:
            path_num = self.q_path_num + self.qv_path_num + node_num
            a_node_idxs = graph.ndata['a'].int() * 2 - 1
            a_node_idxs_list = (node_idxs * a_node_idxs.int()).view(batch_size, -1)
            a_node_idxs_list = [l[l>=0] for l in a_node_idxs_list]
            sampled_a_node_idxs = [torch.tensor(random.choices(a, k=path_num)) for a in a_node_idxs_list]
            sampled_a_node_idxs = torch.cat(sampled_a_node_idxs).cuda().tolist()
            sampled_all_node_idxs = sampled_a_node_idxs

            all_sampled_paths, all_sampled_edges, _ = dgl.sampling.random_walk(graph, sampled_all_node_idxs, length=2, return_eids=True)
            batch_sampled_paths = torch.flip(all_sampled_paths,[1]).reshape(batch_size, -1, all_sampled_paths.shape[1])
            # batch_sampled_edges = torch.flip(all_sampled_edges,[1]).reshape(batch_size, -1, all_sampled_edges.shape[1])
            
        return batch_sampled_paths, path_num

    def path_encoder(self, vil_prediction, node_features, batch_sampled_paths, batch_size, node_num, path_num):
        sampled_paths = batch_sampled_paths.reshape(batch_size * path_num, -1)
        # sampled_edges = batch_sampled_edges.reshape(batch_size * path_num, -1)            

        sampled_paths_emb = node_features[sampled_paths] # (batch_size x path_num) x path_length x concept_dim # 640 x 3 x vil_dim
        sampled_paths_emb[(sampled_paths == -1)] = torch.zeros_like(node_features[0])
        # sampled_edges_emb = edge_features[sampled_edges]
        # sampled_edges_emb[(sampled_edges == -1)] = torch.zeros_like(edge_features[0])
        # sampled_paths_emb = torch.cat((sampled_edges_emb, sampled_paths_emb), 1)

        # Get final_qv_num of paths
        end_point_idxs = torch.stack([[x for x in path if x!=-1][-1] for path in sampled_paths])
        start_point_idxs = torch.stack([[x for x in path if x!=-1][0] for path in sampled_paths])
        batch_end_point_idxs = end_point_idxs.reshape(batch_size, path_num) % node_num
        # print(batch_end_point_idxs)

        # Pad the vil feature to first node

        batch_vil = vil_prediction.unsqueeze(1).expand(batch_size, path_num, vil_prediction.shape[1])
        cat_vil = batch_vil.reshape(-1,vil_prediction.shape[1]).unsqueeze(1)
        sampled_paths_emb = torch.cat((cat_vil, sampled_paths_emb), 1)  # 640 x (1+3) x vil_dim
        # batch_vil_pred_t = pooled_output_t.unsqueeze(1).expand(batch_size, path_num, vil_prediction.shape[1])
        # batch_vil_pred_v = pooled_output_v.unsqueeze(1).expand(batch_size, path_num, vil_prediction.shape[1])
        # cat_vil_t = batch_vil_pred_t.reshape(-1, vil_prediction.shape[1]).unsqueeze(1)
        # cat_vil_v = batch_vil_pred_v.reshape(-1, vil_prediction.shape[1]).unsqueeze(1)
        # sampled_paths_emb = torch.cat((cat_vil_t, cat_vil_v, cat_vil, sampled_paths_emb), 1)  # 640 x (1+3) x vil_dim

        # 1. MLP
        path_outs = sampled_paths_emb.reshape(batch_size*path_num, -1)
        path_outs = self.path_mlp(path_outs)
        # 2. Transformer
        # path_outs = self.transformer(sampled_paths_emb) # 3, 12800, 128
        # 3. LSTM 
        # path_outs, _ = self.lstm(sampled_paths_emb.transpose(0, 1))# length x batch x ...
        # path_outs = path_outs.transpose(0, 1)[:,-1,:] # .mean(dim=1) # # (batch_size x path_num) x 2048

        batch_path_outs = path_outs.reshape(batch_size, path_num, path_outs.shape[1]) # batch_size x path_num x lstm*2

        if 'path_pool' in self.ablation:
            end_point_sorted_, end_point_sorted_indices = torch.sort(end_point_idxs) # (batch_size x path_num) x 1
            end_point_sorted = [list(j) for i, j in groupby(end_point_sorted_.tolist())]
            end_point_sorted_splitidx = torch.cumsum(torch.tensor([len(x) for x in end_point_sorted]), dim=0)
            assert len(end_point_sorted) == batch_size * node_num
            # sampled_paths_sorted = sampled_paths[end_point_sorted_indices]

            path_outs_sorted = path_outs[end_point_sorted_indices]
            # query_vec = vil_prediction.unsqueeze(1).expand(batch_size, node_num, vil_prediction.shape[1]).reshape(-1,vil_prediction.shape[1])
            # query_vec = query_vec[end_point_sorted_indices]
            # idx2paths = {x.item():[-1] for x in node_ids}
            pooled_path_vecs = node_features
            # pooled_path_vecs = torch.zeros(len(node_ids), query_vec.shape[1]).cuda()  # node x 2048
            for cur_start, cur_end, end_pt in zip(torch.cat((torch.tensor([0]), end_point_sorted_splitidx)), end_point_sorted_splitidx, end_point_sorted):
                path_out = path_outs_sorted[cur_start:cur_end,:]
                # path_scores = torch.mv(path_out, query_vec[end_pt[0]])
                # path_att_scores = F.softmax(path_scores, dim=0)
                # pooled_path_vecs[end_pt[0]] = torch.mv(torch.t(path_out), path_att_scores)
                pooled_path_vecs[end_pt[0]] = torch.max(path_out, 0).values
                # idx2paths[node_ids[end_pt[0]].item()] = sampled_paths_sorted[cur_start + torch.argmax(path_att_scores)]

            batch_path_outs = pooled_path_vecs.reshape(batch_size, -1, pooled_path_vecs.shape[1]) #batch x node x vil_dim
        return batch_path_outs, batch_end_point_idxs

    def forward(self, graph, vil_prediction, node_features):
        (batch_size, node_num) = node_features.shape[:2]
        node_features = node_features.reshape(-1, node_features.shape[2])
        batch_sampled_paths, path_num = self.path_sampler(graph, node_features, batch_size, node_num)
        batch_path_outs, batch_end_point_idxs = self.path_encoder(vil_prediction, node_features, batch_sampled_paths, batch_size, node_num, path_num)        
        batch_vil = vil_prediction.unsqueeze(1).expand(-1, node_num if 'path_pool' in self.ablation else path_num, vil_prediction.shape[1])
        logits = self.bilin_func(batch_vil, batch_path_outs).squeeze()
        # logits = self.fc(batch_path_outs).squeeze()
        return logits, batch_path_outs, batch_sampled_paths, batch_end_point_idxs




class GraphRelationEncoder(nn.Module):
    def __init__(self, k, n_type, n_head, n_basis, n_layer, input_size, hidden_size, sent_dim,
                 att_dim, att_layer_num, dropout, diag_decompose, eps=1e-20, ablation=None):
        super().__init__()
        self.layers = nn.ModuleList([GraphRelationLayer(k=k, n_type=n_type, n_head=n_head, n_basis=n_basis,
                                                        input_size=input_size, hidden_size=hidden_size, output_size=input_size,
                                                        sent_dim=sent_dim, att_dim=att_dim, att_layer_num=att_layer_num,
                                                        dropout=dropout, diag_decompose=diag_decompose, eps=eps,
                                                        ablation=ablation) for _ in range(n_layer)])

    def decode(self, end_ids, A):
        bs = end_ids.size(0)
        k = self.layers[0].message_passing.k
        full_path_ids = end_ids.new_zeros((bs, k * 2 * len(self.layers) + 1))
        full_path_ids[:, 0] = end_ids
        full_path_lengths = end_ids.new_ones((bs,))
        for layer in self.layers[::-1]:
            path_ids, path_lengths = layer.decode(end_ids, A)
            for i in range(bs):
                prev_l = full_path_lengths[i]
                inc_l = path_lengths[i]
                path = path_ids[i]
                assert full_path_ids[i, prev_l - 1] == path[inc_l - 1]
                full_path_ids[i, prev_l:prev_l + inc_l - 1] = path_ids[i, :inc_l - 1].flip((0,))
                full_path_lengths[i] = prev_l + inc_l - 1
        for i in range(bs):
            full_path_ids[i, :full_path_lengths[i]] = full_path_ids[i, :full_path_lengths[i]].flip((0,))
        return full_path_ids, full_path_lengths

    def forward(self, S, H, A, node_type_ids, cache_output=False):
        """
        S: tensor of shape (batch_size, d_sent)
            sentence vectors from an encoder
        H: tensor of shape (batch_size, n_node, d_node)
            node features from the previous layer
        A: tensor of shape (batch_size, n_head, n_node, n_node)
            adjacency matrices, if A[:, :, i, j] == 1 then message passing from j to i is allowed
        node_type_ids: long tensor of shape (batch_size, n_node)
            0 == question node; 1 == answer node: 2 == intermediate node
        """
        for layer in self.layers:
            H = layer(S, H, A, node_type_ids, cache_output=cache_output)
        return H


class GraphRelationLayer(nn.Module):
    def __init__(self, k, n_type, n_head, n_basis, input_size, hidden_size, output_size, sent_dim,
                 att_dim, att_layer_num, dropout=0.1, diag_decompose=False, eps=1e-20, ablation=None):
        super().__init__()
        assert input_size == output_size
        self.ablation = ablation

        if 'no_typed_transform' not in self.ablation:
            self.typed_transform = TypedLinear(input_size, hidden_size, n_type)
        else:
            assert input_size == hidden_size

        self.path_attention = PathAttentionLayer(n_type, n_head, sent_dim, att_dim, att_layer_num, dropout, ablation=ablation)
        self.message_passing = MultiHopMessagePassingLayer(k, n_head, hidden_size, diag_decompose, n_basis, eps=eps, ablation=ablation)
        self.aggregator = Aggregator(sent_dim, hidden_size, ablation=ablation)

        self.Vh = nn.Linear(input_size, output_size)
        self.Vz = nn.Linear(hidden_size, output_size)

        self.activation = GELU()
        self.dropout = nn.Dropout(dropout)

    def decode(self, end_ids, A):
        ks = self.len_attn.argmax(2)  # (bs, n_node)
        if 'detach_s_agg' not in self.ablation:
            ks = ks + 1
        ks = ks.gather(1, end_ids.unsqueeze(-1)).squeeze(-1)  # (bs,)
        path_ids, path_lenghts = self.message_passing.decode(end_ids, ks, A, self.start_attn, self.uni_attn, self.trans_attn)
        return path_ids, path_lenghts

    def forward(self, S, H, A, node_type, cache_output=False):
        """
        S: tensor of shape (batch_size, d_sent)
            sentence vectors from an encoder
        H: tensor of shape (batch_size, n_node, d_node)
            node features from the previous layer
        A: tensor of shape (batch_size, n_head, n_node, n_node)
            adjacency matrices, if A[:, :, i, j] == 1 then message passing from j to i is allowed
        node_type: long tensor of shape (batch_size, n_node)
            0 == question node; 1 == answer node: 2 == intermediate node
        """

        if 'no_typed_transform' not in self.ablation:
            X = self.typed_transform(H, node_type)
        else:
            X = H

        start_attn, end_attn, uni_attn, trans_attn = self.path_attention(S, node_type)

        Z_all = self.message_passing(X, A, start_attn, end_attn, uni_attn, trans_attn)
        Z_all = torch.stack(Z_all, 2)  # (bs, n_node, k, h_size) or (bs, n_node, k+1, h_size)
        Z, len_attn = self.aggregator(S, Z_all)

        if cache_output:  # cache intermediate ouputs for decoding
            self.start_attn, self.uni_attn, self.trans_attn = start_attn, uni_attn, trans_attn
            self.len_attn = len_attn  # (bs, n_node, k)

        if 'early_relu' in self.ablation:
            output = self.Vh(H) + self.activation(self.Vz(Z))
        else:
            output = self.activation(self.Vh(H) + self.Vz(Z))

        output = self.dropout(output)
        return output


class GraphRelationNet(nn.Module):
    def __init__(self, k, n_type, n_basis, n_layer, sent_dim, diag_decompose,
                 n_concept, n_relation, concept_dim, concept_in_dim, n_attention_head,
                 fc_dim, n_fc_layer, att_dim, att_layer_num, p_emb, p_gnn, p_fc,
                 pretrained_concept_emb=None, freeze_ent_emb=True, ablation=None,
                 init_range=0.02, eps=1e-20, use_contextualized=False, do_init_rn=False, do_init_identity=False):
        super().__init__()
        self.ablation = ablation
        self.init_range = init_range
        self.do_init_rn = do_init_rn
        self.do_init_identity = do_init_identity

        n_head = 1 if 'no_rel' in self.ablation else n_relation

        self.concept_emb = CustomizedEmbedding(concept_num=n_concept, concept_out_dim=concept_dim,
                                               use_contextualized=use_contextualized, concept_in_dim=concept_in_dim,
                                               pretrained_concept_emb=pretrained_concept_emb, freeze_ent_emb=freeze_ent_emb)

        self.gnn = GraphRelationEncoder(k=k, n_type=n_type, n_head=n_head, n_basis=n_basis, n_layer=n_layer,
                                        input_size=concept_dim, hidden_size=concept_dim, sent_dim=sent_dim,
                                        att_dim=att_dim, att_layer_num=att_layer_num, dropout=p_gnn,
                                        diag_decompose=diag_decompose, eps=eps, ablation=ablation)

        if 'early_trans' in self.ablation:
            self.typed_transform = TypedLinear(concept_dim, concept_dim, n_type=n_type)

        if 'typed_pool' in self.ablation and 'early_trans' not in self.ablation:
            self.pooler = TypedMultiheadAttPoolLayer(n_attention_head, sent_dim, concept_dim, n_type=n_type)
        else:
            self.pooler = MultiheadAttPoolLayer(n_attention_head, sent_dim, concept_dim)

        self.fc = MLP(concept_dim + sent_dim, fc_dim, 1, n_fc_layer, p_fc, layer_norm=True)

        self.dropout_e = nn.Dropout(p_emb)
        self.dropout_fc = nn.Dropout(p_fc)

        if init_range > 0:
            self.apply(self._init_weights)

        if pretrained_concept_emb is not None and not use_contextualized:
            self.concept_emb.emb.weight.data.copy_(pretrained_concept_emb)

    def _init_rn(self, module):
        if hasattr(module, 'typed_transform'):
            h_size = module.typed_transform.out_features
            half_h_size = h_size // 2
            bias = module.typed_transform.bias
            new_bias = bias.data.clone().detach().view(-1, h_size)
            new_bias[:, :half_h_size] = 1
            bias.data.copy_(new_bias.view(-1))

    def _init_identity(self, module):
        if module.diag_decompose:
            module.w_vs.data[:, :, -1] = 1
        elif module.n_basis == 0:
            module.w_vs.data[:, -1, :, :] = torch.eye(module.w_vs.size(-1), device=module.w_vs.device)
        else:
            print('Warning: init_identity not implemented for n_basis > 0')
            pass

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.init_range)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, MultiHopMessagePassingLayer):
            if 'fix_scale' in self.ablation:
                module.w_vs.data.normal_(mean=0.0, std=np.sqrt(np.pi / 2))
            else:
                module.w_vs.data.normal_(mean=0.0, std=self.init_range)
            if hasattr(module, 'w_vs_co'):
                getattr(module, 'w_vs_co').data.fill_(1.0)
            if self.do_init_identity:
                self._init_identity(module)
        elif isinstance(module, PathAttentionLayer):
            if hasattr(module, 'trans_scores'):
                getattr(module, 'trans_scores').data.zero_()
        elif isinstance(module, GraphRelationLayer) and self.do_init_rn:
            self._init_rn(module)

    def decode(self):
        bs, _, n_node, _ = self.adj.size()
        end_ids = self.pool_attn.view(-1, bs, n_node)[0, :, :].argmax(-1)  # use only the first head if multi-head attention
        path_ids, path_lengths = self.gnn.decode(end_ids, self.adj)

        # translate local entity ids (0~200) into global eneity ids (0~7e5)
        entity_ids = path_ids[:, ::2]  # (bs, ?)
        path_ids[:, ::2] = self.concept_ids.gather(1, entity_ids)
        return path_ids, path_lengths

    def forward(self, sent_vecs, concept_ids, node_type_ids, adj_lengths, adj, emb_data=None, cache_output=False):
        """
        sent_vecs: (batch_size, d_sent)
        concept_ids: (batch_size, n_node)
        adj: (batch_size, n_head, n_node, n_node)
        adj_lengths: (batch_size,)
        node_type_ids: (batch_size, n_node)
            0 == question node; 1 == answer node: 2 == intermediate node

        returns: (batch_size, 1)
        """
        gnn_input = self.dropout_e(self.concept_emb(concept_ids, emb_data))
        if 'no_ent' in self.ablation:
            gnn_input[:] = 1.0
        if 'no_rel' in self.ablation:
            adj = adj.sum(1, keepdim=True)
        gnn_output = self.gnn(sent_vecs, gnn_input, adj, node_type_ids, cache_output=cache_output)

        mask = torch.arange(concept_ids.size(1), device=adj.device) >= adj_lengths.unsqueeze(1)
        if 'pool_qc' in self.ablation:
            mask = mask | (node_type_ids != 0)
        elif 'pool_all' in self.ablation:
            mask = mask
        else:  # default is to perform pooling over all the answer concepts (pool_ac)
            mask = mask | (node_type_ids != 1)
        mask[mask.all(1), 0] = 0  # a temporary solution to avoid zero node

        if 'early_trans' in self.ablation:
            gnn_output = self.typed_transform(gnn_output, type_ids=node_type_ids)

        if 'detach_s_pool' in self.ablation:
            sent_vecs_for_pooler = sent_vecs.detach()
        else:
            sent_vecs_for_pooler = sent_vecs

        if 'typed_pool' in self.ablation and 'early_trans' not in self.ablation:
            graph_vecs, pool_attn = self.pooler(sent_vecs_for_pooler, gnn_output, mask, type_ids=node_type_ids)
        else:
            graph_vecs, pool_attn = self.pooler(sent_vecs_for_pooler, gnn_output, mask)

        if cache_output:  # cache for decoding
            self.concept_ids = concept_ids
            self.adj = adj
            self.pool_attn = pool_attn

        concat = self.dropout_fc(torch.cat((graph_vecs, sent_vecs), 1))
        logits = self.fc(concat)
        return logits, pool_attn

class MIEstimator(nn.Module):
    def __init__(self, embd_size):
        super(MIEstimator, self).__init__()
        self.bilin_func = nn.Bilinear(embd_size, embd_size, 1)
        self.linear = nn.Linear(embd_size, 1)
        self.pool = nn.Linear(3, 1)
        sigma = 0.4
        self.sigma = 2*sigma**2
        self.epsilon = 1e-10
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward_(self, uff, vff):
        sc_pos = self.bilin_func(uff, vff)
        # selu = nn.SELU()
        return (sc_pos)

    # path-base Mutual Information
    def path_based_mi(self, feats):
        # feats: path_num x path_length x f_dim
        # sc_pos = self.bilin_func(uff, vff)
        path_num, path_length, f_dim = feats.size()
        P_1l = torch.exp(-0.5*(feats / 2*self.sigma**2).pow(2)) # path_num x path_length x f_dim
        feats = torch.exp(feats)
        mutual_information = []
        for i in range(path_num):
            path = feats[i,:,:]
            p_1l = P_1l[i,:,:]
            J_1l = torch.ones(f_dim, f_dim).cuda()
            for j in range(1,path_length):
                J_xy = (path[j-1,:].unsqueeze(0) * path[j,:].unsqueeze(0).t())
                J_1l = torch.mm(J_1l, J_xy) # f_dim x f_dim
            log_xz = torch.log2(J_1l / p_1l[-1,:])  # f_dim x f_dim
            mi = torch.mm(p_1l,log_xz)# path_length x f_dim
            mi = self.linear(mi).sum(0)
            mutual_information.append(mi)
        mutual_information = torch.stack(mutual_information)
        
        # selu = nn.SELU()
        return (mutual_information)

    # path-base Mutual Information
    def path_based_mi2(self, feats):
        # feats: path_num x path_length x f_dim
        # sc_pos = self.bilin_func(uff, vff)
        path_num, path_length, f_dim = feats.size()
        P_1l = torch.exp(-0.5*(feats / 2*self.sigma**2).pow(2)) # path_num x path_length x f_dim
        feats = torch.exp(feats)
        y = y - y.mean(dim=3, keepdim=True)
        mutual_information = []
        for i in range(path_num):
            y = feats[i,:,:]
            p_1l = P_1l[i,:,:]
            J_1l = torch.ones(f_dim, f_dim).cuda()

            # Subtract mean
            y = y - y.mean(dim=3, keepdim=True)
            p = p - p.mean(dim=3, keepdim=True)

            # Covariances
            y_cov = y @ transpose(y)
            p_cov = p @ transpose(p)
            y_p_cov = y @ transpose(p)

            log_xz = torch.log2(J_1l / p_1l[-1,:])  # f_dim x f_dim
            mi = torch.mm(p_1l,log_xz)# path_length x f_dim
            mi = self.linear(mi).sum(0)
            mutual_information.append(mi)
        mutual_information = torch.stack(mutual_information)
        
        # selu = nn.SELU()
        return (mutual_information)


    # Mutual Information between end_point and start_point
    def path_node_mi(self, feats):
        # feats: path_num x path_length x f_dim
        # sc_pos = self.bilin_func(uff, vff)
        path_num, path_length, f_dim = feats.size()
        mutual_information = []
        for i in range(1,path_length):
            uff = feats[:,i-1,:]
            vff = feats[:,i,:]
            mutual_information.append(self.bilin_func(uff, vff))

        mutual_information = self.pool(torch.cat(mutual_information, 1))
        # selu = nn.SELU()
        return (mutual_information)
    
    # Mutual Information between end_point and start_point
    def forward(self, uff, vff):
        sc_pos = self.bilin_func(uff, vff)
        # selu = nn.SELU()
        return (sc_pos)
    
    def forward__(self, feats):
        mutual_information = self.node_mi(feats)
        return (mutual_information)

    def jointPdf(self, kernel_values1, kernel_values2):
        joint_kernel_values = torch.matmul(kernel_values1.transpose(1, 2), kernel_values2) 
        normalization = torch.sum(joint_kernel_values, dim=(1,2)).view(-1, 1, 1) + self.epsilon
        pdf = joint_kernel_values / normalization
        return pdf


    def nodeMI(self, uff, vff):
        # feats: path_num x path_length x f_dim
        # sc_pos = self.bilin_func(uff, vff)
        kernel_values1 = torch.exp(-0.5*(uff / 2*self.sigma**2).pow(2))
        pdf1 = torch.mean(kernel_values1, dim=1) # path_num x f_dim
        normalization1 = torch.sum(pdf1, dim=1).unsqueeze(1) + self.epsilon # path_num x 1
        pdf1 = pdf1 / normalization1

        kernel_values2 = torch.exp(-0.5*(vff / 2*self.sigma**2).pow(2))
        pdf2 = torch.mean(kernel_values2, dim=1) # path_num x f_dim
        normalization2 = torch.sum(pdf2, dim=1).unsqueeze(1) + self.epsilon # path_num x 1
        pdf2 = pdf2 / normalization2

        pdf_x1x2 = self.jointPdf(kernel_values1, kernel_values2)

        H_x1 = -torch.sum(pdf1*torch.log2(pdf1 + self.epsilon), dim=1)
        H_x2 = -torch.sum(pdf2*torch.log2(pdf2 + self.epsilon), dim=1)
        H_x1x2 = -torch.sum(pdf_x1x2*torch.log2(pdf_x1x2 + self.epsilon), dim=(1,2))
        
        mutual_information = H_x1 + H_x2 - H_x1x2

        # selu = nn.SELU()
        return (mutual_information)

    def forward___(self, uff, vff):
        mi = []
        for i in range(uff.shape[0]):
            for j in range(uff.shape[1]):
                mi.append(self.nodeMI(uff[i,j,:].unsqueeze(0).unsqueeze(0), vff[i,j,:].unsqueeze(0).unsqueeze(0)))
        mi = torch.cat(mi).view(uff.size()[:2])
        # print(mi.shape)
        return mi



def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


def transpose(x):
    return x.transpose(-2, -1)


def inverse(x):
    return torch.inverse(x)


def log_trace(x):
    x = torch.cholesky(x)
    diag = torch.diagonal(x, dim1=-2, dim2=-1)
    return 2 * torch.sum(torch.log(diag + 1e-8), dim=-1)


def log_det(x):
    return torch.logdet(x)