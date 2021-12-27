# the baseline model:
# " [SEP] ".join(ingredients) -> token_ids --BERT--> token-wised encodings --MultiHeadAttention--> sentence-wised encodings --classifier--> prediction --log-likelihood--> loss

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoModel, AutoTokenizer

Constants = {
    "PAD": 0,
    "UNK": 100,
    "BOS": 101,
    "EOS": 102,
    "PAD_WORD": '[PAD]',
    "UNK_WORD": '[UNK]',
    "BOS_WORD": '[CLS]',
    "EOS_WORD": '[SEP]',
}

class Model(nn.Module):
    r""" the baseline model
    """
    def __init__(self, args):
        super().__init__()
        self.bert = AutoModel.from_pretrained(args.pretrained_model_name_or_path)
        if args.freeze_pretrained:
            self.freeze_bert()
        self.aggregator = args.agg_class(input_size = self.bert.config.hidden_size)
        self.head = args.head_class(args, self.bert.config)
        self.args = args
    
    def forward(self, input_ids, label, padding_mask=None):
        batch_size, input_len = input_ids.shape
        if padding_mask is None:
            padding_mask = input_ids.ne(Constants["PAD"])
        
        outputs = self.bert(input_ids)
        hidden_states = outputs.last_hidden_state
        aggregation = self.aggregator(hidden_states, padding_mask)
        loss, prediction = self.head(aggregation, label)
        return loss, prediction
    
    def freeze_bert(self, reverse=False):
        for name,para in self.bert.named_parameters():
            para.requires_grad = False if not reverse else True
    
    def __repr__(self):
        return super(Model, self).__repr__().replace(self.bert.__repr__().replace("\n", "\n  "), self.args.pretrained_model_name_or_path)

class Aggregator(nn.Module):
    r""" a static attention layer to aggregate the information from a length-variable sequence.
    """
    def __init__(self, input_size, hidden_size=512, num_heads=16):
        super().__init__()
        self.hidden_states_to_attention_scores = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=False),
            nn.Tanh(),
            nn.Linear(hidden_size, num_heads, bias=False)
        )
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
    def forward(self, hidden_states, attention_mask):
        r"""
        hidden_states: [batch_size, seq_len, input_size]
        attention_mask: [batch_size, seq_len]
        """
        logits = self.hidden_states_to_attention_scores(hidden_states)
        # [batch_size, seq_len, num_heads]
        
        attention = F.softmax(
            logits - 1e4*(1-attention_mask.float().cuda()).unsqueeze(2).repeat(1, 1, self.num_heads),
            dim = 1).transpose(2, 1)
        # [batch_size, num_heads, seq_len]
        
        aggregation = torch.mean(torch.bmm(attention, hidden_states), dim=1)
        # [batch_size, input_size]
        
        return aggregation

class BasicClassificationHead(nn.Module):
    def __init__(self, args, bert_config):
        super().__init__()
        self.MLP = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(bert_config.hidden_size, args.num_class)
        )
    def forward(self, aggregation, label):
        logits = self.MLP(aggregation)
        loss = F.cross_entropy(logits, label)
        pred = torch.argmax(logits, dim=-1)
        return loss, pred

class TwoLayerClassificationHead(BasicClassificationHead):
    def __init__(self, args, bert_config):
        super().__init__(args, bert_config)
        self.MLP = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(bert_config.hidden_size, bert_config.hidden_size),
            nn.GELU(),
            nn.Linear(bert_config.hidden_size, args.num_class)
        )

class ThreeLayerClassificationHead(BasicClassificationHead):
    def __init__(self, args, bert_config):
        super().__init__(args, bert_config)
        self.MLP = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(bert_config.hidden_size, bert_config.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(bert_config.hidden_size, bert_config.hidden_size),
            nn.GELU(),
            nn.Linear(bert_config.hidden_size, args.num_class)
        )

available_agg_classes = {agg_class.__name__:agg_class for agg_class in [Aggregator]}

available_head_classes = {head_class.__name__:head_class for head_class in [BasicClassificationHead, TwoLayerClassificationHead, ThreeLayerClassificationHead]}