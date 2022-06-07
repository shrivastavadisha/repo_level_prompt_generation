import numpy as np
import torch.nn.functional as F
from torch import nn
import torch
import math


class ScaledDotProductAttention(nn.Module):
  ''' Scaled Dot-Product Attention '''

  def __init__(self, temperature, attn_dropout=0.0): 
    super().__init__()
    self.temperature = temperature
    self.dropout = nn.Dropout(attn_dropout)

  def forward(self, q, k, mask=None):

    attn = torch.matmul(q / self.temperature, k.transpose(2, 3)) #(bs, num_heads, max_len_q, max_len_k)
    #print("attn:", attn.shape)

    if mask is not None:
      attn = attn.masked_fill(mask == 0, -1e9)

    attn = self.dropout(F.softmax(attn, dim=-1))#(bs, num_heads, max_len_q, max_len_k)
    return attn

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head=4, d_model=768, d_k=32, d_v=32,dropout=0.0, include_res_ln=True, return_att_weights=False):
      super(MultiHeadAttention, self).__init__()

      self.include_res_ln = include_res_ln
      self.return_att_weights = return_att_weights
      self.n_head = n_head
      self.d_k = d_k
      self.d_v = d_v
      self.d_model = d_model
      self.w_qs = nn.Linear(d_model, n_head * d_k)
      self.w_ks = nn.Linear(d_model, n_head * d_k)
      self.w_vs = nn.Linear(d_model, n_head * d_v)
      self.fc = nn.Linear(n_head * d_v, d_model)
      self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5, attn_dropout=dropout)
      self.dropout = nn.Dropout(dropout)
      self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, query, keys, values, mask):

      d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
      sz_b = query.size(0) #batch_size
      q = query #(bs, 1, d_model)
      k = keys #(bs, #rules, d_model)
      v = values #(bs, #rules, d_model)
      len_q, len_k, len_v = q.size(1), k.size(1), v.size(1) #(1, n_rules, n_rules)

      residual = q

      # Pass through the pre-attention projection: b x lq x (n*dv)
      # Separate different heads: b x lq x n x dv
      q = self.w_qs(q).view(sz_b, len_q, n_head, d_k) #(bs, 1, n_head, d_k)
      k = self.w_ks(k).view(sz_b, len_k, n_head, d_k) #(bs, n_rules, n_head, d_k)
      v = self.w_vs(v).view(sz_b, len_v, n_head, d_v) #(bs, n_rules , n_head, d_k)

      # Transpose for attention dot product: b x n x lq x dv
      q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2) #(bs, n_head, 1, d_k), (bs, n_head, n_rules, d_k), (bs, n_head, n_rules, d_k)

      if mask is not None:
        mask = mask.unsqueeze(1)   # For head axis broadcasting.

      # calculate attention scores
      attn = self.attention(q, k, mask)#(bs, num_heads, 1, n_rules)

      pred = torch.matmul(attn, v) #(bs, num_heads, 1, d_model)
      pred = pred.transpose(1, 2).contiguous().view(sz_b, len_q, -1) #(bs, 1, num_heads*d_model)
      pred = self.dropout(self.fc(pred)) #(bs, 1, d_model)
      if self.include_res_ln:
        pred += residual
        pred = self.layer_norm(pred)

      pred = pred.squeeze()
      attn = attn.squeeze()
      if self.return_att_weights:
        return pred, attn
      else:
        return pred, None

class PositionwiseFeedForward(nn.Module):
  ''' A two-feed-forward-layer module '''

  def __init__(self, d_in, d_hid, dropout=0.0):
    super().__init__()
    self.w_1 = nn.Linear(d_in, d_hid) # position-wise
    self.w_2 = nn.Linear(d_hid, d_in) # position-wise
    self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    residual = x
    x = self.w_2(F.relu(self.w_1(x)))
    x = self.dropout(x)
    x += residual
    x = self.layer_norm(x)
    return x

class BasicAggModel(nn.Module):

  def __init__(self, include_ff=True, include_res_ln=True, dropout=0.0, d_inner=2048, d_model=768, return_att_weights=False, n_head=8,
                      d_k=96, n_rules=63, device='cpu', is_dense_bias=True):

    super(BasicAggModel, self).__init__()

    self.include_ff = include_ff
    self.include_res_ln = include_res_ln
    self.d_inner = d_inner
    self.d_model = d_model
    self.n_rules = n_rules
    self.device = device
    self.return_att_weights = return_att_weights
    self.mha = MultiHeadAttention(n_head=n_head, d_k=d_k, d_v=d_k, dropout=dropout,\
                    return_att_weights=return_att_weights, include_res_ln=include_res_ln)
    self.ff = PositionwiseFeedForward(self.d_model, self.d_inner, dropout=dropout)
    self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
    self.dropout = nn.Dropout(p=dropout)

    self.rule_encoder = nn.Embedding(self.n_rules, self.d_model)
    if is_dense_bias:
      self.dense_proj = nn.Linear(d_model, n_rules)
    else:
      self.dense_proj = nn.Linear(d_model, n_rules, bias=False)

    for name, p in self.named_parameters():
      if name!='dense_proj.weight' and p.dim() > 1:
        nn.init.xavier_uniform_(p)


  def forward(self, query, keys, mask):

    query = torch.unsqueeze(query, 1) #(bs, 1, d_model)
    bs = query.size(0) #bs

    mask = torch.sum(mask, dim=-1) #(bs, n_rules)
    mask = mask.unsqueeze(1) #(bs, n_rules)

    values = keys

    query = self.layer_norm(query) #(bs, 1, d_model)
    keys = self.layer_norm(keys)    #(bs, n_rules, d_model)
    values = self.layer_norm(values) #(bs, n_rules, d_model)

    #Multi-Head Attention
    pred, att_weights = self.mha(query, keys, values, mask) #(bs, d_model), #(bs, num_heads, n_rules)

    #Positionwise FeedForward
    if self.include_ff:
      pred = self.ff(pred)#(bs, d_model)

    #Final projection to get logits
    pred = self.dense_proj(pred)#(bs, num_rules)

    if self.return_att_weights:
      return pred, att_weights
    else:
      return pred, None


