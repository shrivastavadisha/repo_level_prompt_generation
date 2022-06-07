import os
import numpy as np
import torch
import pickle
from torch.utils.data import Dataset
from transformers import GPT2TokenizerFast, AutoTokenizer
from transformers import DataCollatorWithPadding
from utils import *
import re


class RuleDataset(Dataset):

  def __init__(self, input_data_dir, tokenizer=None, emb_model_type='codebert'):

    self.input_data_dir = input_data_dir
    self.tokenizer = tokenizer
    data_type = input_data_dir.split('/')[-1]
    oracles = {}
    self.data_files = []
    for dp, dn, filenames in os.walk(input_data_dir):
      for f in filenames:
        if f == 'capped_oracle_10000':
          oracle = pickle.load(open(os.path.join(dp, f), 'rb'))
          oracle = self.update_dict(oracle, data_type)
          oracles = {**oracles, **oracle} 

    for dp, dn, filenames in os.walk(input_data_dir):
      if dp.split('/')[-1] == 'capped_'+ emb_model_type + '_mod':
        for f in filenames:
          self.data_files.append(os.path.join(dp, f))

    self.oracles = oracles 
    #print(self.oracles.keys()) 
    print("oracle") 
    print(data_type, len(self.oracles))
    print("data_files")
    print(data_type, len(self.data_files))
    self.data_type = data_type
    self.num_combined = len(combined_to_index)

  def __len__(self):
    return len(self.data_files)

  def __getitem__(self, idx):
    return self.generate_data(self.data_files[idx])

  def update_dict(self, dic, data_type):
    if 'small_' in data_type:
      return dic
    else:
      mod_dic = {}
      for k,v in dic.items():
        mod_k = '/'. join(['rule_classifier_data', data_type] + k.split('/')[2:])
        mod_dic[mod_k] = v
      return mod_dic

  def generate_data(self, data_file):
    data = pickle.load(open(data_file, 'rb'))
    data = self.update_dict(data, self.data_type)
    for hole, rule_context in data.items():
      hole_context = self.get_hole_context(hole)
      if hole in self.oracles:
        combined = self.oracles[hole]['com']
        failure_flag = 1
      else:
        combined = np.zeros(self.num_combined)
        failure_flag = 0
      return hole_context, rule_context, combined, hole, failure_flag

  def get_hole_context(self, hole, num_of_prev_lines=2, num_of_post_lines=2):
    '''
    return the pre_context_len tokens from the current file based on codex tokenization from the position of the cursor
    '''
    hole_parts = hole.split('/')[-1].split('_')
    repo_name = hole.split('/')[2]
    if len(hole_parts) > 3:
        new_hole_parts = hole_parts[:-2]
        filename = '_'.join(new_hole_parts)
        filename = [filename]
    else:
        filename = [hole_parts[0]]
    file = '/'.join(hole.split('/')[:-1] + filename)
    pos = (int(hole_parts[-2]), int(hole_parts[-1]))

    pre_end = pos
    pre_start_line = pos[0] - num_of_prev_lines
    if pre_start_line < 0:
      pre_start_line = 0
    pre_start = (pre_start_line, 0)
    pre_hole_context = get_string(file, pre_start, pre_end)

    post_hole_context = ""
    if num_of_post_lines > 0:    
      file_lines = open(file, encoding="utf8", errors='backslashreplace').readlines()
      post_start_line = pos[0] + 1
      if post_start_line < len(file_lines):
        post_end_line = pos[0] + num_of_post_lines
        if post_end_line >= len(file_lines):
          post_end_line = len(file_lines) - 1
        post_start = (post_start_line, 0)
        post_end = (post_end_line, len(file_lines[post_end_line])) 
        post_hole_context = get_string(file, post_start, post_end)
    hole_context = post_hole_context + " " + pre_hole_context
    hole_context = self.tokenizer(hole_context, truncation=True)
    return hole_context

def collate_fn(data):
  hole_context, rule_contexts, gt_com, hole_id, failure_flag = zip(*data)
  hole_context = data_collator(hole_context)
  rule_contexts = torch.stack(rule_contexts, dim=0)
  #print("rule_contexts:", torch.sum(rule_contexts, dim=-1))
  gt_com = torch.FloatTensor(gt_com)
  failure_flag = torch.IntTensor(failure_flag)
  return hole_context['input_ids'], hole_context['attention_mask'], \
          rule_contexts, \
          gt_com, \
          hole_id, \
          failure_flag

def set_tokenizer(emb_model_type):
  global data_collator
  if emb_model_type == 'codebert':
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
  if emb_model_type == 'graphcodebert':
    tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
  if emb_model_type == 'gpt-2':
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
  return tokenizer



