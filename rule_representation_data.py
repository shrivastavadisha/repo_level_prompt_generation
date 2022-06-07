import os
import numpy as np
import torch
import pickle
from torch.utils.data import Dataset
from transformers import AutoModel
from transformers import GPT2TokenizerFast, AutoTokenizer
from transformers import pipeline
from utils import *
from data_utils import RuleDatasetUtils

class RuleReprDataset(Dataset):

  def __init__(self, input_data_dir, emb_model_type, tokenizer):
    # get all relevant files (in raw form)
    files = []
    oracles = {}
    hole_datas = {}
    parse_datas = {}
    all_duplicate_files = []
    data_type = input_data_dir.split('/')[-1]
    for dp, dn, filenames in os.walk(input_data_dir):
      for f in filenames:
        if f == 'hole_data':
          hole_data = pickle.load(open(os.path.join(dp, f), 'rb'))
          hole_data = self.update_dict(hole_data, data_type)
          hole_datas = {**hole_datas, **hole_data}
        if f == 'parsed_data':
          parse_data = pickle.load(open(os.path.join(dp, f), 'rb'))
          parse_data = self.update_dict(parse_data, data_type)
          parse_datas = {**parse_datas, **parse_data}
        if f == 'duplicates':
          duplicate_files = open(os.path.join(dp, f), 'r').readlines()
          all_duplicate_files.extend([x.strip() for x in duplicate_files]) 
        if os.path.splitext(f)[1] == '.java':          
          files.append(os.path.join(dp, f))
    print(len(all_duplicate_files))
    self.holes = []
    for file in files:
      if file in hole_datas and \
        file not in all_duplicate_files and \
        not file.startswith('rule_classifier_data/train/rsbotownversion/trunk/scripts/'):
        for (l,c) in hole_datas[file]:
          hole_identity = file + '_' + str(l) + '_' + str(c)
          self.holes.append(hole_identity)

    print(len(self.holes))
    self.num_rules = len(combined_to_index)
    self.tokenizer = tokenizer

    self.parse_datas = parse_datas
    self.model_max_length = self.tokenizer.model_max_length
    self.rule_repr_cache = {}
    self.emb_model_type = emb_model_type
    self.set_embedding_model()
    self.repr_size = 768
    self.start = 0
    self.end = 500000

  def update_dict(self, dic, data_type):
    mod_dic = {}
    for k,v in dic.items():
      mod_k = '/'. join(['rule_classifier_data', data_type] + k.split('/')[2:])
      mod_dic[mod_k] = v
    return mod_dic

  def __len__(self):
    return len(self.holes)

  def __getitem__(self, idx):
    if idx >=self.start and idx <= self.end:
      return self.generate_data(self.holes[idx])
    else:
      return None, None, None

  def get_start_index(self, repo, start_offset=0, interval=0):
    count=0
    for i in range(len(self.holes)):
      hole = self.holes[i]
      repo_name = hole.split('/')[2]
      if repo_name == repo:
        count+=1
        repo_end_idx = i

    self.start = repo_end_idx - count + 1
    self.start = self.start + start_offset
    if interval!=0 :
      self.end = self.start + interval
    else:
      self.end = repo_end_idx
    return self.start, self.end

  def is_clear_cache(self):
    if len(self.rule_repr_cache) < 30:
      self.clear_cache = False
    else:
      self.clear_cache = True
      self.rule_repr_cache = {}

  def get_representation(self, inputs, mask):
    outputs = self.emb_model(inputs, attention_mask=mask)
    try:
        representation = outputs.pooler_output
    except:
        representation = outputs.last_hidden_state[:, 0]
    #print(representation.shape)
    return representation

  def get_context_embedding(self, context, attn_mask):
    context_embedding = self.get_representation(context, attn_mask)
    return context_embedding

  def get_rule_context(self, file, hole_pos):
    self.is_clear_cache() 
    rule_dataset_util = RuleDatasetUtils(file, self.parse_datas, hole_pos, self.tokenizer)
    rule_prompts, rule_indexes = rule_dataset_util.get_all_rules_context()
    rule_contexts = self.tokenizer(rule_prompts, truncation=True, padding='max_length')
    rule_inputs = torch.tensor(rule_contexts['input_ids'])
    rule_masks = torch.tensor(rule_contexts['attention_mask'])
    rule_indexes = torch.tensor(rule_indexes)

    # remove rules that are already cached
    rule_prompts = self.tokenizer.batch_decode(rule_inputs)
    filtered_rule_context = []
    filtered_rule_mask = []
    filtered_rule_prompts = []
    filtered_rule_indexes = []
    for i in range(len(rule_prompts)):
      rule_prompt = rule_prompts[i]
      if rule_prompt not in self.rule_repr_cache:
        filtered_rule_indexes.append(rule_indexes[i])
        filtered_rule_context.append(rule_inputs[i])
        filtered_rule_mask.append(rule_masks[i])
        filtered_rule_prompts.append(rule_prompt)

    if filtered_rule_context:
      filtered_rule_context = torch.stack(filtered_rule_context)
      filtered_rule_mask = torch.stack(filtered_rule_mask)

      # get rule representations
      filtered_representations = self.get_context_embedding(filtered_rule_context, filtered_rule_mask)
      # cache the representations
      for i in range(len(filtered_representations)):
        f_repr = filtered_representations[i]
        rule_prompt = filtered_rule_prompts[i]
        self.rule_repr_cache[rule_prompt] = f_repr

    # obtain full representations
    keys = []
    j = 0
    for ind in range(self.num_rules):
      if ind in rule_indexes:
        prompt = rule_prompts[j]
        j+=1
        if prompt in self.rule_repr_cache:
          keys.append(self.rule_repr_cache[prompt])
        else:
          keys.append(torch.zeros(self.repr_size))
      else:
        keys.append(torch.zeros(self.repr_size))

    keys = torch.stack(keys)
    return keys
  
  def generate_data(self, hole):

    hole_parts = hole.split('/')[-1].split('_')
    repo_name = hole.split('/')[2]
    if len(hole_parts) > 3:
        new_hole_parts = hole_parts[:-2]
        filename = '_'.join(new_hole_parts)
        filename = [filename]
    else:
        filename = [hole_parts[0]]
    file = '/'.join(hole.split('/')[:-1] + filename)
    hole_pos = (int(hole_parts[-2]), int(hole_parts[-1]))
    rule_contexts = self.get_rule_context(file, hole_pos)
    return rule_contexts, hole, repo_name

  def set_tokenizer(self):
    if self.emb_model_type == 'codebert':
      self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    if self.emb_model_type == 'graphcodebert':
      self.tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
    if self.emb_model_type == 'gpt-2':
      self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
      self.tokenizer.pad_token = self.tokenizer.eos_token

  def set_embedding_model(self):
    # CodeBERT
    if self.emb_model_type == 'codebert':
      self.emb_model = AutoModel.from_pretrained("microsoft/codebert-base")
    # GraphCodeBERT
    if self.emb_model_type == 'graphcodebert':
      self.emb_model = AutoModel.from_pretrained("microsoft/graphcodebert-base")

def set_tokenizer(emb_model_type):

  if emb_model_type == 'codebert':
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
  if emb_model_type == 'graphcodebert':
    tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
  if emb_model_type == 'gpt-2':
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
  return tokenizer


