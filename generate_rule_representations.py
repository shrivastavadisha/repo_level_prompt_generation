import numpy as np
import os
import torch
import argparse
import random
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
from rule_representation_data import *
from torch import FloatTensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def setup_args():
  """
  Description: Takes in the command-line arguments from user
  """
  parser = argparse.ArgumentParser()

  # data related hyperparameters
  parser.add_argument("--seed", type=int, default=9, help="seed for reproducibility")
  parser.add_argument("--input_data_dir", type=str, default='rule_classifier_data', help="base directory for the data")
  parser.add_argument("--data_split", type=str, default='val', help="train, val, test")

  # model related hyperparameters
  parser.add_argument("--emb_model_type", type=str, default='codebert', help="model to obtain embedding from")
  parser.add_argument("--repo", type=str, default='jata4test', help="model to obtain embedding from")
  return parser.parse_args()

if __name__ == '__main__':

  args = setup_args()

  #Fix seeds
  np.random.seed(args.seed)
  os.environ['PYTHONHASHSEED'] = str(args.seed)
  torch.manual_seed(args.seed)
  random.seed(args.seed)


  # Define dataloaders
  kwargs = {'num_workers': 8, 'pin_memory': True} if device=='cuda' else {}
  tokenizer = set_tokenizer(args.emb_model_type)
  base_dir = os.path.join(args.input_data_dir, args.data_split)
  dataset = RuleReprDataset(base_dir, emb_model_type = args.emb_model_type, tokenizer=tokenizer)
  #for repo in os.listdir(base_dir):
  start, end = dataset.get_start_index(args.repo, start_offset=0, interval=0)
  print(args.repo, start, end)
  for batch, (rule_context, hole, repo_name) in enumerate(dataset):
     if batch > end:
       break
     if repo_name == args.repo:
       save_dir = os.path.join(base_dir, repo_name, args.emb_model_type +'_mod')
       os.makedirs(save_dir, exist_ok=True)
       rule_representation = {hole: rule_context}
       with open(os.path.join(save_dir, str(batch)) , 'wb') as f:
         pickle.dump(rule_representation, f)
