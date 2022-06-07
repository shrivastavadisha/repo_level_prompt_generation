import numpy as np
import os
import sys
import torch
import pickle
import argparse
import random
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from torch import nn
from tqdm import tqdm
from preprocessed_data import *
from model_preprocessed_data import RuleModel
from torch import FloatTensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def setup_args():
  """
  Description: Takes in the command-line arguments from user
  """
  parser = argparse.ArgumentParser()

  parser.add_argument("--seed", type=int, default=9, help="seed for reproducibility")
  parser.add_argument("--input_data_dir", type=str, default='rule_classifier_data', help="base directory for the data")
  parser.add_argument("--data_split", type=str, default='val', help="data_split")
  parser.add_argument("--model_dir", type=str, default='models/', help="base directory for storing the models")
  parser.add_argument("--batch_size", type=int, default=32, help="batch size for training the classifier")
  return parser.parse_args()

def get_accuracy(pred, gold, mask):
  pred = pred.masked_fill(mask==0, 0)
  max_idx = torch.argmax(pred, dim=1, keepdim=True)
  rounded_pred = torch.round(pred)
  max_idx_gold_vals = torch.gather(gold, 1, max_idx)
  mean_highest_success_correct = (max_idx_gold_vals == 1).to(dtype=torch.float).mean()
  return mean_highest_success_correct, pred

def get_prediction(rule_model, info):
  pred, mask = rule_model(info)
  mask = torch.sum(mask, dim=-1) #(bs, #rules)
  return pred, mask

def calculate_loss(rule_model, criterion, info, gt, hole_ids, hole_stats):

  pred, mask = get_prediction(rule_model, info)
  n_valid_entries = torch.sum(mask.view(-1)!=0)
  loss = criterion(pred, gt)
  loss = loss.masked_fill(mask==0, 0)
  mean_highest_success_correct, pred = get_accuracy(pred, gt, mask)
  masked_gt = torch.sum(gt.masked_fill(mask==0, 0), dim=-1)
  mean_oracle_success = masked_gt.masked_fill(masked_gt!=0, 1.0).mean()

  for i in range(len(hole_ids)):
    hid = hole_ids[i]
    hole_loss = torch.sum(loss[i]) 
    n_valid_hole_rules = torch.sum(loss[i]!=0) 
    hole_loss = hole_loss/n_valid_hole_rules  
    hole_prediction = pred[i]    
    hole_stats[hid] = (hole_loss, hole_prediction)

  return {'loss': torch.sum(loss)/n_valid_entries, \
        'mean_highest_success_correct': mean_highest_success_correct}, \
        mean_oracle_success, \
        hole_stats

if __name__ == '__main__':

  args = setup_args()

  #Fix seeds
  np.random.seed(args.seed)
  os.environ['PYTHONHASHSEED'] = str(args.seed)
  torch.manual_seed(args.seed)
  random.seed(args.seed)

  os.makedirs(os.path.join('outputs', args.data_split), exist_ok=True)
  f_out = open(os.path.join('outputs', args.data_split + '_inference'), 'a')

  model_path = args.model_dir
  model_path_parts = model_path.split('/')[-1].split('#')
  emb_model_type = model_path_parts[7]
  n_head = int(model_path_parts[9])
  d_k = int(model_path_parts[11])
  mode = model_path_parts[13]

  # Define train and val dataloaders
  kwargs = {'num_workers': 8, 'pin_memory': True} if device=='cuda' else {}
  tokenizer = set_tokenizer(emb_model_type)
  dataset = RuleDataset(os.path.join(args.input_data_dir, args.data_split), tokenizer=tokenizer, emb_model_type=emb_model_type)  
  data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, **kwargs)

  # Define the model
  rule_model = RuleModel(emb_model_type=emb_model_type, device=device, n_head=n_head, d_k=d_k, mode=mode)

  print("=> loading checkpoint '{}'".format(model_path))
  best_model_path = os.path.join(model_path, 'best_model.th')
  opt_path = os.path.join(model_path, 'optim.th')
  try:
    status_dict = torch.load(opt_path, map_location=torch.device('cpu'))
    rule_model.load_state_dict(torch.load(best_model_path, map_location=torch.device('cpu')))
    print("=> loaded checkpoint '{}' (epoch {})".format(model_path, status_dict['last_epoch']))
    rule_model.to(device)
  except:
    sys.exit("Not a valid model")

  rule_model.eval()
  criterion = nn.BCELoss(reduction='none')

  with torch.no_grad():

    total_highest_success_correct, total_loss = 0.0, 0.0
    total_batches = 0
    total_oracle_success = 0.0
    hole_stats = {}
    count = 0

    for batch in tqdm(data_loader):

      hole_context = Variable(batch[0]).to(device)
      hole_attention_mask = Variable(batch[1]).to(device)
      rule_context = Variable(batch[2]).to(device)
      gt = Variable(batch[3]).to(device)
      hole_id = batch[4]
      failure_flag = Variable(batch[5]).to(device)

      count+= torch.sum(failure_flag)

      batch_metrices, oracle_success, hole_stats = calculate_loss(rule_model, \
                                      criterion, \
                                      (hole_context, hole_attention_mask, rule_context), \
                                      gt, \
                                      hole_id, \
                                      hole_stats)

      batch_loss = batch_metrices['loss']
      total_highest_success_correct += batch_metrices['mean_highest_success_correct']
      total_oracle_success+= oracle_success
      total_loss += batch_loss.item()
      total_batches += 1

  avg_loss = total_loss/ total_batches
  avg_highest_success_accuracy = total_highest_success_correct*100/ total_batches
  avg_oracle_success_accuracy = total_oracle_success*100/total_batches

  f_out.write("\n********************************\n")
  f_out.write(model_path + "\n")
  print("Loss: Total %f" % avg_loss)
  f_out.write("Loss: " + str(avg_loss) + "\n")
  print("Oracle success accuracy: %f" % avg_oracle_success_accuracy)
  f_out.write("Oracle success accuracy: " + str(avg_oracle_success_accuracy) + "\n")
  print("Highest success accuracy:  %f" % avg_highest_success_accuracy)
  f_out.write("Highest success accuracy: " + str(avg_highest_success_accuracy) + "\n")
  f_out.write("\n********************************\n")
  f_out.flush()

  with open(os.path.join('outputs', args.data_split, '/'.join(model_path.split('/')[1:])) , 'wb') as f:
    pickle.dump(hole_stats, f)



      

