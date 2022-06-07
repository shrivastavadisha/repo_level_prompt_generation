import numpy as np
import os
import torch
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

  #data hyperparams
  parser.add_argument("--seed", type=int, default=9, help="seed for reproducibility")
  parser.add_argument("--input_data_dir", type=str, default='rule_classifier_data', help="base directory for the data")
  parser.add_argument("--model_output_path", type=str, default='models/', help="base directory for storing the models")
  parser.add_argument("--output_path", type=str, default='outputs/', help="base directory for storing the models")
  parser.add_argument("--batch_size", type=int, default=64, help="batch size for training the classifier")
  parser.add_argument("--emb_model_type", type=str, default='codebert', help="model to obtain embedding from")
  #training hyperparams
  parser.add_argument("--num_epochs", type=int, default=5000, help="number of epochs for training the classifier")
  parser.add_argument("--optimizer", type=str, default='adam', help="optimizer to use for training")
  parser.add_argument("--lr_scheduler", type=str, default='none', help="optimizer to use for training")
  parser.add_argument("--learning_rate", type=float, default=3e-4, help="learning rate for training the classifier")
  parser.add_argument("--patience", type=int, default=5000, help="patience for early-stop")
  parser.add_argument('--load_from_checkpoint', default=False, action='store_true')
  #model hyperparams
  parser.add_argument("--mode", type=str, default='rlpg-r', help="rule classifier variant: rlpg-h, rlpg-r")
  parser.add_argument("--n_head", type=int, default=4, help="number of heads")
  parser.add_argument("--d_k", type=int, default=32, help="depth of projection")
  parser.add_argument("--dropout", type=float, default=0.25, help="depth of projection")

  
  return parser.parse_args()

def save(model, optimizer, epoch, save_dir):
  to_save = model.module if hasattr(model, "module") else model
  # pytorch_total_params = sum(p.numel() for p in to_save.parameters())
  # pytorch_trainable_params = sum(p.numel() for p in to_save.parameters() if p.requires_grad)
  # print(pytorch_trainable_params, pytorch_total_params)
  torch.save(to_save.state_dict(), os.path.join(save_dir, "best_model.th"))
  torch.save({"optimizer": optimizer.state_dict(), "last_epoch": epoch}, os.path.join(save_dir, "optim.th"))

def get_accuracy(pred, gold, mask):
  pred = pred.masked_fill(mask==0, 0)
  max_idx = torch.argmax(pred, dim=1, keepdim=True)
  rounded_pred = torch.round(pred)
  max_idx_gold_vals = torch.gather(gold, 1, max_idx)
  mean_highest_success_correct = (max_idx_gold_vals == 1).to(dtype=torch.float).mean()
  return mean_highest_success_correct

def get_prediction(rule_model, info):
  pred, mask = rule_model(info)
  mask = torch.sum(mask, dim=-1) #(bs, #rules)
  return pred, mask

def calculate_loss(rule_model, criterion, info, gt):

  pred, mask = get_prediction(rule_model, info)
  n_valid_entries = torch.sum(mask.view(-1)!=0)
  loss = criterion(pred, gt)
  loss = loss.masked_fill(mask==0, 0)
  loss = torch.sum(loss)/n_valid_entries
  mean_highest_success_correct = get_accuracy(pred, gt, mask)
  masked_gt = torch.sum(gt.masked_fill(mask==0, 0), dim=-1)
  mean_oracle_success = masked_gt.masked_fill(masked_gt!=0, 1.0).mean()

  return {'loss': loss, \
        'mean_highest_success_correct': mean_highest_success_correct}, \
          mean_oracle_success


if __name__ == '__main__':

  args = setup_args()

  #Fix seeds
  np.random.seed(args.seed)
  os.environ['PYTHONHASHSEED'] = str(args.seed)
  torch.manual_seed(args.seed)
  random.seed(args.seed)

  #Define paths for storing tensorboard logs
  dir_name = 'optimizer#' + args.optimizer + '#learning_rate#' + str(args.learning_rate) + '#lr_scheduler#' + args.lr_scheduler \
              + '#emb_model_type#' + args.emb_model_type + '#n_head#' + str(args.n_head) + '#d_k#' + str(args.d_k) \
              + '#mode#' + args.mode + '#dropout#' + str(args.dropout)

  save_dir = os.path.join(args.model_output_path, dir_name)
  os.makedirs(save_dir, exist_ok=True)
  tb_writer = SummaryWriter(os.path.join(save_dir, "logs"))
  os.makedirs(args.output_path, exist_ok=True)
  f_out = open(os.path.join(args.output_path, dir_name), 'w')

  # Define train and val dataloaders
  kwargs = {'num_workers': 8, 'pin_memory': True} if device=='cuda' else {}
  tokenizer = set_tokenizer(args.emb_model_type)
  #print(tokenizer)
  train_dataset = RuleDataset(os.path.join(args.input_data_dir, 'train'), tokenizer=tokenizer, emb_model_type=args.emb_model_type)  
  train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, **kwargs)

  val_dataset = RuleDataset(os.path.join(args.input_data_dir, 'val'), tokenizer=tokenizer, emb_model_type=args.emb_model_type)
  val_data_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, **kwargs)

  # Define the model
  rule_model = RuleModel(emb_model_type=args.emb_model_type, device=device, n_head=args.n_head, d_k=args.d_k, \
                         mode = args.mode, dropout=args.dropout)
  rule_model.to(device)

  #Define optimizer and loss
  if args.lr_scheduler == 'none':
    if args.optimizer == 'adam':
      optimizer = torch.optim.Adam(rule_model.parameters(), lr=args.learning_rate)
    if args.optimizer == 'sgd':
      optimizer = torch.optim.SGD(rule_model.parameters(), lr=args.learning_rate)

  if args.lr_scheduler == 'cosine':
    if args.optimizer == 'adam':
      optimizer = torch.optim.Adam(rule_model.parameters(), lr=args.learning_rate)
    if args.optimizer == 'sgd':
      optimizer = torch.optim.SGD(rule_model.parameters(), lr=args.learning_rate)
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)

  if args.lr_scheduler == 'cosinewarm':
    if args.optimizer == 'adam':
      optimizer = torch.optim.Adam(rule_model.parameters(), lr=args.learning_rate)
    if args.optimizer == 'sgd':
      optimizer = torch.optim.SGD(rule_model.parameters(), lr=args.learning_rate)
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)

  if args.lr_scheduler == 'tr2':
    if args.optimizer == 'adam':
      optimizer = torch.optim.Adam(rule_model.parameters(), lr=args.learning_rate)
    if args.att_lr_optimizer == 'sgd':
      optimizer = torch.optim.SGD(rule_model.parameters(), lr=args.learning_rate)
    lr_sched = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=args.learning_rate/2.0, max_lr=0.1,
                  step_size_up=5, mode="triangular2", cycle_momentum=False)

  if args.lr_scheduler == 'reduceonplateau':
    if args.optimizer == 'adam':
      optimizer = torch.optim.Adam(rule_model.parameters(), lr=args.learning_rate)
    if args.optimizer == 'sgd':
      optimizer = torch.optim.SGD(rule_model.parameters(), lr=args.learning_rate)
    lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max')


  if args.load_from_checkpoint:
    print("=> loading checkpoint '{}'".format(save_dir))
    model_path = os.path.join(save_dir, 'best_model.th')
    opt_path = os.path.join(save_dir, 'optim.th')
    status_dict = torch.load(opt_path, map_location=torch.device('cpu'))
    rule_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    optimizer.load_state_dict(status_dict['optimizer'])
    print("=> loaded checkpoint '{}' (epoch {})".format(save_dir, status_dict['last_epoch']))

  criterion = nn.BCELoss(reduction='none')


  best_val_acc = 0

  for epoch in range(args.num_epochs):
    print("Epoch %d" % epoch)
    f_out.write("Epoch: " + str(epoch)+"\n")

    ########################Training Loop#############################################
    total_highest_success_correct, total_loss = 0.0, 0.0

    total_batches = 0
    total_oracle_success = 0.0

    rule_model.train()
    train_count = 0

    for batch in tqdm(train_data_loader):

      hole_context = Variable(batch[0]).to(device)
      hole_attention_mask = Variable(batch[1]).to(device)
      rule_context = Variable(batch[2]).to(device)
      gt = Variable(batch[3]).to(device)
      failure_flag = Variable(batch[5]).to(device)

      train_count+= torch.sum(failure_flag)

      optimizer.zero_grad()

      batch_metrices, oracle_success = calculate_loss(rule_model, \
                                      criterion, \
                                      (hole_context, hole_attention_mask, rule_context), \
                                      gt)

      batch_loss = batch_metrices['loss']
      batch_loss.backward()
      optimizer.step()

      total_highest_success_correct += batch_metrices['mean_highest_success_correct']
      total_oracle_success += oracle_success
      total_loss += batch_loss.item()
      total_batches += 1

    avg_train_loss = total_loss/ total_batches
    avg_highest_success_accuracy = total_highest_success_correct*100/ total_batches
    avg_oracle_success_accuracy = total_oracle_success*100/ total_batches

    tb_writer.add_scalar("metrics/train_loss", avg_train_loss, epoch)
    tb_writer.add_scalar("metrics/train_highest_success_accuracy", avg_highest_success_accuracy, epoch)

    print("Train loss: Total %f" % avg_train_loss)
    f_out.write("Train loss: " + str(avg_train_loss) + "\n")
    print("Train oracle success accuracy: %f" % avg_oracle_success_accuracy)
    f_out.write("Train oracle success accuracy: " + str(avg_oracle_success_accuracy) + "\n")
    print("Train highest success accuracy:  %f" % avg_highest_success_accuracy)
    f_out.write("Train highest success accuracy: " + str(avg_highest_success_accuracy) + "\n")


    ######################################Evaluation Loop############################################
    rule_model.eval()

    with torch.no_grad():

      total_highest_success_correct, total_loss = 0.0, 0.0
      total_batches = 0
      total_oracle_success = 0.0
      val_count = 0

      for batch in tqdm(val_data_loader):

        hole_context = Variable(batch[0]).to(device)
        hole_attention_mask = Variable(batch[1]).to(device)
        rule_context = Variable(batch[2]).to(device)
        gt = Variable(batch[3]).to(device)
        failure_flag = Variable(batch[5]).to(device)

        val_count+= torch.sum(failure_flag)


        batch_metrices, oracle_success = calculate_loss(rule_model, \
                                        criterion, \
                                        (hole_context, hole_attention_mask, rule_context), \
                                        gt)


        batch_loss = batch_metrices['loss']
        total_highest_success_correct += batch_metrices['mean_highest_success_correct']
        total_oracle_success+= oracle_success
        total_loss += batch_loss.item()
        total_batches += 1

    avg_val_loss = total_loss/ total_batches
    avg_highest_success_accuracy = total_highest_success_correct*100/ total_batches
    avg_oracle_success_accuracy = total_oracle_success*100/total_batches

    tb_writer.add_scalar("metrics/val_loss", avg_val_loss, epoch)
    tb_writer.add_scalar("metrics/val_highest_success_accuracy", avg_highest_success_accuracy, epoch)

    print("Val loss: Total %f" % avg_val_loss)
    f_out.write("Val loss: " + str(avg_val_loss) + "\n")
    print("Val oracle success accuracy: %f" % avg_oracle_success_accuracy)
    f_out.write("Val oracle success accuracy: " + str(avg_oracle_success_accuracy) + "\n")
    print("Val highest success accuracy:  %f" % avg_highest_success_accuracy)
    f_out.write("Val highest success accuracy: " + str(avg_highest_success_accuracy) + "\n")

    if args.lr_scheduler =='reduceonplateau':
      lr_sched.step(avg_highest_success_accuracy)
    elif args.lr_scheduler !='none':
      lr_sched.step()

    if avg_highest_success_accuracy > best_val_acc:
      print("Found new best model")
      f_out.write("Found new best model\n")
      best_val_acc = avg_highest_success_accuracy
      save(rule_model, optimizer, epoch, save_dir)
      patience_ctr = 0
    else:
      patience_ctr += 1
      if patience_ctr == args.patience:
        print("Ran out of patience. Stopping training early...")
        f_out.write("Ran out of patience. Stopping training early...\n")
        print("Best Val Acc: ", best_val_acc)
        f_out.write("Best Val Acc: " + str(best_val_acc))
        break
    f_out.write("\n\n")
    f_out.flush()
    print("Best Val Acc: ", best_val_acc)
    f_out.write("Best Val Acc: " + str(best_val_acc))
  f_out.close()

