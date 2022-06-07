import pickle
import os
import torch
import argparse

device = 'cpu'

# This order was obtained based on the decreasing order of success rate on the validation set
rule_order = [5, 7, 6, 1, 20, 22, 2, 0, 25, 3, 23, 24, 28, 26, 4, 21, 62, 31, 27, 29, 30, 8, 34, 32, \
              10, 33, 9, 35, 13, 11, 12, 36, 46, 44, 16, 14, 49, 45, 48, 40, 38, 19, 15, 18, 39, 43, 47,\
               17, 42, 41, 58, 56, 57, 61, 59, 60, 37, 52, 50, 53, 55, 54, 51]

projects = { 'train': [
                        ('largemail' , 1653), 
                        ('ftpserverremoteadmin' , 7323), 
                        ('myt5lib' , 838),
                        ('seamlets' , 4890), 
                        ('gloodb' , 10000),
                        ('jjskit' , 9043),
                        ('mobileexpensetracker' , 2298), 
                        ('gfsfa' , 10000),
                        ('swe574-group3' , 2029), 
                        ('strudem-sicsa' , 6131),
                        ('soap-dtc' , 1370), 
                        ('openprocesslogger' , 7191), 
                        ('tapestry-sesame', 397),
                        ('exogdx' , 735), 
                        ('designpatternjavapedro' , 1069), 
                        ('quidsee' , 3020), 
                        ('healpix-rangeset' , 4734),                      
                        ('sol-agent-platform' , 10000),
                        ('rsbotownversion' , 10000),
                         
                      ], 

              'val': [
                      ('tyrond', 721),
                      ('math-mech-eshop', 2225),
                      ('infinispan-storage-service', 373),
                      ('teammates-shakthi', 7665),
                      ('javasummerframework', 10000),
                      ('tinwiki', 10000),
                      ('jloogle', 3145),
                      ('jcontenedor', 5464),
                      ('sohocms', 772),
                      ('affinity_propagation_java', 1466),
                      ('jata4test', 1921),
                      ('swinagile', 2595),
                      ('navigablep2p', 1322),
                      ('springlime', 879),
                      ],

              'test': [
                        ('dovetaildb', 10000),
                        ('project-pt-diaoc', 10000),
                        ('realtimegc', 2513), 
                        ('fswuniceubtemplates', 2070), 
                        ('qwikioffice-java', 1138),  
                        ('glperaudsimon', 1766),
                        ('xiaonei-java-api', 839),   
                        ('ircrpgbot', 6591), 
                        ('robotsimulator2009w', 7514), 
                        ('gwt-plugindetect',  73),
                        ('apiitfriends',  1385), 
                        ('wicketbits', 754), 
                        ('hucourses',  590), 
                        ('xfuze', 3055),                         
                      ] 
            }

def setup_args():
  """
  Description: Takes in the command-line arguments from user
  """
  parser = argparse.ArgumentParser()

  parser.add_argument("--seed", type=int, default=9, help="seed for reproducibility")
  parser.add_argument("--hole_stats_file", type=str, default='hole', help="name of the prediction file to consider")
  parser.add_argument("--data_split", type=str, default='val', help="data_split")
  parser.add_argument("--base_dir", type=str, default='outputs', help="base dir")
  parser.add_argument("--k", type=int, default=1, help="how many rules to draw")
  return parser.parse_args()


def get_repo_name(hid):
  return hid.split('/')[2]

def update_dict(dic, data_type):
    if 'small_' in data_type:
      return dic
    else:
      mod_dic = {}
      for k,v in dic.items():
        mod_k = '/'. join(['rule_classifier_data', data_type] + k.split('/')[2:])
        mod_dic[mod_k] = v
      return mod_dic

def get_top_k_acc(hole_pred, hole_gt, k=1):
  top_preds, top_pred_indices = torch.topk(hole_pred, k)
  for top_pred_idx in top_pred_indices:
    if hole_gt[top_pred_idx] == 1:
      return 1.0
  return 0.0

def get_rule_wise_nums(oracle):
  rule_success = {}
  for hid, entry in oracle.items():
    hole_gt= entry['com']
    for i in range(len(hole_gt)):
      if hole_gt[i] == 1:
        if i in rule_success:
          rule_success[i]+=1
        else:
          rule_success[i] = 1
  return rule_success

def get_single_rule_acc(hole_gt, k):
  rules = rule_order[:k]
  for rule in rules:
    if hole_gt[rule] == 1:
      return 1.0
  return 0.0

if __name__ == '__main__':

  args = setup_args()

  #Fix seeds
  os.environ['PYTHONHASHSEED'] = str(args.seed)
  torch.manual_seed(args.seed)

  k = args.k
  repo_stats={}
  single_rule_stats = {}
  data = pickle.load(open(os.path.join(args.base_dir, args.data_split , args.hole_stats_file), 'rb'))
  oracle_dir = 'rule_classifier_data/' + args.data_split 
  for repo, repo_count in projects[args.data_split ]:
    oracle = pickle.load(open(os.path.join(oracle_dir, repo, 'capped_oracle_10000'), 'rb'))
    oracle = update_dict(oracle, args.data_split )
    for hid, entry in oracle.items():
      em = get_top_k_acc(data[hid][1], oracle[hid]['com'], k)
      single_rule_em = get_single_rule_acc(oracle[hid]['com'], k)
      if repo in repo_stats:
        repo_stats[repo]+= em
      else:
        repo_stats[repo] = em
      if repo in single_rule_stats:
        single_rule_stats[repo]+= single_rule_em
      else:
        single_rule_stats[repo] = single_rule_em

  repo_success = 0.0
  single_rule_repo_success = 0.0
  for repo, repo_count in projects[args.data_split]:
    repo_success += repo_stats[repo]*100/repo_count
    single_rule_repo_success += single_rule_stats[repo]*100/repo_count


  total_count = 0
  total_success = 0.0
  total_single_rule_success = 0.0
  for repo, repo_count in projects[args.data_split ]:
    total_count+= repo_count
    total_success += repo_stats[repo]
    total_single_rule_success += single_rule_stats[repo]

  print(args.hole_stats_file + "," + str(k) + "," + str(repo_success/len(projects[args.data_split ])) \
        + "," + str(single_rule_repo_success/len(projects[args.data_split ]))\
        + "," + str(total_success*100/total_count) + "," + str(total_single_rule_success*100/total_count))

