import os
import shutil
import pickle
import random
import numpy as np

seed = 9

#Fix seeds
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)

projects = { 'train': [
                    'gfsfa',
                    'sol-agent-platform',
                    'gloodb',
                    'rsbotownversion',
                    'jjskit',
                    'ftpserverremoteadmin', 
                    'openprocesslogger', 
                    'strudem-sicsa',
                    'seamlets', 
                    'healpix-rangeset', 
                    'quidsee', 
                    'mobileexpensetracker', 
                    'swe574-group3', 
                    'largemail', 
                    'soap-dtc', 
                    'designpatternjavapedro', 
                    'myt5lib', 
                    'exogdx', 
                    'tapestry-sesame'                     
                    ], 

            'val': [
                    'javasummerframework',
                    'tinwiki',
                    'teammates-shakthi',
                    'jcontenedor', 
                    'jloogle', 
                    'swinagile', 
                    'math-mech-eshop', 
                    'jata4test', 
                    'affinity_propagation_java', 
                    'navigablep2p', 
                    'springlime', 
                    'sohocms', 
                    'tyrond', 
                    'infinispan-storage-service', 
                    ],
                    
            'test': [
                      'project-pt-diaoc',
                      'dovetaildb',
                      'robotsimulator2009w',
                      'ircrpgbot',
                      'xfuze',
                      'realtimegc',
                      'fswuniceubtemplates', 
                      'glperaudsimon',
                      'apiitfriends',
                      'qwikioffice-java', 
                      'xiaonei-java-api', 
                      'wicketbits', 
                      'hucourses', 
                      'gwt-plugindetect'
                    ] 
          }



repo_split_map = {}
for split, repos in projects.items():
  for repo in repos:
    repo_split_map[repo] = split

max_holes = 10000

def is_move(base_dir, split, repo):
  new_split = repo_split_map[repo]
  if new_split != split:
    shutil.move(os.path.join(base_dir, split, repo), os.path.join(base_dir, new_split, repo))

def find_single_best_rule_success(rule_mapping):
  best_single_rule_success = 0
  for k, v in rule_mapping.items():
    if len(v)> best_single_rule_success:
      best_rule = k
      best_single_rule_success = len(v)
  return best_rule, best_single_rule_success

def find_rule_mapping(oracle):
  rule_mapping = {}
  for hid, entry in oracle.items():
    rules = entry['com']
    success_rule_positions = np.where(rules == 1)[0]
    for s_r_p in success_rule_positions:
      if s_r_p not in rule_mapping:
        rule_mapping[s_r_p] = [hid]
      else:
        rule_mapping[s_r_p].append(hid)
  return rule_mapping

def get_new_oracle_numbers(capped_oracle, repo, total_holes):
  rule_mapping = find_rule_mapping(capped_oracle)
  codex_success = len(rule_mapping[62])
  best_rule, best_rule_success = find_single_best_rule_success(rule_mapping)
  best_single_rule_success = len(rule_mapping[7])
  print(
        repo + ", " + \
        str(total_holes) + ", " + \
        str(float(len(capped_oracle)*100/total_holes)) + ", " + \
        str(float(codex_success*100/total_holes)) + ", " + \
        str(best_rule) + ", " +\
        str(float(best_rule_success*100/total_holes)) + ", " + \
        "in_file_lines_0.75" + ", " +\
        str(float(best_single_rule_success*100/total_holes))
        )

def rewrite_rule_context_data(repo_path, capped_holes, emb_model_type):
  all_files = os.listdir(os.path.join(repo_path, emb_model_type))
  os.makedirs(os.path.join(repo_path, 'capped_'+ emb_model_type), exist_ok=True)
  for file in all_files:
    hole_path = os.path.join(repo_path, emb_model_type, file)
    data = pickle.load(open(hole_path, 'rb'))
    hole = list(data.keys())[0]
    if hole in capped_holes:
      dest_hole_path = os.path.join(repo_path, 'capped_'+ emb_model_type, file)
      shutil.copy(hole_path, dest_hole_path)

def rearrange_data(base_dir, split):
  print(split)
  all_dirs = os.listdir(os.path.join(base_dir, split))
  for repo in all_dirs:
    if repo in repo_split_map:
      print(base_dir, split, repo)
      repo_holes = []
      hole_data = pickle.load(open(os.path.join(base_dir, split, repo, 'hole_data'), 'rb'))
      oracle = pickle.load(open(os.path.join(base_dir, split, repo, 'oracle'), 'rb'))
      duplicate_files = open(os.path.join(base_dir, split, repo, 'duplicates'), 'r').readlines()
      all_duplicate_files = [x.strip() for x in duplicate_files]
      for file, holes in hole_data.items():
        if file not in all_duplicate_files and not file.startswith('rule_classifier_data/val/rsbotownversion/trunk/scripts/'):
          hids = [file + '_' + str(h[0]) + '_' + str(h[1]) for h in holes]
          repo_holes.extend(hids)
      #print(len(repo_holes))
      if len(repo_holes) < max_holes:
        capped_holes = repo_holes
        capped_oracle = oracle
        total_holes = len(repo_holes)
      else:
        capped_holes = random.sample(repo_holes, max_holes)
        capped_oracle = {}
        for hid, entry in oracle.items():
          if hid in capped_holes:
            capped_oracle[hid] = entry
        total_holes = len(capped_holes)

      get_new_oracle_numbers(capped_oracle, repo, total_holes)
      with open(os.path.join(base_dir, split, repo, 'capped_oracle_'+ str(max_holes)), 'wb') as f:
        pickle.dump(capped_oracle, f)

      with open(os.path.join(base_dir, split, repo, 'capped_holes_'+ str(max_holes)), 'w') as f:
        for item in capped_holes:
          f.write("%s\n" %(item,))
      capped_holes = open(os.path.join(base_dir, split, repo, 'capped_holes_10000'), 'r').readlines()
      capped_holes = [x.strip() for x in capped_holes]

      rewrite_rule_context_data(os.path.join(base_dir, split, repo), capped_holes, 'codebert_mod')

      is_move(base_dir, split, repo)

rearrange_data('rule_classifier_data', 'train')
rearrange_data('rule_classifier_data', 'val')
rearrange_data('rule_classifier_data', 'test')


