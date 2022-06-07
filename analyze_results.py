import os
import json
import pickle
from utils import *
import argparse
from rule_config import rule_hyperparams

def get_hole_identities(hole_filename, duplicate_filename):
    hole_data = pickle.load(open(hole_filename, 'rb'))
    duplicate_files = open(duplicate_filename, 'r').readlines()
    duplicate_files = [ x.strip() for x in duplicate_files]
    hole_identities = []
    for (k,v) in hole_data.items():
        if k not in duplicate_files and not k.startswith('rule_classifier_data/train/rsbotownversion/trunk/scripts/'):
          for hole in v:
            h_id = k + '_' + str(hole[0]) + '_' + str(hole[1])
            hole_identities.append(h_id)
    return hole_identities, duplicate_files

def obtain_full_rule_results(result_file, codex_results, hole_rule_mapping):
  rule_results = read_result_file(result_file, hole_rule_mapping)
  j = 0
  all_results = []
  for i in range(len(codex_results)):
    codex_res = codex_results[i]
    hid = codex_res['hole_identity']
    if j < len(rule_results):
      rule_res_hid = rule_results[j]['hole_identity']
      if hid == rule_res_hid:
        all_results.append(rule_results[j])
        j+=1
      else:
        all_results.append(codex_res)
    else:
      all_results.append(codex_res)
  return all_results

def is_match(res_line):
    res_line = json.loads(res_line)
    prediction = res_line['prediction']
    hole = res_line['ground-truth hole']
    pred = prediction.rstrip()
    hole = hole.rstrip()
    # there is an exact match corresponding to this hole id
    if pred == hole:
        return True
    else:
        return False
    
def update_hole_rule_mapping(res_line, hid, hole_rule_mapping, rule_parts):
  if is_match(res_line):
    if hid in hole_rule_mapping:
        hole_rule_mapping[hid].append(rule_parts)
    else:
        hole_rule_mapping[hid] = [rule_parts]
  return hole_rule_mapping

def modify_results(result_lines, duplicate_files):
  if not duplicate_files:
    return result_lines
  else:
    #print(len(duplicate_files), len(result_lines))
    mod_result_lines = []
    for i in range(len(result_lines)):
      res_hid = json.loads(result_lines[i])['hole_identity']
      if is_valid_hole(res_hid, duplicate_files):
        mod_result_lines.append(result_lines[i])
    return mod_result_lines

def read_result_file(rule_result_file, codex_file, hole_identities, hole_rule_mapping, rule_parts, duplicate_files):
    #print(rule_result_file)
    rule_lines = open(rule_result_file, 'r').readlines()
    codex_lines = open(codex_file, 'r').readlines() 
    rule_lines = modify_results(rule_lines, duplicate_files)
    codex_lines = modify_results(codex_lines, duplicate_files)
    j = 0
    for i in range(len(hole_identities)):
        hid = hole_identities[i]
        codex_hid = json.loads(codex_lines[i])['hole_identity']
        codex_hid = alter_hid(codex_hid, hid)

        if j < len(rule_lines):
            try:
              rule_hid = json.loads(rule_lines[j])['hole_identity']
            except:
              print(rule_result_file)
            rule_hid = alter_hid(rule_hid, hid)
            # use rule result
            if hid == rule_hid:
                hole_rule_mapping = update_hole_rule_mapping(rule_lines[j], hid, hole_rule_mapping, rule_parts)
                j+=1
            else:
                # use codex result
                hole_rule_mapping = update_hole_rule_mapping(codex_lines[i], hid, hole_rule_mapping, rule_parts)
        else:
            # use codex result
            hole_rule_mapping = update_hole_rule_mapping(codex_lines[i], hid, hole_rule_mapping, rule_parts)
        
    return hole_rule_mapping

def get_results(base_result_dir, context_location, exclude_codex=True):
  context_result_dir = os.path.join(base_result_dir, context_location)
  result_files = next(os.walk(context_result_dir), (None, None, []))[2]  # [] if no file
  if result_files and exclude_codex and 'codex_4072.json' in result_files:
    result_files.remove('codex_4072.json')
  mod_result_files = [os.path.join(context_result_dir, result_file) for result_file in result_files if result_file]
  result_files = [f for f in mod_result_files if os.path.getsize(f)>0]
  return result_files

def check_validity_by_rule_parts(rule):
  valid = False
  if 'codex_4072' in rule:
    context_location = 'codex'
    context_type = 'codex'
    context_ratio = 0.5
    valid = True
  else:
    context_location = rule.split("/")[-2]
    rule_parts = rule.split("/")[-1].split("_")
    i = 5
    ct = '_'.join(rule_parts[1:5])
    # keep removing the parts joined by _ till it matches a valid context_type
    while(ct not in context_types_to_index):
      i-=1
      ct = '_'.join(rule_parts[1:i])
    context_type = ct

    mod_rule_parts = rule_parts[i:]
    try:
      context_ratio = float(mod_rule_parts[3])/4072
      if check_rule_validity(context_type, mod_rule_parts):
        valid =True
    except:
      valid = False      
  if valid:
    return context_location, context_type, context_ratio
  else:
    return '', '', ''

def get_all_hole_rule_mapping(base_result_dir, hole_identities, duplicate_files):
  in_file_files = get_results(base_result_dir, 'in_file', exclude_codex=False)
  parent_class_files = get_results(base_result_dir, 'parent_class_file')
  import_files = get_results(base_result_dir, 'import_file')
  sibling_files = get_results(base_result_dir, 'sibling_file')
  similar_name_files = get_results(base_result_dir, 'similar_name_file')
  child_class_files = get_results(base_result_dir, 'child_class_file')
  import_of_similar_name_files = get_results(base_result_dir, 'import_of_similar_name_file')
  import_of_sibling_files = get_results(base_result_dir, 'import_of_sibling_file')
  import_of_parent_class_files = get_results(base_result_dir, 'import_of_parent_class_file')
  import_of_child_class_files = get_results(base_result_dir, 'import_of_child_class_file')
  codex_file = os.path.join(base_result_dir, 'in_file', 'codex_4072.json')

  result_files = in_file_files + parent_class_files + import_files + sibling_files + similar_name_files \
                  + child_class_files + import_of_similar_name_files + import_of_sibling_files + import_of_child_class_files + import_of_parent_class_files

  # print(len(in_file_files), len(parent_class_files), len(import_files), len(sibling_files), len(similar_name_files), \
  #       len(child_class_files) , len(import_of_sibling_files), len(import_of_similar_name_files), len(import_of_child_class_files),\
  #       len(import_of_parent_class_files), len(result_files))

  hole_rule_mapping = {}
  for result_file in result_files:
    context_location, context_type, context_ratio = check_validity_by_rule_parts(result_file)
    if context_location:
      hole_rule_mapping = read_result_file(result_file, codex_file, hole_identities, hole_rule_mapping, \
                                          (context_location, context_type, context_ratio), duplicate_files)
  return hole_rule_mapping

def get_failed_holes(successful_holes, hole_identities):
  failed_holes = []
  for hole_identity in hole_identities:
    if hole_identity not in successful_holes:
      failed_holes.append(hole_identity)
  return failed_holes

def find_rule_pattern(rule_pattern, rules):
  found = False
  for rule in rules:
    if rule_pattern in rule:
      found = True
      break
  return found

def find_rule_specific_success(successful_holes, query_file_pattern=''):
  count = 0
  for h_id, rules in successful_holes.items():
    if find_rule_pattern(query_file_pattern, rules):
      count +=1
  return count

def find_complementary_rules(successful_holes):
  other_rules = []
  not_lines_not_iden_codex = []
  not_lines_iden = []
  lines = []
  for h_id, rules in successful_holes.items():
    if not find_rule_pattern('lines', rules):
      if not find_rule_pattern('identifiers', rules):
        if not find_rule_pattern('codex', rules):
          other_rules.append((h_id, rules))
        else:
          not_lines_not_iden_codex.append((h_id, rules))
      else:
        not_lines_iden.append((h_id, rules))
    else:
      lines.append((h_id, rules))
  return lines, not_lines_iden, not_lines_not_iden_codex, other_rules

def check_rule_validity(context_type, rule_parts):
  valid = False
  valid_hyperparams = rule_hyperparams[context_type]
  try:
    rule_context_ratio = float(rule_parts[3])/4072
  except:
    return False
  rule_prompt_separator = rule_parts[-1]
  rule_rule_context_formatting = '_'.join(rule_parts[4:-1])
  if rule_context_ratio in valid_hyperparams['context_ratio']:
    if rule_prompt_separator in valid_hyperparams['prompt_separator']:
      if rule_rule_context_formatting in valid_hyperparams['rule_context_formatting']:
        valid = True
  return valid

def get_rule_templated_version(oracle):
  mod_oracle = {}
  for hid, rules in oracle.items():
    context_locations = []
    context_types = []
    combined = []
    for rule in rules:
      context_location, context_type, context_ratio = rule
      context_locations.append(context_location)
      context_types.append(context_type)
      if context_location != 'codex':
        combined.append(context_location + '#' + context_type + '#' + str(context_ratio))
      else:
        combined.append('codex')        
    context_location = get_multi_hot_vector(context_locations, 'cl')
    context_type = get_multi_hot_vector(context_types, 'ct')
    comb = get_multi_hot_vector(combined, 'com')
    mod_oracle[hid] = {'cl': context_location, 'ct': context_type, 'com': comb}
  return mod_oracle


def find_rule_mapping(successful_holes):
  rule_mapping = {}
  for hid, rules in successful_holes.items():
    for rule in rules:
      if rule not in rule_mapping:
        rule_mapping[rule] = [hid]
      else:
        rule_mapping[rule].append(hid)
  return rule_mapping

def find_single_best_rule_success(rule_mapping):
  best_single_rule_success = 0
  for k, v in rule_mapping.items():
    if len(v)> best_single_rule_success:
      best_rule_parts = k
      best_single_rule_success = len(v)
  best_rule = best_rule_parts[0] + '_' + best_rule_parts[1] + '_' + str(best_rule_parts[2])
  return best_rule, best_single_rule_success


def setup_args():
  """
  Description: Takes in the command-line arguments from user
  """
  parser = argparse.ArgumentParser()
  parser.add_argument("--base_dir", type=str, default='rule_classifier_data', help="base directory for the data")
  parser.add_argument("--data_split", type=str, default='test', help="data split to store the data")
  parser.add_argument("--proj_name", type=str, default='dovetaildb', help="name of the input repo")

  return parser.parse_args()

if __name__ == '__main__':

  args = setup_args()
  hole_filename = os.path.join(args.base_dir, args.data_split, args.proj_name, 'hole_data')
  duplicate_filename = os.path.join(args.base_dir, args.data_split, args.proj_name, 'duplicates')
  hole_identities, duplicate_files = get_hole_identities(hole_filename, duplicate_filename)
  print("Total number of holes:", len(hole_identities))
  base_result_dir = os.path.join('results', args.base_dir, args.data_split, args.proj_name)
  print(len(duplicate_files))
  successful_holes = get_all_hole_rule_mapping(base_result_dir, hole_identities, duplicate_files)
  print("Number of holes that got atleast one rule successful: ", len(successful_holes))

  oracle = get_rule_templated_version(successful_holes)
  with open(os.path.join(args.base_dir, args.data_split, args.proj_name, 'oracle'), 'wb') as f:
    pickle.dump(oracle, f)
  assert len(successful_holes) == len(oracle)
  rule_mapping = find_rule_mapping(successful_holes)
  codex_success = len(rule_mapping[('codex', 'codex', 0.5)])
  best_rule, best_rule_success = find_single_best_rule_success(rule_mapping)
  best_single_rule_success = len(rule_mapping[('in_file', 'lines', 0.75)])
  # print(rule_mapping)
  print(codex_success, best_single_rule_success, best_rule, best_rule_success)
  print(
        args.proj_name + ", " + \
        str(float(len(successful_holes)*100/len(hole_identities))) + ", " + \
        str(float(codex_success*100/len(hole_identities))) + ", " + \
        best_rule + ", " +\
        str(float(best_rule_success*100/len(hole_identities))) + ", " + \
        "in_file_lines_0.75" + ", " +\
        str(float(best_single_rule_success*100/len(hole_identities)))
        )

  # failed_holes = get_failed_holes(successful_holes, hole_identities)
  # print("Number of holes that got no rule successful: ", len(failed_holes))
  # with open(os.path.join(base_result_dir, 'failed_cases'), 'wb') as f:
  #   pickle.dump(failed_holes, f)

  # post_lines_success = find_rule_specific_success(successful_holes, 'lines')
  # codex_success = find_rule_specific_success(successful_holes, 'codex')
  # identifiers_success = find_rule_specific_success(successful_holes, 'identifiers')
  # print("Number of post lines successes: ", post_lines_success)
  # print("Number of codex successes: ", codex_success)
  # print("Number of identifiers successes: ", identifiers_success)

  # lines, not_lines_iden, not_lines_not_iden_codex, other_rules = find_complementary_rules(successful_holes)
  # print("Post Lines: ", len(lines), end=", ")
  # print("Not Post Lines, Identifiers: ", len(not_lines_iden), end=", ")
  # print("Not Post Lines, Not Identifiers, Codex: ", len(not_lines_not_iden_codex), end=", ")
  # print("Other Rules: ", len(other_rules), end=", ")
  # print("No Rules: ", len(failed_holes), end="\n")

