import numpy as np
from itertools import product
from rule_config import *

promptseparator2str = {'space': " ", \
                        'newline': "\n", \
                        'class_names': "class_names",\
                        'class_method_names': "class_method_names",\
                        'method_names': "method_names"}

context_location_to_index = {
                    'in_file':0, \
                    'parent_class_file':1, \
                    'import_file':2,\
                    'sibling_file':3, \
                    'similar_name_file':4, \
                    'child_class_file':5, \
                    'import_of_sibling_file':6, \
                    'import_of_similar_name_file':7, \
                    'import_of_parent_class_file':8, \
                    'import_of_child_class_file':9, \
                    'codex': 10 #codex
                    }


context_types_to_index = {
                      'method_names_and_bodies':0,\
                      'method_names':1,\
                      'identifiers':2, \
                      'type_identifiers':3,\
                      'string_literals':4,\
                      'field_declarations':5, \
                      'lines':6, \
                      'codex': 7 #codex
                      }

count = 0
combined_to_index = {}
cl_keys = list(context_location_to_index.keys())
ct_keys = list(context_types_to_index.keys())
cl_keys.remove('codex')
ct_keys.remove('codex')
for (k1, k2) in product(cl_keys, ct_keys):
  if (k1 != 'in_file' and k2 in ct_keys[:-1]) or (k1 == 'in_file' and k2 in ct_keys[1:]):
    cr_keys = rule_hyperparams[k2]['context_ratio']
    for k3 in cr_keys:
      key = k1 + '#' + k2 + '#' + str(k3)
      combined_to_index[key] = count
      count +=1
combined_to_index['codex'] = count
#print(combined_to_index)

def get_multi_hot_vector(lst, type):
  set_lst = list(set(lst))
  if type == 'cl':
    index_dict = context_location_to_index
  if type == 'ct':
    index_dict = context_types_to_index
  if type == 'com':
    index_dict = combined_to_index
  vector_size = len(index_dict)
  multi_hot_vector = np.zeros(vector_size)
  for entry in lst:
    multi_hot_vector[index_dict[entry]] = 1
  return multi_hot_vector

def is_valid_hole(hole, duplicate_files):
  hole_parts = hole.split('/')[-1].split('_')
  if len(hole_parts) > 3:
    new_hole_parts = hole_parts[:-2]
    filename = '_'.join(new_hole_parts)
    filename = [filename]
  else:
    filename = [hole_parts[0]]
  file = '/'.join(hole.split('/')[:-1] + filename)
  if file in duplicate_files:
    return False
  else:
    return True

def find_intersection(lst1, lst2):
  set_lst1 = set(lst1)
  set_lst2 = set(lst2)
  return set_lst1.intersection(set_lst2)

def alter_hid(orig_hid, hid):
  data_split = hid.split('/')[1]
  if 'gcode-data' in orig_hid:
    new_id = orig_hid.replace('data/gcode-data', 'rule_classifier_data/' + data_split)
    return new_id
  elif 'java-other' in orig_hid:
    new_id = orig_hid.replace('data/java-other', 'rule_classifier_data/' + data_split)
    return new_id
  else:
    return orig_hid

def find_usages(query_att, query_file, lst_key_att, key_file):
  usages = []
  query_str = get_string(query_file, query_att[0], query_att[1])
  for key_att in lst_key_att:
    key_str = get_string(key_file, key_att[0], key_att[1])
    if key_str == query_str:
      usages.append(key_att)
  return usages

def update_list(src_lst, tgt_lst, f, return_type='str'):
  for elem in src_lst:
    elem_str = get_string(f, elem[0], elem[1])
    if elem_str not in tgt_lst:
      if return_type == 'pos':
        tgt_lst.append(elem)
      else:
        tgt_lst.append(elem_str)
  return tgt_lst
  
def find_similar_intersection(file1, file2):
  lst1 = [x.split('/')[-1] for x in file1]
  lst2 = [x.split('/')[-1] for x in file2]
  #print(lst1, lst2)
  return find_intersection(lst1, lst2)

def get_codex_tokenized_string(tokenizer, input_str, context_len, type='back'):
  '''
  get the codex tokenized string
  '''
  if input_str:
    codex_tokens = tokenizer(input_str)['input_ids']
    if type == 'front':
      truncated_codex_tokens = codex_tokens[:context_len]
    else:
      truncated_codex_tokens = codex_tokens[-context_len:]
    out_str = tokenizer.decode(truncated_codex_tokens)
    return out_str, len(truncated_codex_tokens)
  else:
    return '', 0

def join_lines(lst):
  return ''.join(lst)

# take start line as the first non-commented and non-empty line
def modified_start_line(lines):
  for i in range(len(lines)):
    line = lines[i]
    if line and not (line.startswith('/') or line.startswith('*')): # not part of the license text or empty line
      return i

def get_string(filename, start, end):
  '''
  get the string corresponding to the start and end positions in the parse tree
  '''
  lines = open(filename, encoding="utf8", errors='backslashreplace').readlines()
  start_line, start_char = start
  span_str = ''
  if start_line == 0:
    start_line = modified_start_line(lines)
  end_line, end_char = end
  if start_line <= end_line and start_line < len(lines) and start_line!= -1:
    if start_line == end_line:
      if end_char == -1:
        span_str = lines[start_line]
      else:
        span_str = lines[start_line][start_char:end_char]
    else:
      if start_line + 1 < len(lines):
        span_str = lines[start_line][start_char:] + \
                 join_lines(lines[start_line+1: end_line]) + \
                 lines[end_line][:end_char]
  return span_str



def get_context_from_java_github(out_context_len):
  dataset_filename = os.path.join('preprocessed_data/java_github', 'holes_1.val')
  data = pickle.load(open(dataset_filename, 'rb'))
  out_context_prompts = []
  for i in range(len(data)):
    for j in range(len(data[i])):
      for k in range(len(data[i][j])):
        file_data = data[i][j][k]
        if file_data[0]:
          file_token_str = file_data[0]
          out_context_prompts.append(get_codex_tokenized_string(tokenizer, file_token_str, out_context_len))
  return out_context_prompts