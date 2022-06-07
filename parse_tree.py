import os
import pickle
import argparse
from tree_sitter import Language, Parser
from utils import *
import copy

"""
Obtain the parse tree for individual files and collate data at repo-level for rules.
"""

Language.build_library('build/my-languages.so', ['tree-sitter-java']) 

JAVA_LANGUAGE = Language('build/my-languages.so', 'java') 

parser = Parser()
parser.set_language(JAVA_LANGUAGE)


def get_sibling_files(file, all_files):
  file_parts = file.split('/')
  root_dir = '/'.join(file_parts[:-1])
  sibling_files = []
  for f in os.listdir(root_dir):
   if os.path.splitext(f)[1] == '.java' and f != file_parts[-1]:
    sib_file = os.path.join(root_dir, f)
    sibling_files.append(sib_file)
  return sibling_files

def camel_case_split(str):      
  start_idx = [i for i, e in enumerate(str)
               if e.isupper()] + [len(str)]

  start_idx = [0] + start_idx
  return [str[x: y] for x, y in zip(start_idx, start_idx[1:])]

def match_by_parts(file1, file2, split_type):
  # omit .java in the end
  f1 = file1.split('.')[0]
  f2 = file2.split('.')[0]

  if split_type == 'camel-case':
    f1_parts = camel_case_split(f1)
    f2_parts = camel_case_split(f2)

  if split_type == 'underscore':
    f1_parts = f1.split('_')
    f2_parts = f2.split('_')

  for p1 in f1_parts:
    if p1 and p1 in f2_parts:
      #print(split_type, file1, file2, p1, f1_parts, f2_parts)
      return True
  return False


def match_similar_filenames(file1, file2):
  # exactly same name
  if file1 == file2:
    return True

  #camelcase split similar parts
  return match_by_parts(file1, file2, 'camel-case')

  #underscore split similar parts
  return match_by_parts(file1, file2, 'underscore')


def get_similar_name_files(file, all_files):
  filename = file.split('/')[-1]
  similar_name_files = []
  for f in all_files:
    if f != file and match_similar_filenames(f.split('/')[-1], filename):
      similar_name_files.append(f)
  return similar_name_files

def get_tree(filename):
  """
  obtain parse tree for a file
  """
  file_str = open(filename, encoding="utf8", errors='backslashreplace').read()
  tree = parser.parse(bytes(file_str, "utf-8"))
  root_node = tree.root_node
  return root_node

def parse_captures(captures, filename):
  text_spans = []
  for capture in captures:
    #capture[1] = property_name
    start, end = capture[0].start_point, capture[0].end_point
    #text = get_string(filename, start, end)
    text_spans.append((start, end))
  return text_spans

def get_query(attribute_type):

  if attribute_type == 'class_name':
    query = JAVA_LANGUAGE.query("""(class_declaration
                                  name: (identifier) @class_name)""")

  if attribute_type == 'class_body':
    query = JAVA_LANGUAGE.query("""(class_declaration
                                  body: (class_body) @class_body)""")

  if attribute_type == 'parent_class_name':
    query = JAVA_LANGUAGE.query("""(class_declaration
                                  name: (identifier)
                                  superclass: (superclass (type_identifier) @superclass_name))""")

  if attribute_type == 'all_method_name':
    query = JAVA_LANGUAGE.query("""(method_declaration
                                    name: (identifier) @all_method_name)""")

  if attribute_type == 'all_method_body':
    query = JAVA_LANGUAGE.query("""(method_declaration body: (block) @all_method_block)""")

  if attribute_type == 'import_statement':
    query = JAVA_LANGUAGE.query("""(import_declaration (
                                   scoped_identifier
                                   name: (identifier)) @import_statement)""")

  if attribute_type == 'all_field_declaration':
    query = JAVA_LANGUAGE.query("""(field_declaration) @field_declaration""")

  if attribute_type == 'all_string_literal':
    query = JAVA_LANGUAGE.query("""(string_literal) @string_literal""")

  if attribute_type == 'all_identifier':
    query = JAVA_LANGUAGE.query("""(identifier) @identifier""")

  if attribute_type == 'all_type_identifier':
    query = JAVA_LANGUAGE.query("""(type_identifier) @type_identifier""")

  return query

def get_attribute(root_node, filename, attribute_type):

  query = get_query(attribute_type)
  captures = query.captures(root_node)
  if captures:
    attributes = parse_captures(captures, filename)
  else:
    attributes = [((-1, -1), (-1, -1))]
  return attributes

def get_import_path(import_stat, file):
  import_stat_str = get_string(file, import_stat[0], import_stat[1])
  #print(import_stat_str, file)
  import_path_parts = import_stat_str.split(".")
  absolute_import_path = []
  import_path_part = import_path_parts[0]
  if import_path_part != 'java':
    file_path_parts = file.split("/")
    try:
      index_pos = len(file_path_parts) - file_path_parts[::-1].index(import_path_part) - 1
      absolute_import_path = file_path_parts[:index_pos] + import_path_parts
    except ValueError as e:
      print('')
  #print(absolute_import_path)
  if absolute_import_path:
    import_path = '/'.join(absolute_import_path)
    import_path = import_path + '.java'
    return import_path
  else:
    return ''

def get_parent_class_filename(parent_class_name, file_class_info, file):
  parent_class_filename = ''
  if parent_class_name:
    parent_class_name_text = get_string(file, parent_class_name[0], parent_class_name[1])
    # we don't want the current file to be the parent class file
    copy_file_class_info = copy.deepcopy(file_class_info)
    del copy_file_class_info[file]

    if parent_class_name_text:
      # search for the parent class name in all files
      found = False
      for (k,v) in copy_file_class_info.items():
        for val in v:
          if val==parent_class_name_text:
            parent_class_filename = k
            found = True
            break
  return parent_class_filename

def find_relevant_file_identifier(import_identifier, file_identifiers, file):
  candidate_file_identifiers = []
  for file_identifier in file_identifiers:
    if file_identifier:
      file_identifier_str = get_string(file, file_identifier[0], file_identifier[1])
      if file_identifier_str == import_identifier:
        candidate_file_identifiers.append(file_identifier)
  return candidate_file_identifiers[1:]

def get_imports(import_statements, file, all_identifiers, all_type_identifiers):
  imports = {}
  file_identifiers = all_identifiers
  file_identifiers.extend(all_type_identifiers)
  for import_stat in import_statements:
    import_file_path = get_import_path(import_stat, file)
    if import_file_path and os.path.isfile(import_file_path):
      import_identifier = import_file_path.split('/')[-1].split('.')[0]
      candidate_file_identifiers = find_relevant_file_identifier(import_identifier, file_identifiers, file)
      if candidate_file_identifiers:
        imports[import_file_path] = candidate_file_identifiers
  return imports

def check_empty_attribute(attribute):
  if len(attribute) == 1 and attribute[0][0][0] == -1:
      attribute = []
  return attribute

def update_attribute(parse_data, att_type, files):
  count = 0
  for file in files:
    current_file_imports = list(parse_data[file]['imports'].keys())
    att_files = parse_data[file][att_type]
    att_info = []
    for att_file in att_files:
      if att_file:
        att_file_imports = list(parse_data[att_file]['imports'].keys())
        overlapping_imports = find_similar_intersection(att_file_imports, current_file_imports)
        #if len(overlapping_imports) > 0:
        att_info.append((att_file, len(overlapping_imports)))
        #print(file, att_file, overlapping_imports)
    parse_data[file][att_type] = att_info
    if att_info:
      count+=1
      #print(file, parse_data[file][att_type])
  #print(count)
  return parse_data

def update_child_class_info(parse_data, child_class_info):
  for file, file_parse_data in parse_data.items():
    if file in child_class_info:
      parse_data[file]['child_class_filenames'] = child_class_info[file]
    else:
      parse_data[file]['child_class_filenames'] = []
  return parse_data
          
def setup_args():
  """
  Description: Takes in the command-line arguments from user
  """
  parser = argparse.ArgumentParser()

  parser.add_argument("--seed", type=int, default=9, help="seed for reproducibility")
  parser.add_argument("--base_dir", type=str, default='rule_classifier_data/val', \
                            help="base directory for the data")
  parser.add_argument("--proj_name", type=str, default='rsbotownversion', \
                            help="name of the input repo")

  return parser.parse_args()

if __name__ == '__main__':

  args = setup_args()

  #Fix seeds
  np.random.seed(args.seed)
  os.environ['PYTHONHASHSEED']=str(args.seed)

  input_data_path = os.path.join(args.base_dir, args.proj_name)
  os.makedirs(input_data_path, exist_ok=True)

  files = [os.path.join(dp, f) \
            for dp, dn, filenames in os.walk(input_data_path) \
            for f in filenames \
            if os.path.splitext(f)[1] == '.java']

  file_class_info = {}
  for file in files:
    root_node = get_tree(file)
    class_names = get_attribute(root_node, file, 'class_name')
    file_class_names = []
    for cn in class_names:
      start, end = cn
      class_name = get_string(file, start, end)
      file_class_names.append(class_name)
    file_class_info[file] = file_class_names
  #print(file_class_info)

  with open(os.path.join(input_data_path, 'file_class_data'), 'wb') as f:
    pickle.dump(file_class_info, f)

  parse_data = {}
  child_class_info = {}

  similar_count = 0
  sibling_count = 0

  for file in files:
      root_node = get_tree(file)
      sibling_files = get_sibling_files(file, files)
      similar_name_files = get_similar_name_files(file, files)
      if len(similar_name_files) > 0:
        similar_count +=1
      if len(sibling_files) > 0:
        sibling_count +=1

      class_names = get_attribute(root_node, file, 'class_name')
      class_bodies = get_attribute(root_node, file, 'class_body')
      parent_class_names = get_attribute(root_node, file, 'parent_class_name')
      all_field_declarations = get_attribute(root_node, file, 'all_field_declaration')
      all_string_literals = get_attribute(root_node, file, 'all_string_literal')
      all_identifiers = get_attribute(root_node, file, 'all_identifier')
      all_type_identifiers = get_attribute(root_node, file, 'all_type_identifier')
      all_method_names = get_attribute(root_node, file, 'all_method_name')
      all_method_bodies = get_attribute(root_node, file, 'all_method_body')
      import_statements = get_attribute(root_node, file, 'import_statement')

      class_names = check_empty_attribute(class_names)
      class_bodies = check_empty_attribute(class_bodies)
      parent_class_names = check_empty_attribute(parent_class_names)
      all_field_declarations = check_empty_attribute(all_field_declarations)
      all_identifiers = check_empty_attribute(all_identifiers)
      all_type_identifiers = check_empty_attribute(all_type_identifiers)
      all_string_literals = check_empty_attribute(all_string_literals)
      all_method_names = check_empty_attribute(all_method_names)
      all_method_bodies = check_empty_attribute(all_method_bodies)
      import_statements = check_empty_attribute(import_statements)

      # get imports
      imports = get_imports(import_statements, file, all_identifiers, all_type_identifiers)

      parent_class_filenames = []
      mod_parent_class_names = []
      for parent_class_name in parent_class_names:
        parent_class_filename = get_parent_class_filename(parent_class_name, file_class_info, file)
        if parent_class_filename:
          mod_parent_class_names.append(parent_class_name)
          if parent_class_filename in child_class_info:
            child_class_info[parent_class_filename].append(file)
          else:
            child_class_info[parent_class_filename] = [file]
          parent_class_filenames.append(parent_class_filename)

      #print(parent_class_names, parent_class_filenames)
      assert len(mod_parent_class_names) == len(parent_class_filenames)

      #store the data in dict form
      parse_data[file] = {
                          'class_names': class_names,\
                          'class_bodies': class_bodies, \
                          'parent_class_names': mod_parent_class_names, \
                          'parent_class_filenames': parent_class_filenames, \
                          'imports': imports, \
                          'field_declarations': all_field_declarations, \
                          'string_literals': all_string_literals, \
                          'identifiers': all_identifiers, \
                          'type_identifiers': all_type_identifiers, \
                          'all_method_names': all_method_names, \
                          'all_method_bodies': all_method_bodies, \
                          'sibling_files': sibling_files, \
                          'similar_name_files': similar_name_files}

  print(len(files), sibling_count, similar_count)
  print("updating sibling files")
  parse_data = update_attribute(parse_data, 'sibling_files', files)
  print("updating similar_name_files")
  parse_data = update_attribute(parse_data, 'similar_name_files', files)
  print("updating child class filenames")
  parse_data = update_child_class_info(parse_data, child_class_info)
  parse_data = update_attribute(parse_data, 'child_class_filenames', files)
  print("updating parent class filenames")
  parse_data = update_attribute(parse_data, 'parent_class_filenames', files)

  print("Writing parse data...")
  with open(os.path.join(input_data_path, 'parsed_data'), 'wb') as f:
    pickle.dump(parse_data, f)
