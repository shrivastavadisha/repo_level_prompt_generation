from context import *
from utils import *
from rule_config import *
from transformers import GPT2TokenizerFast

context_location_conversion = {
                    'in_file':'in_file', \
                    'parent_class_file':'parent_class_file', \
                    'import_file':'import_file',\
                    'sibling_file':'sibling_files', \
                    'similar_name_file':'similar_name_files', \
                    'child_class_file':'child_class_filenames', \
                    'import_of_sibling_file':'sibling_files', \
                    'import_of_similar_name_file':'similar_name_files', \
                    'import_of_parent_class_file':'parent_class_filenames', \
                    'import_of_child_class_file':'child_class_filenames'
                    }


class RuleDatasetUtils():
  def __init__(self, file, parse_datas, hole_pos, tokenizer):
    super(RuleDatasetUtils, self).__init__()
    #mod_file = '/'. join(['data', 'gcode-data'] + file.split('/')[2:])
    self.file = file
    self.parse_datas = parse_datas
    self.hole_pos = hole_pos
    self.tokenizer = tokenizer
    #self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

  def get_relevant_files(self, context_location):
    if context_location == 'in_file':
      return [self.file]
    else:
      rule_context_obj = getContext(context_location = context_location, file=self.file, parse_data = self.parse_datas)
      rule_context_obj.set_hole_pos(self.hole_pos)
      if rule_context_obj.is_out_files():
        if context_location == 'parent_class_file':
          relevant_file, _ = rule_context_obj.get_parent_class_filename()
          relevant_files = []
        elif context_location == 'import_file':
          relevant_files = rule_context_obj.get_relevant_import_files()
        elif 'import_of_' in context_location:
          relevant_files = rule_context_obj.get_relevant_import_of_att_files(context_location_conversion[context_location])
        else:
          relevant_files = rule_context_obj.get_relevant_files(context_location_conversion[context_location])
        return relevant_files
      else:
        return []

  def get_usages_from_context_location(self, hole_attributes, context_location):
    cl_files = self.get_relevant_files(context_location)
    for hole_att in hole_attributes:
      for cl_file in cl_files:
        print(cl_file)
        file_attributes = self.parse_datas[cl_file]['identifiers']
        att_usages = find_usages(hole_att, self.file, file_attributes, cl_file)
        if not att_usages:
          continue
        else:
          return (cl_file, att_usages)

  def get_all_usages(self, hole_attributes):
    usages = {}
    for context_location in context_location_conversion.keys():
      print(context_location)
      usages[context_location] = self.get_usages_from_context_location(hole_attributes, context_location)      
    return usages

  def get_default_prompt(self, context_len):

    default_context_obj = getContext(context_location='in_file',
                                      tokenizer=self.tokenizer,
                                      file=self.file,
                                      context_len=context_len,
                                      context_scope='pre',\
                                      context_type='lines',\
                                      top_k=-1)

    default_context_obj.set_hole_pos(self.hole_pos)
    default_prompt, default_prompt_len = default_context_obj.get_line_context()
    return default_prompt, default_prompt_len


  def get_all_rules_context(self, num_of_lines_to_exclude=0):
    rule_prompts = []
    rule_indexes = []
    total_context_len = self.tokenizer.model_max_length
    

    for key, val in combined_to_index.items():
      rule_prompt = ''
      if key == 'codex':
        rule_prompt, rule_prompt_len = self.get_default_prompt(context_len=total_context_len)
      if key != 'codex':
        context_location, context_type, context_division_ratio = key.split('#')
        rule_context_formatting = rule_hyperparams[context_type]['rule_context_formatting'][0]
        rule_context_obj = getContext(context_location = context_location, file=self.file, parse_data = self.parse_datas, \
                                      context_type=context_type, rule_context_formatting=rule_context_formatting, \
                                      tokenizer = self.tokenizer, context_len = total_context_len) 

        allocated_rule_context_len = int(rule_context_obj.get_context_len()*float(context_division_ratio))
        rule_context_obj.set_context_len(allocated_rule_context_len)
        rule_context_obj.set_hole_pos(self.hole_pos)

        if context_location == 'in_file':
          rule_prompt, rule_prompt_len = rule_context_obj.get_in_file_context(num_of_lines_to_exclude)

        # there are files for this context location except in_file context location
        is_out_files = rule_context_obj.is_out_files() 
        if is_out_files:
          if context_location == 'parent_class_file':
            rule_prompt, rule_prompt_len = rule_context_obj.get_parent_class_file_context()
          if context_location == 'import_file':
            rule_prompt, rule_prompt_len = rule_context_obj.get_import_file_context()
          if context_location == 'sibling_file':
            rule_prompt, rule_prompt_len = rule_context_obj.get_sibling_file_context()
          if context_location == 'similar_name_file':
            rule_prompt, rule_prompt_len = rule_context_obj.get_similar_name_file_context()
          if context_location == 'child_class_file':
            rule_prompt, rule_prompt_len = rule_context_obj.get_child_class_file_context()
          if context_location == 'import_of_similar_name_file':
            rule_prompt, rule_prompt_len = rule_context_obj.get_import_of_similar_name_file_context()
          if context_location == 'import_of_parent_class_file':
            rule_prompt, rule_prompt_len = rule_context_obj.get_import_of_parent_class_file_context()
          if context_location == 'import_of_child_class_file':
            rule_prompt, rule_prompt_len = rule_context_obj.get_import_of_child_class_file_context()
          if context_location == 'import_of_sibling_file':
            rule_prompt, rule_prompt_len = rule_context_obj.get_import_of_sibling_file_context()

      if rule_prompt:
        rule_prompts.append(rule_prompt)
        rule_indexes.append(val)

    return rule_prompts, rule_indexes



