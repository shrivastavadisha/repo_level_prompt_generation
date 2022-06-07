import os
import json
import glob
import numpy as np
from rule_config import *

base_dirs = os.listdir('rule_classifier_data')

modes = ['codex', 'rule']
context_locations = [
                     'in_file', \
                      'parent_class_file', \
                      'import_file',\
                     'sibling_file', \
                     'similar_name_file', \
                     'child_class_file', \
                     'import_of_sibling_file', \
                     'import_of_similar_name_file', \
                     'import_of_parent_class_file', \
                     'import_of_child_class_file'
                    ]


batch_size = 20
total_context_length = 4072

def main():
  commands = []
  for base_repo in base_dirs:
    base_dir = os.path.join('rule_classifier_data', base_repo)
    for repo in os.listdir(base_dir):
      for mode in modes:
        if mode == 'codex':
          command = "python generate_completions.py --mode " + mode \
                    + " --total_context_len " + str(total_context_length)\
                    + " --base_dir " + base_dir\
                    + " --repo_name " + repo\
                    + " --batch_size " + str(batch_size)
          commands.append(command)

        if mode == 'rule':
          for context_location in context_locations:
            context_types = context_type_dict[context_location]
            for context_type in context_types:
              rule_specific_hyperparams = rule_hyperparams[context_type]
              for context_ratio in rule_specific_hyperparams['context_ratio']:
                for prompt_separator in rule_specific_hyperparams['prompt_separator']:
                  for top_k in rule_specific_hyperparams['top_k']:
                    for rule_context_format in rule_specific_hyperparams['rule_context_formatting']:
                        if top_k == -1:
                          command = "python generate_completions.py --mode " + mode\
                                    + " --context_location " + context_location\
                                    + " --context_type " + context_type\
                                    + " --context_division_ratio " + str(context_ratio) \
                                    + " --prompt_separator " + prompt_separator \
                                    + " --top_k " + str(top_k)\
                                    + " --total_context_len " + str(total_context_length)\
                                    + " --base_dir " + base_dir\
                                    + " --repo_name " + repo\
                                    + " --batch_size " + str(batch_size)\
                                    + " --rule_context_formatting " + rule_context_format\

                          commands.append(command)

                        else:
                          for top_k_type in rule_specific_hyperparams['top_k_type']:
                            final_command = command + " --top_k_type " + top_k_type
                            commands.append(final_command)


  with open("commands_completion", 'w') as f:
    f.writelines("%s\n" % command for command in commands)
  f.close()

if __name__ == '__main__':
  main()