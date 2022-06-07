import numpy as np
import time
import os
import copy
import pickle
import openai
import argparse
import json
import random
from utils import *
from context import *
from transformers import GPT2TokenizerFast

def setup_args():
  """
  Description: Takes in the command-line arguments from user
  """
  parser = argparse.ArgumentParser()

  parser.add_argument("--seed", type=int, default=9, help="seed for reproducibility")
  parser.add_argument("--base_dir", type=str, default='rule_classifier_data/val', help="base directory for the data")
  parser.add_argument("--repo_name", type=str, default='ircrpgbot', help="name of the repo")

  # completion-related hyperparams
  parser.add_argument("--mode", type=str, default='rule', help="codex, rule")

  parser.add_argument("--batch_size", type=int, default=20, \
                        help="batch size of the prompts to be given to Codex")
  parser.add_argument("--completion_len", type=int, default=24, \
                        help=" length of the 24 completions, so total size will be 4096")
  parser.add_argument("--is_run_rule_full", default=False, action='store_true', \
                        help="whether to run rule based method for the full cases or only rule_triggered cases")

  # context related hyperparams
  parser.add_argument("--total_context_len", type=int, default=100, \
                          help="total size of the context: 24 for completions, so total size will be 4096")
  parser.add_argument("--context_division_ratio", type=float, default=0.5, \
                          help="ratio in which the in-file and out-file context are divided")

  parser.add_argument("--context_location", type=str, default='in_file', \
                          help="where to take context from\
                          NOTE that this is always in addition to the previous prompt, \
                          i.e., in addition to the default prompt for a Codex model")

  parser.add_argument("--context_type", type=str, default='field_declarations',\
                      help="the type of context to be taken. \
                      For possible values for each context_locations see rules file")

  # rule-related hyperparameters
  parser.add_argument("--top_k", type=int, default=-1,\
                      help="k value. A value of -1 indicates taking full context of context_type")
  parser.add_argument("--top_k_type", type=str, default='first', \
                      help="first, last")
  parser.add_argument("--prompt_separator", type=str, default='space', \
                        help="space, newline")
  parser.add_argument("--rule_context_formatting", type=str, default='space', \
                        help="space, newline, method_name, class_name, comment, class_method_name")
  return parser.parse_args()

def generate_prediction(prompt):
  '''
  generate predictions using Codex
  '''
  try:
    response = openai.Completion.create(engine='code-davinci-001',\
                                      prompt=prompt,stop='\n',\
                                      temperature=0.0)

  except:
    print ("Waiting")
    response = None
  return response

def check_hole_scope(hole_pos, class_spans):
  '''
  return the class span of the base class where the cursor is present. If there is no base class, return None
  '''
  for class_span in class_spans:
    cs = int(class_span.split('_')[0])
    ce = int(class_span.split('_')[1])
    l, c = hole_pos
    if l == cs or cs == -1:
      return None
    if cs < l <= ce:
      return class_span

def get_default_prompt(hole_pos=(0,0), context_len=0, tokenizer=None, file=''):

  default_context_obj = getContext(context_location='in_file',
                                    tokenizer=tokenizer,
                                    file=file,
                                    context_len=context_len,
                                    context_scope='pre',\
                                    context_type='lines',\
                                    top_k=-1)

  default_context_obj.set_hole_pos(hole_pos)
  default_prompt, default_prompt_len = default_context_obj.get_line_context()
  return default_prompt, default_prompt_len

def get_prompt(rule_context_obj=None, context_location='in_file',
               total_context_len=4072, rule_triggered=False, parent_class_filename='',
               context_division_ratio=0.5, num_of_lines_to_exclude=0):

  # start by assigning half of the total_context_len to the rule prompt
  rule_context_obj.set_context_len(total_context_len)
  allocated_rule_context_len = int(rule_context_obj.get_context_len()*context_division_ratio)
  rule_context_obj.set_context_len(allocated_rule_context_len)

  if context_location == 'in_file':
    rule_prompt, rule_prompt_len = rule_context_obj.get_in_file_context(num_of_lines_to_exclude)
  if context_location == 'parent_class_file':
    rule_prompt, rule_prompt_len = rule_context_obj.get_parent_class_file_context()
  if context_location == 'import_file':
    rule_prompt, rule_prompt_len = rule_context_obj.get_import_file_context()
  if context_location == 'sibling_file' or context_location == 'reverse_sibling_file':
    rule_prompt, rule_prompt_len = rule_context_obj.get_sibling_file_context()
  if context_location == 'similar_name_file' or context_location == 'reverse_similar_name_file':
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

  # if the rule_prompt_len is shorter than the allocated space, use the extra space for the default_prompt
  if rule_prompt_len < allocated_rule_context_len:
    default_context_len = total_context_len - rule_prompt_len
  else:
    default_context_len = total_context_len - allocated_rule_context_len
  # if something is returned by the rule, it means that the rule is triggered
  if rule_prompt_len > 0:
    rule_triggered = True
  default_prompt, default_prompt_len = get_default_prompt(
                                      hole_pos=getattr(rule_context_obj, 'hole_pos'),
                                      context_len=default_context_len,
                                      tokenizer=getattr(rule_context_obj, 'tokenizer'),
                                      file=getattr(rule_context_obj, 'file')
                                      )
  return rule_prompt, default_prompt, rule_triggered

if __name__ == '__main__':

  args = setup_args()

  #Fix seeds
  np.random.seed(args.seed)
  os.environ['PYTHONHASHSEED'] = str(args.seed)

  os.environ["OPENAI_API_KEY"] = open('openai_api_key', 'r').read().strip()
  openai.api_key = os.getenv("OPENAI_API_KEY")

  #directory for storing results
  input_data_dir = os.path.join(args.base_dir, args.repo_name)
  result_path = os.path.join('results', args.base_dir, args.repo_name)
  os.makedirs(result_path, exist_ok=True)


  # get tokenizer
  tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

  #get stored parsed data
  parsed_data_filename = os.path.join(args.base_dir, args.repo_name, 'parsed_data')
  parse_data = pickle.load(open(parsed_data_filename, 'rb'))
  #get the holes
  hole_filename = os.path.join(args.base_dir, args.repo_name, 'hole_data')
  hole_data = pickle.load(open(hole_filename, 'rb'))

  # get all relevant files (in raw form)
  files = [os.path.join(dp, f) \
                for dp, dn, filenames in os.walk(input_data_dir) \
                for f in filenames \
                if os.path.splitext(f)[1] == '.java']

  # Create the result file for writing the predictions
  result_dir = os.path.join(result_path, args.context_location)
  os.makedirs(result_dir, exist_ok=True)

  if args.mode == 'rule':
    result_filename = args.mode + '_' + args.context_type + '_' + str(args.top_k_type)\
                      + '_' + str(args.top_k) \
                      + '_' + str(int(args.total_context_len * args.context_division_ratio)) \
                      + '_' + args.rule_context_formatting \
                      + '_' + args.prompt_separator 

  if args.mode =='codex':
    result_filename = args.mode + '_' + str(args.total_context_len)

  full_result_filename = os.path.join(result_dir,  result_filename + '.json')
  print(full_result_filename)
  if os.path.isfile(full_result_filename) and os.path.getsize(full_result_filename) > 0:
    print(full_result_filename, " : result file already exists")
    exit()

  f = open(full_result_filename, 'w')

  all_holes = []
  rule_prompts = []
  default_prompts = []
  all_hole_identities = []
  all_rules_triggered = []
  total_count = 0
  rule_triggered_count = 0

  # get the prompts for all files
  for file in files:
    if file in hole_data:
      file_lines = open(file, encoding="utf8", errors='backslashreplace').readlines()

      # define the rule context object. Depends on the file
      rule_context_obj = getContext(context_location = args.context_location, \
                      tokenizer=tokenizer, file=file,
                      parse_data = parse_data,
                      context_type=args.context_type,
                      top_k=args.top_k,top_k_type=args.top_k_type,
                      rule_context_formatting=args.rule_context_formatting,
                      file_lines=file_lines)

      is_out_file = rule_context_obj.is_out_files()

      # go through the holes in the file
      for (l,c) in hole_data[file]: # l = line no, c = character offset within line l
        if total_count%1000 == 0:
          print("Total Count:", total_count)
        hole = file_lines[l][c:]
        hole_identity = file + '_' + str(l) + '_' + str(c)
        hole_pos = (l, c)

        # if mode is codex or we have no parent_class_files or import_files,
        # then get the default prompt directly
        if args.mode == 'codex' or \
        (args.mode == 'rule' and args.context_location != 'in_file' and not is_out_file):
          default_prompt, default_prompt_len = get_default_prompt(hole_pos, args.total_context_len,
                                                                  tokenizer, file)
          rule_triggered = False
          rule_prompt = ''

        else:
          rule_context_obj.set_hole_pos(hole_pos)
          rule_prompt, default_prompt, rule_triggered = \
                                  get_prompt(
                                  rule_context_obj=rule_context_obj, \
                                  context_location=args.context_location, \
                                  total_context_len=args.total_context_len, \
                                  context_division_ratio=args.context_division_ratio)

        #print("RP: ", rule_prompt)
        #print("DP: ", default_prompt)
        if rule_triggered == True:
          rule_triggered_count+=1
        rule_prompts.append(rule_prompt)
        default_prompts.append(default_prompt)
        all_holes.append(hole)
        all_hole_identities.append(hole_identity)
        all_rules_triggered.append(rule_triggered)
        #all_parent_class_filenames.append(parent_class_filename)

        total_count += 1

  print(total_count, rule_triggered_count)
  print(len(all_holes))


  # create prompts only for the cases where the rules are triggered.
  # other cases will be the same as codex, so they can be directly copied from the pre results
  prompts = []
  for i in range(len(all_holes)):
    if (args.mode != 'codex' and args.is_run_rule_full) \
      or (args.mode != 'codex' and not args.is_run_rule_full and all_rules_triggered[i])\
      or (args.mode == 'codex'):
      rule_p = rule_prompts[i]
      def_p = default_prompts[i]
      prompt_separator = args.prompt_separator
      # if rule is empty
      if not rule_p and prompt_separator == 'newline':
        prompt_separator == 'space'
      prompt = rule_p + promptseparator2str[prompt_separator] + def_p

      # make sure that the length of the prompt is less than or equal to the total_context_len
      codex_tokens = tokenizer(prompt)['input_ids']
      if len(codex_tokens) > args.total_context_len:
        codex_tokens = codex_tokens[-args.total_context_len:]
        prompt = tokenizer.decode(codex_tokens)
      if prompt:
        assert len(codex_tokens) <= args.total_context_len, 'prompt length exceeds the maximum length'
        #print("Hole:", all_holes[i])
        #print("Prompt:", prompt)
        prompts.append((i, prompt))

  print(len(prompts))

  # prompt the codex model in batches to generate completions with the prompts created before
  count = 0
  i = 0
  while (i < len(prompts)):
    print(i)
    batch_prompts = prompts[i:i+args.batch_size]
    batch_prompt_texts = [x[1] for x in batch_prompts]
    #print(batch_prompt_post_texts)
    batch_prompt_indexes = [x[0] for x in batch_prompts] # index within the all_* arrays
    batch_responses = generate_prediction(batch_prompt_texts)
    if batch_responses != None:
      for j in range(len(batch_prompts)):
        response = batch_responses.choices[j]
        prediction = response.text
        #prediction_tokens = response.logprobs.tokens
        #prediction_token_logprobs = response.logprobs.token_logprobs
        hole = all_holes[batch_prompt_indexes[j]]
        hole_identity = all_hole_identities[batch_prompt_indexes[j]]
        rule_triggered = all_rules_triggered[batch_prompt_indexes[j]]
        if rule_triggered:
          count+=1
        batch_suffix = ''
        result = {
              'hole_identity': hole_identity, \
              'prediction': prediction, \
              'ground-truth hole': hole, \
              'prompt': batch_prompt_texts[j], \
              'post_prompt': batch_suffix, \
              'rule_triggered': rule_triggered, \
              'index': batch_prompt_indexes[j] + 1 # this index corresponds to the global index
            }
        f.write(json.dumps(result))
        f.write("\n")
        f.flush()
      i = i + args.batch_size
    else:
      # wait for 60s before calling the API again
      time.sleep(60)

f.close()
print(i, j, count, len(prompts))
