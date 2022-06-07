
# context_location and context_type define a rule
context_location = [
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


all_context_types = [ 
                      'method_names_and_bodies',\
                      'method_names',\
                      'identifiers', \
                      'type_identifiers',\
                      'string_literals',\
                      'field_declarations', \
                      'lines'
                      ]

context_type_dict = {}
for con_loc in context_location:
  if con_loc == 'in_file':
    context_types = all_context_types[1:]
  else:
    context_types = all_context_types[:-1]
  context_type_dict[con_loc] = context_types

# rule-specific hyperparams to run. Make changes here to run different configurations
rule_hyperparams = {
                    'lines':
                    {
                    'context_ratio': [0.5, 0.25, 0.75],
                    'top_k': [-1],
                    'prompt_separator': ['space'],
                    'top_k_type':['first'],
                    'rule_context_formatting':['space']
                    },

                    'identifiers':
                    {
                    'context_ratio': [0.5],
                    'top_k': [-1],
                    'prompt_separator': ['newline'],
                    'top_k_type':['first'],
                    'rule_context_formatting':['class_name']
                    },

                    'type_identifiers':
                    {
                    'context_ratio': [0.5],
                    'top_k': [-1],
                    'prompt_separator': ['newline'],
                    'top_k_type':['first'],
                    'rule_context_formatting':['class_name']
                    },

                    'string_literals':
                    {
                    'context_ratio': [0.5],
                    'top_k': [-1],
                    'prompt_separator': ['newline'],
                    'top_k_type':['first'],
                    'rule_context_formatting':['class_name']
                    },

                    'method_names':
                    {
                    'context_ratio': [0.5],
                    'top_k': [-1],
                    'prompt_separator': ['newline'],
                    'top_k_type':['first'],
                    'rule_context_formatting':['class_name']
                    },

                    'field_declarations':
                    {
                    'context_ratio': [0.5],
                    'top_k': [-1],
                    'prompt_separator': ['newline'],
                    'top_k_type':['first'],
                    'rule_context_formatting':['class_name']
                    },

                    'method_names_and_bodies':
                    {
                    'context_ratio': [0.5],
                    'top_k': [-1],
                    'prompt_separator': ['newline'],
                    'top_k_type':['first'],
                    'rule_context_formatting':['class_method_name']
                    }


}
