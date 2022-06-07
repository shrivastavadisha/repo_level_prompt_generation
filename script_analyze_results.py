import os
base_dir = 'rule_classifier_data'

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

commands = []
for data_split, data_split_repos in projects.items():
  for proj in data_split_repos:
    proj_name = proj.strip()
    command = "python analyze_results.py --proj_name " + proj_name \
              + " --base_dir " + base_dir + " --data_split " + data_split
    commands.append(command)

with open("commands_analyze_results", 'w') as f:
  f.writelines("%s\n" % command for command in commands)
f.close()

