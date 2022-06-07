import os
import sys
import hashlib
import numpy as np
import argparse

comments = ['*', '/']

def chunk_reader(fobj, chunk_size=1024):
    """Generator that reads a file in chunks of bytes"""
    while True:
        chunk = fobj.read(chunk_size)
        if not chunk:
            return
        yield chunk


def get_hole_count(full_path):
    hole_count = 0
    file_lines = open(full_path, encoding="utf8", errors='backslashreplace').readlines()
    for line in file_lines:
        line = line.strip()
        # omitting comments and empty lines (heuristic: NEED TO DOUBLE CHECK FOR WIDE APPLICABILITY)
        if line and not (np.any([line.startswith(comment) for comment in comments])):
            hole_count+=1
    return hole_count

def check_for_duplicates(paths, hash=hashlib.sha1):
    duplicate_files = {}
    file_count = 0
    hole_count = 0
    hashes = {}
    duplicate_file_paths = []
    for path in paths:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
              if os.path.splitext(filename)[1] == '.java':
                full_path = os.path.join(dirpath, filename)
                hashobj = hash()
                for chunk in chunk_reader(open(full_path, 'rb')):
                    hashobj.update(chunk)
                file_id = (hashobj.digest(), os.path.getsize(full_path))
                duplicate = hashes.get(file_id, None)
                if duplicate:
                    if duplicate not in duplicate_files:
                        duplicate_files[duplicate] = [full_path]
                    else:
                        duplicate_files[duplicate].append(full_path)
                    # print("Duplicate found: ")

                else:
                    hashes[file_id] = full_path

    for k,v in duplicate_files.items():
        num_files = len(v)
        file_count+= num_files
        ind_hole_count = get_hole_count(k)
        hole_count += num_files * ind_hole_count
        duplicate_file_paths.append(k)
        for path in v:
            duplicate_file_paths.append(path)


    print(paths)
    print(len(duplicate_file_paths))
    print(str(file_count) + ", " + str(hole_count))
    with open(os.path.join(paths[0], "duplicates"), "w") as outfile:
        outfile.write("\n".join(duplicate_file_paths))

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

  repo_path = os.path.join(args.base_dir, args.proj_name)
  if os.path.isdir(repo_path):
    check_for_duplicates([repo_path])
