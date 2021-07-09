import re
import json
import os

mapper = {"non-covid": 0,
         "covid": 1}


regex = re.compile("/([^/]*)/ct_scan_(\d+)")

def extract_patient_metadata(path, regex=regex, mapper=mapper):
    metadata = re.findall(regex, path)
    patient_id = 0
    label = 0
    
    if len(metadata) > 0:
        patient_id = int(metadata[0][1])
        label = mapper[metadata[0][0]]
    else:
        print("Unable to access for path {}".format(path))
    
    return {"patient_id": patient_id,
           "label": label}

fold_splitter_train = {"1": {"paths": [],
                            "metadata": [],
                            "frames": []}, 
                }

fold_splitter_valid = {"1": {"paths": [],
                            "metadata": [],
                            "frames": []}, 
                }

ix_fold = 1
total_folds = 1
dir_data = "/home/sentic/storage2/iccv_madu/"

while ix_fold <= total_folds:
    fold_path = os.path.join(dir_data, "fold_" + str(ix_fold))
    
    fold_path_train = os.path.join(fold_path, "train")
    fold_path_valid = os.path.join(fold_path, "val")
    
    fold_path_train_covid = os.path.join(fold_path_train, "covid")
    fold_path_train_noncovid = os.path.join(fold_path_train, "non-covid")
    
    fold_path_valid_covid = os.path.join(fold_path_valid, "covid")
    fold_path_valid_noncovid = os.path.join(fold_path_valid, "non-covid")
    
    for fname in os.listdir(fold_path_train_covid):
        full_path = os.path.join(fold_path_train_covid, fname)
        fold_splitter_train[str(ix_fold)]["paths"].append(full_path)
        fold_splitter_train[str(ix_fold)]["metadata"].append(1) # is covid |-> label := 1
        fold_splitter_train[str(ix_fold)]["frames"].append(len(os.listdir(full_path)))
        
    for fname in os.listdir(fold_path_train_noncovid):
        full_path = os.path.join(fold_path_train_noncovid, fname)
        fold_splitter_train[str(ix_fold)]["paths"].append(full_path)
        fold_splitter_train[str(ix_fold)]["metadata"].append(0) # is non-covid |-> label := 1
        fold_splitter_train[str(ix_fold)]["frames"].append(len(os.listdir(full_path)))
        
    for fname in os.listdir(fold_path_valid_covid):
        full_path = os.path.join(fold_path_valid_covid, fname)
        fold_splitter_valid[str(ix_fold)]["paths"].append(full_path)
        fold_splitter_valid[str(ix_fold)]["metadata"].append(1) # is covid |-> label := 1
        fold_splitter_valid[str(ix_fold)]["frames"].append(len(os.listdir(full_path)))
        
    for fname in os.listdir(fold_path_valid_noncovid):
        full_path = os.path.join(fold_path_valid_noncovid, fname)
        fold_splitter_valid[str(ix_fold)]["paths"].append(full_path)
        fold_splitter_valid[str(ix_fold)]["metadata"].append(0) # is non-covid |-> label := 1
        fold_splitter_valid[str(ix_fold)]["frames"].append(len(os.listdir(full_path)))
    
    
import json

with open("./train_folding.json", "w") as fhandle:
    json.dump(fold_splitter_train, fhandle)
    
with open("./valid_folding.json", "w") as fhandle:
    json.dump(fold_splitter_valid, fhandle)
    
import json

with open("./train_folding.json", "r") as fhandle:
    fold_splitter_train = json.load(fhandle)
    
with open("./valid_folding.json", "r") as fhandle:
    fold_splitter_valid = json.load(fhandle)
    
    
# I) Make sure you don t put in valid what you already had some previous fold
# II) Sample from the training fold to create the distribution in the valid dataset
#   bin    |   val covid    |  val non-covid |
#_________________________________________________
#  20/120   |     77        |       117      |
#  120/240  |     13        |        9       |
#  240/360  |     47        |       76       |
#  360/480  |     27        |       30       |
#  480/600  |     2         |        6       |
#  600/*    |     1         |        1       |

already_in_valid = []

for ix_fold in fold_splitter_valid:
    already_in_valid += fold_splitter_valid[ix_fold]["paths"]
    
initial_training_paths = fold_splitter_train["1"]["paths"]
initial_training_labels = fold_splitter_train["1"]["metadata"]
initial_training_frames = fold_splitter_train["1"]["frames"]

initial_valid_paths = fold_splitter_valid["1"]["paths"]
initial_valid_labels = fold_splitter_valid["1"]["metadata"]
initial_valid_frames = fold_splitter_valid["1"]["frames"]



covid_1 = [(p, l, f) for (p, l, f) in zip(initial_training_paths, initial_training_labels, initial_training_frames)
          if l == 1 and 20 <= f and f < 120]

covid_2 = [(p, l, f) for (p, l, f) in zip(initial_training_paths, initial_training_labels, initial_training_frames)
          if l == 1 and 120 <= f and f < 240]

covid_3 = [(p, l, f) for (p, l, f) in zip(initial_training_paths, initial_training_labels, initial_training_frames)
          if l == 1 and 240 <= f and f < 360]

covid_4 = [(p, l, f) for (p, l, f) in zip(initial_training_paths, initial_training_labels, initial_training_frames)
          if l == 1 and 360 <= f and f < 480]

covid_5 = [(p, l, f) for (p, l, f) in zip(initial_training_paths, initial_training_labels, initial_training_frames)
          if l == 1 and 480 <= f and f < 512]
###########################################################################################################
non_covid_1 = [(p, l, f) for (p, l, f) in zip(initial_training_paths, initial_training_labels, initial_training_frames)
          if l == 0 and 20 <= f and f < 120]

non_covid_2 = [(p, l, f) for (p, l, f) in zip(initial_training_paths, initial_training_labels, initial_training_frames)
          if l == 0 and 120 <= f and f < 240]

non_covid_3 = [(p, l, f) for (p, l, f) in zip(initial_training_paths, initial_training_labels, initial_training_frames)
          if l == 0 and 240 <= f and f < 360]

non_covid_4 = [(p, l, f) for (p, l, f) in zip(initial_training_paths, initial_training_labels, initial_training_frames)
          if l == 0 and 360 <= f and f < 480]

non_covid_5 = [(p, l, f) for (p, l, f) in zip(initial_training_paths, initial_training_labels, initial_training_frames)
          if l == 0 and 480 <= f and f < 512]

import random
list_splits = [covid_1, covid_2, covid_3, covid_4, covid_5,
               non_covid_1, non_covid_2, non_covid_3, non_covid_4, non_covid_5]
constraints = {"1": 74, "2": 8, "3": 56, "4": 28, "5": 2, # covid distribution 
               "6": 107, "7": 13, "8": 60, "9": 29, "10": 5 # non-covid distribution
                }
new_fold_idxs = ["2", "3", "4", "5"]

for enum_ix_fold, new_fold_idx in enumerate(new_fold_idxs):
    fold_splitter_train[new_fold_idx] = {"paths": [],
                                        "metadata": [],
                                        "frames": []}
    fold_splitter_valid[new_fold_idx] = {"paths": [],
                                        "metadata": [],
                                        "frames": []}
    for ix, sample in enumerate(list_splits):
        num_from_sample = constraints[str(ix + 1)]
        to_add_in_val = random.sample(sample, num_from_sample)
        fold_splitter_valid[new_fold_idx]["paths"] += [p for (p, _, _) in to_add_in_val]
        fold_splitter_valid[new_fold_idx]["metadata"] += [l for (_, l, _) in to_add_in_val]
        fold_splitter_valid[new_fold_idx]["frames"] += [f for (_, _, f) in to_add_in_val]
        
        #print(len(fold_splitter_train[new_fold_idx]["paths"]))
        
        fold_splitter_train[new_fold_idx]["paths"] += [p for (p, l, f) in sample if (p, l, f) not in to_add_in_val]
        fold_splitter_train[new_fold_idx]["metadata"] += [l for (p, l, f) in sample if (p, l, f) not in to_add_in_val]
        fold_splitter_train[new_fold_idx]["frames"] += [f for (p, l, f) in sample if (p, l, f) not in to_add_in_val]        

    fold_splitter_train[new_fold_idx]["paths"] += [p for p in initial_valid_paths]
    fold_splitter_train[new_fold_idx]["metadata"] += [l for l in initial_valid_labels]
    fold_splitter_train[new_fold_idx]["frames"] += [f for f in initial_valid_frames]
    
import json

with open("./train_folding.json", "w") as fhandle:
    json.dump(fold_splitter_train, fhandle)
    
with open("./valid_folding.json", "w") as fhandle:
    json.dump(fold_splitter_valid, fhandle)
    
fold_splitter_test = {"1": {"paths": [],
                            "metadata": [],
                            "frames": []}, 
                }
dir_test_imags = "/home/sentic/storage2/iccv_test_madu"
import os

for dir_name in os.listdir(dir_test_imags):
    fold_splitter_test["1"]["paths"].append(os.path.join(dir_test_imags, dir_name))
    fold_splitter_test["1"]["metadata"].append(2)
    fold_splitter_test["1"]["frames"].append(len(os.listdir(os.path.join(dir_test_imags, dir_name))))
    
import json

with open("./test_set_folding.json", "w") as fhandle:
    json.dump(fold_splitter_test, fhandle)
    ix_fold += 1
    
