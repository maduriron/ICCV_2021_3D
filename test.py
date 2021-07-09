import os
import torch 
from models_scripts import i3_res50, i3_res50_nl, disable_bn, enable_bn
from utilities_scripts import SAM, LR_Scheduler, get_criterion, LoadingBar, Log, initialize, RandAugment
from dataset_scripts import CTDataset, CTDatasetTestSimple
import json

from torch.utils.data import DataLoader
import torchvision

import pandas as pd
####### PARAMETERS ######
batch_size = 1
cuda_device_index = 0
n_class = 2 # extend number of classes
root = "/home/sentic/storage2/iccv_test_madu"
num_workers = 2 # workers for dataloader
fold_test_path = "./test_set_folding.json"
fold_id = "1"
model_paths = [
              "/home/sentic/storage2/iccv_madu/checkpoints/model1_basicAUG_fold1_extended/checkpoint_model8E_60_1_43.pth",
              "/home/sentic/storage2/iccv_madu/checkpoints/model1_basicAUG_fold1_1/checkpoint_model1_150_1_141.pth",
              "/home/sentic/storage2/iccv_madu/checkpoints/model1_basicAUG_fold1_0/checkpoint_model1_1_39.pth",
              "/home/sentic/storage2/iccv_madu/checkpoints/model1_basicAUG_fold2/checkpoint_model4_2_93.pth",
              "/home/sentic/storage2/iccv_madu/checkpoints/model1_basicAUG_fold3/checkpoint_model5_3_99.pth",
              "/home/sentic/storage2/iccv_madu/checkpoints/model1_basicAUG_fold3/checkpoint_model5_3_74.pth",
              "/home/sentic/storage2/iccv_madu/checkpoints/model1_basicAUG_fold4/checkpoint_model6_4_99.pth",
              "/home/sentic/storage2/iccv_madu/checkpoints/model1_basicAUG_fold4/checkpoint_model6_4_98.pth",
              "/home/sentic/storage2/iccv_madu/checkpoints/model1_basicAUG_fold5/checkpoint_model7_5_96.pth",
              "/home/sentic/storage2/iccv_madu/checkpoints/model1_basicAUG_fold5/checkpoint_model7_5_81.pth",

              ]
device = torch.device("cuda:" + str(cuda_device_index) if torch.cuda.is_available() else "cpu")
prepath = ""
replacer = ""
clip_len = 128

output_covid_file = "./results/covid.csv"
output_non_covid_file = "./results/non_covid.csv"
######################################

############## UTILS #################
with open(fold_test_path) as fhandle:
    fold_splitter_test = json.load(fhandle)
    
def find_frames_by_name(fname, paths=fold_splitter_test["1"]["paths"],
                        frames=fold_splitter_test["1"]["frames"]):
    for ix, (p, f) in enumerate(zip(paths, frames)):
        if p == fname:
            return f
    return 0

def transfer_negative_to_positive(old_list, threshold=0.5):
    new_list = []
    for tup in old_list:
        if tup[0] == 1:
            new_list.append(tup)
        else:
            if tup[1] >= threshold:
                new_list.append(tup)
    return new_list

def choose_best_option(predictions_for_input, method="highest",
                      threshold=None):
    if method == "highest":
        if threshold is not None:
            predictions_for_input = [x for x in predictions_for_input if x[0] == 1 or (x[0] == 0 and x[1] >= threshold)]
        return sorted(predictions_for_input, key=lambda x: x[1], reverse=True)[0]
    elif method == "frequent":
        if threshold is None:
            threshold = 0.5
        list_positive = [x for x in predictions_for_input if x[0] == 1 if x[1] >= threshold]
        list_negative = [x for x in predictions_for_input if x[0] == 0 if x[1] >= threshold]
        if len(list_positive) > len(list_negative):
            if len(list_positive) > 0:
                return sorted(list_positive, key=lambda x: x[1], reverse=True)[0]
        else:
            if len(list_negative) > 0:
                return sorted(list_negative, key=lambda x: x[1], reverse=True)[0]
        return sorted(predictions_for_input, key=lambda x: x[1], reverse=True)[0]
    elif method == "weight":
        if threshold is not None:
            predictions_for_input = [x for x in predictions_for_input if x[0] == 1 or (x[0] == 0 and x[1] >= threshold)]
        list_positive = [x for x in predictions_for_input if x[0] == 1]
        list_negative = [x for x in predictions_for_input if x[0] == 0]
        score_positive, score_negative = 0, 0
        if len(list_positive) > 0:
            score_positive = sum([x[1] for x in list_positive]) / len(list_positive)
        if len(list_negative) > 0:
            score_negative = sum([x[1] for x in list_negative]) / len(list_negative)
        if score_positive >= score_negative:
            return (1, score_positive)
        else:
            return (0, score_negative)
        return sorted(predictions_for_input, key=lambda x: x[1], reverse=True)[0]
    
def simple_inference(inputs, leap=0, backward=False, 
                     flipx=False, flipy=False, 
                     original_num_frames=0, 
                     offset=0):
    inputs_clone = inputs.clone().detach()
    if backward == True:
        inputs1 = inputs_clone[:, :, :original_num_frames, :, :]
        inputs1 = torch.flip(inputs1, (2,))
        inputs2 = inputs_clone[:, :, original_num_frames:, :, :]
        inputs_clone = torch.cat([inputs1, inputs2], axis=2)
        del inputs1
        del inputs2
        
    if flipx == True:
        inputs_clone = torch.flip(inputs_clone, (3,))
        
    if flipy == True:
        inputs_clone = torch.flip(inputs_clone, (4,))
        
    if leap != 0:
        inputs_clone = inputs_clone[:, :, offset::leap, :, :]
    return inputs_clone

def decide_score(predictions, predictions_for_input):
    scores = torch.nn.functional.softmax(predictions, dim=1)
    predictions_for_input.append((1, scores[0][1].item()))
    predictions_for_input.append((0, scores[0][0].item()))
    return predictions_for_input
    
########################

###### DATASET STUFF #########
dataset_test = CTDatasetTestSimple(root=root, 
                      fold_id=fold_id, 
                      fold_splitter=fold_splitter_test,
                      transforms=None,
                      replacer="",
                      prepath="",
                      clip_len=clip_len,
                      split="test"
                      )

dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=num_workers)

#########################

##### MAIN LOOP ########

import torch.nn as nn
import torch
from tqdm import tqdm 
d = {}
scores_list = []

threshold_mapper = {0: [0.6, 0.38],
                    1: [0.6, 0.35],
                    2: [0.65, 0.48],
                    
                    3: [0.6, 0.35],
                    
                    4: [0.6, 0.35], 
                    5: [0.75, 0.45],
                    
                    6: [0.55, 0.45],
                    7: [0.55, 0.45],
                    
                    8: [0.65, 0.45],
                    9: [0.7, 0.4]
                   }

for ix_model, path_checkpoint in tqdm(enumerate(model_paths)):
    dict_checkpoint = torch.load(path_checkpoint, map_location='cuda:' + str(cuda_device_index))
    net_state_dict = dict_checkpoint['model_state_dict']

    model = i3_res50_nl(n_class)


    model.load_state_dict(net_state_dict)
    model.to(device)
    model.eval()
    ######################
    with torch.no_grad():
        for batch in dataloader_test:
            inputs, targets = (b.to(device) for b in batch[:2])
            fname = batch[2][0]
            
            if fname not in d:
                d[fname] = {"true": [targets.item()], "predicted": []}
                
            T = inputs.shape[2]
            predictions_for_input = []
            original_num_frames = inputs.shape[2]
            
            if T <= clip_len:
                original_num_frames = find_frames_by_name(fname)
                #################################################
                inputs1 = simple_inference(inputs, leap=0, backward=False, flipx=False, flipy=False, 
                         original_num_frames=original_num_frames, offset=0)
                predictions = model(inputs1) # forward input
                del inputs1
                predictions_for_input = decide_score(predictions, predictions_for_input)
                ##################################################
                inputs1 = simple_inference(inputs, leap=0, backward=True, flipx=False, flipy=False, 
                         original_num_frames=original_num_frames, offset=0)
                predictions = model(inputs1) # forward input
                del inputs1
                predictions_for_input = decide_score(predictions, predictions_for_input)
                ##################################################

                inputs1 = simple_inference(inputs, leap=0, backward=False, flipx=True, flipy=False, 
                         original_num_frames=original_num_frames, offset=0)
                predictions = model(inputs1) # forward input
                del inputs1
                predictions_for_input = decide_score(predictions, predictions_for_input)
                ##################################################

                inputs1 = simple_inference(inputs, leap=0, backward=False, flipx=False, flipy=True, 
                         original_num_frames=original_num_frames, offset=0)
                predictions = model(inputs1) # forward input
                del inputs1
                predictions_for_input = decide_score(predictions, predictions_for_input)
                ##################################################

                inputs1 = simple_inference(inputs, leap=0, backward=False, flipx=True, flipy=True, 
                         original_num_frames=original_num_frames, offset=0)
                predictions = model(inputs1) # forward input
                del inputs1
                predictions_for_input = decide_score(predictions, predictions_for_input)
                ##################################################

                inputs1 = simple_inference(inputs, leap=0, backward=True, flipx=True, flipy=False, 
                         original_num_frames=original_num_frames, offset=0)
                predictions = model(inputs1) # forward input
                del inputs1
                predictions_for_input = decide_score(predictions, predictions_for_input)
                ##################################################

                inputs1 = simple_inference(inputs, leap=0, backward=True, flipx=False, flipy=True, 
                         original_num_frames=original_num_frames, offset=0)
                predictions = model(inputs1) # forward input
                del inputs1
                predictions_for_input = decide_score(predictions, predictions_for_input)
                ##################################################

                inputs1 = simple_inference(inputs, leap=0, backward=True, flipx=True, flipy=True, 
                         original_num_frames=original_num_frames, offset=0)
                predictions = model(inputs1) # forward input
                del inputs1
                predictions_for_input = decide_score(predictions, predictions_for_input)
                ##################################################

            elif T > clip_len and T <= 2 * clip_len:
                inputs1 = simple_inference(inputs, leap=0, backward=False, flipx=False, flipy=False, 
                         original_num_frames=original_num_frames, offset=0)
                predictions = model(inputs1) # forward input
                del inputs1
                predictions_for_input = decide_score(predictions, predictions_for_input)
                ##################################################
                inputs1 = simple_inference(inputs, leap=0, backward=True, flipx=False, flipy=False, 
                         original_num_frames=original_num_frames, offset=0)
                predictions = model(inputs1) # forward input
                del inputs1
                predictions_for_input = decide_score(predictions, predictions_for_input)
                ###################################################


                inputs1 = simple_inference(inputs, leap=0, backward=False, flipx=True, flipy=False, 
                         original_num_frames=original_num_frames, offset=0)
                predictions = model(inputs1) # forward input
                del inputs1
                predictions_for_input = decide_score(predictions, predictions_for_input)
                ##################################################

                inputs1 = simple_inference(inputs, leap=0, backward=False, flipx=False, flipy=True, 
                         original_num_frames=original_num_frames, offset=0)
                predictions = model(inputs1) # forward input
                del inputs1
                predictions_for_input = decide_score(predictions, predictions_for_input)
                ##################################################

                inputs1 = simple_inference(inputs, leap=0, backward=False, flipx=True, flipy=True, 
                         original_num_frames=original_num_frames, offset=0)
                predictions = model(inputs1) # forward input
                del inputs1
                predictions_for_input = decide_score(predictions, predictions_for_input)
                ##################################################

                inputs1 = simple_inference(inputs, leap=0, backward=True, flipx=True, flipy=False, 
                         original_num_frames=original_num_frames, offset=0)
                predictions = model(inputs1) # forward input
                del inputs1
                predictions_for_input = decide_score(predictions, predictions_for_input)
                ##################################################

                inputs1 = simple_inference(inputs, leap=0, backward=True, flipx=False, flipy=True, 
                         original_num_frames=original_num_frames, offset=0)
                predictions = model(inputs1) # forward input
                del inputs1
                predictions_for_input = decide_score(predictions, predictions_for_input)
                ##################################################

                inputs1 = simple_inference(inputs, leap=0, backward=True, flipx=True, flipy=True, 
                         original_num_frames=original_num_frames, offset=0)
                predictions = model(inputs1) # forward input
                del inputs1
                predictions_for_input = decide_score(predictions, predictions_for_input)
                ##################################################
            elif T > 2 * clip_len and T <= 4 * clip_len:
                leap = 2
                for offset in range(leap):
                    inputs1 = simple_inference(inputs, leap=leap, backward=False, flipx=False, flipy=False, 
                                               original_num_frames=original_num_frames, offset=offset)
                    predictions = model(inputs1)
                    del inputs1
                    predictions_for_input = decide_score(predictions, predictions_for_input)

                    inputs1 = simple_inference(inputs, leap=leap, backward=True, flipx=False, flipy=False, 
                                               original_num_frames=original_num_frames, offset=offset)
                    predictions = model(inputs1)
                    del inputs1
                    predictions_for_input = decide_score(predictions, predictions_for_input)
                    ### 
                    inputs1 = simple_inference(inputs, leap=leap, backward=False, flipx=True, flipy=False, 
                             original_num_frames=original_num_frames, offset=offset)
                    predictions = model(inputs1) # forward input
                    del inputs1
                    predictions_for_input = decide_score(predictions, predictions_for_input)
                    ##################################################

                    inputs1 = simple_inference(inputs, leap=leap, backward=False, flipx=False, flipy=True, 
                             original_num_frames=original_num_frames, offset=offset)
                    predictions = model(inputs1) # forward input
                    del inputs1
                    predictions_for_input = decide_score(predictions, predictions_for_input)
                    ##################################################

                    inputs1 = simple_inference(inputs, leap=leap, backward=False, flipx=True, flipy=True, 
                             original_num_frames=original_num_frames, offset=offset)
                    predictions = model(inputs1) # forward input
                    del inputs1
                    predictions_for_input = decide_score(predictions, predictions_for_input)
                    ##################################################

                    inputs1 = simple_inference(inputs, leap=leap, backward=True, flipx=True, flipy=False, 
                             original_num_frames=original_num_frames, offset=offset)
                    predictions = model(inputs1) # forward input
                    del inputs1
                    predictions_for_input = decide_score(predictions, predictions_for_input)
                    ##################################################

                    inputs1 = simple_inference(inputs, leap=leap, backward=True, flipx=False, flipy=True, 
                             original_num_frames=original_num_frames, offset=offset)
                    predictions = model(inputs1) # forward input
                    del inputs1
                    predictions_for_input = decide_score(predictions, predictions_for_input)
                    ##################################################

                    inputs1 = simple_inference(inputs, leap=leap, backward=True, flipx=True, flipy=True, 
                             original_num_frames=original_num_frames, offset=offset)
                    predictions = model(inputs1) # forward input
                    del inputs1
                    predictions_for_input = decide_score(predictions, predictions_for_input)
                    ##################################################
            elif T > 4 * clip_len:
                leap = 3
                for offset in range(leap):
                    inputs1 = simple_inference(inputs, leap=leap, backward=False, flipx=False, flipy=False, 
                                               original_num_frames=original_num_frames, offset=offset)
                    predictions = model(inputs1)
                    del inputs1
                    predictions_for_input = decide_score(predictions, predictions_for_input)

                    inputs1 = simple_inference(inputs, leap=leap, backward=True, flipx=False, flipy=False, 
                                               original_num_frames=original_num_frames, offset=offset)
                    predictions = model(inputs1)
                    del inputs1
                    predictions_for_input = decide_score(predictions, predictions_for_input)

                    ### 
                    inputs1 = simple_inference(inputs, leap=leap, backward=False, flipx=True, flipy=False, 
                             original_num_frames=original_num_frames, offset=offset)
                    predictions = model(inputs1) # forward input
                    del inputs1
                    predictions_for_input = decide_score(predictions, predictions_for_input)
                    ##################################################

                    inputs1 = simple_inference(inputs, leap=leap, backward=False, flipx=False, flipy=True, 
                             original_num_frames=original_num_frames, offset=offset)
                    predictions = model(inputs1) # forward input
                    del inputs1
                    predictions_for_input = decide_score(predictions, predictions_for_input)
                    ##################################################

                    inputs1 = simple_inference(inputs, leap=leap, backward=False, flipx=True, flipy=True, 
                             original_num_frames=original_num_frames, offset=offset)
                    predictions = model(inputs1) # forward input
                    del inputs1
                    predictions_for_input = decide_score(predictions, predictions_for_input)
                    ##################################################

                    inputs1 = simple_inference(inputs, leap=leap, backward=True, flipx=True, flipy=False, 
                             original_num_frames=original_num_frames, offset=offset)
                    predictions = model(inputs1) # forward input
                    del inputs1
                    predictions_for_input = decide_score(predictions, predictions_for_input)
                    ##################################################

                    inputs1 = simple_inference(inputs, leap=leap, backward=True, flipx=False, flipy=True, 
                             original_num_frames=original_num_frames, offset=offset)
                    predictions = model(inputs1) # forward input
                    del inputs1
                    predictions_for_input = decide_score(predictions, predictions_for_input)
                    ##################################################

                    inputs1 = simple_inference(inputs, leap=leap, backward=True, flipx=True, flipy=True, 
                             original_num_frames=original_num_frames, offset=offset)
                    predictions = model(inputs1) # forward input
                    del inputs1
                    predictions_for_input = decide_score(predictions, predictions_for_input)
                    ##################################################
            else:
                print("NOT GOOD")
            new_list = transfer_negative_to_positive(predictions_for_input, 
                                                     threshold=threshold_mapper[ix_model][0])
            best_option = choose_best_option(new_list, method="frequent",
                                             threshold=threshold_mapper[ix_model][1])
            d[fname]["predicted"].append(best_option)
#########################################################################

covid_fnames = []
non_covid_fnames = []

for fname in d:
    new_list = transfer_negative_to_positive(d[fname]["predicted"], threshold=0.6)
    best_option = choose_best_option(new_list, method="frequent", threshold=0.45)
    if best_option[0] == 1:
        covid_fnames.append(os.path.basename(fname))
    elif best_option[0] == 0:
        non_covid_fnames.append(os.path.basename(fname))
        
df_covid  = pd.DataFrame()
df_covid['co1'] = covid_fnames

df_non_covid = pd.DataFrame()
df_non_covid['col1'] = non_covid_fnames

df_covid.T.to_csv(output_covid_file, mode='w', index=False, header=False, sep=",")
df_non_covid.T.to_csv(output_non_covid_file, mode='w', index=False, header=False, sep=",")
