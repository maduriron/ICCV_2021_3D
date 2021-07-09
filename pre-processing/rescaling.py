### FOR TRAIN ###
import os
import SimpleITK as sitk
import numpy as np

from tqdm import tqdm
import cv2

import json 
import pandas as pd
import shutil

dir_train = "/home/sentic/storage2/iccv_test_data/" # directory with the images
out_dir = "/home/sentic/storage2/iccv_test_madu/fold_1/" # directory where to save the images 

for phase_name in tqdm(os.listdir(dir_train)): # train or val
    directory_phase_in = os.path.join(dir_train, phase_name)
    directory_phase_out = os.path.join(out_dir, phase_name)
    
    if os.path.isdir(directory_phase_out) == False:
        os.makedirs(directory_phase_out)
    
    for class_name in os.listdir(directory_phase_in): # covid or non-covid
        class_directory_in = os.path.join(directory_phase_in, class_name)
        class_directory_out = os.path.join(directory_phase_out, class_name)
        
        if os.path.isdir(class_directory_out) == False:
            os.makedirs(class_directory_out)

        for ct_name in os.listdir(class_directory_in):
            ct_path_in = os.path.join(class_directory_in, ct_name)
            ct_path_out = os.path.join(class_directory_out, ct_name)
            
            if os.path.isdir(ct_path_out) == False:
                os.makedirs(ct_path_out)
                
            if len(os.listdir(ct_path_in)) <= 20 or len(os.listdir(ct_path_in)) > 512:
                print("eliminating patient {}".format(ct_name))
                shutil.rmtree(ct_path_out)
            else:
                for fname in os.listdir(ct_path_in):
                    if fname.endswith(".jpg") == False:
                        continue
                    image_path_in = os.path.join(ct_path_in, fname)
                    image_path_out = os.path.join(ct_path_out, fname)

                    slice_array = cv2.imread(image_path_in)
                    scale_percent = 224 / slice_array.shape[1]
                    width = int(slice_array.shape[1] * scale_percent)
                    height = int(slice_array.shape[0] * scale_percent)
                    dim = (width, height)
                    resized = cv2.resize(slice_array, dim, interpolation = cv2.INTER_AREA)
                    cv2.imwrite(image_path_out, resized)

############################################################                    
### FOR TEST ###
directory_in = "/home/sentic/storage2/iccv_test_data"
directory_out = "/home/sentic/storage2/iccv_test_madu"

import os
import SimpleITK as sitk
import numpy as np

from tqdm import tqdm
import cv2

import json 
import pandas as pd
import shutil

if os.path.isdir(directory_out) == False:
    os.makedirs(directory_out)

    
for ct_name in tqdm(os.listdir(directory_in)):
    ct_path_in = os.path.join(directory_in, ct_name)
    ct_path_out = os.path.join(directory_out, ct_name)

    if os.path.isdir(ct_path_out) == False:
        os.makedirs(ct_path_out)
    else:
        continue

    for fname in os.listdir(ct_path_in):
        if fname.endswith(".jpg") == False:
            continue
        image_path_in = os.path.join(ct_path_in, fname)
        image_path_out = os.path.join(ct_path_out, fname)

        slice_array = cv2.imread(image_path_in)
        if slice_array is None:
            print("Problem with {}".format(image_path_in))
            continue
        scale_percent = 224 / slice_array.shape[1]
        width = int(slice_array.shape[1] * scale_percent)
        height = int(slice_array.shape[0] * scale_percent)
        dim = (width, height)
        resized = cv2.resize(slice_array, dim, interpolation = cv2.INTER_AREA)
        cv2.imwrite(image_path_out, resized)
###################################################