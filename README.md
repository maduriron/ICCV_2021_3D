# ICCV_2021_3D
This repository contains code for the approach of SenticLAB.UAIC team on the Covid Detection competition on CTs.

The train script is train.py.
The test script is test.py.
You need to have all the images rescaled to 224X224 (so I don't waste time to use the Resize Transform each time I train/inference). Check pre-processing/rescale.py.
You also need to have the specific folding files for fold training. Check pre-processing/making_fold.py
