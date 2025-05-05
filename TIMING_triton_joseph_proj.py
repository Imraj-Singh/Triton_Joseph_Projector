from triton_joseph_proj import *
import numpy as np
import matplotlib.pyplot as plt
import time
import array_api_compat.torch as xp


##### CHECK if the test data was created glob slices_*_lors_*.npy

from glob import glob
all_test_data = glob("slices_*_lors_*.npy")
if len(all_test_data) == 0:
    print("No test data found. Please run the create_test_data.py script to generate it.")
    exit()

import os
if os.path.exists("TIMING_triton_joseph_proj.txt"):
    os.remove("TIMING_triton_joseph_proj.txt")
    print("Deleted existing TIMING_triton_joseph_proj.txt file.")

# extract slices and lors from the filename
slices = []
lors = []
for test_data in all_test_data:
    parts = test_data.split("_")
    slices.append(int(parts[1]))
    lors.append(int(parts[3].split(".")[0]))
    print(f"Found test data: {test_data} with slices: {slices[-1]} and lors: {lors[-1]}")

    dict_data = np.load(test_data, allow_pickle=True).item()
    xstart = xp.asarray(dict_data['xstart']).cuda()
    xend = xp.asarray(dict_data['xend']).cuda()
    img_origin = xp.asarray(dict_data['img_origin']).cuda()
    voxel_size = xp.asarray(dict_data['voxel_size']).cuda()
    img_shape = dict_data['img_shape']
    x_gt = xp.asarray(dict_data['x']).cuda()
    meas = xp.asarray(dict_data['meas']).cuda() + 100

    fwd = lambda x: joseph3d_fwd_vec(xstart.reshape(-1,3), xend.reshape(-1,3), x, img_origin, voxel_size, img_shape).reshape(meas.shape)

    back = lambda y: joseph3d_back_vec(xstart.reshape(-1,3), xend.reshape(-1,3), y.ravel(), img_origin, voxel_size, img_shape)

    tmp = fwd(x_gt)
    torch.cuda.synchronize(x_gt.device)
    start = time.time()
    for i in range(10):
        tmp = fwd(x_gt)
        torch.cuda.synchronize(x_gt.device)
    end_t = time.time()
    forward_timing = (end_t - start) / 10

    sens_img = back(xp.ones_like(meas))
    x = xp.ones_like(sens_img)
    torch.cuda.synchronize(x_gt.device)

    start = time.time()
    for i in range(10):
        tmp = back(meas)
        torch.cuda.synchronize(x_gt.device)
    end_t = time.time()
    backward_timing = (end_t - start) / 10

    start = time.time()
    for i in range(10):
        em_precond = x/sens_img
        ratio = meas/(fwd(x)+100.)
        torch.cuda.synchronize(x_gt.device)
        bp_ratio = back(ratio)
        torch.cuda.synchronize(x_gt.device)
        x = em_precond*bp_ratio
    end_t = time.time()
    mlem_timing = (end_t - start) / 10

    number_of_lors = xstart.shape[0]*xstart.shape[1]*xstart.shape[2]
    number_of_voxels = x.shape[0]*x.shape[1]*x.shape[2]

    # save the timings to a file
    with open("TIMING_triton_joseph_proj.txt", "a") as f:
        f.write(f"{lors[-1]} {slices[-1]} {number_of_lors} {number_of_voxels} {forward_timing} {backward_timing} {mlem_timing}\n")
        print(f"Saved timings for slices: {slices[-1]} and lors: {lors[-1]} to timings.txt")