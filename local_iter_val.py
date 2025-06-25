"""
The code was adapted from the CVPR24 Segment Anything in Medical Images on a Laptop Challenge
https://www.codabench.org/competitions/1847/ 

pip install connected-components-3d
pip install cupy-cuda12x
pip install cucim-cu12


The testing images will be evaluated one by one.

Folder structure:
CVPR25_iter_eval.py
--docker_folder path # submitted docker containers from participants
    - docker_dir
        - teamname_1.tar.gz 
        - teamname_2.tar.gz
        - ...
--test_img_path # test images
    - imgs
        - case1.npz  # test image
        - case2.npz  
        - ...   
--save_path  # segmentation results
    - output
        - case1.npz  # segmentation file name is the same as the testing image name
        - case2.npz  
        - ...
--validation_gts_path # path to validation / test set GT files
    -   Contains the npz files with the same name as the images but only 'gts' key is available in each file instead of storing it in the image itself. This is done to prevent label leakage during the challenge.
    - validation_gts
        - case1.npz  # file containing only the 'gts' key
        - case2.npz  
        - ...
--verbose
    -   Whether to have a more detailed output, e.g. coordinates of generated clicks


This script is designed for evaluating docker submissions for the CVPR25: Foundation Models for Interactive 3D Biomedical Image Segmentation Challenge Challenge

##########################################################
######### Docker Submission Evaluation Process ###########
##########################################################
Submissions for the CVPR 2025: Foundation Models for Interactive 3D Biomedical Image Segmentation Challenge will be evaluated using an iterative refinement approach. 
Each participant's Docker container will be tested on a set of medical images provided as .npz files. 
The evaluation process follows these key steps:
    -   Initial Prediction: Image +  Bounding Box Prompt (1 prediction)
        -   Each test case begins with a bounding box prompt, specified in the 'bbox' key of the test image. This serves as the starting point for the segmentation.
    -   Iterative Click Refinements: Image + Bounding Box + 1-5 Clicks (5 predictions)
        -   After the initial segmentation, we iteratively simulate 5 refinement clicks to address segmentation errors. These clicks are automatically generated based on the center of the largest error region in the current prediction:
            -   If the center of the largest error is an undersegmentation, we simulate and place a foreground click.
            -   If the center of the largest error is an oversegmentation, we simulate and place a background click.
        -   The clicks are stored in the clicks key of the 'npz' file and progressively updated during the second step of the evaluation.

###############################################################
######### How are interactions (bbox, clicks) stored? #########
###############################################################
The interactions are stored in the 'bbox' and 'clicks' keys of each input .npz image.
    - The bounding box is stored in the 'bbox' key as a list of dictionaries [{'z_min': 27, 'z_max': 396, 'z_mid': 311, 'z_mid_x_min': 175, 'z_mid_y_min': 94, 'z_mid_x_max': 278, 'z_mid_y_max': 233}, ...] containing bbox coordinates for each class.
    - The clicks are provided in the 'clicks' key as a list of dictionaries [{'fg': [click_fg_1, clicks_fg_2,...], 'bg': [click_bg_1, click_bg_2,...]}, ...] 
where click_fg_i and click_bg_i are 3-element arrays with the 3D click coordinates [x, y, z].

#######################################
######### Performance Metrics #########
#######################################
For each image, multi-class segmentation quality is evaluated using:
-   Dice Similarity Coefficient (DSC) and Normalized Surface Dice (NSD), calculated iteratively over the 6 steps (bounding box + 5 clicks).
-   AUC (Area Under the Curve) for DSC and NSD to measure cumulative improvement with more interactions.
-   Final DSC and NSD after all interactions.
-   Inference Time averaged over all 6 steps.

##########################
######### Output #########
##########################
Results are saved in .npz format with metrics compiled into a CSV file for each submission. 5 metrics are stored: DSC_AUC, NSD_AUC, Final_DSC, Final_NSD, Inference Time.


################################
######### Script Steps #########
################################
This script executes the following steps:
1. Docker Submission Handling:
   - Loads docker containers submitted by participants.
   - Executes inference for each test image using the participant's docker container. Images are infered one by one.

2. Iterative Refinement:
   - The initial bounding box prediction is refined iteratively by simulating user clicks at the centers of segmentation errors for each class in the image.
   - The Euclidean Distance Transform (EDT) is computed for error regions to identify the distance to the boundary of each error component,
     ensuring clicks are placed at locations at the center of the largest error for refinement.
   - For each image, the docker is run 6 times for inference:
        - 1) Bounding Box initial prediction
        - 2)-6) Click refinement predictions (each new click is placed in the center of the largest error component)
            - If the center of the largest error is part of the background --> a background click is placed
            - Otherwise, a foreground click is placed
        - Steps 1)-6) are done in parallel for all segmentation classes in 6 interaction steps (6 docker runs) 

3. GPU vs. CPU Computation:
   - If a GPU is available, the script uses `cupy` and `cucim` for accelerated EDT computation.
   - For CPU-only environments, `scipy.ndimage.distance_transform_edt` is used as a fallback.

4. Metrics Calculation:
   - Computes multi-class DSC and NSD for each image.
   - For the final metrics, the AUC (Area Under the Curve) for the DSC and NSD are computed for iterative improvement across the 6 interactive iterations. 
        - The AUC quantifies the cumulative performance improvement over the 6 successive iterations (bbox + 5 clicks) providing a holistic view of the segmentation refinement process.
   - The final DSC and NSD after all 6 interactive steps are also computed.
        - These metrics reflect the final segmentation quality achieved after all refinements, indicating the model's final performance.
   - The last metric is the inference time which is the average inference time over the 6 interactive steps.

5. Output:
   - Segmentation results are saved in the specified output directory. 
        -   Final prediction in the 'segs' key
        -   Intermediate prediction in the 'all_segs' key
   - Metrics for each test case are compiled into a CSV file.

#################################
############## Misc##############
#################################
- The input image also contains the 'prev_pred' key which stores the prediction from the previous iteration. This is used only to help with submission that are using the previous prediction as an additional input and is not
a mandatory input.
"""

import os
join = os.path.join
import shutil
import time
import torch
import argparse
from collections import OrderedDict
import pandas as pd
import numpy as np
import traceback

from scipy.ndimage import distance_transform_edt 
import cc3d
from SurfaceDice import compute_surface_distances, compute_surface_dice_at_tolerance, compute_dice_coefficient
from scipy import integrate
from tqdm import tqdm

# Taken from CVPR24 challenge code with change to np.unique
def compute_multi_class_dsc(gt, seg):
    dsc = []
    for i in np.sort(pd.unique(gt.ravel()))[1:]: # skip bg
        gt_i = gt == i
        seg_i = seg == i
        dsc.append(compute_dice_coefficient(gt_i, seg_i))
    return np.mean(dsc)

# Taken from CVPR24 challenge code with change to np.unique
def compute_multi_class_nsd(gt, seg, spacing, tolerance=2.0):
    nsd = []
    for i in np.sort(pd.unique(gt.ravel()))[1:]: # skip bg
        gt_i = gt == i
        seg_i = seg == i
        surface_distance = compute_surface_distances(
            gt_i, seg_i, spacing_mm=spacing
        )
        nsd.append(compute_surface_dice_at_tolerance(surface_distance, tolerance))
    return np.mean(nsd)

def patched_np_load(*args, **kwargs):
    with np.load(*args, **kwargs) as f:
        return dict(f) 

def sample_coord(edt):
    # Find all coordinates with max EDT value
    np.random.seed(42)

    max_val = edt.max()
    max_coords = np.argwhere(edt == max_val)

    # Uniformly choose one of them
    chosen_index = max_coords[np.random.choice(len(max_coords))]

    center = tuple(chosen_index)
    return center

# Compute the EDT with same shape as the image
def compute_edt(error_component):
    # Get bounding box of the largest error component to limit computation
    coords = np.argwhere(error_component)
    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0) + 1

    crop_shape = max_coords - min_coords

    # Compute padding (25% of crop size in each dimension)
    padding =  np.maximum((crop_shape * 0.25).astype(int), 1)


    # Define new padded shape
    padded_shape = crop_shape + 2 * padding

    # Create new empty array with padding
    center_crop = np.zeros(padded_shape, dtype=np.uint8)

    # Fill center region with actual cropped data
    center_crop[
        padding[0]:padding[0] + crop_shape[0],
        padding[1]:padding[1] + crop_shape[1],
        padding[2]:padding[2] + crop_shape[2]
    ] = error_component[
        min_coords[0]:max_coords[0],
        min_coords[1]:max_coords[1],
        min_coords[2]:max_coords[2]
    ]

    large_roi = False
    if center_crop.shape[0] * center_crop.shape[1] * center_crop.shape[2] > 60000000:
        from skimage.measure import block_reduce
        print(f'ROI too large {center_crop.shape} --> 2x downsampling for EDT')
        center_crop = block_reduce(center_crop, block_size=(2, 2, 2), func=np.max)
        large_roi = True

    # Compute EDT on the padded array
    if torch.cuda.is_available() and not large_roi: # GPU available
        import cupy as cp
        from cucim.core.operations import morphology
        error_mask_cp = cp.array(center_crop)
        edt_cp = morphology.distance_transform_edt(error_mask_cp, return_distances=True)
        edt = cp.asnumpy(edt_cp)
    else: # CPU available only
        edt = distance_transform_edt(center_crop)
    
    if large_roi: # upsample
        edt = edt.repeat(2, axis=0).repeat(2, axis=1).repeat(2, axis=2)

    # Crop out the center (remove padding)
    dist_cropped = edt[
        padding[0]:padding[0] + crop_shape[0],
        padding[1]:padding[1] + crop_shape[1],
        padding[2]:padding[2] + crop_shape[2]
    ]

    # Create full-sized EDT result array and splat back 
    dist_full = np.zeros_like(error_component, dtype=dist_cropped.dtype)
    dist_full[
        min_coords[0]:max_coords[0],
        min_coords[1]:max_coords[1],
        min_coords[2]:max_coords[2]
    ] = dist_cropped

    dist_transformed = dist_full

    return dist_transformed

# -*- encoding: utf-8 -*-
'''
@File    :   infer_with_medim.py
@Time    :   2024/09/08 11:31:02
@Author  :   Haoyu Wang 
@Contact :   small_dark@sina.com
@Brief   :   Example code for inference with MedIM
'''

import medim
import torch
import numpy as np
import torch.nn.functional as F
import torchio as tio
import os.path as osp
import os
from torchio.data.io import sitk_to_nib
import SimpleITK as sitk
from os.path import join
from glob import glob
from collections import defaultdict


def random_sample_next_click(prev_mask, gt_mask):
    """
    Randomly sample one click from ground-truth mask and previous seg mask

    Arguements:
        prev_mask: (torch.Tensor) [H,W,D] previous mask that SAM-Med3D predict
        gt_mask: (torch.Tensor) [H,W,D] ground-truth mask for this image
    """
    prev_mask = prev_mask > 0
    true_masks = gt_mask > 0

    if (not true_masks.any()):
        raise ValueError("Cannot find true value in the ground-truth!")

    fn_masks = torch.logical_and(true_masks, torch.logical_not(prev_mask))
    fp_masks = torch.logical_and(torch.logical_not(true_masks), prev_mask)

    to_point_mask = torch.logical_or(fn_masks, fp_masks)

    all_points = torch.argwhere(to_point_mask)
    point = all_points[np.random.randint(len(all_points))]

    if fn_masks[point[0], point[1], point[2]]:
        is_positive = True
    else:
        is_positive = False

    sampled_point = point.clone().detach().reshape(1, 1, 3)
    sampled_label = torch.tensor([
        int(is_positive),
    ]).reshape(1, 1)

    return sampled_point, sampled_label


def sam_model_infer(model,
                    roi_image,
                    prompt_generator=random_sample_next_click,
                    roi_gt=None,
                    prev_low_res_mask=None):
    '''
    Inference for SAM-Med3D, inputs prompt points with its labels (positive/negative for each points)

    # roi_image: (torch.Tensor) cropped image, shape [1,1,128,128,128]
    # prompt_points_and_labels: (Tuple(torch.Tensor, torch.Tensor))
    '''

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print("using device", device)
    model = model.to(device)
    
    # import pdb; pdb.set_trace()

    with torch.no_grad():
        input_tensor = roi_image.to(device)
        image_embeddings = model.image_encoder(input_tensor)

        points_coords, points_labels = torch.zeros(1, 0,
                                                   3).to(device), torch.zeros(
                                                       1, 0).to(device)
        new_points_co, new_points_la = torch.Tensor(
            [[[64, 64, 64]]]).to(device), torch.Tensor([[1]]).to(torch.int64)
        if (roi_gt is not None):
            prev_low_res_mask = prev_low_res_mask if (
                prev_low_res_mask is not None) else torch.zeros(
                    1, 1, roi_image.shape[2] // 4, roi_image.shape[3] //
                    4, roi_image.shape[4] // 4)
            prev_low_res_mask = F.interpolate(prev_low_res_mask,
                                              size=(roi_image.shape[2] // 4, roi_image.shape[3] // 4, roi_image.shape[4] // 4),
                                              mode='nearest').to(torch.float32)
            new_points_co, new_points_la = prompt_generator(
                torch.zeros_like(roi_image)[0, 0], roi_gt[0, 0])
            new_points_co, new_points_la = new_points_co.to(
                device), new_points_la.to(device)
        points_coords = torch.cat([points_coords, new_points_co], dim=1)
        points_labels = torch.cat([points_labels, new_points_la], dim=1)

        sparse_embeddings, dense_embeddings = model.prompt_encoder(
            points=[points_coords, points_labels],
            boxes=None,  # we currently not support bbox prompt
            masks=prev_low_res_mask.to(device),
            # masks=None,
        )

        low_res_masks, _ = model.mask_decoder(
            image_embeddings=image_embeddings,  # (1, 384, 8, 8, 8)
            image_pe=model.prompt_encoder.get_dense_pe(),  # (1, 384, 8, 8, 8)
            sparse_prompt_embeddings=sparse_embeddings,  # (1, 2, 384)
            dense_prompt_embeddings=dense_embeddings,  # (1, 384, 8, 8, 8)
            multimask_output=False
        )

        prev_mask = F.interpolate(low_res_masks,
                                  size=roi_image.shape[-3:],
                                  mode='trilinear',
                                  align_corners=False)

    # convert prob to mask
    medsam_seg_prob = torch.sigmoid(prev_mask)  # (1, 1, 64, 64, 64)
    medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
    medsam_seg_mask = (medsam_seg_prob > 0.5).astype(np.uint8)

    return medsam_seg_mask


def save_numpy_to_nifti(in_arr: np.array, out_path, meta_info):
    # torchio turn 1xHxWxD -> DxWxH
    # so we need to squeeze and transpose back to HxWxD
    ori_arr = np.transpose(in_arr.squeeze(), (2, 1, 0))
    out = sitk.GetImageFromArray(ori_arr)
    sitk_meta_translator = lambda x: [float(i) for i in x]
    out.SetOrigin(sitk_meta_translator(meta_info["origin"]))
    out.SetDirection(sitk_meta_translator(meta_info["direction"]))
    out.SetSpacing(sitk_meta_translator(meta_info["spacing"]))
    sitk.WriteImage(out, out_path)


def resample_nii(imgs: np.array,
                 gts: np.array,
                 prev_seg: np.array,
                 target_spacing: tuple = (1.5, 1.5, 1.5),
                ):
    """
    Resample a nii.gz file to a specified spacing using torchio.

    Parameters:
    - input_path: Path to the input .nii.gz file.
    - output_path: Path to save the resampled .nii.gz file.
    - target_spacing: Desired spacing for resampling. Default is (1.5, 1.5, 1.5).
    """
    # Load the nii.gz file using torchio
    subject = tio.Subject(
                    image=tio.ScalarImage(tensor=imgs[None]), 
                    label=tio.LabelMap(tensor=gts[None]),
                    prev_seg=tio.LabelMap(tensor=prev_seg[None]),
                    )
    resampler = tio.Resample(target=target_spacing)
    resampled_subject = resampler(subject)
    return resampled_subject


def read_data_from_subject(subject):
    # sitk_image = sitk.ReadImage(img_path)
    # sitk_label = sitk.ReadImage(gt_path)

    # if sitk_image.GetOrigin() != sitk_label.GetOrigin():
    #     sitk_image.SetOrigin(sitk_label.GetOrigin())
    # if sitk_image.GetDirection() != sitk_label.GetDirection():
    #     sitk_image.SetDirection(sitk_label.GetDirection())

    # sitk_image_arr, _ = sitk_to_nib(sitk_image)
    # sitk_label_arr, _ = sitk_to_nib(sitk_label)

    # subject = tio.Subject(
    #     image=tio.ScalarImage(tensor=sitk_image_arr),
    #     label=tio.LabelMap(tensor=sitk_label_arr),
    # )
    # import pdb; pdb.set_trace()
    crop_transform = tio.CropOrPad(mask_name='label',
                                   target_shape=(128, 128, 128))
    padding_params, cropping_params = crop_transform.compute_crop_or_pad(
        subject)
    if (cropping_params is None): cropping_params = (0, 0, 0, 0, 0, 0)
    if (padding_params is None): padding_params = (0, 0, 0, 0, 0, 0)

    infer_transform = tio.Compose([
        crop_transform,
        tio.ZNormalization(masking_method=lambda x: x > 0),
    ])
    subject_roi = infer_transform(subject)

    # import pdb; pdb.set_trace()
    img3D_roi, gt3D_roi = subject_roi.image.data.clone().detach().unsqueeze(
        1), subject_roi.label.data.clone().detach().unsqueeze(1)
    prev_seg3D_roi = subject_roi.prev_seg.data.clone().detach().unsqueeze(1)
    ori_roi_offset = (
        cropping_params[0],
        cropping_params[0] + 128 - padding_params[0] - padding_params[1],
        cropping_params[2],
        cropping_params[2] + 128 - padding_params[2] - padding_params[3],
        cropping_params[4],
        cropping_params[4] + 128 - padding_params[4] - padding_params[5],
    )

    meta_info = {
    #     "image_path": img_path,
    #     "origin": sitk_label.GetOrigin(),
    #     "direction": sitk_label.GetDirection(),
    #     "spacing": sitk_label.GetSpacing(),
        "padding_params": padding_params,
        "cropping_params": cropping_params,
        "ori_roi": ori_roi_offset,
    }
    return (
        img3D_roi,
        gt3D_roi,
        prev_seg3D_roi,
        meta_info,
    )


def data_preprocess(imgs, cls_gt, cls_prev_seg, orig_spacing, category_index):
    subject = resample_nii(imgs, cls_gt, cls_prev_seg, target_spacing=[t/o for o, t in zip(orig_spacing, [1.5, 1.5, 1.5])])
    roi_image, roi_label, roi_prev_seg, meta_info = read_data_from_subject(subject)
    
    meta_info["orig_shape"] = imgs.shape
    meta_info["resampled_shape"] = subject.spatial_shape,
    return roi_image, roi_label, roi_prev_seg, meta_info


def data_postprocess(roi_pred, meta_info, output_dir='outputs'):
    os.makedirs(output_dir, exist_ok=True)
    pred3D_full = np.zeros(*meta_info["resampled_shape"])
    padding_params = meta_info["padding_params"]
    unpadded_pred = roi_pred[padding_params[0] : 128-padding_params[1],
                             padding_params[2] : 128-padding_params[3],
                             padding_params[4] : 128-padding_params[5]]
    ori_roi = meta_info["ori_roi"]
    pred3D_full[ori_roi[0]:ori_roi[1], ori_roi[2]:ori_roi[3],
                ori_roi[4]:ori_roi[5]] = unpadded_pred

    # sitk_image = sitk.ReadImage(ori_img_path)
    # ori_meta_info = {
    #     "image_path": ori_img_path,
    #     "image_shape": sitk_image.GetSize(),
    #     "origin": sitk_image.GetOrigin(),
    #     "direction": sitk_image.GetDirection(),
    #     "spacing": sitk_image.GetSpacing(),
    # }
    pred3D_full_ori = F.interpolate(
        torch.Tensor(pred3D_full)[None][None],
        size=meta_info["orig_shape"],
        mode='nearest').cpu().numpy().squeeze()
    # save_numpy_to_nifti(pred3D_full_ori, output_path, meta_info)
    return pred3D_full_ori


def read_data_from_npz(npz_file):
    data = np.load(npz_file, allow_pickle=True)
    imgs = data.get('imgs', None)
    gts = data.get('gts', None)

    sitk_spacing = data.get('spacing', None)
    # spacing = sitk_spacing
    spacing = [sitk_spacing[2], sitk_spacing[0], sitk_spacing[1]]

    # z-score normalize imgs
    imgs = imgs.astype(np.float32)

    # parsing boxes/clicks tensor, allow category to has more than 1 clicks
    all_clicks = defaultdict(list)
    # get bbox click first
    boxes = data.get('boxes', None)
    if (boxes is not None):
        for cls_idx, bbox in enumerate(boxes):
            all_clicks[cls_idx].append((
            ((bbox['z_min']+bbox['z_max'])/2, (bbox['z_mid_y_min']+bbox['z_mid_y_max'])/2, (bbox['z_mid_x_min']+bbox['z_mid_x_max'])/2,), # center of bbox
            [1], # positive click
            ))
        # all_clicks = [
        #     [
        #     #  ([bbox['z_mid'], (bbox['z_mid_y_min']+bbox['z_mid_y_max'])/2, (bbox['z_mid_x_min']+bbox['z_mid_x_max'])/2,], # center of bbox
        #      ([(bbox['z_min']+bbox['z_max'])/2, (bbox['z_mid_y_min']+bbox['z_mid_y_max'])/2, (bbox['z_mid_x_min']+bbox['z_mid_x_max'])/2,
        #     ], # center of bbox
        #     [1], # positive click
        #     )] for bbox in boxes
        # ] # [[(clickl_cls1, tag), (click2_cls1, tag), ...], []]

    # get point click then
    prev_pred = data.get('prev_pred', np.zeros_like(imgs, dtype=np.uint8))
    clicks = data.get('clicks', None)
    if (clicks is not None):
        for cls_idx, cls_click_dict in enumerate(clicks):
            for click in cls_click_dict['fg']:
            #     all_clicks[cls_idx].append(((click[2], click[1], click[0]), [1]))
                all_clicks[cls_idx].append((click, [1]))
            for click in cls_click_dict['bg']:
                all_clicks[cls_idx].append((click, [0]))
            # if len(cls_click_dict['fg'])<1 :
            #     center = np.unravel_index(np.argmax(prev_pred==(cls_idx+1)), prev_pred.shape)
            #     all_clicks[cls_idx].append((center, [1]))
        # print(all_clicks)

    # import pdb; pdb.set_trace()
    return imgs, spacing, all_clicks, prev_pred


def create_gt_arr(shape, point, category_index, square_size=20):
    # Create an empty array with the same shape as the input array
    gt_array = np.zeros(shape)
    
    # Extract the coordinates of the point
    z, y, x = point
    
    # Calculate the half size of the square
    half_size = square_size // 2
    
    # Calculate the coordinates of the square around the point
    z_min = max(int(z - half_size), 0)
    z_max = min(int(z + half_size) + 1, shape[0])
    y_min = max(int(y - half_size), 0)
    y_max = min(int(y + half_size) + 1, shape[1])
    x_min = max(int(x - half_size), 0)
    x_max = min(int(x + half_size) + 1, shape[2])
    
    # Set the values within the square to 1
    gt_array[z_min:z_max, y_min:y_max, x_min:x_max] = category_index
    
    return gt_array


from segment_anything.build_sam3D import sam_model_registry3D
ckpt_path = "./ckpt/sam_med3d_turbo_cvpr_alldata.pth" # all data
model = sam_model_registry3D["vit_b_ori"](checkpoint=None)
state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
model.load_state_dict(state_dict['model_state_dict'])
model.eval()

def medim_infer_main():
    ''' 2. read and pre-process your input data '''
    npz_file = glob("inputs/*.npz")[0]
    out_dir = "./outputs"
    imgs, spacing, all_clicks, prev_pred = read_data_from_npz(npz_file)
    final_pred = np.zeros_like(imgs, dtype=np.uint8)
    for idx, cls_clicks in all_clicks.items():
        category_index = idx + 1
        # import pdb; pdb.set_trace()
        pred_ori = prev_pred==category_index
        final_pred[pred_ori!=0] = category_index
        if (cls_clicks[-1][1][0] == 1):
            cls_gt = create_gt_arr(imgs.shape, cls_clicks[-1][0], category_index=category_index)
            # print(category_index, imgs.shape, spacing, cls_clicks, (cls_gt==category_index).sum())
            # continue
            cls_prev_seg = prev_pred==category_index
            roi_image, roi_label, roi_prev_seg, meta_info = data_preprocess(imgs, cls_gt, cls_prev_seg,
                                                            orig_spacing=spacing, 
                                                            category_index=category_index)

            # import pdb; pdb.set_trace()
            ''' 3. infer with the pre-trained SAM-Med3D model '''
            
            roi_pred = sam_model_infer(model, roi_image, roi_gt=roi_label, prev_low_res_mask=roi_prev_seg)

            # import pdb; pdb.set_trace()
            ''' 4. post-process and save the result '''
            pred_ori = data_postprocess(roi_pred, meta_info, out_dir)
            final_pred[pred_ori!=0] = category_index

    output_path = osp.join(out_dir, osp.basename(npz_file))
    np.savez_compressed(output_path, segs=final_pred)
    # print("result saved to", output_path)



parser = argparse.ArgumentParser('Segmentation iterative refinement with clicks eavluation for docker containers', add_help=False)
parser.add_argument('-i', '--test_img_path', default='3D_val_npz', type=str, help='testing data path')
parser.add_argument('-o','--save_path', default='./seg', type=str, help='segmentation output path')
parser.add_argument('-d','--docker_folder_path', default='./team_docker', type=str, help='team docker path')
parser.add_argument('-val_gts','--validation_gts_path', default='3D_val_gt_interactive_seg', type=str, help='path to validation set (or final test set) GT files')
parser.add_argument('-v','--verbose', default=False, action='store_true', help="Verbose output, e.g., print coordinates of generated clicks")

args = parser.parse_args()

test_img_path = args.test_img_path
save_path = args.save_path
docker_path = args.docker_folder_path
validation_gts_path = args.validation_gts_path
verbose = args.verbose

if not os.path.exists(validation_gts_path):
    validation_gts_path = None
    print('[WARNING] Validation path does not exist for your GT data! Make sure you supplied the correct path or your .npz inputs have a gts key!')

input_temp = './inputs/'
output_temp = './outputs'
os.makedirs(save_path, exist_ok=True)

dockers = sorted(os.listdir(docker_path))
test_cases = sorted(os.listdir(test_img_path))

for docker in dockers:
    try:
        # create temp folers for inference one-by-one
        if os.path.exists(input_temp):
            shutil.rmtree(input_temp)
        if os.path.exists(output_temp):
            shutil.rmtree(output_temp)
        os.makedirs(input_temp)
        os.makedirs(output_temp)

        # load docker and create a new folder to save segmentation results
        teamname = docker.split('.')[0].lower()
        print('teamname docker: ', docker)
        # os.system('docker image load -i {}'.format(join(docker_path, docker)))
        team_outpath = join(save_path, teamname)
        # if os.path.exists(team_outpath):
        #     shutil.rmtree(team_outpath)
        # os.makedirs(team_outpath)
        # os.system(f'chmod -R 777 ./* >/dev/null 2>&1') # ignore output warnings/errors of this command with >/dev/null 2>&1
        
        # Evaluation Metrics
        metric = OrderedDict()
        metric['CaseName'] = []
        # 5 Metrics
        metric['TotalRunningTime'] = []
        metric['RunningTime_1'] = []
        metric['RunningTime_2'] = []
        metric['RunningTime_3'] = []
        metric['RunningTime_4'] = []
        metric['RunningTime_5'] = []
        metric['RunningTime_6'] = []
        metric['DSC_AUC'] = []
        metric['NSD_AUC'] = []
        metric['DSC_Final'] = []
        metric['NSD_Final'] = []
        metric['DSC_1'] = []
        metric['DSC_2'] = []    
        metric['DSC_3'] = []
        metric['DSC_4'] = []
        metric['DSC_5'] = []
        metric['DSC_6'] = []
        metric['NSD_1'] = []
        metric['NSD_2'] = []
        metric['NSD_3'] = []
        metric['NSD_4'] = []
        metric['NSD_5'] = []
        metric['NSD_6'] = []
        metric['num_class'] = []
        metric['runtime_upperbound'] = []
        n_clicks = 5
        time_warning = False

        # To obtain the running time for each case, testing cases are inferred one-by-one
        for case in tqdm(test_cases):
            if os.path.exists(join(team_outpath, case)):
                print(f'Skipping {case} as it already exists in {input_temp} or {output_temp}')
                continue
            metric_temp = {}
            real_running_time = 0
            dscs = []
            nsds = []
            all_segs = []
            no_bbox = False

            # copy input image to accumulate clicks in its dict
            shutil.copy(join(test_img_path, case), input_temp)
            if validation_gts_path is  None: # for training images
                gts = patched_np_load(join(input_temp, case), allow_pickle=True)['gts']
            else: # for validation or test images --> gts are in separate files to avoid label leakage during the course of the challenge
                gts = patched_np_load(join(validation_gts_path, case), allow_pickle=True)['gts']
                
            unique_gts = np.sort(pd.unique(gts.ravel()))
            num_classes = len(unique_gts) - 1
            metric_temp['num_class'] = num_classes
            metric_temp['runtime_upperbound'] = num_classes * 90


            # foreground and background clicks for each class
            clicks_cls = [{'fg': [], 'bg': []} for _ in unique_gts[1:]] # skip background class 0 
            clicks_order = [[] for _ in unique_gts[1:]]
            if "boxes" in patched_np_load(join(input_temp, case), allow_pickle=True).keys():
                boxes = patched_np_load(join(input_temp, case), allow_pickle=True)['boxes']
            

            for it in range(n_clicks + 1): # + 1 due to bbox pred at iteration 0
                if it == 0:
                    if "boxes" not in patched_np_load(join(input_temp, case), allow_pickle=True).keys():
                        if verbose:
                            print(f'This sample does not use a Bounding Box for the initial iteration {it}') 
                        no_bbox = True
                        metric_temp["RunningTime_1"] = 0
                        metric_temp["DSC_1"] = 0
                        metric_temp["NSD_1"] = 0
                        dscs.append(0)
                        nsds.append(0)
                        continue
                    if verbose:
                        print(f'Using Bounding Box for iteration {it}') 
                else:
                    if verbose:
                        print(f'Using Clicks for iteration {it}')
                    if os.path.isfile(join(output_temp, case)):
                        segs = patched_np_load(join(output_temp, case), allow_pickle=True)['segs'].astype(np.uint8) # previous prediction
                    else:
                        segs = np.zeros_like(gts).astype(np.uint8) # in case the bbox prediction did not produce a result
                    all_segs.append(segs.astype(np.uint8))

                    # Refinement clicks
                    for ind, cls in enumerate(sorted(unique_gts[1:])):
                        if cls == 0:
                            continue # skip background

                        segs_cls = (segs == cls).astype(np.uint8)
                        gts_cls = (gts == cls).astype(np.uint8)

                        # Compute error mask
                        error_mask = (segs_cls != gts_cls).astype(np.uint8)
                        if np.sum(error_mask) > 0:
                            errors = cc3d.connected_components(error_mask, connectivity=26)  # 26 for 3D connectivity

                            # Calculate the sizes of connected error components
                            component_sizes = np.bincount(errors.flat)

                            # Ignore non-error regions 
                            component_sizes[0] = 0

                            # Find the largest error component
                            largest_component_error = np.argmax(component_sizes)

                            # Find the voxel coordinates of the largest error component
                            largest_component = (errors == largest_component_error)

                            edt = compute_edt(largest_component)
                            edt *= largest_component # make sure correct voxels have a distance of 0
                            if np.sum(edt) == 0: # no valid voxels to sample
                                if verbose:
                                    print("Error is extremely small --> Sampling uniformly instead of using EDT")
                                edt = largest_component # in case EDT is empty (due to artifacts in resizing, simply sample a random voxel from the component), happens only for extremely small errors

                            center = sample_coord(edt)

                            if gts_cls[center] == 0: # oversegmentation -> place background click
                                assert segs_cls[center] == 1
                                clicks_cls[ind]['bg'].append(list(center))
                                clicks_order[ind].append('bg')
                            else: # undersegmentation -> place foreground click
                                assert segs_cls[center] == 0
                                clicks_cls[ind]['fg'].append(list(center))
                                clicks_order[ind].append('fg')

                            assert largest_component[center] # click within error

                            if verbose:
                                print(f"Class {cls}: Largest error component center is at {center}")
                        else:
                            clicks_order[ind].append(None)
                            if verbose:
                                print(f"Class {cls}: No error connected components found. Prediction is perfect! No clicks were added.")
                    
                    # update model input with new click
                    input_img = patched_np_load(join(input_temp, case), allow_pickle=True)

                    if validation_gts_path is None:
                        if no_bbox:
                            np.savez_compressed(
                                join(input_temp, case),
                                imgs=input_img['imgs'],
                                gts=input_img['gts'], # only for training images
                                spacing=input_img['spacing'],
                                clicks=clicks_cls,
                                clicks_order=clicks_order, 
                                prev_pred=segs,
                            ) 
                        else:
                            np.savez_compressed(
                                join(input_temp, case),
                                imgs=input_img['imgs'],
                                gts=input_img['gts'], # only for training images
                                spacing=input_img['spacing'],
                                clicks=clicks_cls, 
                                clicks_order=clicks_order, 
                                prev_pred=segs,
                                boxes=boxes,
                            ) 
                    else:
                        if no_bbox:
                            np.savez_compressed(
                                join(input_temp, case),
                                imgs=input_img['imgs'],
                                spacing=input_img['spacing'],
                                clicks=clicks_cls, 
                                clicks_order=clicks_order, 
                                prev_pred=segs,
                            ) 
                        else:
                            np.savez_compressed(
                                join(input_temp, case),
                                imgs=input_img['imgs'],
                                spacing=input_img['spacing'],
                                clicks=clicks_cls, 
                                clicks_order=clicks_order, 
                                prev_pred=segs,
                                boxes=boxes,
                            ) 

                # Model inference on the current input
                # if torch.cuda.is_available(): # GPU available
                #     cmd = 'docker container run --gpus "device=0" -m 16G --name {} --rm -v $PWD/inputs/:/workspace/inputs/ -v $PWD/outputs/:/workspace/outputs/ blueyo0/{}:latest /bin/bash -c "sh predict.sh" '.format(teamname.replace('/', '_'), teamname)
                # else:
                #     cmd = 'docker container run -m 32G --name {} --rm -v $PWD/inputs/:/workspace/inputs/ -v $PWD/outputs/:/workspace/outputs/ {}:latest /bin/bash -c "sh predict.sh" '.format(teamname.replace('/', '_'), teamname)
                
                # cmd = "python medim_infer.py"
                if verbose:
                    print(teamname, ' docker command:', cmd, '\n', 'testing image name:', case)
                start_time = time.time()
                # os.system(cmd)
                medim_infer_main()
                infer_time = time.time() - start_time
                real_running_time += infer_time # only add the inference time without the click generation time
                print(f"{case} finished! Inference time: {infer_time}")
                metric_temp[f"RunningTime_{it + 1}"] = infer_time

                if not os.path.isfile(join(output_temp, case)):
                    print(f"[WARNING] Failed / Skipped prediction for iteration {it}! Setting prediction to zeros...")
                    segs = np.zeros_like(gts).astype(np.uint8)
                else:
                    segs = patched_np_load(join(output_temp, case), allow_pickle=True)['segs']
                all_segs.append(segs.astype(np.uint8))

                dsc = compute_multi_class_dsc(gts, segs)
                # compute nsd
                if dsc > 0.2:
                    # only compute nsd when dice > 0.2 because NSD is also low when dice is too low
                    try:
                        nsd = compute_multi_class_nsd(gts, segs, patched_np_load(join(input_temp, case), allow_pickle=True)['spacing'])
                    except Exception as e:
                        nsd = 0.0 # assume model performs poor on this sample
                else:
                    nsd = 0.0 # Assume model performs poor on this sample
                dscs.append(dsc)
                nsds.append(nsd)
                metric_temp[f'DSC_{it + 1}'] = dsc
                metric_temp[f'NSD_{it + 1}'] = nsd
                print('Dice', dsc, 'NSD', nsd)
                seg_name = case


                # Copy temp prediction to the final folder
                try:
                    shutil.copy(join(output_temp, seg_name), join(team_outpath, seg_name))
                    segs = patched_np_load(join(team_outpath, seg_name), allow_pickle=True)['segs']
                    np.savez_compressed(
                        join(team_outpath, seg_name),
                        segs=segs,
                        all_segs=all_segs, # store all intermediate predictions
                    ) 
                except:
                    print(f"{join(output_temp, seg_name)}, {join(team_outpath, seg_name)}")
                    if os.path.exists(join(team_outpath, seg_name)):
                        os.remove(team_outpath, seg_name) # clean up cached files if model has failed
                    print("Final prediction could not be copied!")
            

            if real_running_time > 90 * (len(unique_gts) - 1):
                print("[WARNING] Your model seems to take more than 90 seconds per class during inference! The final test set will have a time constraint of 90s per class --> Make sure to optimize your approach!")
                time_warning = True
            # Compute interactive metrics
            dsc_auc = integrate.cumulative_trapezoid(np.array(dscs[-n_clicks:]), np.arange(n_clicks))[-1] # AUC is only over the point prompts since the bbox prompt is optional
            nsd_auc = integrate.cumulative_trapezoid(np.array(nsds[-n_clicks:]), np.arange(n_clicks))[-1] 
            dsc_final = dscs[-1]
            nsd_final = nsds[-1]
            if os.path.exists(join(team_outpath, seg_name)): # add to csv only if final prediction is successful
                for k, v in metric_temp.items():
                    metric[k].append(v)
                metric['CaseName'].append(case)
                metric['TotalRunningTime'].append(real_running_time)
                metric['DSC_AUC'].append(dsc_auc)
                metric['NSD_AUC'].append(nsd_auc)
                metric['DSC_Final'].append(dsc_final)
                metric['NSD_Final'].append(nsd_final)
            os.remove(join(input_temp, case))  

            metric_df = pd.DataFrame(metric)
            metric_df.to_csv(join(team_outpath, teamname + '_metrics.csv'), index=False)

        # Clean up for next docker
        torch.cuda.empty_cache()
        # os.system("docker rmi {}:latest".format(teamname.split('_')[0]))
        shutil.rmtree(input_temp)
        shutil.rmtree(output_temp)
        if time_warning: # repeat warning at the end as well
            print("[WARNING] Your model seems to take more than 90 seconds per class during inference for some images! The final test set will have a time constraint of 90s per class --> Make sure to optimize your approach!")
    except Exception as e:
        print(e)
        traceback.print_exc()
        print(f"Error processing {case} with docker {docker}. Skipping this docker.")
