# -*- encoding: utf-8 -*-

import os.path as osp
from glob import glob

import medim
from tqdm import tqdm

from utils.infer_utils import validate_paired_img_gt
from segment_anything.build_sam3D import sam_model_registry3D
import torch
import argparse
import os

def validate_on_dataset(args):
    model = sam_model_registry3D[args.model_type](checkpoint=None)
    state_dict = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict['model_state_dict'])

    gt_fname_list = sorted(glob(osp.join(args.gt_dir, "*.nii.gz")))
    for gt_fname in tqdm(gt_fname_list):
        case_name = osp.basename(gt_fname).replace(".nii.gz", "")
        img_path = osp.join(args.img_dir, f"{case_name}.nii.gz")
        gt_path = gt_fname
        out_path = osp.join(args.out_dir, f"{case_name}.nii.gz")
        if (os.path.exists(out_path)): continue
        validate_paired_img_gt(model, img_path, gt_path, out_path, num_clicks=args.num_clicks)

if __name__ == "__main__":
    ''' prepare the pre-trained model with local path or huggingface url '''

    parser = argparse.ArgumentParser(description="Run 3D medical image segmentation inference.")
    
    # Define command-line arguments
    parser.add_argument('--img_dir', type=str, required=True, help='Directory containing input images (e.g., ./data/ct_AMOS/imagesVal)')
    parser.add_argument('--gt_dir', type=str, required=True, help='Directory containing ground truth labels (e.g., ./data/ct_AMOS/labelsVal)')
    parser.add_argument('--out_dir', type=str, required=True, help='Directory to save prediction outputs (e.g., ./data/ct_AMOS/pred_ablation_base3d)')
    parser.add_argument('--model_type', type=str, required=True, help='Type of the SAM model (e.g., vit_b_ori)')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to the model checkpoint file (e.g., ./ckpt/base3d_dice_best.pth)')
    
    # Add any other parameters you might need, e.g., num_clicks if it should be variable
    parser.add_argument('--num_clicks', type=int, default=1, help='Number of clicks for segmentation guidance')

    args = parser.parse_args()
  
    validate_on_dataset(args)
