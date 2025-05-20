# -*- encoding: utf-8 -*-

import os.path as osp
from glob import glob

import medim
from tqdm import tqdm

from utils.infer_utils import validate_paired_img_gt
from segment_anything.build_sam3D import sam_model_registry3D
import torch

if __name__ == "__main__":
    ''' prepare the pre-trained model with local path or huggingface url '''
    # ckpt_path = "https://huggingface.co/blueyo0/SAM-Med3D/blob/main/sam_med3d_turbo.pth"
    # or you can use a local path like:
    # ckpt_path = "./ckpt/sam_med3d_turbo.pth"

    test_data_list = [
        dict(
            img_dir="./data/ct_AMOS/imagesVal",
            gt_dir="./data/ct_AMOS/labelsVal",
            out_dir="./data/ct_AMOS/pred_ablation_base3d",
            model_type = 'vit_b_ori',
            ckpt_path = "./ckpt/base3d_dice_best.pth",
        ),
        dict(
            img_dir="./data/ct_AMOS/imagesVal",
            gt_dir="./data/ct_AMOS/labelsVal",
            out_dir="./data/ct_AMOS/pred_ablation_pe2d",
            model_type = 'vit_b_pe2d',
            ckpt_path = "./ckpt/pe2d_dice_best.pth",
        ),
        dict(
            img_dir="./data/ct_AMOS/imagesVal",
            gt_dir="./data/ct_AMOS/labelsVal",
            out_dir="./data/ct_AMOS/pred_ablation_att2d",
            model_type = 'vit_b_att2d',
            ckpt_path = "./ckpt/att2d_dice_best.pth",
        ),
        dict(
            img_dir="./data/ct_AMOS/imagesVal",
            gt_dir="./data/ct_AMOS/labelsVal",
            out_dir="./data/ct_AMOS/pred_ablation_pe2d_att2d",
            model_type = 'vit_b_pe2d_att2d',
            ckpt_path = "./ckpt/pe2d_att2d_dice_best.pth",
        ),
    ]
    for test_data in test_data_list:
        model = sam_model_registry3D[test_data['model_type']](checkpoint=None)
        state_dict = torch.load(test_data['ckpt_path'], map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict['model_state_dict'])

        gt_fname_list = sorted(glob(osp.join(test_data["gt_dir"], "*.nii.gz")))
        for gt_fname in tqdm(gt_fname_list):
            case_name = osp.basename(gt_fname).replace(".nii.gz", "")
            img_path = osp.join(test_data["img_dir"], f"{case_name}.nii.gz")
            gt_path = gt_fname
            out_path = osp.join(test_data["out_dir"], f"{case_name}.nii.gz")
            validate_paired_img_gt(model, img_path, gt_path, out_path, num_clicks=1)
