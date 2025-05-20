# -*- encoding: utf-8 -*-

from utils.infer_utils import validate_paired_img_gt
from utils.metric_utils import compute_metrics, print_computed_metrics
from segment_anything.build_sam3D import sam_model_registry3D
import torch

if __name__ == "__main__":
    ''' 1. prepare the pre-trained model with local path or huggingface url '''
    # model_type = 'vit_b_ori'
    # ckpt_path = "./ckpt/base3d_dice_best.pth"
    model_type = 'vit_b_pe2d'
    ckpt_path = "./ckpt/pe2d_dice_best.pth"
    model_type = 'vit_b_att2d'
    ckpt_path = "./ckpt/att2d_dice_best.pth"
    model_type = 'vit_b_pe2d_att2d'
    ckpt_path = "./ckpt/pe2d_att2d_dice_best.pth"

    model = sam_model_registry3D[model_type](checkpoint=None)
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict['model_state_dict'])

    ''' 2. read and pre-process your input data '''
    img_path = "./test_data/amos_val_toy_data/imagesVa/amos_0013.nii.gz"
    gt_path = "./test_data/amos_val_toy_data/labelsVa/amos_0013.nii.gz"
    out_path = "./test_data/amos_val_toy_data/pred/amos_0013.nii.gz"
    
    ''' 3. infer with the pre-trained SAM-Med3D model '''
    print("Validation start! plz wait for some times.")
    validate_paired_img_gt(model, img_path, gt_path, out_path, num_clicks=1)
    print("Validation finish! plz check your prediction.")

    ''' 4. compute the metrics of your prediction with the ground truth '''
    metrics = compute_metrics(
        gt_path=gt_path,
        pred_path=out_path,
        metrics=['dice'],
        classes=None,
    )
    print_computed_metrics(metrics)
