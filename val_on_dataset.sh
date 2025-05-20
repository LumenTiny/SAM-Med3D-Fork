python medim_val_dataset.py

python utils/compute_dataset_metrics.py \
    --gt_dir ./data/ct_AMOS/labelsVal \
    --pred_dir ./data/ct_AMOS/pred_sammed3d

python utils/compute_dataset_metrics.py \
    --gt_dir ./data/ct_TotalSeg/labelsTs \
    --pred_dir ./data/ct_TotalSeg/pred_sammed3d