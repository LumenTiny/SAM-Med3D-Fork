#!/bin/bash

# Script to run the Python inference script with different configurations
# and then compute dataset metrics.

# Python script for inference
PYTHON_SCRIPT_NAME="local_val_dataset.py" 
# Utility script for metrics
METRICS_SCRIPT_NAME="utils/compute_dataset_metrics.py"

# Common directories
COMMON_IMG_DIR="./data/ct_AMOS/imagesVal"
COMMON_GT_DIR="./data/ct_AMOS/labelsVal"
COMMON_PRED_BASE_DIR="./data/ct_AMOS" # Base for prediction output directories
COMMON_CKPT_DIR="./ckpt"

# --- Configurations ---
# These arrays define the parameters for each configuration.
# Ensure the arrays have the same number of elements and correspond to each other.

# Suffix for prediction directory and log file, also used for display
config_suffixes=("base3d" "pe2d" "att2d" "pe2d_att2d")

# Model types for the inference script
model_types=("vit_b_ori" "vit_b_pe2d" "vit_b_att2d" "vit_b_pe2d_att2d")

# Checkpoint file name stems (without .pth extension)
ckpt_stems=("base3d_dice_best" "pe2d_dice_best" "att2d_dice_best" "pe2d_att2d_dice_best")

num_clicks=5

# --- Run Inference for all configurations ---
echo "Starting inference for all configurations..."

for i in "${!config_suffixes[@]}"; do
    config_suffix="${config_suffixes[i]}"
    model_type="${model_types[i]}"
    ckpt_stem="${ckpt_stems[i]}"

    # Construct paths for the current configuration
    current_pred_out_dir="${COMMON_PRED_BASE_DIR}/c${num_clicks}_pred_ablation_${config_suffix}"
    current_ckpt_path="${COMMON_CKPT_DIR}/${ckpt_stem}.pth"

    echo "" # Newline for readability
    echo "---------------------------------------------------------------------"
    echo "Running Inference: Configuration $((i+1)) - ${config_suffix}"
    echo "Model Type: ${model_type}"
    echo "Checkpoint: ${current_ckpt_path}"
    echo "Output Dir: ${current_pred_out_dir}"
    echo "---------------------------------------------------------------------"

    # Create output directory for predictions if it doesn't exist
    mkdir -p "${current_pred_out_dir}"

    python "${PYTHON_SCRIPT_NAME}" \
        --img_dir "${COMMON_IMG_DIR}" \
        --gt_dir "${COMMON_GT_DIR}" \
        --out_dir "${current_pred_out_dir}" \
        --model_type "${model_type}" \
        --ckpt_path "${current_ckpt_path}" \
        --num_clicks "${num_clicks}"
    
    if [ $? -ne 0 ]; then
        echo "Error during inference for configuration ${config_suffix}. Exiting."
        exit 1
    fi
done

echo ""
echo "All inference configurations processed successfully."
echo "====================================================================="


# --- Compute Dataset Metrics for all configurations ---
METRICS_RESULTS_DIR="data/c${num_clicks}_results" # Directory to store metrics log files

echo "Starting metrics calculation..."
echo "Metrics results will be saved in: ${METRICS_RESULTS_DIR}"

# Create the base directory for metrics results if it doesn't exist
mkdir -p "${METRICS_RESULTS_DIR}"

for i in "${!config_suffixes[@]}"; do
    config_suffix="${config_suffixes[i]}"

    # Construct path to the prediction directory for the current configuration
    current_pred_dir_for_metrics="${COMMON_PRED_BASE_DIR}/c${num_clicks}_pred_ablation_${config_suffix}"
    # Construct log file path
    log_file="${METRICS_RESULTS_DIR}/metrics_pred_ablation_${config_suffix}.log"

    echo "" # Newline for readability
    echo "---------------------------------------------------------------------"
    echo "Calculating Metrics for: ${config_suffix}"
    echo "Prediction Dir: ${current_pred_dir_for_metrics}"
    echo "Log File: ${log_file}"
    echo "---------------------------------------------------------------------"

    python "${METRICS_SCRIPT_NAME}" \
        --gt_dir "${COMMON_GT_DIR}" \
        --pred_dir "${current_pred_dir_for_metrics}" \
        > "${log_file}"
    
    if [ $? -ne 0 ]; then
        echo "Error during metrics calculation for ${config_suffix}. Check log: ${log_file}"
        # Decide if you want to exit or continue with other metrics
        # exit 1 
    else
        echo "Metrics for ${config_suffix} saved to ${log_file}"
    fi
done

echo ""
echo "All metrics calculations processed."
echo "====================================================================="
echo "Script finished."
