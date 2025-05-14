import os
import glob
import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm
from SurfaceDice import compute_surface_distances, compute_surface_dice_at_tolerance, compute_dice_coefficient
import concurrent.futures
import json
import datetime

dataset = './data/ct_AMOS'
gt_folder = os.path.join(dataset, 'labelsVal')  # GT 文件夹路径

# 多个预测文件夹
# prediction_folders = ['ACDC_test_nnUNet', 'ACDC_test_SegVol/text_prompt1/', "ACDC_test_SegVol/text_box_prompt", 'ACDC_test_SAT', 'ACDC_test_MedSAM2']
prediction_folders = ['ct_AMOS_pred']

class_list = list(range(1, 15+1))  # 类别 1 到 3

# Initialize dictionaries to store DSC and NSD values for each class and each folder
dsc_results = {folder: {i: [] for i in class_list} for folder in prediction_folders}
nsd_results = {folder: {i: [] for i in class_list} for folder in prediction_folders}
processed_files = {folder: [] for folder in prediction_folders}
errors = []

def process_file(prediction_file, prediction_folder):
    try:
        filename = os.path.basename(prediction_file)
        case_id = os.path.splitext(filename)[0].split('.')[0]  # 提取样本ID
        gt_file = os.path.join(gt_folder, filename)

        if not os.path.exists(gt_file):
            print(f'No GT found for {filename}')
            errors.append(f"Missing GT: {filename}")
            return prediction_folder, None, None, None
        
        # 读取预测和GT文件
        prediction = nib.load(prediction_file).get_fdata().astype(np.uint8)
        gt = nib.load(gt_file).get_fdata().astype(np.uint8)

        gt_nii = nib.load(gt_file)
        case_spacing = gt_nii.header.get_zooms()

        DSC_values = {}
        NSD_values = {}

        for i in class_list:
            if np.sum(gt == i) == 0 and np.sum(prediction == i) == 0:
                DSC_i = np.nan
                NSD_i = np.nan
            elif np.sum(gt == i) == 0 and np.sum(prediction == i) > 0:
                DSC_i = 0
                NSD_i = 0
            else:
                organ_i_gt, organ_i_seg = gt == i, prediction == i         
                surface_distances = compute_surface_distances(organ_i_gt, organ_i_seg, case_spacing)
                DSC_i = compute_dice_coefficient(organ_i_gt, organ_i_seg)
                NSD_i = compute_surface_dice_at_tolerance(surface_distances, 1)

            DSC_values[i] = DSC_i
            NSD_values[i] = NSD_i

        return prediction_folder, case_id, DSC_values, NSD_values
    except Exception as e:
        error_msg = f"Error processing {prediction_file}: {str(e)}"
        print(error_msg)
        errors.append(error_msg)
        return prediction_folder, None, None, None

# 获取所有预测结果文件夹中的预测文件
prediction_files = {folder: glob.glob(os.path.join(dataset, folder, '*.nii.gz')) for folder in prediction_folders}

# 使用 ThreadPoolExecutor 进行并行处理
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    futures = []
    for prediction_folder, files in prediction_files.items():
        for prediction_file in files:
            futures.append(executor.submit(process_file, prediction_file, prediction_folder))

    # 逐个获取处理结果
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc='Processing files', unit='file'):
        result = future.result()
        if result is None:
            continue
            
        prediction_folder, case_id, DSC_values, NSD_values = result
        if case_id is None:
            continue
            
        processed_files[prediction_folder].append(case_id)
        
        for i in class_list:
            dsc_results[prediction_folder][i].append(DSC_values[i])
            nsd_results[prediction_folder][i].append(NSD_values[i])

# 计算每个文件夹中的每个类别的平均值和标准差
dsc_mean = {folder: {i: np.nanmean(dsc_results[folder][i]) for i in class_list} for folder in prediction_folders}
dsc_std = {folder: {i: np.nanstd(dsc_results[folder][i]) for i in class_list} for folder in prediction_folders}

nsd_mean = {folder: {i: np.nanmean(nsd_results[folder][i]) for i in class_list} for folder in prediction_folders}
nsd_std = {folder: {i: np.nanstd(nsd_results[folder][i]) for i in class_list} for folder in prediction_folders}

# 计算每个方法的整体平均值和标准差
# 首先将每个方法所有类别的所有值合并到一个列表中
dsc_all_values = {folder: [] for folder in prediction_folders}
nsd_all_values = {folder: [] for folder in prediction_folders}

for folder in prediction_folders:
    for i in class_list:
        dsc_all_values[folder].extend([v for v in dsc_results[folder][i] if not np.isnan(v)])
        nsd_all_values[folder].extend([v for v in nsd_results[folder][i] if not np.isnan(v)])

# 计算总体平均值和标准差
dsc_overall_mean = {folder: np.mean(dsc_all_values[folder]) for folder in prediction_folders}
dsc_overall_std = {folder: np.std(dsc_all_values[folder]) for folder in prediction_folders}

nsd_overall_mean = {folder: np.mean(nsd_all_values[folder]) for folder in prediction_folders}
nsd_overall_std = {folder: np.std(nsd_all_values[folder]) for folder in prediction_folders}

# 合并结果为一个字典
results_summary = {
    "metadata": {
        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "prediction_folders": prediction_folders,
        "class_list": class_list,
        "processed_files": processed_files,
        "errors": errors
    },
    "per_class_results": {
        "DSC": dsc_results,
        "NSD": nsd_results
    },
    "average_results": {
        "DSC_mean": dsc_mean,
        "DSC_std": dsc_std,
        "NSD_mean": nsd_mean,
        "NSD_std": nsd_std
    },
    "overall_results": {
        "DSC_overall_mean": dsc_overall_mean,
        "DSC_overall_std": dsc_overall_std,
        "NSD_overall_mean": nsd_overall_mean,
        "NSD_overall_std": nsd_overall_std
    }
}

# Save the results to a JSON file
json_output_path = os.path.join(dataset, 'ACDC_results_summary.json')
with open(json_output_path, 'w') as json_file:
    json.dump(results_summary, json_file, indent=4)

print(f"Results saved to {json_output_path}")
print(f"Processed {sum(len(files) for files in processed_files.values())} files successfully")
if errors:
    print(f"Encountered {len(errors)} errors during processing")

# 打印每个方法的总体DSC和NSD值（mean±std格式）
print("\n====== 总体评估结果 ======")
for folder in prediction_folders:
    short_name = folder.split('/')[-1] if '/' in folder else folder
    print(f"{short_name}:")
    print(f"  DSC: {dsc_overall_mean[folder]:.4f}±{dsc_overall_std[folder]:.4f}")
    print(f"  NSD: {nsd_overall_mean[folder]:.4f}±{nsd_overall_std[folder]:.4f}")
print("=========================")
