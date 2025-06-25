FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

WORKDIR /workspace

COPY segment_anything ./segment_anything/
COPY utils ./utils/
COPY medim_infer.py .
COPY predict.sh .

COPY ckpt/sam_med3d_turbo_cvpr_alldata.pth ./ckpt/sam_med3d_turbo_cvpr_alldata.pth

RUN chmod +x predict.sh
RUN pip install --no-cache-dir opencv-python-headless matplotlib monai torchio SimpleITK -i https://pypi.tuna.tsinghua.edu.cn/simple

# COPY custom_MedIM ./custom_MedIM_src/
# RUN pip install --no-cache-dir ./custom_MedIM_src/