docker build -t sammed3d_baseline_cu128_alldata:latest . 
docker save sammed3d_baseline_cu128_alldata:latest | gzip -c > team_docker/sammed3d_baseline_cu128_alldata.tar.gz 
python CVPR25_iter_eval.py