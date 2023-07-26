gpu_ids=$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ',')
gpu_ids=${gpu_ids%?}  # 
# gpu_ids="0" # only use one gpu for debug
gpu_num=$(echo $gpu_ids | tr ',' '\n' | wc -l)


exp_id="main_train"
CUDA_VISIBLE_DEVICES=$gpu_ids \
python3 -m torch.distributed.launch \
--master_port 29521 \
--nproc_per_node $gpu_num \
eval.py \
--cfg ./assets/yaml/main_train.yml \
--distributed \