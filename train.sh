gpu_ids=$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ',')
gpu_ids=${gpu_ids%?}  # 
# gpu_ids="0" # only use one gpu for debug
gpu_num=$(echo $gpu_ids | tr ',' '\n' | wc -l)


lr=1e-4
wd=1e-4
exp_id="main_train"
CUDA_VISIBLE_DEVICES=$gpu_ids \
python3 -m torch.distributed.launch \
--master_port 29521 \
--nproc_per_node $gpu_num \
train.py \
--cfg ./assets/yaml/main_train.yml \
--distributed \
--amp \
--lr $lr \
--lr_backbone $lr \
--weight_decay $wd \
--exp_id $exp_id



lr=5e-5
wd=1e-3
exp_id="fine_tune"
CUDA_VISIBLE_DEVICES=$gpu_ids \
python3 -m torch.distributed.launch \
--master_port 29521 \
--nproc_per_node $gpu_num \
train.py \
--cfg ./assets/yaml/fine_tune.yml \
--distributed \
--amp \
--with_contrastive \
--total_steps 10000 \
--lr $lr \
--lr_backbone $lr \
--weight_decay $wd \
--exp_id $exp_id \
--resume_ckpt JOTR/output/JOTR/checkpoint/snapshot_17.pth.tar
